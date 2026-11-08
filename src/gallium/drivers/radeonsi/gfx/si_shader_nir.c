/*
 * Copyright 2017 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"
#include "nir.h"
#include "si_pipe.h"
#include "si_shader_internal.h"

void si_nir_opts(struct si_screen *sscreen, struct nir_shader *nir, bool has_array_temps)
{
   bool progress;

   do {
      progress = false;
      bool lower_alu_to_scalar = false;
      bool lower_phis_to_scalar = false;

      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_lower_alu_to_scalar, nir->options->lower_to_scalar_filter, NULL);
      NIR_PASS(progress, nir, nir_lower_phis_to_scalar, NULL, NULL);

      if (has_array_temps) {
         NIR_PASS(progress, nir, nir_split_array_vars, nir_var_function_temp);
         NIR_PASS(lower_alu_to_scalar, nir, nir_shrink_vec_array_vars, nir_var_function_temp);
         NIR_PASS(progress, nir, nir_opt_find_array_copies);
      }
      NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
      NIR_PASS(progress, nir, nir_opt_dead_write_vars);

      NIR_PASS(lower_alu_to_scalar, nir, nir_opt_loop);
      /* (Constant) copy propagation is needed for txf with offsets. */
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      /* nir_opt_if_optimize_phi_true_false is disabled on LLVM14 (#6976) */
      NIR_PASS(lower_phis_to_scalar, nir, nir_opt_if,
               nir_opt_if_optimize_phi_true_false);
      NIR_PASS(progress, nir, nir_opt_dead_cf);

      if (lower_alu_to_scalar) {
         NIR_PASS(_, nir, nir_lower_alu_to_scalar, nir->options->lower_to_scalar_filter, NULL);
      }
      if (lower_phis_to_scalar)
         NIR_PASS(_, nir, nir_lower_phis_to_scalar, NULL, NULL);
      progress |= lower_alu_to_scalar | lower_phis_to_scalar;

      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 8,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_select_options);

      /* Needed for algebraic lowering */
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_generate_bfi);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      if (!nir->info.flrp_lowered) {
         unsigned lower_flrp = (nir->options->lower_flrp16 ? 16 : 0) |
                               (nir->options->lower_flrp32 ? 32 : 0) |
                               (nir->options->lower_flrp64 ? 64 : 0);
         assert(lower_flrp);
         NIR_PASS(progress, nir, nir_lower_flrp, lower_flrp, false /* always_precise */);

         /* Nothing should rematerialize any flrps, so we only
          * need to do this lowering once.
          */
         nir->info.flrp_lowered = true;
      }

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_shrink_vectors, true);

      nir_opt_peephole_select_options peephole_discard_options = {
         .limit = 0,
         .discard_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_discard_options);
      if (nir->options->max_unroll_iterations) {
         NIR_PASS(progress, nir, nir_opt_loop_unroll);
      }

      if (nir->info.stage == MESA_SHADER_FRAGMENT)
         NIR_PASS(_, nir, nir_opt_move_discards_to_top);
   } while (progress);

   NIR_PASS(_, nir, nir_lower_var_copies);
}

void si_nir_late_opts(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_VERTEX && nir->info.stage != MESA_SHADER_GEOMETRY)
      NIR_PASS(_, nir, nir_opt_reassociate_for_fma);

   bool more_late_algebraic = true;
   while (more_late_algebraic) {
      more_late_algebraic = false;
      NIR_PASS(more_late_algebraic, nir, nir_opt_algebraic_late);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_opt_cse);
   }
}

/* Buffer objects expose native CPU scalars, unlike internal GPU scratch. */
static nir_def *si_nir_bswap_scalar(nir_builder *b, nir_def *value)
{
   if (value->bit_size == 8)
      return value;
   if (value->bit_size == 16)
      return nir_ior(b, nir_ushr_imm(b, value, 8), nir_ishl_imm(b, value, 8));
   if (value->bit_size == 64) {
      nir_def *lo = nir_unpack_64_2x32_split_x(b, value);
      nir_def *hi = nir_unpack_64_2x32_split_y(b, value);
      return nir_pack_64_2x32_split(b, si_nir_bswap_scalar(b, hi), si_nir_bswap_scalar(b, lo));
   }
   assert(value->bit_size == 32);
   return nir_ior(b,
      nir_ior(b, nir_ushr_imm(b, value, 24),
                 nir_iand_imm(b, nir_ushr_imm(b, value, 8), 0x0000ff00)),
      nir_ior(b, nir_iand_imm(b, nir_ishl_imm(b, value, 8), 0x00ff0000),
                 nir_ishl_imm(b, value, 24)));
}

static nir_def *si_native_atomic_value(nir_builder *b, nir_atomic_op op,
                                        nir_def *old, nir_def *data)
{
   switch (op) {
   case nir_atomic_op_iadd: return nir_iadd(b, old, data);
   case nir_atomic_op_imin: return nir_imin(b, old, data);
   case nir_atomic_op_umin: return nir_umin(b, old, data);
   case nir_atomic_op_imax: return nir_imax(b, old, data);
   case nir_atomic_op_umax: return nir_umax(b, old, data);
   case nir_atomic_op_iand: return nir_iand(b, old, data);
   case nir_atomic_op_ior: return nir_ior(b, old, data);
   case nir_atomic_op_ixor: return nir_ixor(b, old, data);
   case nir_atomic_op_xchg: return data;
   case nir_atomic_op_fadd: return nir_fadd(b, old, data);
   case nir_atomic_op_fmin: return nir_fmin(b, old, data);
   case nir_atomic_op_fmax: return nir_fmax(b, old, data);
   case nir_atomic_op_inc_wrap:
      return nir_bcsel(b, nir_uge(b, old, data), nir_imm_intN_t(b, 0, old->bit_size),
                        nir_iadd_imm(b, old, 1));
   case nir_atomic_op_dec_wrap:
      return nir_bcsel(b, nir_ior(b, nir_ieq_imm(b, old, 0), nir_ult(b, data, old)),
                        data, nir_iadd_imm(b, old, -1));
   default: UNREACHABLE("unexpected native atomic operation");
   }
}

/* A helper invocation's atomic has no effect, so its compare/exchange never
 * succeeds and the retry loop would hang the GPU.
 */
static nir_def *si_native_atomic_done(nir_builder *b, nir_def *observed, nir_def *old)
{
   nir_def *done = nir_ieq(b, observed, old);
   if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
      done = nir_ior(b, done, nir_is_helper_invocation(b, 1));
   return done;
}

static bool si_lower_native_be_atomic(nir_builder *b, nir_intrinsic_instr *intr)
{
   const unsigned bits = intr->def.bit_size;
   nir_atomic_op op = nir_intrinsic_atomic_op(intr);
   nir_def *index = intr->src[0].ssa, *offset = intr->src[1].ssa;
   enum gl_access_qualifier access = nir_intrinsic_access(intr);
   b->cursor = nir_instr_remove(&intr->instr);
   nir_def *result;
   if (intr->intrinsic == nir_intrinsic_ssbo_atomic_swap) {
      result = nir_ssbo_atomic_swap(b, bits, index, offset,
                                    si_nir_bswap_scalar(b, intr->src[2].ssa),
                                    si_nir_bswap_scalar(b, intr->src[3].ssa),
                                    .access = access, .atomic_op = nir_atomic_op_cmpxchg);
   } else {
      /* Keep the shared buffer in native byte order throughout the operation.
       * A compare/exchange loop also preserves arithmetic under contention.
       */
      nir_def *zero = nir_imm_intN_t(b, 0, bits);
      nir_def *initial = nir_ssbo_atomic_swap(b, bits, index, offset, zero, zero,
                                             .access = access, .atomic_op = nir_atomic_op_cmpxchg);
      nir_variable *raw = nir_local_variable_create(b->impl, glsl_uintN_t_type(bits), "native_atomic");
      nir_store_var(b, raw, initial, 1);
      nir_loop *loop = nir_push_loop(b);
      nir_def *old = nir_load_var(b, raw);
      nir_def *native_old = si_nir_bswap_scalar(b, old);
      nir_def *native_new = si_native_atomic_value(b, op, native_old, intr->src[2].ssa);
      nir_def *observed = nir_ssbo_atomic_swap(b, bits, index, offset, old,
                                              si_nir_bswap_scalar(b, native_new),
                                              .access = access, .atomic_op = nir_atomic_op_cmpxchg);
      nir_store_var(b, raw, observed, 1);
      nir_if *done = nir_push_if(b, si_native_atomic_done(b, observed, old));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, done);
      nir_pop_loop(b, loop);
      result = nir_load_var(b, raw);
   }
   result = si_nir_bswap_scalar(b, result);
   nir_def_rewrite_uses(&intr->def, result);
   return true;
}

static bool si_lower_native_be_ssbo(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      return si_lower_native_be_atomic(b, intr);
   case nir_intrinsic_load_ubo: {
      /* Generated utility shaders bind LE32 user constants directly. GL's
       * explicit UBOs are identified by its reserved default-uniform slot.
       */
      if (!b->shader->info.first_ubo_is_default_ubo)
         return false;
      unsigned bits = intr->def.bit_size;
      if (bits <= 8)
         return false;
      bool has_default = b->shader->info.first_ubo_is_default_ubo;
      if (has_default && nir_src_is_const(intr->src[0]) &&
          nir_src_as_uint(intr->src[0]) == 0 && bits <= 32)
         return false;
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *native = si_nir_bswap_scalar(b, &intr->def);
      if (has_default) {
         /* Default uniforms are already uploaded as LE dwords. Doubles
          * still have their two native dwords in high/low order.
          */
         nir_def *uploaded = &intr->def;
         if (bits == 64)
            uploaded = nir_pack_64_2x32_split(b,
                         nir_unpack_64_2x32_split_y(b, &intr->def),
                         nir_unpack_64_2x32_split_x(b, &intr->def));
         native = nir_bcsel(b, nir_ieq_imm(b, intr->src[0].ssa, 0), uploaded, native);
      }
      nir_def_rewrite_uses_after(&intr->def, native);
      return true;
   }
   case nir_intrinsic_load_ssbo: {
      if (intr->def.bit_size <= 8)
         return false;
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *value = si_nir_bswap_scalar(b, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, value);
      return true;
   }
   case nir_intrinsic_store_ssbo:
      if (intr->src[0].ssa->bit_size <= 8)
         return false;
      b->cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(&intr->src[0], si_nir_bswap_scalar(b, intr->src[0].ssa));
      return true;
   default:
      return false;
   }
}

void si_finalize_nir(struct pipe_screen *screen, struct nir_shader *nir,
                     bool optimize)
{
   struct si_screen *sscreen = (struct si_screen *)screen;

   if (nir->info.io_lowered) {
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out) {
         UNREACHABLE("no IO variables should be present with lowered IO");
      }
   } else {
      /* This always recomputes FS output bases. */
      nir_lower_io_passes(nir, false);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_in | nir_var_shader_out, NULL);
   }

   /* Big endian GFX8: buffer objects hold native CPU scalars (see si_lower_native_be_ssbo). */
   if (UTIL_ARCH_BIG_ENDIAN && sscreen->info.gfx_level == GFX8 && !nir->info.internal)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, si_lower_native_be_ssbo,
               nir_metadata_none, NULL);

   if (optimize) {
      si_nir_opts(sscreen, nir, true);
      /* This reduces code size for some shaders. */
      si_nir_late_opts(nir);
   } else {
      /* These are needed to prevent regressing Max Waves 16 -> 8 for alien_isolation/832.shader_test. */
      NIR_PASS(_, nir, nir_lower_alu_to_scalar, nir->options->lower_to_scalar_filter, NULL);
      NIR_PASS(_, nir, nir_opt_copy_prop);
      /* nir_find_inlinable_uniforms can't find anything without these. */
      NIR_PASS(_, nir, nir_opt_algebraic);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      /* This reduces code size for some shaders. */
      NIR_PASS(_, nir, nir_opt_algebraic_late);
      /* Not sure why we need this, but it returns progress. */
      NIR_PASS(_, nir, nir_opt_dce);
   }

   NIR_PASS_ASSERT_NO_PROGRESS(nir, nir_opt_intrinsics);
   NIR_PASS_ASSERT_NO_PROGRESS(nir, nir_lower_system_values);

   /* Remove uniforms because those should have been lowered to UBOs already. */
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_uniform) {
      if (!glsl_type_get_image_count(var->type) &&
          !glsl_type_get_texture_count(var->type) &&
          !glsl_type_get_sampler_count(var->type))
         exec_node_remove(&var->node);
   }

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   if (sscreen->options.inline_uniforms)
      nir_find_inlinable_uniforms(nir);
}
