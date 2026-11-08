/*
 * Copyright © 2020 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */
#include "util/u_endian.h"
#include "nir_format_convert.h"
#include "ac_descriptors.h"
#include "ac_shader_util.h"
#include "aco_nir_call_attribs.h"
#include "nir.h"
#include "nir_builder.h"
#include "radv_descriptor_set.h"
#include "radv_nir.h"
#include "radv_physical_device.h"
#include "radv_shader.h"
#include "radv_shader_args.h"

typedef struct {
   enum amd_gfx_level gfx_level;
   uint32_t address32_hi;
   uint32_t sampled_image_desc_size;
   uint32_t combined_image_sampler_desc_size;
   uint32_t combined_image_sampler_offset;
   bool disable_aniso_single_level;
   bool has_image_load_dcc_bug;
   bool disable_tg4_trunc_coord;
   bool has_desc_resource_level;

   const struct radv_shader_args *args;
   const struct radv_shader_info *info;
   const struct radv_shader_layout *layout;
} lower_descriptors_state;

static nir_def *
get_scalar_arg(nir_builder *b, unsigned size, struct ac_arg arg)
{
   assert(arg.used);
   return nir_load_scalar_arg_amd(b, size, .base = arg.arg_index);
}

static nir_def *
get_indirect_descriptors_addr(nir_builder *b, lower_descriptors_state *state)
{
   if (mesa_shader_stage_is_rt(b->shader->info.stage))
      return nir_load_param(b, RT_ARG_DESCRIPTORS);
   return get_scalar_arg(b, 1, state->args->descriptors[0]);
}

static nir_def *
get_indirect_push_constants_addr(nir_builder *b, lower_descriptors_state *state)
{
   if (mesa_shader_stage_is_rt(b->shader->info.stage))
      return nir_load_param(b, RT_ARG_PUSH_CONSTANTS);
   return get_scalar_arg(b, 1, state->args->ac.push_constants);
}

static nir_def *
get_dynamic_descriptors_addr(nir_builder *b, lower_descriptors_state *state)
{
   if (mesa_shader_stage_is_rt(b->shader->info.stage))
      return nir_load_param(b, RT_ARG_DYNAMIC_DESCRIPTORS);
   return get_scalar_arg(b, 1, state->args->ac.dynamic_descriptors);
}

static nir_def *
convert_pointer_to_64_bit(nir_builder *b, lower_descriptors_state *state, nir_def *ptr)
{
   return nir_pack_64_2x32_split(b, ptr, nir_imm_int(b, state->address32_hi));
}

static nir_def *
get_dynamic_descriptors_offset(nir_builder *b, lower_descriptors_state *state, uint32_t desc_set, uint32_t binding)
{
   struct radv_descriptor_set_layout *layout = state->layout->set[desc_set].layout;
   nir_def *dynamic_offset_start;

   if (state->layout->independent_sets) {
      nir_def *addr = get_scalar_arg(b, 1, state->args->ac.dynamic_descriptors_offset_addr);
      addr = convert_pointer_to_64_bit(b, state, addr);
      dynamic_offset_start = ac_nir_load_smem(b, 1, addr, nir_imm_int(b, desc_set * 4), 4, 0);
   } else {
      dynamic_offset_start = nir_imm_int(b, state->layout->set[desc_set].dynamic_offset_start);
   }

   nir_def *offset = nir_iadd_imm(b, dynamic_offset_start, layout->binding[binding].dynamic_offset_offset);

   return nir_imul_imm(b, offset, 16);
}

static nir_def *
load_desc_ptr(nir_builder *b, lower_descriptors_state *state, unsigned set)
{
   const struct radv_userdata_locations *user_sgprs_locs = &state->info->user_sgprs_locs;
   if (user_sgprs_locs->shader_data[AC_UD_INDIRECT_DESCRIPTORS].sgpr_idx != -1 ||
       mesa_shader_stage_is_rt(b->shader->info.stage)) {
      nir_def *addr = get_indirect_descriptors_addr(b, state);
      addr = convert_pointer_to_64_bit(b, state, addr);
      return ac_nir_load_smem(b, 1, addr, nir_imm_int(b, set * 4), 4, 0);
   }

   assert(state->args->descriptors[set].used);
   nir_def *res = get_scalar_arg(b, 1, state->args->descriptors[set]);
   nir_intrinsic_set_arg_num_lsb_zero(nir_def_as_intrinsic(res), 2);
   return res;
}

static nir_def *
load_heap_ptr(nir_builder *b, lower_descriptors_state *state, unsigned heap_idx)
{
   nir_def *res;
   if (mesa_shader_stage_is_rt(b->shader->info.stage)) {
      res = nir_load_param(b, heap_idx == RADV_HEAP_RESOURCE ? RT_ARG_HEAP_RESOURCE : RT_ARG_HEAP_SAMPLER);
   } else {
      assert(state->args->descriptors[heap_idx].used);
      res = get_scalar_arg(b, 1, state->args->descriptors[heap_idx]);
   }

   nir_intrinsic_set_arg_num_lsb_zero(nir_def_as_intrinsic(res), 2);
   return res;
}

static void
visit_vulkan_resource_index(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   unsigned desc_set = nir_intrinsic_desc_set(intrin);
   unsigned binding = nir_intrinsic_binding(intrin);
   struct radv_descriptor_set_layout *layout = state->layout->set[desc_set].layout;
   unsigned stride;

   nir_def *set_ptr, *offset;
   if (vk_descriptor_type_is_dynamic(layout->binding[binding].type)) {
      set_ptr = get_dynamic_descriptors_addr(b, state);
      offset = get_dynamic_descriptors_offset(b, state, desc_set, binding);
      stride = 16;
   } else {
      set_ptr = load_desc_ptr(b, state, desc_set);
      offset = nir_imm_int(b, layout->binding[binding].offset);
      stride = layout->binding[binding].size;
   }

   nir_def *binding_ptr = nir_imul_imm_nuw(b, intrin->src[0].ssa, stride);
   binding_ptr = nir_iadd_nuw(b, binding_ptr, offset);

   nir_def_replace(&intrin->def, nir_vec3(b, set_ptr, binding_ptr, nir_imm_int(b, stride)));
}

static void
visit_vulkan_resource_reindex(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);
   assert(desc_type == nir_descriptor_type_uniform_buffer || desc_type == nir_descriptor_type_storage_buffer ||
          desc_type == nir_descriptor_type_acceleration_structure);

   nir_def *binding_ptr = nir_channel(b, intrin->src[0].ssa, 1);
   nir_def *stride = nir_channel(b, intrin->src[0].ssa, 2);

   nir_def *index = nir_imul_nuw(b, intrin->src[1].ssa, stride);
   binding_ptr = nir_iadd_nuw(b, binding_ptr, index);

   nir_def_replace(&intrin->def, nir_vector_insert_imm(b, intrin->src[0].ssa, binding_ptr, 1));
}

static void
visit_load_vulkan_descriptor(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   if (nir_intrinsic_desc_type(intrin) == nir_descriptor_type_acceleration_structure) {
      nir_def *addr = convert_pointer_to_64_bit(
         b, state, nir_iadd(b, nir_channel(b, intrin->src[0].ssa, 0), nir_channel(b, intrin->src[0].ssa, 1)));
      nir_def *desc = nir_load_global(b, 1, 64, addr, .access = ACCESS_NON_WRITEABLE);

      nir_def_rewrite_uses(&intrin->def, desc);
   } else {
      nir_def_rewrite_uses(&intrin->def, nir_vector_insert_imm(b, intrin->src[0].ssa, nir_imm_int(b, 0), 2));
   }
   nir_instr_remove(&intrin->instr);
}

static nir_def *
load_inline_buffer_descriptor(nir_builder *b, lower_descriptors_state *state, nir_def *rsrc)
{
   uint32_t desc[4];

   ac_build_raw_buffer_descriptor(state->gfx_level, state->has_desc_resource_level, (uint64_t)state->address32_hi << 32,
                                  0xffffffff, desc);

   return nir_vec4(b, rsrc, nir_imm_int(b, desc[1]), nir_imm_int(b, desc[2]), nir_imm_int(b, desc[3]));
}

static nir_def *
load_buffer_descriptor(nir_builder *b, lower_descriptors_state *state, nir_def *rsrc, unsigned access)
{
   if (access & ACCESS_NON_UNIFORM)
      return nir_iadd(b, nir_channel(b, rsrc, 0), nir_channel(b, rsrc, 1));

   nir_def *desc_set = convert_pointer_to_64_bit(b, state, nir_channel(b, rsrc, 0));
   return ac_nir_load_smem(b, 4, desc_set, nir_channel(b, rsrc, 1), 4, 0);
}

static nir_def *
load_ubo_descriptor(nir_builder *b, lower_descriptors_state *state, nir_def *rsrc, unsigned access)
{
   nir_binding binding = nir_chase_binding(nir_src_for_ssa(rsrc));

   /* If binding.success=false, then this is a variable pointer, which we don't support with
    * VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK.
    */
   if (binding.success) {
      struct radv_descriptor_set_layout *layout = state->layout->set[binding.desc_set].layout;
      if (layout->binding[binding.binding].type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         rsrc = nir_iadd(b, nir_channel(b, rsrc, 0), nir_channel(b, rsrc, 1));
         return load_inline_buffer_descriptor(b, state, rsrc);
      }
   }

   return load_buffer_descriptor(b, state, rsrc, access);
}

static void
visit_ssbo_descriptor_amd(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   nir_def *rsrc = intrin->src[0].ssa;
   nir_def *desc;

   if (nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM) {
      nir_def *ptr = nir_iadd(b, nir_channel(b, rsrc, 0), nir_channel(b, rsrc, 1));
      ptr = convert_pointer_to_64_bit(b, state, ptr);
      desc = nir_load_global(b, 4, 32, ptr, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER, .align_mul = 16);
   } else {
      /* load the entire descriptor so it can be CSE'd */
      nir_def *ptr = convert_pointer_to_64_bit(b, state, nir_channel(b, rsrc, 0));
      desc = ac_nir_load_smem(b, 4, ptr, nir_channel(b, rsrc, 1), 4, 0);
   }

   nir_def_replace(&intrin->def, desc);
}

static nir_def *
get_sampler_desc(nir_builder *b, lower_descriptors_state *state, nir_deref_instr *deref, nir_def *index,
                 enum ac_descriptor_type desc_type, bool non_uniform, nir_tex_instr *tex, bool write)
{
   nir_def *desc_ptr = NULL;
   uint32_t offset = 0;
   bool indirect = false;
   uint32_t plane_offset;

   if (deref) {
      nir_variable *var = nir_deref_instr_get_variable(deref);
      assert(var && !index);
      unsigned desc_set = var->data.descriptor_set;
      unsigned binding_index = var->data.binding;
      indirect = nir_deref_instr_has_indirect(deref);

      struct radv_descriptor_set_layout *layout = state->layout->set[desc_set].layout;
      struct radv_descriptor_set_binding_layout *binding = &layout->binding[binding_index];

      if (desc_type == AC_DESC_SAMPLER) {
         /* Immutable/embedded samplers are lowered earlier. */
         assert(!binding->immutable_samplers_offset || indirect);
      }

      while (deref->deref_type != nir_deref_type_var) {
         assert(deref->deref_type == nir_deref_type_array);
         unsigned array_size = MAX2(glsl_get_aoa_size(deref->type), 1);
         array_size *= binding->size;

         nir_def *tmp = nir_imul_imm_nuw(b, deref->arr.index.ssa, array_size);

         if (index) {
            index = nir_iadd_nuw(b, tmp, index);
         } else {
            index = tmp;
         }

         deref = nir_deref_instr_parent(deref);
      }

      offset = binding->offset;
      if (desc_type == AC_DESC_SAMPLER && binding->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         offset += state->combined_image_sampler_offset;

      desc_ptr = load_desc_ptr(b, state, desc_set);
      plane_offset = state->combined_image_sampler_desc_size;
   } else {
      desc_ptr = load_heap_ptr(b, state, desc_type == AC_DESC_SAMPLER ? RADV_HEAP_SAMPLER : RADV_HEAP_RESOURCE);
      plane_offset = state->sampled_image_desc_size;
   }

   unsigned size = 8;
   switch (desc_type) {
   case AC_DESC_IMAGE:
   case AC_DESC_PLANE_0:
      break;
   case AC_DESC_FMASK:
      offset += 32;
      break;
   case AC_DESC_PLANE_1:
      offset += plane_offset;
      break;
   case AC_DESC_SAMPLER:
      size = RADV_SAMPLER_DESC_SIZE / 4;
      break;
   case AC_DESC_BUFFER:
      size = RADV_BUFFER_DESC_SIZE / 4;
      break;
   case AC_DESC_PLANE_2:
      offset += 2 * plane_offset;
      break;
   }

   nir_def *index_offset = index ? nir_iadd_imm_nuw(b, index, offset) : nir_imm_int(b, offset);

   if (non_uniform)
      return nir_iadd(b, desc_ptr, index_offset);

   nir_def *addr = convert_pointer_to_64_bit(b, state, desc_ptr);
   nir_def *desc = ac_nir_load_smem(b, size, addr, index_offset, size * 4u, 0);

   if (desc_type == AC_DESC_IMAGE && state->has_image_load_dcc_bug && !tex && !write) {
      nir_def *comp[8];
      for (unsigned i = 0; i < 8; i++)
         comp[i] = nir_channel(b, desc, i);

      /* WRITE_COMPRESS_ENABLE must be 0 for all image loads to workaround a
       * hardware bug.
       */
      comp[6] = nir_iand_imm(b, comp[6], C_00A018_WRITE_COMPRESS_ENABLE);

      return nir_vec(b, comp, 8);
   } else if (desc_type == AC_DESC_SAMPLER && tex->op == nir_texop_tg4 && state->disable_tg4_trunc_coord) {
      nir_def *comp[4];
      for (unsigned i = 0; i < 4; i++)
         comp[i] = nir_channel(b, desc, i);

      /* We want to always use the linear filtering truncation behaviour for
       * nir_texop_tg4, even if the sampler uses nearest/point filtering.
       */
      comp[0] = nir_iand_imm(b, comp[0], C_008F30_TRUNC_COORD);

      return nir_vec(b, comp, 4);
   }

   return desc;
}

static void
update_image_intrinsic(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(deref->type);
   bool is_load =
      intrin->intrinsic == nir_intrinsic_image_deref_load || intrin->intrinsic == nir_intrinsic_image_deref_sparse_load;

   nir_def *desc = get_sampler_desc(b, state, deref, NULL, dim == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE,
                                    nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM, NULL, !is_load);

   if (intrin->intrinsic == nir_intrinsic_image_deref_descriptor_amd) {
      nir_def_replace(&intrin->def, desc);
   } else {
      nir_rewrite_image_intrinsic(intrin, desc, nir_image_intrinsic_type_bindless);
   }
}

static void
lower_load_heap_descriptor(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   nir_resource_type resource_type = nir_intrinsic_resource_type(intrin);
   assert(resource_type != nir_resource_type_sampler);

   nir_def *heap_ptr = load_heap_ptr(b, state, RADV_HEAP_RESOURCE);
   nir_def *desc;

   if (resource_type == nir_resource_type_acceleration_structure) {
      nir_def *addr = convert_pointer_to_64_bit(b, state, nir_iadd(b, heap_ptr, intrin->src[0].ssa));
      desc = nir_build_load_global(b, 1, 64, addr, .access = ACCESS_NON_WRITEABLE);
   } else {
      desc = nir_vec3(b, heap_ptr, intrin->src[0].ssa, nir_imm_int(b, 0));
   }

   nir_def_replace(&intrin->def, desc);
}

static void
lower_load_resource_heap_data(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   nir_def *heap_ptr = load_heap_ptr(b, state, RADV_HEAP_RESOURCE);
   nir_def *addr = convert_pointer_to_64_bit(b, state, nir_iadd(b, heap_ptr, intrin->src[0].ssa));

   nir_def *data = nir_build_load_global(b, intrin->def.num_components, intrin->def.bit_size, addr,
                                         .access = ACCESS_NON_WRITEABLE, .align_mul = nir_intrinsic_align_mul(intrin),
                                         .align_offset = nir_intrinsic_align_offset(intrin));

   nir_def_replace(&intrin->def, data);
}

static void
lower_image_heap_intrinsic(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   const bool is_load =
      intrin->intrinsic == nir_intrinsic_image_heap_load || intrin->intrinsic == nir_intrinsic_image_heap_sparse_load;
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intrin);

   nir_def *desc =
      get_sampler_desc(b, state, NULL, intrin->src[0].ssa, dim == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE,
                       nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM, NULL, !is_load);

   if (intrin->intrinsic == nir_intrinsic_image_heap_descriptor_amd) {
      nir_def_replace(&intrin->def, desc);
   } else {
      nir_rewrite_image_intrinsic(intrin, desc, nir_image_intrinsic_type_bindless);
   }
}

static bool
can_increase_load_size(nir_intrinsic_instr *intrin, unsigned offset, unsigned old, unsigned new)
{
   /* Only increase the size of loads if doing so won't extend into a new page/cache-line. */
   unsigned align_mul = MIN2(nir_intrinsic_align_mul(intrin), 64u);
   unsigned end = (nir_intrinsic_align_offset(intrin) + offset + old) & (align_mul - 1);
   return (new - old) <= (align_mul - end);
}

static nir_def *
load_push_constant(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   unsigned base = nir_intrinsic_base(intrin);
   unsigned bit_size = intrin->def.bit_size;
   unsigned count = intrin->def.num_components * (bit_size / 32u);
   assert(bit_size >= 32);

   nir_def *addr = NULL;
   nir_def *offset = NULL;
   unsigned const_offset = -1;
   if (nir_src_is_const(intrin->src[0]))
      const_offset = (base + nir_src_as_uint(intrin->src[0])) / 4u;

   const unsigned max_push_constant = sizeof(state->args->ac.inline_push_const_mask) * 8u;

   nir_component_mask_t comps_read = nir_def_components_read(&intrin->def);

   nir_def *data[NIR_MAX_VEC_COMPONENTS * 2];
   unsigned num_loads = 0;
   for (unsigned start = 0; start < count;) {
      if (!(comps_read & BITFIELD64_BIT(start >> (bit_size == 64 ? 1 : 0)))) {
         data[num_loads++] = nir_undef(b, 1, 32);
         start += 1;
         continue;
      }

      /* Try to use inline push constants when possible. */
      unsigned inline_idx = const_offset + start;
      if (const_offset != -1 && inline_idx < max_push_constant &&
          (state->args->ac.inline_push_const_mask & BITFIELD64_BIT(inline_idx))) {
         inline_idx = util_bitcount64(state->args->ac.inline_push_const_mask & BITFIELD64_MASK(inline_idx));
         data[num_loads++] = get_scalar_arg(b, 1, state->args->ac.inline_push_consts[inline_idx]);
         start += 1;
         continue;
      }

      if (!state->args->ac.push_constants.used && !mesa_shader_stage_is_rt(b->shader->info.stage)) {
         /* Assume this is an inlined push constant load which was expanded to include dwords which are not inlined.
          * RT stages use neither shader args nor inlined push constants, so skip this for RT shaders.
          */
         assert(const_offset != -1);
         data[num_loads++] = nir_undef(b, 1, 32);
         start += 1;
         continue;
      }

      if (!offset) {
         addr = get_indirect_push_constants_addr(b, state);
         addr = convert_pointer_to_64_bit(b, state, addr);
         offset = nir_iadd_imm_nuw(b, intrin->src[0].ssa, base);
      }

      /* Decrease to supported size. */
      unsigned size = count - start;
      size = state->gfx_level >= GFX12 && size == 3 ? 3 : (1 << (util_last_bit(size) - 1));

      /* Try to round up to power of two instead. */
      if (size < (count - start) && can_increase_load_size(intrin, start * 4, size, size * 2))
         size *= 2;

      data[num_loads++] = ac_nir_load_smem(b, size, addr, nir_iadd_imm_nuw(b, offset, start * 4), 4, 0);
      start += size;
   }
   return nir_extract_bits(b, data, num_loads, 0, intrin->def.num_components, bit_size);
}

static bool
lower_descriptors_early_intrin(nir_builder *b, nir_intrinsic_instr *intrin, void *_state)
{
   lower_descriptors_state *state = _state;
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *rsrc;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ubo:
      rsrc = load_ubo_descriptor(b, state, intrin->src[0].ssa, nir_intrinsic_access(intrin));
      nir_src_rewrite(&intrin->src[0], rsrc);
      break;
   default:
      return false;
   }
   return true;
}

static bool
lower_descriptors_intrin(nir_builder *b, lower_descriptors_state *state, nir_intrinsic_instr *intrin)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *rsrc;
   switch (intrin->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      visit_vulkan_resource_index(b, state, intrin);
      break;
   case nir_intrinsic_vulkan_resource_reindex:
      visit_vulkan_resource_reindex(b, state, intrin);
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      visit_load_vulkan_descriptor(b, state, intrin);
      break;
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      rsrc = load_buffer_descriptor(b, state, intrin->src[0].ssa, nir_intrinsic_access(intrin));
      nir_src_rewrite(&intrin->src[0], rsrc);
      break;
   case nir_intrinsic_store_ssbo:
      rsrc = load_buffer_descriptor(b, state, intrin->src[1].ssa, nir_intrinsic_access(intrin));
      nir_src_rewrite(&intrin->src[1], rsrc);
      break;
   case nir_intrinsic_ssbo_descriptor_amd:
      visit_ssbo_descriptor_amd(b, state, intrin);
      break;
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_descriptor_amd:
      update_image_intrinsic(b, state, intrin);
      break;
   case nir_intrinsic_load_push_constant: {
      nir_def_replace(&intrin->def, load_push_constant(b, state, intrin));
      break;
   }
   case nir_intrinsic_load_heap_descriptor:
      lower_load_heap_descriptor(b, state, intrin);
      break;
   case nir_intrinsic_load_resource_heap_data:
      lower_load_resource_heap_data(b, state, intrin);
      break;
   case nir_intrinsic_image_heap_load:
   case nir_intrinsic_image_heap_sparse_load:
   case nir_intrinsic_image_heap_store:
   case nir_intrinsic_image_heap_atomic:
   case nir_intrinsic_image_heap_atomic_swap:
   case nir_intrinsic_image_heap_size:
   case nir_intrinsic_image_heap_samples:
   case nir_intrinsic_image_heap_descriptor_amd:
      lower_image_heap_intrinsic(b, state, intrin);
      break;
   default:
      return false;
   }

   return true;
}

static bool
lower_descriptors_tex(nir_builder *b, lower_descriptors_state *state, nir_tex_instr *tex)
{
   b->cursor = nir_before_instr(&tex->instr);

   nir_deref_instr *texture_deref_instr = NULL;
   nir_deref_instr *sampler_deref_instr = NULL;
   nir_def *texture_heap_offset = NULL;
   nir_def *sampler_heap_offset = NULL;
   int plane = -1;

   nir_def *image = NULL;
   nir_def *sampler = NULL;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref:
         texture_deref_instr = nir_src_as_deref(tex->src[i].src);
         break;
      case nir_tex_src_sampler_deref:
         sampler_deref_instr = nir_src_as_deref(tex->src[i].src);
         break;
      case nir_tex_src_plane:
         plane = nir_src_as_int(tex->src[i].src);
         break;
      case nir_tex_src_sampler_handle:
         sampler = tex->src[i].src.ssa;
         break;
      case nir_tex_src_texture_heap_offset:
         texture_heap_offset = tex->src[i].src.ssa;
         break;
      case nir_tex_src_sampler_heap_offset:
         sampler_heap_offset = tex->src[i].src.ssa;
         break;
      default:
         break;
      }
   }

   if (plane >= 0) {
      assert(tex->op != nir_texop_txf_ms && tex->op != nir_texop_samples_identical);
      assert(tex->sampler_dim != GLSL_SAMPLER_DIM_BUF);
      image = get_sampler_desc(b, state, texture_deref_instr, texture_heap_offset, AC_DESC_PLANE_0 + plane,
                               tex->texture_non_uniform, tex, false);
   } else if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      image = get_sampler_desc(b, state, texture_deref_instr, texture_heap_offset, AC_DESC_BUFFER,
                               tex->texture_non_uniform, tex, false);
   } else if (tex->op == nir_texop_fragment_mask_fetch_amd) {
      image = get_sampler_desc(b, state, texture_deref_instr, texture_heap_offset, AC_DESC_FMASK,
                               tex->texture_non_uniform, tex, false);
   } else {
      image = get_sampler_desc(b, state, texture_deref_instr, texture_heap_offset, AC_DESC_IMAGE,
                               tex->texture_non_uniform, tex, false);
   }

   if (sampler_deref_instr || sampler_heap_offset) {
      assert(!sampler);
      sampler = get_sampler_desc(b, state, sampler_deref_instr, sampler_heap_offset, AC_DESC_SAMPLER,
                                 tex->sampler_non_uniform, tex, false);
   }

   if (sampler && state->disable_aniso_single_level && tex->sampler_dim < GLSL_SAMPLER_DIM_RECT) {
      /* Disable anisotropic filtering if BASE_LEVEL == LAST_LEVEL.
       *
       * GFX6-GFX7:
       *   If BASE_LEVEL == LAST_LEVEL, the shader must disable anisotropic
       *   filtering manually. The driver sets img7 to a mask clearing
       *   MAX_ANISO_RATIO if BASE_LEVEL == LAST_LEVEL. The shader must do:
       *     s_and_b32 samp0, samp0, img7
       *
       * GFX8:
       *   The ANISO_OVERRIDE sampler field enables this fix in TA.
       */
      /* TODO: This is unnecessary for combined image+sampler.
       * We can do this when updating the desc set. */
      nir_def *comp[4];
      for (unsigned i = 0; i < 4; i++)
         comp[i] = nir_channel(b, sampler, i);
      comp[0] = nir_iand(b, comp[0], nir_channel(b, image, 7));

      sampler = nir_vec(b, comp, 4);
   }

   if (tex->op == nir_texop_descriptor_amd) {
      nir_def_replace(&tex->def, image);
      return true;
   }

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref:
         tex->src[i].src_type = nir_tex_src_texture_handle;
         nir_src_rewrite(&tex->src[i].src, image);
         break;
      case nir_tex_src_sampler_deref:
         tex->src[i].src_type = nir_tex_src_sampler_handle;
         nir_src_rewrite(&tex->src[i].src, sampler);
         break;
      case nir_tex_src_sampler_handle:
         nir_src_rewrite(&tex->src[i].src, sampler);
         break;
      case nir_tex_src_texture_heap_offset:
         tex->src[i].src_type = nir_tex_src_texture_handle;
         nir_src_rewrite(&tex->src[i].src, image);
         break;
      case nir_tex_src_sampler_heap_offset:
         tex->src[i].src_type = nir_tex_src_sampler_handle;
         nir_src_rewrite(&tex->src[i].src, sampler);
         break;
      default:
         break;
      }
   }

   return true;
}

static bool
lower_descriptors_instr(nir_builder *b, nir_instr *instr, void *_state)
{
   lower_descriptors_state *state = _state;

   if (instr->type == nir_instr_type_tex)
      return lower_descriptors_tex(b, state, nir_instr_as_tex(instr));
   else if (instr->type == nir_instr_type_intrinsic)
      return lower_descriptors_intrin(b, state, nir_instr_as_intrinsic(instr));

   return false;
}

/* Texel buffers are host-mapped, so they hold host-order elements that the typed
 * hardware paths would decode as little-endian. The format is only known from the
 * descriptor at run time, so read it there and handle elements wider than a byte in
 * the shader.
 */
#define TEXEL_DST_SEL(word3, chan) nir_ubfe_imm(b, word3, 3 * (chan), 3)
#define TEXEL_NUM_FORMAT(word3)    nir_ubfe_imm(b, word3, 12, 3)
#define TEXEL_DATA_FORMAT(word3)   nir_ubfe_imm(b, word3, 15, 4)

static nir_def *
texel_bswap16(nir_builder *b, nir_def *value)
{
   return nir_iand_imm(b, nir_ior(b, nir_ushr_imm(b, value, 8), nir_ishl_imm(b, value, 8)), 0xffff);
}

static nir_def *
texel_bswap32(nir_builder *b, nir_def *value)
{
   return nir_ior(b, nir_ior(b, nir_ushr_imm(b, value, 24), nir_iand_imm(b, nir_ushr_imm(b, value, 8), 0xff00)),
                  nir_ior(b, nir_iand_imm(b, nir_ishl_imm(b, value, 8), 0xff0000), nir_ishl_imm(b, value, 24)));
}

/* Swaps a 32-bit or 64-bit value. */
static nir_def *
texel_bswap(nir_builder *b, nir_def *value)
{
   if (value->bit_size == 32)
      return texel_bswap32(b, value);
   return nir_pack_64_2x32_split(b, texel_bswap32(b, nir_unpack_64_2x32_split_y(b, value)),
                                 texel_bswap32(b, nir_unpack_64_2x32_split_x(b, value)));
}

/* R64 texels hold their value in the first channel; the others are constants. */
static nir_def *
texel_swap_r64(nir_builder *b, nir_def *value)
{
   nir_def *c[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < value->num_components; i++)
      c[i] = i ? nir_channel(b, value, i) : texel_bswap(b, nir_channel(b, value, 0));
   return nir_vec(b, c, value->num_components);
}

static nir_def *
texel_is_any(nir_builder *b, nir_def *value, unsigned count, const unsigned *options)
{
   nir_def *result = nir_imm_false(b);
   for (unsigned i = 0; i < count; i++)
      result = nir_ior(b, result, nir_ieq_imm(b, value, options[i]));
   return result;
}

static const unsigned texel_32_formats[] = {V_008F0C_BUF_DATA_FORMAT_32, V_008F0C_BUF_DATA_FORMAT_32_32,
                                            V_008F0C_BUF_DATA_FORMAT_32_32_32, V_008F0C_BUF_DATA_FORMAT_32_32_32_32};
static const unsigned texel_16_formats[] = {V_008F0C_BUF_DATA_FORMAT_16, V_008F0C_BUF_DATA_FORMAT_16_16,
                                            V_008F0C_BUF_DATA_FORMAT_16_16_16_16};

static nir_def *
texel_needs_native(nir_builder *b, nir_def *data_format)
{
   return nir_ior(b, nir_ior(b, texel_is_any(b, data_format, ARRAY_SIZE(texel_32_formats), texel_32_formats),
                             texel_is_any(b, data_format, ARRAY_SIZE(texel_16_formats), texel_16_formats)),
                  nir_ior(b, nir_ieq_imm(b, data_format, V_008F0C_BUF_DATA_FORMAT_2_10_10_10),
                          nir_ieq_imm(b, data_format, V_008F0C_BUF_DATA_FORMAT_10_11_11)));
}

static nir_def *
texel_decode(nir_builder *b, nir_def *bits, unsigned size, nir_def *num_format)
{
   nir_def *sext = nir_ibfe_imm(b, bits, 0, size);
   nir_def *fu = nir_u2f32(b, bits), *fs = nir_i2f32(b, sext);
   nir_def *result = bits; /* UINT */
   result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_SINT), sext, result);
   result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_USCALED), fu, result);
   result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_SSCALED), fs, result);
   result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_UNORM),
                      nir_fdiv_imm(b, fu, (double)((1u << size) - 1)), result);
   result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_SNORM),
                      nir_fmax(b, nir_fdiv_imm(b, fs, (double)((1u << (size - 1)) - 1)), nir_imm_float(b, -1.0f)),
                      result);
   if (size == 16)
      result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_FLOAT),
                         nir_channel(b, nir_unpack_half_2x16(b, bits), 0), result);
   return result;
}

static nir_def *
texel_encode(nir_builder *b, nir_def *value, unsigned size, nir_def *num_format)
{
   const unsigned mask = (1u << size) - 1;
   nir_def *result = nir_iand_imm(b, value, mask); /* UINT and SINT */
   nir_def *unorm = nir_f2u32(b, nir_fadd_imm(b, nir_fmul_imm(b, nir_fsat(b, value), (double)mask), 0.5));
   nir_def *snorm = nir_iand_imm(b,
                                 nir_f2i32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fclamp(b, value, nir_imm_float(b, -1.0f), nir_imm_float(b, 1.0f)),
                                                                              (double)(mask >> 1)))),
                                 mask);
   result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_UNORM), unorm, result);
   result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_SNORM), snorm, result);
   if (size == 16)
      result = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_FLOAT),
                         nir_pack_half_2x16(b, nir_vec2(b, value, nir_imm_float(b, 0.0f))), result);
   return result;
}

static nir_def *
texel_raw_load(nir_builder *b, nir_def *desc, nir_def *index, unsigned offset, enum pipe_format format)
{
   nir_def *zero = nir_imm_int(b, 0);
   return nir_load_typed_buffer_amd(b, 1, 32, desc, zero, zero, index, .base = offset, .format = format,
                                    .align_mul = format == PIPE_FORMAT_R16_UINT ? 2 : 4, .align_offset = 0,
                                    .memory_modes = nir_var_image);
}

/* Returns the texel as 32-bit channel bits, already swizzled as the descriptor asks. */
static nir_def *
texel_native_fetch(nir_builder *b, nir_def *desc, nir_def *index)
{
   nir_def *word3 = nir_channel(b, desc, 3);
   nir_def *data_format = TEXEL_DATA_FORMAT(word3);
   nir_def *num_format = TEXEL_NUM_FORMAT(word3);
   nir_variable *var = nir_local_variable_create(b->impl, glsl_uvec4_type(), "native_texel");

   nir_push_if(b, texel_is_any(b, data_format, ARRAY_SIZE(texel_32_formats), texel_32_formats));
   {
      nir_def *c[4];
      for (unsigned i = 0; i < 4; i++)
         c[i] = texel_bswap32(b, texel_raw_load(b, desc, index, i * 4, PIPE_FORMAT_R32_UINT));
      nir_store_var(b, var, nir_vec(b, c, 4), 0xf);
   }
   nir_push_else(b, NULL);
   nir_push_if(b, texel_is_any(b, data_format, ARRAY_SIZE(texel_16_formats), texel_16_formats));
   {
      nir_def *c[4];
      for (unsigned i = 0; i < 4; i++) {
         nir_def *bits = texel_bswap16(b, texel_raw_load(b, desc, index, i * 2, PIPE_FORMAT_R16_UINT));
         c[i] = texel_decode(b, bits, 16, num_format);
      }
      nir_store_var(b, var, nir_vec(b, c, 4), 0xf);
   }
   nir_push_else(b, NULL);
   {
      nir_def *word = texel_bswap32(b, texel_raw_load(b, desc, index, 0, PIPE_FORMAT_R32_UINT));
      nir_def *c[4];
      nir_push_if(b, nir_ieq_imm(b, data_format, V_008F0C_BUF_DATA_FORMAT_10_11_11));
      nir_def *r = nir_ubfe_imm(b, word, 0, 11), *g = nir_ubfe_imm(b, word, 11, 11), *bl = nir_ubfe_imm(b, word, 22, 10);
      /* Small floats share the half-float layout once aligned to its mantissa. */
      nir_def *packed = nir_channel(b, nir_unpack_half_2x16(b, nir_ishl_imm(b, r, 4)), 0);
      nir_store_var(b, var,
                    nir_vec4(b, packed, nir_channel(b, nir_unpack_half_2x16(b, nir_ishl_imm(b, g, 4)), 0),
                             nir_channel(b, nir_unpack_half_2x16(b, nir_ishl_imm(b, bl, 5)), 0), nir_imm_int(b, 0)),
                    0xf);
      nir_push_else(b, NULL);
      for (unsigned i = 0; i < 4; i++) {
         const unsigned size = i == 3 ? 2 : 10;
         c[i] = texel_decode(b, nir_ubfe_imm(b, word, i * 10, size), size, num_format);
      }
      nir_store_var(b, var, nir_vec(b, c, 4), 0xf);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);

   nir_def *raw = nir_load_var(b, var);
   nir_def *is_int = nir_ior(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_UINT),
                             nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_SINT));
   nir_def *one = nir_bcsel(b, is_int, nir_imm_int(b, 1), nir_imm_int(b, 0x3f800000));
   nir_def *out[4];
   for (unsigned i = 0; i < 4; i++) {
      nir_def *sel = TEXEL_DST_SEL(word3, i);
      nir_def *value = nir_imm_int(b, 0);
      value = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F0C_SQ_SEL_1), one, value);
      value = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F0C_SQ_SEL_X), nir_channel(b, raw, 0), value);
      value = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F0C_SQ_SEL_Y), nir_channel(b, raw, 1), value);
      value = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F0C_SQ_SEL_Z), nir_channel(b, raw, 2), value);
      out[i] = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F0C_SQ_SEL_W), nir_channel(b, raw, 3), value);
   }
   return nir_vec(b, out, 4);
}

static void
texel_store_word(nir_builder *b, nir_def *desc, nir_def *index, unsigned offset, nir_def *value)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_store_buffer_amd(b, value, desc, zero, zero, index, .base = offset, .memory_modes = nir_var_image);
}

static void
texel_native_store(nir_builder *b, nir_def *desc, nir_def *index, nir_def *value)
{
   nir_def *word3 = nir_channel(b, desc, 3);
   nir_def *data_format = TEXEL_DATA_FORMAT(word3);
   nir_def *num_format = TEXEL_NUM_FORMAT(word3);

   nir_push_if(b, texel_is_any(b, data_format, ARRAY_SIZE(texel_32_formats), texel_32_formats));
   {
      for (unsigned i = 0; i < 4; i++) {
         /* Channels past the format's count would land in the next element. */
         nir_push_if(b, i == 0 ? nir_imm_true(b)
                               : texel_is_any(b, data_format, 4 - i, texel_32_formats + i));
         texel_store_word(b, desc, index, i * 4, texel_bswap32(b, nir_channel(b, value, i)));
         nir_pop_if(b, NULL);
      }
   }
   nir_push_else(b, NULL);
   nir_push_if(b, texel_is_any(b, data_format, ARRAY_SIZE(texel_16_formats), texel_16_formats));
   {
      for (unsigned i = 0; i < 4; i++) {
         nir_def *present = i == 0   ? nir_imm_true(b)
                            : i == 1 ? texel_is_any(b, data_format, 2, texel_16_formats + 1)
                                     : nir_ieq_imm(b, data_format, V_008F0C_BUF_DATA_FORMAT_16_16_16_16);
         nir_push_if(b, present);
         nir_def *bits = texel_encode(b, nir_channel(b, value, i), 16, num_format);
         texel_store_word(b, desc, index, i * 2, nir_u2u16(b, texel_bswap16(b, bits)));
         nir_pop_if(b, NULL);
      }
   }
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, data_format, V_008F0C_BUF_DATA_FORMAT_10_11_11));
   {
      nir_def *word = nir_format_pack_11f11f10f(b, nir_trim_vector(b, value, 3));
      texel_store_word(b, desc, index, 0, texel_bswap32(b, word));
   }
   nir_push_else(b, NULL);
   {
      nir_def *word = nir_imm_int(b, 0);
      for (unsigned i = 0; i < 4; i++) {
         const unsigned size = i == 3 ? 2 : 10;
         word = nir_ior(b, word, nir_ishl_imm(b, texel_encode(b, nir_channel(b, value, i), size, num_format), i * 10));
      }
      texel_store_word(b, desc, index, 0, texel_bswap32(b, word));
   }
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static bool
lower_native_texel_buffer(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   nir_def *desc, *index, *def;

   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF || tex->op != nir_texop_txf || tex->def.bit_size != 32)
         return false;
      int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (handle < 0 || coord < 0)
         return false;
      desc = tex->src[handle].src.ssa;
      index = tex->src[coord].src.ssa;
      def = &tex->def;
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_buffer_amd:
         /* Since Mesa 25, ac_nir_lower_image_tex turns texel buffer fetches and image loads
          * into formatted buffer loads before descriptor lowering.
          * TODO: R64 texel buffers arrive as 32-bit pairs here and are not half-swapped.
          */
         if (!(nir_intrinsic_access(intr) & ACCESS_USES_FORMAT_AMD) ||
             (nir_intrinsic_access(intr) & ACCESS_SPARSE) ||
             (nir_intrinsic_memory_modes(intr) & nir_var_shader_in) || intr->def.bit_size != 32)
            return false;
         desc = intr->src[0].ssa;
         index = intr->src[3].ssa;
         def = &intr->def;
         break;
      case nir_intrinsic_bindless_image_load:
         if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF && intr->def.bit_size == 64) {
            b->cursor = nir_after_instr(instr);
            nir_def *swapped = texel_swap_r64(b, &intr->def);
            nir_def_rewrite_uses_after(&intr->def, swapped);
            return true;
         }
         if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF || intr->def.bit_size != 32)
            return false;
         desc = intr->src[0].ssa;
         index = intr->src[1].ssa;
         def = &intr->def;
         break;
      case nir_intrinsic_bindless_image_store: {
         if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF && nir_src_bit_size(intr->src[3]) == 64) {
            b->cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[3], texel_swap_r64(b, intr->src[3].ssa));
            return true;
         }
         if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF || nir_src_bit_size(intr->src[3]) != 32)
            return false;
         b->cursor = nir_before_instr(instr);
         nir_def *handle = intr->src[0].ssa;
         nir_def *element = nir_channel(b, intr->src[1].ssa, 0);
         nir_def *value = intr->src[3].ssa;
         b->cursor = nir_instr_remove(instr);
         nir_def *data_format = TEXEL_DATA_FORMAT(nir_channel(b, handle, 3));
         nir_push_if(b, texel_needs_native(b, data_format));
         /* Earlier passes may have shrunk the value to the format's channel count. */
         texel_native_store(b, handle, element, nir_pad_vector_imm_int(b, value, 0, 4));
         nir_push_else(b, NULL);
         nir_builder_instr_insert(b, instr);
         nir_pop_if(b, NULL);
         return true;
      }
      case nir_intrinsic_bindless_image_atomic:
      case nir_intrinsic_bindless_image_atomic_swap: {
         if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF)
            return false;
         /* Only R32 and R64 formats allow atomics, so values just need their bytes swapped. */
         const unsigned bits = intr->def.bit_size;
         if (bits != 32 && bits != 64)
            return false;
         nir_atomic_op op = nir_intrinsic_atomic_op(intr);
         enum pipe_format format = nir_intrinsic_format(intr);
         nir_def *handle = intr->src[0].ssa, *coord = intr->src[1].ssa, *sample = intr->src[2].ssa;
         b->cursor = nir_instr_remove(instr);
         nir_def *zero = nir_imm_intN_t(b, 0, bits);
#define TEXEL_CMPXCHG(cmp, val)                                                                                        \
   nir_bindless_image_atomic_swap(b, bits, handle, coord, sample, cmp, val, .image_dim = GLSL_SAMPLER_DIM_BUF,           \
                                  .image_array = false, .format = format, .access = 0,                                 \
                                  .atomic_op = nir_atomic_op_cmpxchg)
         nir_def *result;
         if (intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap) {
            result = TEXEL_CMPXCHG(texel_bswap(b, intr->src[3].ssa), texel_bswap(b, intr->src[4].ssa));
         } else {
            nir_variable *raw = nir_local_variable_create(b->impl, glsl_uintN_t_type(bits), "native_texel_atomic");
            nir_store_var(b, raw, TEXEL_CMPXCHG(zero, zero), 1);
            nir_loop *loop = nir_push_loop(b);
            nir_def *old = nir_load_var(b, raw);
            nir_def *native_old = texel_bswap(b, old), *arg = intr->src[3].ssa, *updated;
            switch (op) {
            case nir_atomic_op_iadd: updated = nir_iadd(b, native_old, arg); break;
            case nir_atomic_op_imin: updated = nir_imin(b, native_old, arg); break;
            case nir_atomic_op_umin: updated = nir_umin(b, native_old, arg); break;
            case nir_atomic_op_imax: updated = nir_imax(b, native_old, arg); break;
            case nir_atomic_op_umax: updated = nir_umax(b, native_old, arg); break;
            case nir_atomic_op_iand: updated = nir_iand(b, native_old, arg); break;
            case nir_atomic_op_ior: updated = nir_ior(b, native_old, arg); break;
            case nir_atomic_op_ixor: updated = nir_ixor(b, native_old, arg); break;
            case nir_atomic_op_fmin: updated = nir_fmin(b, native_old, arg); break;
            case nir_atomic_op_fmax: updated = nir_fmax(b, native_old, arg); break;
            default: updated = arg; break;
            }
            nir_def *observed = TEXEL_CMPXCHG(old, texel_bswap(b, updated));
            nir_store_var(b, raw, observed, 1);
            nir_push_if(b, nir_ieq(b, observed, old));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);
            nir_pop_loop(b, loop);
            result = nir_load_var(b, raw);
         }
#undef TEXEL_CMPXCHG
         nir_def_rewrite_uses(&intr->def, texel_bswap(b, result));
         return true;
      }
      default:
         return false;
      }
   } else {
      return false;
   }

   b->cursor = nir_before_instr(instr);
   nir_variable *var = nir_local_variable_create(b->impl, glsl_uvec4_type(), "native_texel_result");
   index = nir_channel(b, index, 0);
   b->cursor = nir_instr_remove(instr);

   /* Uses move to the merged result before the original instruction is reinserted. */
   nir_def *result_placeholder = nir_load_var(b, var);
   nir_def_rewrite_uses(def, nir_trim_vector(b, result_placeholder, def->num_components));
   b->cursor = nir_before_instr(nir_def_instr(result_placeholder));

   nir_def *data_format = TEXEL_DATA_FORMAT(nir_channel(b, desc, 3));
   nir_push_if(b, texel_needs_native(b, data_format));
   nir_store_var(b, var, texel_native_fetch(b, desc, index), 0xf);
   nir_push_else(b, NULL);
   nir_builder_instr_insert(b, instr);
   nir_store_var(b, var, nir_pad_vector_imm_int(b, def, 0, 4), 0xf);
   nir_pop_if(b, NULL);
   return true;
}

bool
radv_nir_lower_descriptors(nir_shader *shader, const struct radv_compiler_info *compiler_info,
                           const struct radv_shader_stage *stage)
{
   bool progress = false;

   lower_descriptors_state state = {
      .gfx_level = compiler_info->ac->gfx_level,
      .address32_hi = compiler_info->hw.address32_hi,
      .sampled_image_desc_size = compiler_info->sampled_image_desc_size,
      .combined_image_sampler_desc_size = compiler_info->combined_image_sampler_desc_size,
      .combined_image_sampler_offset = compiler_info->combined_image_sampler_offset,
      .disable_aniso_single_level =
         compiler_info->key.disable_aniso_single_level && compiler_info->ac->gfx_level < GFX8,
      .has_image_load_dcc_bug = compiler_info->ac->has_image_load_dcc_bug,
      .disable_tg4_trunc_coord = !compiler_info->ac->conformant_trunc_coord && !compiler_info->key.disable_trunc_coord,
      .has_desc_resource_level = compiler_info->ac->has_desc_resource_level,
      .args = &stage->args,
      .info = &stage->info,
      .layout = &stage->layout,
   };

   /* Do an early pass over intrinsics so load_ubo lowering
    * can look at the vulkan_resource_index to tell if it's
    * an inline ubo.
    */
   progress |= nir_shader_intrinsics_pass(shader, lower_descriptors_early_intrin, nir_metadata_control_flow, &state);
   progress |= nir_shader_instructions_pass(shader, lower_descriptors_instr, nir_metadata_control_flow, &state);

   /* Big endian GFX8: host-order texel buffers. This runs after the regular variable
    * lowering, so finish its own variables.
    */
   if (UTIL_ARCH_BIG_ENDIAN && compiler_info->ac->gfx_level == GFX8 && !shader->info.internal &&
       nir_shader_instructions_pass(shader, lower_native_texel_buffer, nir_metadata_none, NULL)) {
      NIR_PASS(_, shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, shader, nir_opt_dce);
      progress = true;
   }

   return progress;
}
