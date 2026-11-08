/*
 * Copyright 2022 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * This lowering pass converts index based buffer/image/texture access to
 * explicite descriptor based, which simplify the compiler backend translation.
 *
 * For example: load_ubo(1) -> load_ubo(vec4), where the vec4 is the buffer
 * descriptor with index==1, so compiler backend don't need to do index-to-descriptor
 * finding which is the most complicated part (move to nir now).
 */

#include "nir_builder.h"
#include "nir_format_convert.h"

#include "ac_nir.h"
#include "si_pipe.h"
#include "si_shader_internal.h"

struct lower_resource_state {
   struct si_shader *shader;
   struct si_shader_args *args;
};

static nir_def *load_ubo_desc_fast_path(nir_builder *b, nir_def *addr_lo,
                                            struct si_shader_selector *sel)
{
   const struct ac_buffer_state buffer_state = {
      .va = (uint64_t)sel->screen->info.address32_hi << 32,
      .size = sel->info.constbuf0_num_slots * 16,
      .format = PIPE_FORMAT_R32_FLOAT,
      .swizzle =
         {
            PIPE_SWIZZLE_X,
            PIPE_SWIZZLE_Y,
            PIPE_SWIZZLE_Z,
            PIPE_SWIZZLE_W,
         },
      .gfx10_oob_select = V_008F0C_OOB_SELECT_RAW,
      .has_desc_resource_level = sel->screen->info.compiler_info.has_desc_resource_level,
   };
   uint32_t desc[4];

   ac_build_buffer_descriptor(sel->screen->info.gfx_level, &buffer_state, desc);

   return nir_vec4(b, addr_lo, nir_imm_int(b, desc[1]), nir_imm_int(b, desc[2]),
                   nir_imm_int(b, desc[3]));
}

static nir_def *clamp_index(nir_builder *b, nir_def *index, unsigned max)
{
   if (util_is_power_of_two_or_zero(max))
      return nir_iand_imm(b, index, max - 1);
   else {
      nir_def *clamp = nir_imm_int(b, max - 1);
      nir_def *cond = nir_uge(b, clamp, index);
      return nir_bcsel(b, cond, index, clamp);
   }
}

static nir_def *load_ubo_desc(nir_builder *b, nir_def *index,
                                  struct lower_resource_state *s)
{
   struct si_shader_selector *sel = s->shader->selector;


   if (b->shader->info.num_ubos == 1 && b->shader->info.num_ssbos == 0) {
      nir_def *addr = ac_nir_load_arg(b, &s->args->ac, s->args->const_and_shader_buffers);
      return load_ubo_desc_fast_path(b, addr, sel);
   }

   nir_def *addr = si_nir_load_addr32_arg(s->shader->selector->screen, s->args,
                                          b, s->args->const_and_shader_buffers);
   index = clamp_index(b, index, b->shader->info.num_ubos);
   index = nir_iadd_imm(b, index, SI_NUM_SHADER_BUFFERS);

   nir_def *offset = nir_ishl_imm(b, index, 4);
   return ac_nir_load_smem(b, 4, addr, offset, 16, ACCESS_CAN_SPECULATE);
}

static nir_def *load_ssbo_desc(nir_builder *b, nir_src *index,
                                   struct lower_resource_state *s)
{
   /* Fast path if the shader buffer is in user SGPRs. */
   if (nir_src_is_const(*index)) {
      unsigned slot = nir_src_as_uint(*index);
      if (slot < s->shader->info.cs_num_shaderbufs_in_user_sgprs)
         return ac_nir_load_arg(b, &s->args->ac, s->args->cs_shaderbuf[slot]);
   }

   nir_def *addr = si_nir_load_addr32_arg(s->shader->selector->screen, s->args,
                                          b, s->args->const_and_shader_buffers);
   nir_def *slot = clamp_index(b, index->ssa, b->shader->info.num_ssbos);
   slot = nir_isub_imm(b, SI_NUM_SHADER_BUFFERS - 1, slot);

   nir_def *offset = nir_ishl_imm(b, slot, 4);
   return ac_nir_load_smem(b, 4, addr, offset, 16, ACCESS_CAN_SPECULATE);
}

static nir_def *fixup_image_desc(nir_builder *b, nir_def *rsrc, bool uses_store,
                                     struct lower_resource_state *s)
{
   struct si_shader_selector *sel = s->shader->selector;
   struct si_screen *screen = sel->screen;

   /**
    * Given a 256-bit resource descriptor, force the DCC enable bit to off.
    *
    * At least on Tonga, executing image stores on images with DCC enabled and
    * non-trivial can eventually lead to lockups. This can occur when an
    * application binds an image as read-only but then uses a shader that writes
    * to it. The OpenGL spec allows almost arbitrarily bad behavior (including
    * program termination) in this case, but it doesn't cost much to be a bit
    * nicer: disabling DCC in the shader still leads to undefined results but
    * avoids the lockup.
    */
   if (uses_store &&
       screen->info.gfx_level <= GFX9 &&
       screen->info.gfx_level >= GFX8) {
      nir_def *tmp = nir_channel(b, rsrc, 6);
      tmp = nir_iand_imm(b, tmp, C_008F28_COMPRESSION_EN);
      rsrc = nir_vector_insert_imm(b, rsrc, tmp, 6);
   }

   if (!uses_store &&
       screen->info.compiler_info.has_image_load_dcc_bug &&
       screen->always_allow_dcc_stores) {
      nir_def *tmp = nir_channel(b, rsrc, 6);
      tmp = nir_iand_imm(b, tmp, C_00A018_WRITE_COMPRESS_ENABLE);
      rsrc = nir_vector_insert_imm(b, rsrc, tmp, 6);
   }

   return rsrc;
}

/* AC_DESC_FMASK is handled exactly like AC_DESC_IMAGE. The caller should
 * adjust "index" to point to FMASK.
 */
static nir_def *load_image_desc(nir_builder *b, nir_def *list, nir_def *index,
                                    enum ac_descriptor_type desc_type, bool uses_store,
                                    bool bindless, struct lower_resource_state *s)
{
   /* index is in uvec8 unit, convert to offset in bytes */
   nir_def *offset = nir_ishl_imm(b, index, 5);

   unsigned num_channels;
   if (desc_type == AC_DESC_BUFFER) {
      num_channels = 4;
   } else {
      assert(desc_type == AC_DESC_IMAGE || desc_type == AC_DESC_FMASK);
      num_channels = 8;
   }

   nir_def *rsrc = ac_nir_load_smem(b, num_channels, list, offset, 16,
                                    bindless ? 0 : ACCESS_CAN_SPECULATE);

   if (desc_type == AC_DESC_IMAGE)
      rsrc = fixup_image_desc(b, rsrc, uses_store, s);

   return rsrc;
}

static nir_def *deref_to_index(nir_builder *b,
                                   nir_deref_instr *deref,
                                   unsigned max_slots,
                                   nir_def **dynamic_index_ret,
                                   unsigned *const_index_ret)
{
   unsigned const_index = 0;
   nir_def *dynamic_index = NULL;
   while (deref->deref_type != nir_deref_type_var) {
      assert(deref->deref_type == nir_deref_type_array);
      unsigned array_size = MAX2(glsl_get_aoa_size(deref->type), 1);

      if (nir_src_is_const(deref->arr.index)) {
         const_index += array_size * nir_src_as_uint(deref->arr.index);
      } else {
         nir_def *tmp = nir_imul_imm(b, deref->arr.index.ssa, array_size);
         dynamic_index = dynamic_index ? nir_iadd(b, dynamic_index, tmp) : tmp;
      }

      deref = nir_deref_instr_parent(deref);
   }

   unsigned base_index = deref->var->data.binding;
   const_index += base_index;

   /* Redirect invalid resource indices to the first array element. */
   if (const_index >= max_slots)
      const_index = base_index;

   nir_def *index = nir_imm_int(b, const_index);
   if (dynamic_index) {
      index = nir_iadd(b, dynamic_index, index);

      /* From the GL_ARB_shader_image_load_store extension spec:
       *
       *    If a shader performs an image load, store, or atomic
       *    operation using an image variable declared as an array,
       *    and if the index used to select an individual element is
       *    negative or greater than or equal to the size of the
       *    array, the results of the operation are undefined but may
       *    not lead to termination.
       */
      index = clamp_index(b, index, max_slots);
   }

   if (dynamic_index_ret)
      *dynamic_index_ret = dynamic_index;
   if (const_index_ret)
      *const_index_ret = const_index;

   return index;
}

static nir_def *load_deref_image_desc(nir_builder *b, nir_deref_instr *deref,
                                          enum ac_descriptor_type desc_type, bool is_load,
                                          struct lower_resource_state *s)
{
   unsigned const_index;
   nir_def *dynamic_index;
   nir_def *index = deref_to_index(b, deref, b->shader->info.num_images,
                                       &dynamic_index, &const_index);

   nir_def *desc;
   if (!dynamic_index && desc_type != AC_DESC_FMASK &&
       const_index < s->shader->info.cs_num_images_in_user_sgprs) {
      /* Fast path if the image is in user SGPRs. */
      desc = ac_nir_load_arg(b, &s->args->ac, s->args->cs_image[const_index]);

      if (desc_type == AC_DESC_IMAGE)
         desc = fixup_image_desc(b, desc, !is_load, s);
   } else {
      /* FMASKs are separate from images. */
      if (desc_type == AC_DESC_FMASK)
         index = nir_iadd_imm(b, index, SI_NUM_IMAGES);

      index = nir_isub_imm(b, SI_NUM_IMAGE_SLOTS - 1, index);

      nir_def *list = si_nir_load_addr32_arg(s->shader->selector->screen, s->args,
                                             b, s->args->samplers_and_images);
      desc = load_image_desc(b, list, index, desc_type, !is_load, false, s);
   }

   return desc;
}

static nir_def *load_bindless_image_desc(nir_builder *b, nir_def *index,
                                             enum ac_descriptor_type desc_type, bool is_load,
                                             struct lower_resource_state *s)
{
   /* Bindless image descriptors use 16-dword slots. */
   index = nir_ishl_imm(b, index, 1);

   /* FMASK is right after the image. */
   if (desc_type == AC_DESC_FMASK)
      index = nir_iadd_imm(b, index, 1);

   nir_def *list = si_nir_load_addr32_arg(s->shader->selector->screen, s->args,
                                          b, s->args->bindless_samplers_and_images);
   return load_image_desc(b, list, index, desc_type, !is_load, true, s);
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

static void si_native_imagebuffer32(nir_builder *, nir_intrinsic_instr *, struct lower_resource_state *);

static bool si_native_wide_image_access(nir_builder *b, nir_intrinsic_instr *intrin,
                                      struct lower_resource_state *s)
{
   if (!UTIL_ARCH_BIG_ENDIAN || s->shader->selector->screen->info.gfx_level != GFX8 ||
       b->shader->info.internal ||
       (intrin->intrinsic != nir_intrinsic_image_deref_load &&
        intrin->intrinsic != nir_intrinsic_bindless_image_load &&
        intrin->intrinsic != nir_intrinsic_image_deref_store &&
        intrin->intrinsic != nir_intrinsic_bindless_image_store &&
        intrin->intrinsic != nir_intrinsic_image_deref_atomic &&
        intrin->intrinsic != nir_intrinsic_image_deref_atomic_swap &&
        intrin->intrinsic != nir_intrinsic_bindless_image_atomic &&
        intrin->intrinsic != nir_intrinsic_bindless_image_atomic_swap))
      return false;
   enum glsl_sampler_dim dim = (intrin->intrinsic == nir_intrinsic_image_deref_load ||
                                intrin->intrinsic == nir_intrinsic_image_deref_store ||
                                intrin->intrinsic == nir_intrinsic_image_deref_atomic ||
                                intrin->intrinsic == nir_intrinsic_image_deref_atomic_swap) ?
      glsl_get_sampler_dim(nir_src_as_deref(intrin->src[0])->type) : nir_intrinsic_image_dim(intrin);
   if (dim == GLSL_SAMPLER_DIM_BUF)
      return false;
   const struct util_format_description *format = util_format_description(nir_intrinsic_format(intrin));
   return nir_intrinsic_format(intrin) == PIPE_FORMAT_NONE ||
          nir_intrinsic_format(intrin) == PIPE_FORMAT_R11G11B10_FLOAT ||
          nir_intrinsic_format(intrin) == PIPE_FORMAT_R10G10B10A2_UNORM ||
          nir_intrinsic_format(intrin) == PIPE_FORMAT_R10G10B10A2_UINT ||
          (format && format->layout == UTIL_FORMAT_LAYOUT_PLAIN && format->is_array &&
           (format->channel[0].size == 16 || format->channel[0].size == 32));
}

static void si_lower_native_wide_image_load(nir_builder *, nir_intrinsic_instr *, nir_def *);
static void si_lower_native_wide_image_store(nir_builder *, nir_intrinsic_instr *, nir_def *);
static void si_lower_native_wide_image_atomic(nir_builder *, nir_intrinsic_instr *, nir_def *);


static bool lower_resource_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                                     struct lower_resource_state *s)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ubo: {
      assert(!(nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM));

      /* Check if the instruction already sources a descriptor and doesn't need to be lowered. */
      if (intrin->src[0].ssa->num_components == 4 && intrin->src[0].ssa->bit_size == 32)
         return false;

      nir_def *desc = load_ubo_desc(b, intrin->src[0].ssa, s);
      nir_src_rewrite(&intrin->src[0], desc);
      break;
   }
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: {
      assert(!(nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM));

      /* Check if the instruction already sources a descriptor and doesn't need to be lowered. */
      if (intrin->src[0].ssa->num_components == 4 && intrin->src[0].ssa->bit_size == 32)
         return false;

      nir_def *desc = load_ssbo_desc(b, &intrin->src[0], s);
      nir_src_rewrite(&intrin->src[0], desc);
      break;
   }
   case nir_intrinsic_store_ssbo: {
      assert(!(nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM));

      /* Check if the instruction already sources a descriptor and doesn't need to be lowered. */
      if (intrin->src[1].ssa->num_components == 4 && intrin->src[1].ssa->bit_size == 32)
         return false;

      nir_def *desc = load_ssbo_desc(b, &intrin->src[1], s);
      nir_src_rewrite(&intrin->src[1], desc);
      break;
   }
   case nir_intrinsic_load_ssbo_address: {
      assert(nir_src_as_uint(intrin->src[1]) == 0);
      nir_def *desc = load_ssbo_desc(b, &intrin->src[0], s);
      nir_def *lo = nir_channel(b, desc, 0);
      nir_def *hi = nir_i2i32(b, nir_u2u16(b, nir_channel(b, desc, 1)));
      nir_def_rewrite_uses(&intrin->def, nir_pack_64_2x32_split(b, lo, hi));
      nir_instr_remove(&intrin->instr);
      break;
   }
   case nir_intrinsic_ssbo_descriptor_amd: {
      assert(!(nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM));
      nir_def_replace(&intrin->def, load_ssbo_desc(b, &intrin->src[0], s));
      break;
   }
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_fragment_mask_load_amd:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_descriptor_amd: {
      assert(!(nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM));

      nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);

      enum ac_descriptor_type desc_type;
      if (intrin->intrinsic == nir_intrinsic_image_deref_fragment_mask_load_amd) {
         desc_type = AC_DESC_FMASK;
      } else {
         enum glsl_sampler_dim dim = glsl_get_sampler_dim(deref->type);
         desc_type = dim == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE;
      }

      bool is_load =
         intrin->intrinsic == nir_intrinsic_image_deref_load ||
         intrin->intrinsic == nir_intrinsic_image_deref_sparse_load ||
         intrin->intrinsic == nir_intrinsic_image_deref_fragment_mask_load_amd ||
         intrin->intrinsic == nir_intrinsic_image_deref_descriptor_amd;

      nir_def *desc = load_deref_image_desc(b, deref, desc_type, is_load, s);
      nir_def *native_width = NULL;
      if (si_native_wide_image_access(b, intrin, s))
         native_width = nir_channel(b, load_deref_image_desc(b, deref, AC_DESC_FMASK, true, s),
                                   glsl_get_sampler_dim(deref->type) == GLSL_SAMPLER_DIM_MS ? 7 : 0);
      if (UTIL_ARCH_BIG_ENDIAN && s->shader->selector->screen->info.gfx_level == GFX8 &&
          desc_type == AC_DESC_FMASK)
         desc = nir_vector_insert_imm(b, desc, nir_imm_int(b, 0), 7);

      if (intrin->intrinsic == nir_intrinsic_image_deref_descriptor_amd) {
         nir_def_replace(&intrin->def, desc);
      } else {
         nir_intrinsic_set_image_dim(intrin, glsl_get_sampler_dim(deref->type));
         nir_intrinsic_set_image_array(intrin, glsl_sampler_type_is_array(deref->type));
         nir_rewrite_image_intrinsic(intrin, desc, nir_image_intrinsic_type_bindless);
         if (native_width) {
            if (intrin->intrinsic == nir_intrinsic_bindless_image_store)
               si_lower_native_wide_image_store(b, intrin, native_width);
            else if (intrin->intrinsic == nir_intrinsic_bindless_image_atomic ||
                     intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap)
               si_lower_native_wide_image_atomic(b, intrin, native_width);
            else
               si_lower_native_wide_image_load(b, intrin, native_width);
         }
         else
            si_native_imagebuffer32(b, intrin, s);
      }
      break;
   }
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_fragment_mask_load_amd:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
   case nir_intrinsic_bindless_image_descriptor_amd: {
      assert(!(nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM));

      enum ac_descriptor_type desc_type;
      if (intrin->intrinsic == nir_intrinsic_bindless_image_fragment_mask_load_amd) {
         desc_type = AC_DESC_FMASK;
      } else {
         enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intrin);
         desc_type = dim == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE;
      }

      /* Check if the instruction already sources a descriptor and doesn't need to be lowered. */
      if (intrin->src[0].ssa->num_components == (desc_type == AC_DESC_BUFFER ? 4 : 8) &&
          intrin->src[0].ssa->bit_size == 32)
         return false;

      bool is_load =
         intrin->intrinsic == nir_intrinsic_bindless_image_load ||
         intrin->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
         intrin->intrinsic == nir_intrinsic_bindless_image_fragment_mask_load_amd ||
         intrin->intrinsic == nir_intrinsic_bindless_image_descriptor_amd;

      nir_def *index = intrin->src[0].ssa;
      nir_def *desc = load_bindless_image_desc(b, index, desc_type, is_load, s);
      nir_def *native_width = NULL;
      if (si_native_wide_image_access(b, intrin, s))
         native_width = nir_channel(b, load_bindless_image_desc(b, index, AC_DESC_FMASK, true, s),
                                   nir_intrinsic_image_dim(intrin) == GLSL_SAMPLER_DIM_MS ? 7 : 0);
      if (UTIL_ARCH_BIG_ENDIAN && s->shader->selector->screen->info.gfx_level == GFX8 &&
          desc_type == AC_DESC_FMASK)
         desc = nir_vector_insert_imm(b, desc, nir_imm_int(b, 0), 7);

      if (intrin->intrinsic == nir_intrinsic_bindless_image_descriptor_amd) {
         nir_def_replace(&intrin->def, desc);
      } else {
         nir_src_rewrite(&intrin->src[0], desc);
         if (native_width) {
            if (intrin->intrinsic == nir_intrinsic_bindless_image_store)
               si_lower_native_wide_image_store(b, intrin, native_width);
            else if (intrin->intrinsic == nir_intrinsic_bindless_image_atomic ||
                     intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap)
               si_lower_native_wide_image_atomic(b, intrin, native_width);
            else
               si_lower_native_wide_image_load(b, intrin, native_width);
         }
      }
      break;
   }
   default:
      return false;
   }

   return true;
}

static nir_def *load_sampler_desc(nir_builder *b, nir_def *list, nir_def *index,
                                  enum ac_descriptor_type desc_type, bool bindless)
{
   /* index is in 16 dword unit, convert to offset in bytes */
   nir_def *offset = nir_ishl_imm(b, index, 6);

   unsigned num_channels = 0;
   switch (desc_type) {
   case AC_DESC_IMAGE:
      /* The image is at [0:7]. */
      num_channels = 8;
      break;
   case AC_DESC_BUFFER:
      /* The buffer is in [0:3]. */
      num_channels = 4;
      break;
   case AC_DESC_FMASK:
      /* The FMASK is at [8:15]. */
      offset = nir_iadd_imm(b, offset, 32);
      num_channels = 8;
      break;
   case AC_DESC_SAMPLER:
      /* The sampler state is at [12:15]. */
      offset = nir_iadd_imm(b, offset, 48);
      num_channels = 4;
      break;
   default:
      UNREACHABLE("invalid desc type");
      break;
   }

   return ac_nir_load_smem(b, num_channels, list, offset, 16,
                           bindless ? 0 : ACCESS_CAN_SPECULATE);
}

static nir_def *load_deref_sampler_desc(nir_builder *b, nir_deref_instr *deref,
                                            enum ac_descriptor_type desc_type,
                                            struct lower_resource_state *s,
                                            bool return_descriptor)
{
   unsigned max_slots = BITSET_LAST_BIT(b->shader->info.textures_used);
   nir_def *index = deref_to_index(b, deref, max_slots, NULL, NULL);
   index = nir_iadd_imm(b, index, SI_NUM_IMAGE_SLOTS / 2);

   /* return actual desc when required by caller */
   if (return_descriptor) {
      nir_def *list = si_nir_load_addr32_arg(s->shader->selector->screen, s->args,
                                             b, s->args->samplers_and_images);
      return load_sampler_desc(b, list, index, desc_type, false);
   }

   /* Just use index here and let nir-to-llvm backend to translate to actual
    * descriptor. This is because we need waterfall to handle non-dynamic-uniform
    * index there.
    */
   return index;
}

static nir_def *load_bindless_sampler_desc(nir_builder *b, nir_def *index,
                                               enum ac_descriptor_type desc_type,
                                               struct lower_resource_state *s)
{
   nir_def *list = si_nir_load_addr32_arg(s->shader->selector->screen, s->args,
                                          b, s->args->bindless_samplers_and_images);

   return load_sampler_desc(b, list, index, desc_type, true);
}

static nir_def *load_tex_descriptor(nir_builder *b, nir_deref_instr *texture_deref,
                                    nir_def *texture_handle, enum ac_descriptor_type desc_type,
                                    struct lower_resource_state *s, bool return_descriptor,
                                    unsigned backend_flags)
{
   if (backend_flags & AC_NIR_TEX_BACKEND_FLAG_IS_IMAGE) {
      if (texture_deref)
         return load_deref_image_desc(b, texture_deref, desc_type, true, s);
      else
         return load_bindless_image_desc(b, texture_handle, desc_type, true, s);
   } else {
      if (texture_deref)
         return load_deref_sampler_desc(b, texture_deref, desc_type, s, return_descriptor);
      else
         return load_bindless_sampler_desc(b, texture_handle, desc_type, s);
   }
}

static nir_def *fixup_sampler_desc(nir_builder *b,
                                       nir_tex_instr *tex,
                                       nir_def *sampler,
                                       struct lower_resource_state *s)
{
   const struct si_shader_selector *sel = s->shader->selector;

   if (tex->op != nir_texop_tg4 || sel->screen->info.compiler_info.conformant_trunc_coord)
      return sampler;

   /* Set TRUNC_COORD=0 for textureGather(). */
   nir_def *dword0 = nir_channel(b, sampler, 0);
   dword0 = nir_iand_imm(b, dword0, C_008F30_TRUNC_COORD);
   sampler = nir_vector_insert_imm(b, sampler, dword0, 0);
   return sampler;
}

static nir_def *si_tbo_bswap(nir_builder *b, nir_def *x)
{
   if (x->bit_size == 8)
      return x;
   if (x->bit_size == 16)
      return nir_ior(b, nir_ushr_imm(b, x, 8), nir_ishl_imm(b, x, 8));
   return nir_ior(b,
      nir_ior(b, nir_ushr_imm(b, x, 24), nir_iand_imm(b, nir_ushr_imm(b, x, 8), 0xff00)),
      nir_ior(b, nir_iand_imm(b, nir_ishl_imm(b, x, 8), 0xff0000), nir_ishl_imm(b, x, 24)));
}

/* Wider native image views need byte permutations inside each component,
 * followed by the destination format conversion. The hardware descriptor
 * stays typed except for loads that require this permutation.
 */
static void si_lower_native_wide_image_load(nir_builder *b, nir_intrinsic_instr *intrin,
                                            nir_def *metadata)
{
   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *source_width = nir_iand_imm(b, metadata, 0xff);
   nir_def *target_width = nir_iand_imm(b, nir_ushr_imm(b, metadata, 8), 0xff);
   nir_def *channels = nir_iand_imm(b, nir_ushr_imm(b, metadata, 16), 0xff);
   nir_def *packed_target = nir_ushr_imm(b, metadata, 24);
   nir_def *active = nir_ine_imm(b, source_width, 0);
   nir_def *desc = intrin->src[0].ssa;
   nir_def *word = nir_channel(b, desc, 1);
   unsigned shift = util_logbase2(S_008F14_NUM_FORMAT(1));
   nir_def *num_format = nir_iand_imm(b, nir_ushr_imm(b, word, shift),
                                       S_008F14_NUM_FORMAT(~0u) >> shift);
   nir_def *raw_word = nir_ior_imm(b, nir_iand_imm(b, word, C_008F14_NUM_FORMAT),
                                    S_008F14_NUM_FORMAT(V_008F14_IMG_NUM_FORMAT_UINT));
   word = nir_bcsel(b, active, raw_word, word);
   desc = nir_vector_insert_imm(b, desc, word, 1);
   nir_src_rewrite(&intrin->src[0], desc);
   b->cursor = nir_after_instr(&intrin->instr);
   nir_def *values[4];
   bool floating = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intrin)) == nir_type_float;
   bool signed_int = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intrin)) == nir_type_int;
   nir_def *packed_raw = nir_channel(b, &intrin->def, 0);
   nir_def *packed_word = nir_bcsel(b, nir_ieq_imm(b, source_width, 8), si_tbo_bswap(b, packed_raw),
      nir_ior(b, nir_ushr_imm(b, packed_raw, 16), nir_ishl_imm(b, packed_raw, 16)));
   nir_def *packed_float = floating ? nir_format_unpack_11f11f10f(b, packed_word) : NULL;
   for (unsigned i = 0; i < intrin->def.num_components; i++) {
      nir_def *raw = nir_channel(b, &intrin->def, i);
      nir_def *half_swap = nir_ior(b, nir_ushr_imm(b, raw, 16), nir_ishl_imm(b, raw, 16));
      nir_def *wide = nir_bcsel(b, nir_ieq_imm(b, source_width, 8), si_tbo_bswap(b, raw), half_swap);
      nir_def *swapped = si_tbo_bswap(b, nir_u2u16(b, raw));
      nir_def *narrow;
      if (floating) {
         nir_def *unorm = nir_fmul_imm(b, nir_u2f32(b, swapped), 1.0 / 65535);
         nir_def *snorm = nir_fmax(b, nir_fmul_imm(b, nir_i2f32(b, swapped), 1.0 / 32767),
                                    nir_imm_float(b, -1));
         narrow = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F14_IMG_NUM_FORMAT_SNORM), snorm, unorm);
         narrow = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F14_IMG_NUM_FORMAT_FLOAT),
                               nir_f2f32(b, swapped), narrow);
      } else {
         narrow = signed_int ? nir_i2i32(b, swapped) : nir_u2u32(b, swapped);
      }
      nir_def *converted = nir_bcsel(b, nir_ieq_imm(b, target_width, 16), narrow, wide);
      unsigned packed_bits = i == 3 ? 2 : 10;
      unsigned packed_max = (1u << packed_bits) - 1;
      nir_def *packed_component = nir_iand_imm(b, nir_ushr_imm(b, packed_word, i * 10), packed_max);
      if (floating) {
         packed_component = nir_fmul_imm(b, nir_u2f32(b, packed_component), 1.0 / packed_max);
         nir_def *float_component = i < 3 ? nir_channel(b, packed_float, i) : nir_imm_float(b, 1);
         packed_component = nir_bcsel(b, nir_ieq_imm(b, packed_target, 3), float_component, packed_component);
      }
      converted = nir_bcsel(b, nir_ine_imm(b, packed_target, 0), packed_component, converted);
      nir_def *missing = floating ? nir_imm_float(b, i == 3 ? 1 : 0) : nir_imm_int(b, i == 3 ? 1 : 0);
      converted = nir_bcsel(b, nir_ult(b, nir_imm_int(b, i), channels), converted, missing);
      values[i] = nir_bcsel(b, active, converted, raw);
   }
   nir_def *result = nir_vec(b, values, intrin->def.num_components);
   nir_def_rewrite_uses_after(&intrin->def, result);
}

/* Reverse the native wider-view conversion before the typed image store.
 * UINT descriptors preserve the permuted bits without a second conversion.
 */
static void si_lower_native_wide_image_store(nir_builder *b, nir_intrinsic_instr *intrin,
                                             nir_def *metadata)
{
   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *source_width = nir_iand_imm(b, metadata, 0xff);
   nir_def *target_width = nir_iand_imm(b, nir_ushr_imm(b, metadata, 8), 0xff);
   nir_def *packed_target = nir_ushr_imm(b, metadata, 24);
   nir_def *active = nir_ine_imm(b, source_width, 0);
   nir_def *desc = intrin->src[0].ssa;
   nir_def *word = nir_channel(b, desc, 1);
   unsigned shift = util_logbase2(S_008F14_NUM_FORMAT(1));
   nir_def *num_format = nir_iand_imm(b, nir_ushr_imm(b, word, shift),
                                       S_008F14_NUM_FORMAT(~0u) >> shift);
   nir_def *raw_word = nir_ior_imm(b, nir_iand_imm(b, word, C_008F14_NUM_FORMAT),
                                    S_008F14_NUM_FORMAT(V_008F14_IMG_NUM_FORMAT_UINT));
   desc = nir_vector_insert_imm(b, desc, nir_bcsel(b, active, raw_word, word), 1);
   nir_src_rewrite(&intrin->src[0], desc);

   nir_def *data = intrin->src[3].ssa;
   bool floating = nir_alu_type_get_base_type(nir_intrinsic_src_type(intrin)) == nir_type_float;
   bool signed_int = nir_alu_type_get_base_type(nir_intrinsic_src_type(intrin)) == nir_type_int;
   nir_def *packed_word = nir_imm_int(b, 0);
   for (unsigned i = 0; i < 4; i++) {
      unsigned max = i == 3 ? 3 : 1023;
      nir_def *x = nir_channel(b, data, i);
      nir_def *part = floating ?
         nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, x), max))) :
         nir_umin(b, x, nir_imm_int(b, max));
      packed_word = nir_ior(b, packed_word, nir_ishl_imm(b, part, i * 10));
   }
   if (floating)
      packed_word = nir_bcsel(b, nir_ieq_imm(b, packed_target, 3),
                                nir_format_pack_11f11f10f(b, nir_channels(b, data, 7)), packed_word);
   packed_word = nir_bcsel(b, nir_ieq_imm(b, source_width, 8), si_tbo_bswap(b, packed_word),
      nir_ior(b, nir_ushr_imm(b, packed_word, 16), nir_ishl_imm(b, packed_word, 16)));

   nir_def *values[4];
   for (unsigned i = 0; i < 4; i++) {
      nir_def *x = nir_channel(b, data, i);
      nir_def *wide = nir_bcsel(b, nir_ieq_imm(b, source_width, 8), si_tbo_bswap(b, x),
         nir_ior(b, nir_ushr_imm(b, x, 16), nir_ishl_imm(b, x, 16)));
      nir_def *half;
      if (floating) {
         nir_def *unorm = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, x), 65535)));
         nir_def *bounded = nir_fmin(b, nir_fmax(b, x, nir_imm_float(b, -1)), nir_imm_float(b, 1));
         nir_def *snorm = nir_f2i32(b, nir_fround_even(b, nir_fmul_imm(b, bounded, 32767)));
         half = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F14_IMG_NUM_FORMAT_SNORM), snorm, unorm);
         half = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F14_IMG_NUM_FORMAT_FLOAT),
                            nir_u2u32(b, nir_f2f16(b, x)), half);
      } else {
         half = signed_int ?
            nir_imin(b, nir_imax(b, x, nir_imm_int(b, -32768)), nir_imm_int(b, 32767)) :
            nir_umin(b, x, nir_imm_int(b, 65535));
      }
      half = nir_u2u32(b, si_tbo_bswap(b, nir_u2u16(b, half)));
      nir_def *value = nir_bcsel(b, nir_ieq_imm(b, target_width, 16), half, wide);
      nir_def *packed_component = i == 0 ? packed_word : nir_imm_int(b, i == 3 ? 1 : 0);
      value = nir_bcsel(b, nir_ine_imm(b, packed_target, 0), packed_component, value);
      values[i] = nir_bcsel(b, active, value, x);
   }
   nir_src_rewrite(&intrin->src[3], nir_vec(b, values, 4));
}

/* An atomic operates on the native integer, not its byte-permuted storage.
 * Use a compare/exchange loop so carries and signed comparisons remain correct.
 * Bindings without conversion retain their original hardware atomic.
 */
static nir_def *si_wide_image_word(nir_builder *b, nir_def *x, nir_def *source_width)
{
   return nir_bcsel(b, nir_ieq_imm(b, source_width, 8), si_tbo_bswap(b, x),
                      nir_ior(b, nir_ushr_imm(b, x, 16), nir_ishl_imm(b, x, 16)));
}

static void si_lower_native_wide_image_atomic(nir_builder *b, nir_intrinsic_instr *intrin,
                                              nir_def *metadata)
{
   nir_def *desc = intrin->src[0].ssa, *coord = intrin->src[1].ssa;
   nir_def *sample = intrin->src[2].ssa, *arg = intrin->src[3].ssa;
   bool cmp = intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
   nir_def *second = cmp ? intrin->src[4].ssa : NULL;
   nir_atomic_op op = nir_intrinsic_atomic_op(intrin);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intrin);
   bool array = nir_intrinsic_image_array(intrin);
   enum pipe_format format = nir_intrinsic_format(intrin);
   enum gl_access_qualifier access = nir_intrinsic_access(intrin);
   nir_intrinsic_instr *original = nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intrin->instr));
   b->cursor = nir_instr_remove(&intrin->instr);
   nir_def *source_width = nir_iand_imm(b, metadata, 0xff);
   nir_if *converted = nir_push_if(b, nir_ine_imm(b, source_width, 0));
   nir_def *result;
   if (cmp) {
      result = nir_bindless_image_atomic_swap(b, 32, desc, coord, sample,
                  si_wide_image_word(b, arg, source_width),
                  si_wide_image_word(b, second, source_width),
                  .image_dim = dim, .image_array = array, .format = format,
                  .access = access, .atomic_op = nir_atomic_op_cmpxchg);
   } else {
      nir_def *zero = nir_imm_int(b, 0);
      nir_def *initial = nir_bindless_image_atomic_swap(b, 32, desc, coord, sample, zero, zero,
                  .image_dim = dim, .image_array = array, .format = format,
                  .access = access, .atomic_op = nir_atomic_op_cmpxchg);
      nir_variable *raw = nir_local_variable_create(b->impl, glsl_uint_type(), "wide_image_atomic");
      nir_store_var(b, raw, initial, 1);
      nir_loop *loop = nir_push_loop(b);
      nir_def *old = nir_load_var(b, raw), *native = si_wide_image_word(b, old, source_width), *next;
      switch (op) {
      case nir_atomic_op_iadd: next = nir_iadd(b, native, arg); break;
      case nir_atomic_op_imin: next = nir_imin(b, native, arg); break;
      case nir_atomic_op_umin: next = nir_umin(b, native, arg); break;
      case nir_atomic_op_imax: next = nir_imax(b, native, arg); break;
      case nir_atomic_op_umax: next = nir_umax(b, native, arg); break;
      case nir_atomic_op_iand: next = nir_iand(b, native, arg); break;
      case nir_atomic_op_ior: next = nir_ior(b, native, arg); break;
      case nir_atomic_op_ixor: next = nir_ixor(b, native, arg); break;
      case nir_atomic_op_xchg: next = arg; break;
      default: UNREACHABLE("unsupported native wider image atomic");
      }
      nir_def *observed = nir_bindless_image_atomic_swap(b, 32, desc, coord, sample,
                  old, si_wide_image_word(b, next, source_width),
                  .image_dim = dim, .image_array = array, .format = format,
                  .access = access, .atomic_op = nir_atomic_op_cmpxchg);
      nir_store_var(b, raw, observed, 1);
      nir_if *done = nir_push_if(b, si_native_atomic_done(b, observed, old));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, done);
      nir_pop_loop(b, loop);
      result = nir_load_var(b, raw);
   }
   result = si_wide_image_word(b, result, source_width);
   nir_push_else(b, converted);
   nir_builder_instr_insert(b, &original->instr);
   nir_pop_if(b, converted);
   nir_def_rewrite_uses(&intrin->def, nir_if_phi(b, result, &original->def));
}

static nir_def *si_native_tbo_values(nir_builder *b, nir_def *desc, nir_def *coord,
                                    nir_def *num_format, nir_def *channels,
                                    nir_alu_type type, unsigned bits, nir_variable_mode modes,
                                    enum gl_access_qualifier access)
{
   nir_def *values[4];
   bool floating = nir_alu_type_get_base_type(type) == nir_type_float;
   nir_def *zero = nir_imm_int(b, 0);
   for (unsigned i = 0; i < 4; i++) {
      nir_def *raw = nir_load_buffer_amd(b, 1, bits, desc, zero,
                                        nir_imm_int(b, i * (bits / 8)), coord,
                                        .memory_modes = modes, .access = access);
      raw = si_tbo_bswap(b, raw);
      nir_def *value;
      if (!floating) {
         value = nir_alu_type_get_base_type(type) == nir_type_int ?
                    nir_i2i32(b, raw) : nir_u2u32(b, raw);
      } else if (bits == 32) {
         value = raw;
      } else {
         nir_def *unorm = nir_fmul_imm(b, nir_u2f32(b, raw), 1.0 / ((1u << bits) - 1));
         nir_def *snorm = nir_fmax(b, nir_fmul_imm(b, nir_i2f32(b, raw),
                                                  1.0 / ((1u << (bits - 1)) - 1)),
                                     nir_imm_float(b, -1));
         value = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_SNORM),
                            snorm, unorm);
         if (bits == 16)
            value = nir_bcsel(b, nir_ieq_imm(b, num_format, V_008F0C_BUF_NUM_FORMAT_FLOAT),
                                nir_f2f32(b, raw), value);
      }
      nir_def *missing = floating ? nir_imm_float(b, i == 3 ? 1 : 0) :
                                   nir_imm_int(b, i == 3 ? 1 : 0);
      values[i] = nir_bcsel(b, nir_ult(b, nir_imm_int(b, i), channels), value, missing);
   }
   return nir_vec4(b, values[0], values[1], values[2], values[3]);
}

/* Decode one element of a big-endian texel buffer from raw loads. The format comes from
 * descriptor word 3, so this serves texture buffers and image buffers alike.
 */
static nir_def *si_native_formatted_buffer_load(nir_builder *b, nir_def *desc, nir_def *coord,
                                                nir_alu_type dest_type, nir_variable_mode modes,
                                                enum gl_access_qualifier access)
{
   nir_def *word = nir_channel(b, desc, 3);
   unsigned fmt_shift = util_logbase2(S_008F0C_DATA_FORMAT(1));
   unsigned num_shift = util_logbase2(S_008F0C_NUM_FORMAT(1));
   nir_def *fmt = nir_iand_imm(b, nir_ushr_imm(b, word, fmt_shift),
                               S_008F0C_DATA_FORMAT(~0u) >> fmt_shift);
   nir_def *num = nir_iand_imm(b, nir_ushr_imm(b, word, num_shift),
                               S_008F0C_NUM_FORMAT(~0u) >> num_shift);
   nir_def *is8 = nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_8),
      nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_8_8),
                  nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_8_8_8_8)));
   nir_def *is16 = nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_16),
      nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_16_16),
                  nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_16_16_16_16)));
   nir_def *four = nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_8_8_8_8),
      nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_16_16_16_16),
                  nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_32_32_32_32)));
   nir_def *two = nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_8_8),
      nir_ior(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_16_16),
                  nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_32_32)));
   nir_def *channels = nir_bcsel(b, four, nir_imm_int(b, 4),
      nir_bcsel(b, two, nir_imm_int(b, 2),
         nir_bcsel(b, nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_32_32_32),
                     nir_imm_int(b, 3), nir_imm_int(b, 1))));
   bool floating = nir_alu_type_get_base_type(dest_type) == nir_type_float;
   nir_def *is1010102 = nir_ieq_imm(b, fmt, V_008F0C_BUF_DATA_FORMAT_2_10_10_10);
   nir_if *if8 = nir_push_if(b, is8);
   nir_def *value8 = si_native_tbo_values(b, desc, coord, num, channels, dest_type, 8, modes, access);
   nir_push_else(b, if8);
   nir_if *if16 = nir_push_if(b, is16);
   nir_def *value16 = si_native_tbo_values(b, desc, coord, num, channels, dest_type, 16, modes, access);
   nir_push_else(b, if16);
   nir_if *if10 = nir_push_if(b, is1010102);
   nir_def *value10;
   {
      nir_def *zero = nir_imm_int(b, 0);
      nir_def *packed = si_tbo_bswap(b, nir_load_buffer_amd(b, 1, 32, desc, zero, zero, coord,
                                                            .memory_modes = modes, .access = access));
      nir_def *parts[4];
      for (unsigned i = 0; i < 4; i++) {
         unsigned bits = i == 3 ? 2 : 10, max = (1u << bits) - 1;
         parts[i] = nir_ubfe_imm(b, packed, i * 10, bits);
         if (floating)
            parts[i] = nir_fmul_imm(b, nir_u2f32(b, parts[i]), 1.0 / max);
      }
      value10 = nir_vec(b, parts, 4);
   }
   nir_push_else(b, if10);
   nir_def *value32 = si_native_tbo_values(b, desc, coord, num, channels, dest_type, 32, modes, access);
   nir_pop_if(b, if10);
   nir_def *value_wide = nir_if_phi(b, value10, value32);
   nir_pop_if(b, if16);
   nir_def *wide = nir_if_phi(b, value16, value_wide);
   nir_pop_if(b, if8);
   nir_def *value = nir_if_phi(b, value8, wide);
   nir_def *swizzled[4];
   nir_def *one = floating ? nir_imm_float(b, 1) : nir_imm_int(b, 1);
   for (unsigned i = 0; i < 4; i++) {
      nir_def *sel = nir_ubfe_imm(b, word, 3 * i, 3);
      nir_def *component = nir_imm_int(b, 0);
      for (unsigned j = 0; j < 4; j++)
         component = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F0C_SQ_SEL_X + j),
                                 nir_channel(b, value, j), component);
      swizzled[i] = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F0C_SQ_SEL_1), one, component);
   }
   value = nir_vec4(b, swizzled[0], swizzled[1], swizzled[2], swizzled[3]);
   nir_def *stride = nir_ubfe_imm(b, nir_channel(b, desc, 1), 16, 14);
   nir_def *valid = nir_iand(b, nir_ine_imm(b, stride, 0),
                              nir_ult(b, coord, nir_udiv(b, nir_channel(b, desc, 2), stride)));
   return nir_bcsel(b, valid, value, nir_imm_ivec4(b, 0, 0, 0, 0));
}

/* Texel-buffer txf and image-buffer loads arrive here as formatted load_buffer_amd
 * (ac_nir_lower_image_tex); the hardware format conversion would read host-order data
 * as little-endian, so replace them with raw loads and a native decode.
 */
static bool si_lower_native_formatted_load(nir_builder *b, nir_intrinsic_instr *intrin)
{
   unsigned access = nir_intrinsic_access(intrin);
   /* Only texel/image buffers (see ac_nir_lower_image_tex); vertex fetch has its own path. */
   if (!(nir_intrinsic_memory_modes(intrin) & (nir_var_uniform | nir_var_image)) ||
       (nir_intrinsic_memory_modes(intrin) & nir_var_shader_in))
      return false;
   if (!(access & ACCESS_USES_FORMAT_AMD) || (access & ACCESS_SPARSE) ||
       intrin->def.bit_size != 32 || intrin->src[0].ssa->num_components != 4 ||
       !nir_src_is_const(intrin->src[1]) || nir_src_as_uint(intrin->src[1]) != 0 ||
       !nir_src_is_const(intrin->src[2]) || nir_src_as_uint(intrin->src[2]) != 0)
      return false;
   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *value = si_native_formatted_buffer_load(b, intrin->src[0].ssa, intrin->src[3].ssa,
                                                    nir_intrinsic_dest_type(intrin),
                                                    nir_intrinsic_memory_modes(intrin),
                                                    access & ~(ACCESS_USES_FORMAT_AMD));
   value = nir_trim_vector(b, value, intrin->def.num_components);
   nir_def_replace(&intrin->def, value);
   return true;
}

/* Turn the raw result of a view wider than its texture into the view's values; the
 * descriptor word is described at si_native_wide_view_meta().
 */
static nir_def *si_lower_native_wide_tex(nir_builder *b, nir_def *raw, nir_def *meta, bool floating)
{
   nir_def *kind = nir_ubfe_imm(b, meta, 1, 2);
   nir_def *type = nir_ubfe_imm(b, meta, 3, 4);
   nir_def *comp[4];

   for (unsigned i = 0; i < 4; i++) {
      nir_def *x = nir_channel(b, raw, i);
      nir_def *bytes16 = nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, x, 8), 0xff),
                                 nir_ishl_imm(b, nir_iand_imm(b, x, 0xff), 8));
      nir_def *halves = nir_ior(b, nir_ushr_imm(b, x, 16), nir_ishl_imm(b, x, 16));
      x = nir_bcsel(b, nir_ieq_imm(b, kind, 0), x,
          nir_bcsel(b, nir_ieq_imm(b, kind, 1), bytes16,
                    nir_bcsel(b, nir_ieq_imm(b, kind, 2), si_tbo_bswap(b, x), halves)));
      if (floating) {
         nir_def *sx = nir_ishr_imm(b, nir_ishl_imm(b, x, 16), 16);
         nir_def *unorm = nir_fmul_imm(b, nir_u2f32(b, x), 1.0 / 65535.0);
         nir_def *snorm = nir_fmax(b, nir_fmul_imm(b, nir_i2f32(b, sx), 1.0 / 32767.0),
                                   nir_imm_float(b, -1.0));
         nir_def *half = nir_channel(b, nir_unpack_half_2x16(b, x), 0);
         comp[i] = nir_bcsel(b, nir_ieq_imm(b, type, 3), unorm,
                   nir_bcsel(b, nir_ieq_imm(b, type, 4), snorm,
                   nir_bcsel(b, nir_ieq_imm(b, type, 5), half,
                   i < 3 ? nir_bcsel(b, nir_ieq_imm(b, type, 10), nir_format_srgb_to_linear(b, x), x) : x)));
      } else {
         nir_def *sx = nir_ishr_imm(b, nir_ishl_imm(b, x, 16), 16);
         comp[i] = nir_bcsel(b, nir_ieq_imm(b, type, 2), sx, x);
      }
   }

   /* Packed formats use one word, in the first channel. */
   nir_def *word = comp[0];
   nir_def *packed[4];
   if (floating) {
      nir_def *w = nir_bcsel(b, nir_uge_imm(b, type, 6), word, nir_imm_int(b, 0));
      nir_def *f11 = nir_format_unpack_11f11f10f(b, w);
      nir_def *exponent = nir_iadd_imm(b, nir_ushr_imm(b, w, 27), -24);
      nir_def *scale = nir_ldexp(b, nir_imm_float(b, 1.0), exponent);
      for (unsigned i = 0; i < 4; i++) {
         nir_def *unorm = nir_fmul_imm(b, nir_u2f32(b, nir_ubfe_imm(b, w, i * 10, i == 3 ? 2 : 10)),
                                       i == 3 ? 1.0 / 3.0 : 1.0 / 1023.0);
         nir_def *e5 = i == 3 ? nir_imm_float(b, 1.0)
                              : nir_fmul(b, nir_u2f32(b, nir_ubfe_imm(b, w, i * 9, 9)), scale);
         nir_def *f = i == 3 ? nir_imm_float(b, 1.0) : nir_channel(b, f11, i);
         packed[i] = nir_bcsel(b, nir_ieq_imm(b, type, 6), unorm,
                     nir_bcsel(b, nir_ieq_imm(b, type, 8), f,
                     nir_bcsel(b, nir_ieq_imm(b, type, 9), e5, comp[i])));
      }
   } else {
      for (unsigned i = 0; i < 4; i++)
         packed[i] = nir_bcsel(b, nir_ieq_imm(b, type, 7),
                               nir_ubfe_imm(b, word, i * 10, i == 3 ? 2 : 10), comp[i]);
   }

   /* The swizzle the descriptor would have applied. */
   nir_def *one = floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *channels = nir_ubfe_imm(b, meta, 7, 3);
   nir_def *out[4];
   for (unsigned i = 0; i < 4; i++) {
      nir_def *sel = nir_ubfe_imm(b, meta, 10 + 3 * i, 3);
      nir_def *value = zero;
      for (unsigned j = 0; j < 4; j++) {
         nir_def *present = nir_ult(b, nir_imm_int(b, j), channels);
         value = nir_bcsel(b, nir_iand(b, nir_ieq_imm(b, sel, V_008F1C_SQ_SEL_X + j), present),
                           packed[j], value);
      }
      out[i] = nir_bcsel(b, nir_ieq_imm(b, sel, V_008F1C_SQ_SEL_1), one, value);
   }
   return nir_vec4(b, out[0], out[1], out[2], out[3]);
}

static bool lower_resource_tex(nir_builder *b, nir_tex_instr *tex,
                               struct lower_resource_state *s)
{
   nir_deref_instr *texture_deref = NULL;
   nir_deref_instr *sampler_deref = NULL;
   nir_def *texture_handle = NULL;
   nir_def *sampler_handle = NULL;
   bool has_sampler = tex->sampler_dim != GLSL_SAMPLER_DIM_BUF &&
                      tex->sampler_dim != GLSL_SAMPLER_DIM_MS;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref:
         texture_deref = nir_src_as_deref(tex->src[i].src);
         break;
      case nir_tex_src_sampler_deref:
         if (has_sampler) {
            sampler_deref = nir_src_as_deref(tex->src[i].src);
         } else {
            nir_tex_instr_remove_src(tex, i);
            i--;
         }
         break;
      case nir_tex_src_texture_handle:
         texture_handle = tex->src[i].src.ssa;
         break;
      case nir_tex_src_sampler_handle:
         if (has_sampler) {
            sampler_handle = tex->src[i].src.ssa;
         } else {
            nir_tex_instr_remove_src(tex, i);
            i--;
         }
         break;
      default:
         break;
      }
   }

   enum ac_descriptor_type desc_type;
   if (tex->op == nir_texop_fragment_mask_fetch_amd)
      desc_type = AC_DESC_FMASK;
   else
      desc_type = tex->sampler_dim == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE;

   if (tex->op == nir_texop_descriptor_amd) {
      nir_def *image = load_tex_descriptor(b, texture_deref, texture_handle, desc_type, s, true,
                                           tex->backend_flags);
      nir_def_replace(&tex->def, image);
      return true;
   }

   if (tex->op == nir_texop_sampler_descriptor_amd) {
      assert(has_sampler);
      nir_def *sampler = load_tex_descriptor(b, sampler_deref, sampler_handle, AC_DESC_SAMPLER, s,
                                             true, tex->backend_flags);
      nir_def_replace(&tex->def, sampler);
      return true;
   }

   nir_def *image = load_tex_descriptor(b, texture_deref, texture_handle, desc_type, s,
                                        !tex->texture_non_uniform, tex->backend_flags);

   nir_def *sampler = NULL;
   if (sampler_deref || sampler_handle) {
      sampler = load_tex_descriptor(b, sampler_deref, sampler_handle, AC_DESC_SAMPLER, s,
                                    !tex->sampler_non_uniform, tex->backend_flags);
   }

   if (sampler && sampler->num_components > 1)
      sampler = fixup_sampler_desc(b, tex, sampler, s);

   /* Word 0 of the unused FMASK slot says whether this view needs converting. */
   nir_def *native_wide = NULL;
   if (UTIL_ARCH_BIG_ENDIAN && s->shader->selector->screen->info.gfx_level == GFX8 &&
       !b->shader->info.internal && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_MS && !tex->is_shadow && !tex->texture_non_uniform &&
       tex->def.bit_size == 32 &&
       (tex->op == nir_texop_tex || tex->op == nir_texop_txb || tex->op == nir_texop_txl ||
        tex->op == nir_texop_txd || tex->op == nir_texop_txf)) {
      nir_def *fmask = texture_deref ?
         load_deref_sampler_desc(b, texture_deref, AC_DESC_FMASK, s, true) :
         load_bindless_sampler_desc(b, texture_handle, AC_DESC_FMASK, s);
      native_wide = nir_channel(b, fmask, 0);
   }

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref:
         tex->src[i].src_type = nir_tex_src_texture_handle;
         FALLTHROUGH;
      case nir_tex_src_texture_handle:
         nir_src_rewrite(&tex->src[i].src, image);
         break;
      case nir_tex_src_sampler_deref:
         tex->src[i].src_type = nir_tex_src_sampler_handle;
         FALLTHROUGH;
      case nir_tex_src_sampler_handle:
         nir_src_rewrite(&tex->src[i].src, sampler);
         break;
      default:
         break;
      }
   }

   if (native_wide) {
      b->cursor = nir_after_instr(&tex->instr);
      const unsigned n = tex->def.num_components;
      const bool floating = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
      /* A copy, so that rewriting the texture's users below leaves the conversion alone. */
      nir_def *copy = nir_mov(b, &tex->def);
      nir_def *raw = nir_pad_vector_imm_int(b, copy, 0, 4);
      nir_if *wide = nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, native_wide, 1), 0));
      nir_def *converted = si_lower_native_wide_tex(b, raw, native_wide, floating);
      nir_push_else(b, wide);
      nir_pop_if(b, wide);
      nir_def *phi = nir_if_phi(b, converted, raw);
      nir_def *result = n == 4 ? phi : nir_trim_vector(b, phi, n);
      nir_def_rewrite_uses_after_instr(&tex->def, result, nir_def_instr(copy));
   }

   return true;
}

/* A writeonly image without a format qualifier only learns its element layout from
 * the descriptor. Without this the values reach the buffer little-endian, and a draw
 * reading them back indirectly sees enormous vertex counts and hangs the GPU.
 */
static void si_native_imagebuffer_noformat_store(nir_builder *b, nir_intrinsic_instr *intrin)
{
   nir_def *desc = intrin->src[0].ssa;
   nir_def *coord = nir_channel(b, intrin->src[1].ssa, 0);
   nir_def *value = intrin->src[3].ssa;
   nir_def *zero = nir_imm_int(b, 0);
   enum gl_access_qualifier access = nir_intrinsic_access(intrin);

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *word3 = nir_channel(b, desc, 3);
   nir_def *dfmt = nir_iand_imm(b, nir_ushr_imm(b, word3, 15), 0xf);
   nir_def *num = nir_iand_imm(b, nir_ushr_imm(b, word3, 12), 0x7);

   nir_def *is32 = nir_ior(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_32),
                       nir_ior(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_32_32),
                       nir_ior(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_32_32_32),
                               nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_32_32_32_32))));
   nir_def *is16 = nir_ior(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_16),
                       nir_ior(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_16_16),
                               nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_16_16_16_16)));
   nir_def *channels =
      nir_bcsel(b, nir_ior(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_32),
                           nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_16)), nir_imm_int(b, 1),
      nir_bcsel(b, nir_ior(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_32_32),
                           nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_16_16)), nir_imm_int(b, 2),
      nir_bcsel(b, nir_ieq_imm(b, dfmt, V_008F0C_BUF_DATA_FORMAT_32_32_32), nir_imm_int(b, 3),
                nir_imm_int(b, 4))));

   nir_instr_remove(&intrin->instr);

   nir_push_if(b, is32);
   for (unsigned i = 0; i < 4; i++) {
      nir_push_if(b, nir_ult(b, nir_imm_int(b, i), channels));
      nir_store_buffer_amd(b, si_tbo_bswap(b, nir_channel(b, value, i)), desc, zero,
                           nir_imm_int(b, i * 4), coord, .write_mask = 1,
                           .memory_modes = nir_var_image, .access = access);
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   nir_push_if(b, is16);
   for (unsigned i = 0; i < 4; i++) {
      nir_push_if(b, nir_ult(b, nir_imm_int(b, i), channels));
      nir_def *c = nir_channel(b, value, i);
      nir_def *fc = c;
      nir_def *unorm = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, fc), 65535.0)));
      nir_def *snorm = nir_f2i32(b, nir_fround_even(b, nir_fmul_imm(b,
                          nir_fmin(b, nir_fmax(b, fc, nir_imm_float(b, -1.0)), nir_imm_float(b, 1.0)),
                          32767.0)));
      nir_def *uint = nir_umin(b, c, nir_imm_int(b, 65535));
      nir_def *sint = nir_imin(b, nir_imax(b, c, nir_imm_int(b, -32768)), nir_imm_int(b, 32767));
      nir_def *half = nir_u2u32(b, nir_f2f16(b, fc));
      nir_def *raw =
         nir_bcsel(b, nir_ieq_imm(b, num, V_008F0C_BUF_NUM_FORMAT_UNORM), unorm,
         nir_bcsel(b, nir_ieq_imm(b, num, V_008F0C_BUF_NUM_FORMAT_SNORM), snorm,
         nir_bcsel(b, nir_ieq_imm(b, num, V_008F0C_BUF_NUM_FORMAT_FLOAT), half,
         nir_bcsel(b, nir_ieq_imm(b, num, V_008F0C_BUF_NUM_FORMAT_SINT), sint, uint))));
      nir_store_buffer_amd(b, si_tbo_bswap(b, nir_u2u16(b, raw)), desc, zero,
                           nir_imm_int(b, i * 2), coord, .write_mask = 1,
                           .memory_modes = nir_var_image, .access = access);
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   /* Byte-sized formats have no order to fix; the hardware converts them. */
   nir_instr_insert(b->cursor, &intrin->instr);
   b->cursor = nir_after_instr(&intrin->instr);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

/* Convert native 32-bit array elements for typed image buffer access. */
static void si_native_imagebuffer32(nir_builder *b, nir_intrinsic_instr *intrin,
                                    struct lower_resource_state *s)
{
   if (!UTIL_ARCH_BIG_ENDIAN || s->shader->selector->screen->info.gfx_level != GFX8 ||
       b->shader->info.internal || nir_intrinsic_image_dim(intrin) != GLSL_SAMPLER_DIM_BUF)
      return;
   if (nir_intrinsic_format(intrin) == PIPE_FORMAT_NONE &&
       intrin->intrinsic == nir_intrinsic_bindless_image_store) {
      si_native_imagebuffer_noformat_store(b, intrin);
      return;
   }
   const struct util_format_description *format = util_format_description(nir_intrinsic_format(intrin));
   if (!format || format->layout != UTIL_FORMAT_LAYOUT_PLAIN)
      return;
   if (format->format == PIPE_FORMAT_R10G10B10A2_UNORM ||
       format->format == PIPE_FORMAT_R10G10B10A2_UINT) {
      nir_def *desc = intrin->src[0].ssa;
      nir_def *coord = nir_channel(b, intrin->src[1].ssa, 0);
      nir_def *zero = nir_imm_int(b, 0);
      bool normalized = format->format == PIPE_FORMAT_R10G10B10A2_UNORM;
      b->cursor = nir_before_instr(&intrin->instr);
      if (intrin->intrinsic == nir_intrinsic_bindless_image_load) {
         nir_def *word = nir_load_buffer_amd(b, 1, 32, desc, zero, zero, coord,
                                            .memory_modes = nir_var_image,
                                            .access = nir_intrinsic_access(intrin));
         word = si_tbo_bswap(b, word);
         nir_def *components[4];
         for (unsigned i = 0; i < 4; i++) {
            unsigned bits = i == 3 ? 2 : 10, max = (1u << bits) - 1;
            nir_def *part = nir_iand_imm(b, nir_ushr_imm(b, word, i * 10), max);
            components[i] = normalized ? nir_fmul_imm(b, nir_u2f32(b, part), 1.0 / max) : part;
         }
         nir_def *value = nir_vec(b, components, 4);
         nir_def_rewrite_uses(&intrin->def, value);
         nir_instr_remove(&intrin->instr);
      } else if (intrin->intrinsic == nir_intrinsic_bindless_image_store) {
         nir_def *word = zero;
         for (unsigned i = 0; i < 4; i++) {
            unsigned bits = i == 3 ? 2 : 10, max = (1u << bits) - 1;
            nir_def *part = nir_channel(b, intrin->src[3].ssa, i);
            if (normalized) {
               part = nir_fsat(b, part);
               part = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, part, max)));
            } else {
               part = nir_umin(b, part, nir_imm_int(b, max));
            }
            part = nir_ishl_imm(b, nir_iand_imm(b, part, max), i * 10);
            word = nir_ior(b, word, part);
         }
         nir_store_buffer_amd(b, si_tbo_bswap(b, word), desc, zero, zero, coord,
                              .write_mask = 1, .memory_modes = nir_var_image,
                              .access = nir_intrinsic_access(intrin));
         nir_instr_remove(&intrin->instr);
      }
      return;
   }
   if (!format->is_array)
      return;
   if (format->channel[0].size == 16 &&
       intrin->intrinsic == nir_intrinsic_bindless_image_load) {
      nir_def *desc = intrin->src[0].ssa;
      nir_def *coord = nir_channel(b, intrin->src[1].ssa, 0);
      nir_def *word = nir_channel(b, desc, 3);
      unsigned shift = util_logbase2(S_008F0C_NUM_FORMAT(1));
      nir_def *num = nir_iand_imm(b, nir_ushr_imm(b, word, shift),
                                  S_008F0C_NUM_FORMAT(~0u) >> shift);
      nir_def *channels = nir_imm_int(b, format->nr_channels);
      b->cursor = nir_before_instr(&intrin->instr);
      nir_def *value = si_native_tbo_values(b, desc, coord, num, channels,
                                            nir_intrinsic_dest_type(intrin), 16, nir_var_image,
                                            nir_intrinsic_access(intrin));
      nir_def_rewrite_uses(&intrin->def, value);
      nir_instr_remove(&intrin->instr);
      return;
   }
   if (format->channel[0].size == 16 &&
       intrin->intrinsic == nir_intrinsic_bindless_image_store) {
      nir_def *desc = intrin->src[0].ssa;
      nir_def *coord = nir_channel(b, intrin->src[1].ssa, 0);
      nir_def *zero = nir_imm_int(b, 0);
      b->cursor = nir_before_instr(&intrin->instr);
      for (unsigned i = 0; i < format->nr_channels; ++i) {
         nir_def *component = nir_channel(b, intrin->src[3].ssa, i);
         nir_def *raw;
         if (format->channel[0].type == UTIL_FORMAT_TYPE_FLOAT) {
            raw = nir_f2f16(b, component);
         } else if (format->channel[0].normalized) {
            bool sign = format->channel[0].type == UTIL_FORMAT_TYPE_SIGNED;
            nir_def *bounded = nir_fmin(b, nir_fmax(b, component, nir_imm_float(b, sign ? -1.0 : 0.0)), nir_imm_float(b, 1.0));
            nir_def *scaled = nir_fround_even(b, nir_fmul_imm(b, bounded, sign ? 32767.0 : 65535.0));
            raw = nir_u2u16(b, sign ? nir_f2i32(b, scaled) : nir_f2u32(b, scaled));
         } else {
            if (format->channel[0].type == UTIL_FORMAT_TYPE_SIGNED)
               component = nir_imin(b, nir_imax(b, component, nir_imm_int(b, -32768)), nir_imm_int(b, 32767));
            else
               component = nir_umin(b, component, nir_imm_int(b, 65535));
            raw = nir_u2u16(b, component);
         }
         raw = si_tbo_bswap(b, raw);
         nir_store_buffer_amd(b, raw, desc, zero, nir_imm_int(b, i * 2), coord,
                              .write_mask = 1, .memory_modes = nir_var_image,
                              .access = nir_intrinsic_access(intrin));
      }
      nir_instr_remove(&intrin->instr);
      return;
   }
   if (format->channel[0].size != 32)
      return;
   if (intrin->intrinsic == nir_intrinsic_bindless_image_atomic ||
       intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap) {
      nir_def *desc = intrin->src[0].ssa, *coord = intrin->src[1].ssa;
      nir_def *sample = intrin->src[2].ssa, *arg = intrin->src[3].ssa;
      nir_atomic_op op = nir_intrinsic_atomic_op(intrin);
      enum gl_access_qualifier access = nir_intrinsic_access(intrin);
      bool cmp = intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
      nir_def *second = cmp ? intrin->src[4].ssa : NULL;
      b->cursor = nir_instr_remove(&intrin->instr);
      nir_def *result;
      if (cmp) {
         result = nir_bindless_image_atomic_swap(b, 32, desc, coord, sample,
                    si_tbo_bswap(b, arg), si_tbo_bswap(b, second),
                    .image_dim = GLSL_SAMPLER_DIM_BUF, .format = format->format,
                    .access = access, .atomic_op = nir_atomic_op_cmpxchg);
      } else {
         nir_def *zero = nir_imm_int(b, 0);
         nir_def *initial = nir_bindless_image_atomic_swap(b, 32, desc, coord, sample, zero, zero,
                    .image_dim = GLSL_SAMPLER_DIM_BUF, .format = format->format,
                    .access = access, .atomic_op = nir_atomic_op_cmpxchg);
         nir_variable *raw = nir_local_variable_create(b->impl, glsl_uint_type(), "native_image_atomic");
         nir_store_var(b, raw, initial, 1);
         nir_loop *loop = nir_push_loop(b);
         nir_def *old = nir_load_var(b, raw), *native = si_tbo_bswap(b, old), *next;
         switch (op) {
         case nir_atomic_op_iadd: next = nir_iadd(b, native, arg); break;
         case nir_atomic_op_imin: next = nir_imin(b, native, arg); break;
         case nir_atomic_op_umin: next = nir_umin(b, native, arg); break;
         case nir_atomic_op_imax: next = nir_imax(b, native, arg); break;
         case nir_atomic_op_umax: next = nir_umax(b, native, arg); break;
         case nir_atomic_op_iand: next = nir_iand(b, native, arg); break;
         case nir_atomic_op_ior: next = nir_ior(b, native, arg); break;
         case nir_atomic_op_ixor: next = nir_ixor(b, native, arg); break;
         case nir_atomic_op_xchg: next = arg; break;
         case nir_atomic_op_fadd: next = nir_fadd(b, native, arg); break;
         case nir_atomic_op_fmin: next = nir_fmin(b, native, arg); break;
         case nir_atomic_op_fmax: next = nir_fmax(b, native, arg); break;
         case nir_atomic_op_inc_wrap: next = nir_bcsel(b, nir_uge(b, native, arg), nir_imm_int(b, 0), nir_iadd_imm(b, native, 1)); break;
         case nir_atomic_op_dec_wrap: next = nir_bcsel(b, nir_ior(b, nir_ieq_imm(b, native, 0), nir_ult(b, arg, native)), arg, nir_iadd_imm(b, native, -1)); break;
         default: UNREACHABLE("unsupported native image buffer atomic");
         }
         nir_def *observed = nir_bindless_image_atomic_swap(b, 32, desc, coord, sample,
                    old, si_tbo_bswap(b, next), .image_dim = GLSL_SAMPLER_DIM_BUF,
                    .format = format->format, .access = access, .atomic_op = nir_atomic_op_cmpxchg);
         nir_store_var(b, raw, observed, 1);
         nir_if *done = nir_push_if(b, si_native_atomic_done(b, observed, old));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, done);
         nir_pop_loop(b, loop);
         result = nir_load_var(b, raw);
      }
      nir_def_rewrite_uses(&intrin->def, si_tbo_bswap(b, result));
      return;
   }
   nir_def *original;
   bool load = intrin->intrinsic == nir_intrinsic_bindless_image_load;
   if (load) {
      b->cursor = nir_after_instr(&intrin->instr);
      original = &intrin->def;
   } else if (intrin->intrinsic == nir_intrinsic_bindless_image_store) {
      b->cursor = nir_before_instr(&intrin->instr);
      original = intrin->src[3].ssa;
   } else {
      return;
   }
   nir_def *components[4];
   for (unsigned i = 0; i < original->num_components; ++i) {
      nir_def *x = nir_channel(b, original, i);
      components[i] = i < format->nr_channels ? si_tbo_bswap(b, x) : x;
   }
   nir_def *value = nir_vec(b, components, original->num_components);
   if (load)
      nir_def_rewrite_uses_after(&intrin->def, value);
   else
      nir_src_rewrite(&intrin->src[3], value);
}

static bool lower_resource_instr(nir_builder *b, nir_instr *instr, void *state)
{
   struct lower_resource_state *s = (struct lower_resource_state *)state;

   b->cursor = nir_before_instr(instr);

   switch (instr->type) {
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (UTIL_ARCH_BIG_ENDIAN && intrin->intrinsic == nir_intrinsic_load_buffer_amd &&
          s->shader->selector->screen->info.gfx_level == GFX8 && !b->shader->info.internal)
         return si_lower_native_formatted_load(b, intrin);
      return lower_resource_intrinsic(b, intrin, s);
   }
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      return lower_resource_tex(b, tex, s);
   }
   default:
      return false;
   }
}

bool si_nir_lower_resource(nir_shader *nir, struct si_shader *shader,
                           struct si_shader_args *args)
{
   struct lower_resource_state state = {
      .shader = shader,
      .args = args,
   };

   nir_metadata preserved = UTIL_ARCH_BIG_ENDIAN && shader->selector->screen->info.gfx_level == GFX8 ?
                               nir_metadata_none : nir_metadata_control_flow;
   return nir_shader_instructions_pass(nir, lower_resource_instr, preserved, &state);
}
