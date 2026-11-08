/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir/radv_meta_nir.h"
#include "radv_entrypoints.h"
#include "radv_formats.h"
#include "radv_meta.h"
#include "vk_shader_module.h"

/*
 * GFX queue: Compute shader implementation of image->buffer copy
 * Compute queue: implementation also of buffer->image, image->image, and image clear.
 */

static VkResult
get_itob_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER;

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 2,
      .pBindings = bindings,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 16,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_copy_buffer_image_key {
   enum radv_meta_object_key_type type;
   bool is_3d;
};

static VkResult
get_itob_pipeline(struct radv_device *device, const struct radv_image *image, VkPipeline *pipeline_out,
                  VkPipelineLayout *layout_out)
{
   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   struct radv_copy_buffer_image_key key;
   VkResult result;

   result = get_itob_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER;
   key.is_3d = is_3d;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_itob_compute_shader(is_3d);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static VkResult
get_btoi_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_BUFFER_TO_IMAGE;

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 2,
      .pBindings = bindings,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 16,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

static VkResult
get_btoi_pipeline(struct radv_device *device, const struct radv_image *image, VkPipeline *pipeline_out,
                  VkPipelineLayout *layout_out)
{
   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   struct radv_copy_buffer_image_key key;
   VkResult result;

   result = get_btoi_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_COPY_BUFFER_TO_IMAGE;
   key.is_3d = is_3d;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_btoi_compute_shader(is_3d);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static VkResult
get_itoi_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_IMAGE;

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 2,
      .pBindings = bindings,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 24,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_copy_image_key {
   enum radv_meta_object_key_type type;
   bool src_3d;
   bool dst_3d;
   uint8_t samples_log2;
};

static VkResult
get_itoi_pipeline(struct radv_device *device, const struct radv_image *src_image, const struct radv_image *dst_image,
                  uint32_t samples, VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   const bool src_3d = src_image->vk.image_type == VK_IMAGE_TYPE_3D;
   const bool dst_3d = dst_image->vk.image_type == VK_IMAGE_TYPE_3D;
   const uint32_t samples_log2 = ffs(samples) - 1;
   VkResult result;
   struct radv_copy_image_key key;

   result = get_itoi_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_COPY_IMAGE;
   key.src_3d = src_3d;
   key.dst_3d = dst_3d;
   key.samples_log2 = samples_log2;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_itoi_compute_shader(src_3d, dst_3d, samples);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static VkResult
get_cleari_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_CLEAR_CS;

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 20,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_clear_key {
   enum radv_meta_object_key_type type;
   bool is_3d;
   uint8_t samples_log2;
};

static VkResult
get_cleari_pipeline(struct radv_device *device, const struct radv_image *image, VkPipeline *pipeline_out,
                    VkPipelineLayout *layout_out)
{
   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   const uint32_t samples = image->vk.samples;
   const uint32_t samples_log2 = ffs(samples) - 1;
   struct radv_clear_key key;
   VkResult result;

   result = get_cleari_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_CLEAR_CS;
   key.is_3d = is_3d;
   key.samples_log2 = samples_log2;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_cleari_compute_shader(is_3d, samples);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static VkResult
get_cleari_96bit_pipeline(struct radv_device *device, VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_CLEAR_CS_96BIT;
   VkResult result;

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 16,
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key,
                                        sizeof(key), layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_cleari_96bit_compute_shader();

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static void
create_iview(struct radv_cmd_buffer *cmd_buffer, struct radv_meta_blit2d_surf *surf, struct radv_image_view *iview,
             VkFormat format, VkImageAspectFlagBits aspects, VkImageUsageFlags2KHR usage)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);

   if (format == VK_FORMAT_UNDEFINED)
      format = surf->format;

   const VkImageViewUsage2CreateInfoKHR iview_usage_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
      .usage = usage,
   };

   radv_image_view_init(iview, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .pNext = &iview_usage_info,
                           .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                           .image = radv_image_to_handle(surf->image),
                           .viewType = radv_meta_get_view_type(surf->image),
                           .format = format,
                           .subresourceRange = {.aspectMask = aspects,
                                                .baseMipLevel = surf->level,
                                                .levelCount = 1,
                                                .baseArrayLayer = surf->layer,
                                                .layerCount = 1},
                        },
                        &(struct radv_image_view_extra_create_info){
                           .disable_compression = surf->disable_compression,
                        });
}

/* GFX9+ has an issue where the HW does not calculate mipmap degradations
 * for block-compressed images correctly (see the comment in
 * radv_image_view_init). Some texels are unaddressable and cannot be copied
 * to/from by a compute shader. Here we will perform a buffer copy to copy the
 * texels that the hardware missed.
 *
 * GFX10+ will not use this workaround because it can be fixed by adjusting its
 * image view descriptors instead.
 *
 * GFX12+ is not affected (see NO_EDGE_CLAMP).
 */
static void
fixup_gfx9_cs_copy(struct radv_cmd_buffer *cmd_buffer, const struct radv_meta_blit2d_buffer *buf_bsurf,
                   const struct radv_meta_blit2d_surf *img_bsurf, const VkOffset3D *offset, const VkExtent3D *extent,
                   bool to_image)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const unsigned mip_level = img_bsurf->level;
   const struct radv_image *image = img_bsurf->image;
   const struct radeon_surf *surf = &image->planes[0].surface;
   const struct radeon_info *gpu_info = &pdev->info;
   struct ac_surf_info surf_info = radv_get_ac_surf_info(device, image);
   VkAddressCopyFlagsKHR img_copy_flags = 0, mem_copy_flags = 0;

   if (gpu_info->gfx_level < GFX9 || gpu_info->gfx_level >= GFX12)
      return;

   if (image->vk.mip_levels == 1 || !vk_format_is_block_compressed(image->vk.format))
      return;

   /* GFX10 will use a different workaround unless this is not a 2D image */
   if (gpu_info->gfx_level >= GFX10 && image->vk.image_type == VK_IMAGE_TYPE_2D)
      return;

   /* The physical extent of the base mip */
   VkExtent2D hw_base_extent = {surf->u.gfx9.base_mip_width, surf->u.gfx9.base_mip_height};

   /* The hardware-calculated extent of the selected mip
    * (naive divide-by-two integer math)
    */
   VkExtent2D hw_mip_extent = {u_minify(hw_base_extent.width, mip_level), u_minify(hw_base_extent.height, mip_level)};

   /* The actual extent we want to copy */
   VkExtent2D mip_extent = {extent->width, extent->height};

   VkOffset2D mip_offset = {offset->x, offset->y};

   if (hw_mip_extent.width >= mip_offset.x + mip_extent.width &&
       hw_mip_extent.height >= mip_offset.y + mip_extent.height)
      return;

   if (!to_image) {
      /* If we are writing to a buffer, then we need to wait for the compute
       * shader to finish because it may write over the unaddressable texels
       * while we're fixing them. If we're writing to an image, we do not need
       * to wait because the compute shader cannot write to those texels
       */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_L2 | RADV_CMD_FLAG_INV_VCACHE;
   }

   if (image->bindings[0].bo)
      img_copy_flags |= radv_get_copy_flags_from_bo(image->bindings[0].bo);
   mem_copy_flags |= buf_bsurf->copy_flags;

   for (uint32_t y = 0; y < mip_extent.height; y++) {
      uint32_t coordY = y + mip_offset.y;
      /* If the default copy algorithm (done previously) has already seen this
       * scanline, then we can bias the starting X coordinate over to skip the
       * region already copied by the default copy.
       */
      uint32_t x = (coordY < hw_mip_extent.height) ? hw_mip_extent.width : 0;
      for (; x < mip_extent.width; x++) {
         uint32_t coordX = x + mip_offset.x;
         uint64_t img_offset =
            ac_surface_addr_from_coord(pdev->addrlib, gpu_info, surf, &surf_info, mip_level, coordX, coordY,
                                       img_bsurf->layer, image->vk.image_type == VK_IMAGE_TYPE_3D);
         const uint64_t img_va = image->bindings[0].addr + img_offset;
         /* buf_bsurf->offset already includes the layer offset */
         const uint64_t mem_va = buf_bsurf->addr + buf_bsurf->offset + y * buf_bsurf->pitch * surf->bpe + x * surf->bpe;
         if (to_image) {
            radv_copy_memory(cmd_buffer, mem_va, img_va, surf->bpe, mem_copy_flags, img_copy_flags);
         } else {
            radv_copy_memory(cmd_buffer, img_va, mem_va, surf->bpe, img_copy_flags, mem_copy_flags);
         }
      }
   }
}

void
radv_meta_image_to_buffer(struct radv_cmd_buffer *cmd_buffer, struct radv_meta_blit2d_surf *src,
                          struct radv_meta_blit2d_buffer *dst, const VkOffset3D *offset, const VkExtent3D *extent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_image_view src_view;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_itob_pipeline(device, src->image, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   create_iview(cmd_buffer, src, &src_view, VK_FORMAT_UNDEFINED, src->aspect_mask, VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR);

   radv_meta_bind_descriptors(
      cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 2,
      (VkDescriptorGetInfoEXT[]){{
                                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                    .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                    .data.pSampledImage =
                                       &(VkDescriptorImageInfo){
                                          .sampler = VK_NULL_HANDLE,
                                          .imageView = radv_image_view_to_handle(&src_view),
                                          .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                       },
                                 },
                                 {
                                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                    .type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                                    .data.pStorageTexelBuffer =
                                       &(VkDescriptorAddressInfoEXT){
                                          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                                          .address = dst->addr + dst->offset,
                                          .range = dst->size - dst->offset,
                                          .format = dst->format,
                                       },
                                 }});

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   unsigned push_constants[4] = {offset->x, offset->y, src->layer, dst->pitch};

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), push_constants);

   radv_unaligned_dispatch(cmd_buffer, extent->width, extent->height, 1);
   fixup_gfx9_cs_copy(cmd_buffer, dst, src, offset, extent, false);

   radv_image_view_finish(&src_view);
}

void
radv_meta_buffer_to_image_cs(struct radv_cmd_buffer *cmd_buffer, struct radv_meta_blit2d_buffer *src,
                             struct radv_meta_blit2d_surf *dst, const VkOffset3D *offset, const VkExtent3D *extent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_image_view dst_view;
   uint32_t texel_scale = 1;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   if (vk_format_is_96bit(dst->image->vk.format)) {
      src->format = radv_meta_get_96bit_channel_format(src->format);
      dst->format = src->format;
      texel_scale = 3;
   }

   result = get_btoi_pipeline(device, dst->image, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   create_iview(cmd_buffer, dst, &dst_view, VK_FORMAT_UNDEFINED, dst->aspect_mask, VK_IMAGE_USAGE_2_STORAGE_BIT_KHR);

   radv_meta_bind_descriptors(
      cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 2,
      (VkDescriptorGetInfoEXT[]){{
                                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                    .type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                                    .data.pStorageTexelBuffer =
                                       &(VkDescriptorAddressInfoEXT){
                                          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                                          .address = src->addr + src->offset,
                                          .range = src->size - src->offset,
                                          .format = src->format,
                                       },
                                 },
                                 {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                  .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  .data.pStorageImage = (VkDescriptorImageInfo[]){
                                     {
                                        .sampler = VK_NULL_HANDLE,
                                        .imageView = radv_image_view_to_handle(&dst_view),
                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                     },
                                  }}});

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   unsigned push_constants[4] = {
      offset->x * texel_scale,
      offset->y,
      dst->layer,
      src->pitch * texel_scale,
   };

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), push_constants);

   radv_unaligned_dispatch(cmd_buffer, extent->width * texel_scale, extent->height, 1);
   fixup_gfx9_cs_copy(cmd_buffer, src, dst, offset, extent, true);

   radv_image_view_finish(&dst_view);
}

void
radv_meta_image_to_image_cs(struct radv_cmd_buffer *cmd_buffer, struct radv_meta_blit2d_surf *src,
                            struct radv_meta_blit2d_surf *dst, const VkOffset3D *src_offset,
                            const VkOffset3D *dst_offset, const VkExtent3D *extent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_image_view src_view, dst_view;
   uint32_t samples = src->image->vk.samples;
   uint32_t texel_scale = 1;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   if (vk_format_is_96bit(src->format)) {
      /* 96-bit formats are only compatible to themselves. */
      assert(vk_format_is_96bit(dst->format));

      src->format = radv_meta_get_96bit_channel_format(src->format);
      dst->format = src->format;
      texel_scale = 3;
   }

   result = get_itoi_pipeline(device, src->image, dst->image, samples, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   u_foreach_bit (i, dst->aspect_mask) {
      unsigned dst_aspect_mask = 1u << i;
      unsigned src_aspect_mask = dst_aspect_mask;
      VkFormat depth_format = 0;
      if (dst_aspect_mask == VK_IMAGE_ASPECT_STENCIL_BIT)
         depth_format = vk_format_stencil_only(dst->image->vk.format);
      else if (dst_aspect_mask == VK_IMAGE_ASPECT_DEPTH_BIT)
         depth_format = vk_format_depth_only(dst->image->vk.format);
      else {
         /*
          * "Multi-planar images can only be copied on a per-plane basis, and the subresources used in each region when
          * copying to or from such images must specify only one plane, though different regions can specify different
          * planes."
          */
         assert((dst->aspect_mask & (dst->aspect_mask - 1)) == 0);
         assert((src->aspect_mask & (src->aspect_mask - 1)) == 0);
         src_aspect_mask = src->aspect_mask;
      }

      /* Adjust the aspect for color to depth/stencil image copies. */
      if (vk_format_is_color(src->image->vk.format) && vk_format_is_depth_or_stencil(dst->image->vk.format)) {
         assert(src->aspect_mask == VK_IMAGE_ASPECT_COLOR_BIT);
         src_aspect_mask = src->aspect_mask;
      } else if (vk_format_is_depth_or_stencil(src->image->vk.format) && vk_format_is_color(dst->image->vk.format)) {
         if (src->aspect_mask == VK_IMAGE_ASPECT_STENCIL_BIT) {
            depth_format = vk_format_stencil_only(src->image->vk.format);
         } else {
            assert(src->aspect_mask == VK_IMAGE_ASPECT_DEPTH_BIT);
            depth_format = vk_format_depth_only(src->image->vk.format);
         }
      }

      create_iview(cmd_buffer, src, &src_view,
                   (src_aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) ? depth_format : 0,
                   src_aspect_mask, VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR);
      create_iview(cmd_buffer, dst, &dst_view,
                   dst_aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) ? depth_format : 0,
                   dst_aspect_mask, VK_IMAGE_USAGE_2_STORAGE_BIT_KHR);

      radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 2,
                                 (VkDescriptorGetInfoEXT[]){{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                                             .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                                             .data.pSampledImage =
                                                                (VkDescriptorImageInfo[]){
                                                                   {
                                                                      .sampler = VK_NULL_HANDLE,
                                                                      .imageView = radv_image_view_to_handle(&src_view),
                                                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                                   },
                                                                }},
                                                            {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                                             .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                             .data.pStorageImage = (VkDescriptorImageInfo[]){
                                                                {
                                                                   .sampler = VK_NULL_HANDLE,
                                                                   .imageView = radv_image_view_to_handle(&dst_view),
                                                                   .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                                },
                                                             }}});

      radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

      unsigned push_constants[6] = {
         src_offset->x * texel_scale, src_offset->y, src->layer, dst_offset->x * texel_scale, dst_offset->y, dst->layer,
      };

      radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                               push_constants);

      radv_unaligned_dispatch(cmd_buffer, extent->width * texel_scale, extent->height, 1);

      radv_image_view_finish(&src_view);
      radv_image_view_finish(&dst_view);
   }
}

static void
radv_meta_clear_image_cs_96bit(struct radv_cmd_buffer *cmd_buffer, struct radv_meta_blit2d_surf *dst,
                               const VkClearColorValue *clear_color)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   unsigned stride;
   VkResult result;

   result = get_cleari_96bit_pipeline(device, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_cs_add_buffer(device->ws, cs->b, dst->image->bindings[0].bo);

   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1,
                              (VkDescriptorGetInfoEXT[]){{
                                 .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                 .type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                                 .data.pStorageTexelBuffer =
                                    &(VkDescriptorAddressInfoEXT){
                                       .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                                       .address = dst->image->bindings[0].addr,
                                       .range = dst->image->size,
                                       .format = radv_meta_get_96bit_channel_format(dst->format),
                                    },
                              }});

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   stride = radv_get_image_stride_for_96bit(device, dst->image);

   unsigned push_constants[4] = {
      clear_color->uint32[0],
      clear_color->uint32[1],
      clear_color->uint32[2],
      stride,
   };

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), push_constants);

   radv_unaligned_dispatch(cmd_buffer, dst->image->vk.extent.width, dst->image->vk.extent.height, 1);
}

/* Big endian: a view of four 16-bit channels over storage of 32-bit words has each pair
 * reversed in memory, and color attachments cannot write that order (COMP_SWAP has no
 * g,r,a,b). So rendering swaps the pairs in memory on entry and swaps them back on exit.
 */
static VkResult
be_swap_pairs_create(struct radv_device *device)
{
   struct radv_meta_state *state = &device->meta_state;
   VkResult result = VK_SUCCESS;

   mtx_lock(&state->mtx);
   if (state->be_swap_pairs.pipeline)
      goto out;

   const VkDescriptorSetLayoutCreateInfo ds_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = 1,
      .pBindings = &(VkDescriptorSetLayoutBinding){.binding = 0,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   result = radv_CreateDescriptorSetLayout(radv_device_to_handle(device), &ds_info, &state->alloc,
                                           &state->be_swap_pairs.ds_layout);
   if (result != VK_SUCCESS)
      goto out;

   const VkPipelineLayoutCreateInfo pl_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &state->be_swap_pairs.ds_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_COMPUTE_BIT, 0, 12},
   };
   result = radv_CreatePipelineLayout(radv_device_to_handle(device), &pl_info, &state->alloc,
                                      &state->be_swap_pairs.p_layout);
   if (result != VK_SUCCESS)
      goto out;

   const struct glsl_type *img_type = glsl_image_type(GLSL_SAMPLER_DIM_2D, true, GLSL_TYPE_UINT);
   nir_builder b = radv_meta_init_shader(device, MESA_SHADER_COMPUTE, "meta_be_swap_pairs");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   nir_variable *img = nir_variable_create(b.shader, nir_var_image, img_type, "img");
   img->data.descriptor_set = 0;
   img->data.binding = 0;
   img->data.image.format = PIPE_FORMAT_R32G32_UINT;

   nir_def *id = get_global_ids(&b, 2);
   nir_def *push = nir_load_push_constant(&b, 3, 32, nir_imm_int(&b, 0), .range = 12);
   nir_def *coord = nir_vec4(&b, nir_iadd(&b, nir_channel(&b, id, 0), nir_channel(&b, push, 0)),
                             nir_iadd(&b, nir_channel(&b, id, 1), nir_channel(&b, push, 1)),
                             nir_channel(&b, push, 2), nir_undef(&b, 1, 32));
   nir_def *value = nir_image_deref_load(&b, 4, 32, &nir_build_deref_var(&b, img)->def, coord,
                                         nir_undef(&b, 1, 32), nir_imm_int(&b, 0),
                                         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = true,
                                         .dest_type = nir_type_uint32);
   nir_def *x = nir_channel(&b, value, 0), *y = nir_channel(&b, value, 1);
   nir_def *swapped = nir_vec4(&b, nir_ior(&b, nir_ushr_imm(&b, x, 16), nir_ishl_imm(&b, x, 16)),
                               nir_ior(&b, nir_ushr_imm(&b, y, 16), nir_ishl_imm(&b, y, 16)),
                               nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_image_deref_store(&b, &nir_build_deref_var(&b, img)->def, coord, nir_undef(&b, 1, 32), swapped,
                         nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = true,
                         .src_type = nir_type_uint32);

   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = vk_shader_module_handle_from_nir(b.shader),
                .pName = "main"},
      .layout = state->be_swap_pairs.p_layout,
   };
   result = radv_compute_pipeline_create(radv_device_to_handle(device), state->cache, &info, NULL,
                                         &state->be_swap_pairs.pipeline);
   ralloc_free(b.shader);

out:
   mtx_unlock(&state->mtx);
   return result;
}

void
radv_meta_be_swap_16bit_pairs(struct radv_cmd_buffer *cmd_buffer, struct radv_image_view *iview)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   struct radv_meta_saved_state saved_state;
   struct radv_image_view view;

   if (be_swap_pairs_create(device) != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   radv_meta_save(&saved_state, cmd_buffer,
                  RADV_META_SAVE_COMPUTE_PIPELINE | RADV_META_SAVE_CONSTANTS | RADV_META_SAVE_DESCRIPTORS);

   radv_image_view_init(&view, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .image = radv_image_to_handle(iview->image),
                           .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                           .format = VK_FORMAT_R32G32_UINT,
                           .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                .baseMipLevel = iview->vk.base_mip_level,
                                                .levelCount = 1,
                                                .baseArrayLayer = iview->vk.base_array_layer,
                                                .layerCount = iview->vk.layer_count},
                        },
                        0, NULL);

   radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer), VK_PIPELINE_BIND_POINT_COMPUTE,
                        device->meta_state.be_swap_pairs.pipeline);
   radv_meta_push_descriptor_set(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, device->meta_state.be_swap_pairs.p_layout,
                                 0, 1,
                                 (VkWriteDescriptorSet[]){
                                    {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                     .dstBinding = 0,
                                     .descriptorCount = 1,
                                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                     .pImageInfo = (VkDescriptorImageInfo[]){{
                                        .imageView = radv_image_view_to_handle(&view),
                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                     }}},
                                 });

   const uint32_t layers = render->view_mask ? util_last_bit(render->view_mask) : render->layer_count;
   for (uint32_t layer = 0; layer < layers; layer++) {
      if (render->view_mask && !(render->view_mask & (1u << layer)))
         continue;
      const uint32_t push[3] = {(uint32_t)render->area.offset.x, (uint32_t)render->area.offset.y, layer};
      vk_common_CmdPushConstants(radv_cmd_buffer_to_handle(cmd_buffer), device->meta_state.be_swap_pairs.p_layout,
                                 VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      radv_unaligned_dispatch(cmd_buffer, render->area.extent.width, render->area.extent.height, 1);
   }

   radv_image_view_finish(&view);
   radv_meta_restore(&saved_state, cmd_buffer);
}

/* Big endian: a view of four 16-bit channels over storage of 32-bit words has each pair
 * reversed in memory, and color attachments cannot write that order (COMP_SWAP has no
 * g,r,a,b). So rendering swaps the pairs in memory on entry and swaps them back on exit.
 */
static nir_shader *
build_be_swap_pairs_shader(void)
{
   const struct glsl_type *img_type = glsl_image_type(GLSL_SAMPLER_DIM_2D, true, GLSL_TYPE_UINT);
   nir_builder b = radv_meta_nir_init_shader(MESA_SHADER_COMPUTE, "meta_be_swap_pairs");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   nir_variable *img = nir_variable_create(b.shader, nir_var_image, img_type, "img");
   img->data.descriptor_set = 0;
   img->data.binding = 0;
   img->data.image.format = PIPE_FORMAT_R32G32_UINT;

   nir_def *id = radv_meta_nir_get_global_ids(&b, 2);
   nir_def *push = nir_load_push_constant(&b, 3, 32, nir_imm_int(&b, 0), .range = 12);
   nir_def *coord = nir_vec4(&b, nir_iadd(&b, nir_channel(&b, id, 0), nir_channel(&b, push, 0)),
                             nir_iadd(&b, nir_channel(&b, id, 1), nir_channel(&b, push, 1)), nir_channel(&b, push, 2),
                             nir_undef(&b, 1, 32));
   nir_def *value =
      nir_image_deref_load(&b, 4, 32, &nir_build_deref_var(&b, img)->def, coord, nir_undef(&b, 1, 32),
                           nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = true,
                           .dest_type = nir_type_uint32);
   nir_def *x = nir_channel(&b, value, 0), *y = nir_channel(&b, value, 1);
   nir_def *swapped = nir_vec4(&b, nir_ior(&b, nir_ushr_imm(&b, x, 16), nir_ishl_imm(&b, x, 16)),
                               nir_ior(&b, nir_ushr_imm(&b, y, 16), nir_ishl_imm(&b, y, 16)), nir_imm_int(&b, 0),
                               nir_imm_int(&b, 0));
   nir_image_deref_store(&b, &nir_build_deref_var(&b, img)->def, coord, nir_undef(&b, 1, 32), swapped,
                         nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = true,
                         .src_type = nir_type_uint32);
   return b.shader;
}

static VkResult
get_be_swap_pairs_pipeline(struct radv_device *device, VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_BE_SWAP_16BIT_PAIRS;
   VkResult result;

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 12,
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key,
                                        sizeof(key), layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = build_be_swap_pairs_shader();
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = vk_shader_module_handle_from_nir(cs),
            .pName = "main",
         },
      .layout = *layout_out,
   };
   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);
   ralloc_free(cs);
   return result;
}

void
radv_meta_be_swap_16bit_pairs(struct radv_cmd_buffer *cmd_buffer, struct radv_image_view *iview)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   struct radv_image_view view;
   VkPipelineLayout layout;
   VkPipeline pipeline;

   VkResult result = get_be_swap_pairs_pipeline(device, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   const bool nested = cmd_buffer->state.meta.inside_meta_op;
   if (!nested)
      radv_meta_begin(cmd_buffer);

   radv_image_view_init(&view, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                           .image = radv_image_to_handle(iview->image),
                           .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                           .format = VK_FORMAT_R32G32_UINT,
                           .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                .baseMipLevel = iview->vk.base_mip_level,
                                                .levelCount = 1,
                                                .baseArrayLayer = iview->vk.base_array_layer,
                                                .layerCount = iview->vk.layer_count},
                        },
                        NULL);

   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1,
                              (VkDescriptorGetInfoEXT[]){
                                 {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                  .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  .data.pStorageImage =
                                     (VkDescriptorImageInfo[]){
                                        {
                                           .sampler = VK_NULL_HANDLE,
                                           .imageView = radv_image_view_to_handle(&view),
                                           .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        },
                                     }},
                              });
   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   const uint32_t layers = render->view_mask ? util_last_bit(render->view_mask) : render->layer_count;
   for (uint32_t layer = 0; layer < layers; layer++) {
      if (render->view_mask && !(render->view_mask & (1u << layer)))
         continue;
      const uint32_t push[3] = {(uint32_t)render->area.offset.x, (uint32_t)render->area.offset.y, layer};
      radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      radv_unaligned_dispatch(cmd_buffer, render->area.extent.width, render->area.extent.height, 1);
   }

   radv_image_view_finish(&view);
   if (!nested)
      radv_meta_end(cmd_buffer);
}

void
radv_meta_clear_image_cs(struct radv_cmd_buffer *cmd_buffer, struct radv_meta_blit2d_surf *dst,
                         const VkClearColorValue *clear_color)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_image_view dst_iview;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   if (vk_format_is_96bit(dst->format)) {
      radv_meta_clear_image_cs_96bit(cmd_buffer, dst, clear_color);
      return;
   }

   result = get_cleari_pipeline(device, dst->image, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   create_iview(cmd_buffer, dst, &dst_iview, VK_FORMAT_UNDEFINED, dst->aspect_mask, VK_IMAGE_USAGE_2_STORAGE_BIT_KHR);

   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1,
                              (VkDescriptorGetInfoEXT[]){
                                 {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                  .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  .data.pStorageImage =
                                     (VkDescriptorImageInfo[]){
                                        {
                                           .sampler = VK_NULL_HANDLE,
                                           .imageView = radv_image_view_to_handle(&dst_iview),
                                           .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        },
                                     }},
                              });

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   unsigned push_constants[5] = {
      clear_color->uint32[0], clear_color->uint32[1], clear_color->uint32[2], clear_color->uint32[3], dst->layer,
   };

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), push_constants);

   radv_unaligned_dispatch(cmd_buffer, dst->image->vk.extent.width, dst->image->vk.extent.height, 1);

   radv_image_view_finish(&dst_iview);
}
