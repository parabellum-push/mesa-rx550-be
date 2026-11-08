/*
 * Copyright © 2016 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_formats.h"
#include "radv_meta.h"
#include "radv_meta_native_indices.h"
#include "radv_sdma.h"
#include "vk_format.h"

VkFormat
vk_format_for_size(int bs)
{
   switch (bs) {
   case 1:
      return VK_FORMAT_R8_UINT;
   case 2:
      return VK_FORMAT_R8G8_UINT;
   case 4:
      return VK_FORMAT_R8G8B8A8_UINT;
   case 8:
      return VK_FORMAT_R16G16B16A16_UINT;
   case 12:
      return VK_FORMAT_R32G32B32_UINT;
   case 16:
      return VK_FORMAT_R32G32B32A32_UINT;
   default:
      UNREACHABLE("Invalid format block size");
   }
}

struct radv_meta_blit2d_surf
radv_blit_surf_for_image_level_layer(struct radv_image *image, VkImageLayout layout,
                                     const VkImageSubresourceLayers *subres)
{
   VkFormat format = radv_get_aspect_format(image, subres->aspectMask);

   if (!radv_dcc_enabled(image, subres->mipLevel) && !(radv_tc_compat_htile_enabled(image, subres->mipLevel)))
      format = vk_format_for_size(vk_format_get_blocksize(format));

   format = vk_format_no_srgb(format);

   return (struct radv_meta_blit2d_surf){
      .format = format,
      .bs = vk_format_get_blocksize(format),
      .level = subres->mipLevel,
      .layer = subres->baseArrayLayer,
      .image = image,
      .aspect_mask = subres->aspectMask,
      .current_layout = layout,
   };
}

static bool
alloc_transfer_temp_bo(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   if (!cmd_buffer->transfer.copy_temp) {
      const VkResult r =
         radv_bo_create(device, &cmd_buffer->vk.base, RADV_SDMA_TRANSFER_TEMP_BYTES, 4096, RADEON_DOMAIN_VRAM,
                        RADEON_FLAG_NO_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING, RADV_BO_PRIORITY_SCRATCH, 0,
                        true, &cmd_buffer->transfer.copy_temp);

      if (r != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd_buffer->vk, r);
         return false;
      }
   }

   radv_cs_add_buffer(device->ws, cs->b, cmd_buffer->transfer.copy_temp);
   return true;
}

static void gfx_or_compute_copy_memory_to_image(struct radv_cmd_buffer *cmd_buffer,
                                                VkAddressCopyFlagsKHR src_copy_flags, struct radv_image *image,
                                                const VkDeviceMemoryImageCopyKHR *region, bool use_compute);

static void compute_copy_image_to_memory(struct radv_cmd_buffer *cmd_buffer, VkAddressCopyFlagsKHR dst_copy_flags,
                                         struct radv_image *image, const VkDeviceMemoryImageCopyKHR *region);

static void
transfer_copy_memory_image(struct radv_cmd_buffer *cmd_buffer, VkAddressCopyFlagsKHR buffer_flags,
                           struct radv_image *image, const VkDeviceMemoryImageCopyKHR *region, bool to_image)
{
   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   if (!radv_sdma_supports_image(device, image, to_image)) {
      if (!radv_gang_init(cmd_buffer))
         return;

      if (radv_flush_gang_leader_semaphore(cmd_buffer)) {
         radv_wait_gang_leader(cmd_buffer);
         cmd_buffer->gang.flush_bits |= RADV_CMD_FLAG_INV_L2;
      }

      radv_gang_cache_flush(cmd_buffer);

      if (to_image) {
         gfx_or_compute_copy_memory_to_image(cmd_buffer, buffer_flags, image, region, true);
      } else {
         compute_copy_image_to_memory(cmd_buffer, buffer_flags, image, region);
      }

      return;
   }

   if (cmd_buffer->gang.cs && radv_flush_gang_follower_semaphore(cmd_buffer))
      radv_wait_gang_follower(cmd_buffer);

   const VkOffset3D img_offset_el = vk_image_offset_to_elements(&image->vk, region->imageOffset);
   const VkExtent3D img_extent_el = vk_image_extent_to_elements(&image->vk, region->imageExtent);

   struct ac_sdma_surf buf = radv_sdma_get_buf_surf(image, region);
   const struct ac_sdma_surf img =
      radv_sdma_get_surf(cmd_buffer, image, region->imageLayout, region->imageSubresource, img_offset_el);
   const VkExtent3D extent = radv_sdma_get_copy_extent(image, region->imageSubresource, img_extent_el);

   if (radv_sdma_use_unaligned_buffer_image_copy(device, &buf, &img, extent)) {
      if (!alloc_transfer_temp_bo(cmd_buffer))
         return;

      radv_sdma_copy_buffer_image_unaligned(device, cs, &buf, &img, extent, cmd_buffer->transfer.copy_temp, to_image);
      return;
   }

   radv_sdma_copy_buffer_image(device, cs, &buf, &img, extent, to_image);
}

static void
radv_fixup_copy_dst_htile_metadata(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image,
                                   VkImageLayout image_layout, const VkImageSubresourceLayers *subresource,
                                   const VkOffset3D *offset, const VkExtent3D *extent, bool before_copy)
{
   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);

   const uint32_t queue_mask = radv_image_queue_family_mask(image, cmd_buffer->qf, cmd_buffer->qf);
   if (!radv_layout_is_htile_compressed(device, image, subresource->mipLevel, image_layout, queue_mask))
      return;

   if (radv_image_decompress_htile_on_image_stores(device, image))
      return;

   const bool is_partial_copy = offset->x || offset->y || offset->z || extent->width != image->vk.extent.width ||
                                extent->height != image->vk.extent.height || extent->depth != image->vk.extent.depth;

   if (before_copy) {
      /* For partial copies, HTILE is decompressed before because image stores don't write the
       * uncompressed DWORD to HTILE. And then it's needed to re-initialize HTILE to its
       * uncompressed state after the copy.
       */
      if (is_partial_copy) {
         radv_describe_barrier_start(cmd_buffer, RGP_BARRIER_UNKNOWN_REASON);

         u_foreach_bit (i, subresource->aspectMask) {
            unsigned aspect_mask = 1u << i;
            radv_expand_depth_stencil(cmd_buffer, image,
                                      &(VkImageSubresourceRange){
                                         .aspectMask = aspect_mask,
                                         .baseMipLevel = subresource->mipLevel,
                                         .levelCount = 1,
                                         .baseArrayLayer = subresource->baseArrayLayer,
                                         .layerCount = vk_image_subresource_layer_count(&image->vk, subresource),
                                      },
                                      NULL);
         }

         radv_describe_barrier_end(cmd_buffer);
      }
   } else {
      if (!is_partial_copy) {
         /* Fixup HTILE after a copy on compute, but not for partial copies because decompressing the
          * image also means that HTILE is re-initialized to its uncompressed state.
          */
         const VkImageSubresourceRange range = {
            .aspectMask = subresource->aspectMask,
            .baseMipLevel = subresource->mipLevel,
            .levelCount = 1,
            .baseArrayLayer = subresource->baseArrayLayer,
            .layerCount = vk_image_subresource_layer_count(&image->vk, subresource),
         };
         const uint32_t htile_value = radv_get_htile_initial_value(device, image);

         cmd_buffer->state.flush_bits |= radv_clear_htile(cmd_buffer, image, &range, htile_value, false);
      }
   }
}

static void
gfx_or_compute_copy_memory_to_image(struct radv_cmd_buffer *cmd_buffer, VkAddressCopyFlagsKHR src_copy_flags,
                                    struct radv_image *image, const VkDeviceMemoryImageCopyKHR *region,
                                    bool use_compute)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);

   /* The Vulkan 1.0 spec says "dstImage must have a sample count equal to
    * VK_SAMPLE_COUNT_1_BIT."
    */
   assert(image->vk.samples == 1);

   if (use_compute) {
      radv_fixup_copy_dst_htile_metadata(cmd_buffer, image, region->imageLayout, &region->imageSubresource,
                                         &region->imageOffset, &region->imageExtent, true);
   }

   /**
    * From the Vulkan 1.0.6 spec: 18.3 Copying Data Between Images
    *    extent is the size in texels of the source image to copy in width,
    *    height and depth. 1D images use only x and width. 2D images use x, y,
    *    width and height. 3D images use x, y, z, width, height and depth.
    *
    *
    * Also, convert the offsets and extent from units of texels to units of
    * blocks - which is the highest resolution accessible in this command.
    */
   const VkOffset3D img_offset_el = vk_image_offset_to_elements(&image->vk, region->imageOffset);

   /* Start creating blit rect */
   const VkExtent3D img_extent_el = vk_image_extent_to_elements(&image->vk, region->imageExtent);

   /* Create blit surfaces */
   struct radv_meta_blit2d_surf img_bsurf =
      radv_blit_surf_for_image_level_layer(image, region->imageLayout, &region->imageSubresource);

   if (!radv_is_buffer_format_supported(img_bsurf.format, NULL)) {
      const uint32_t queue_mask = radv_image_queue_family_mask(image, cmd_buffer->qf, cmd_buffer->qf);
      const VkFormat raw_format = vk_format_for_size(vk_format_get_blocksize(img_bsurf.format));

      if (!radv_dcc_formats_compatible(pdev->info.gfx_level, img_bsurf.format, raw_format, NULL) &&
          radv_layout_dcc_compressed(device, image, region->imageSubresource.mipLevel, region->imageLayout,
                                     queue_mask)) {
         radv_describe_barrier_start(cmd_buffer, RGP_BARRIER_UNKNOWN_REASON);

         radv_decompress_dcc(cmd_buffer, image,
                             &(VkImageSubresourceRange){
                                .aspectMask = region->imageSubresource.aspectMask,
                                .baseMipLevel = region->imageSubresource.mipLevel,
                                .levelCount = 1,
                                .baseArrayLayer = region->imageSubresource.baseArrayLayer,
                                .layerCount = vk_image_subresource_layer_count(&image->vk, &region->imageSubresource),
                             });
         img_bsurf.disable_compression = true;

         radv_describe_barrier_end(cmd_buffer);
      }

      img_bsurf.format = raw_format;
   }

   const struct vk_image_buffer_layout buf_layout = vk_image_memory_copy_layout(&image->vk, region);
   struct radv_meta_blit2d_buffer buf_bsurf = {
      .addr = region->addressRange.address,
      .size = region->addressRange.size,
      .format = img_bsurf.format,
      .pitch = buf_layout.row_stride_B / buf_layout.element_size_B,
      .copy_flags = src_copy_flags,
   };

   if (image->vk.image_type == VK_IMAGE_TYPE_3D)
      img_bsurf.layer = img_offset_el.z;
   /* Loop through each 3D or array slice */
   unsigned num_slices_3d = img_extent_el.depth;
   unsigned num_slices_array = vk_image_subresource_layer_count(&image->vk, &region->imageSubresource);
   unsigned slice_3d = 0;
   unsigned slice_array = 0;
   while (slice_3d < num_slices_3d && slice_array < num_slices_array) {
      /* Perform Blit */
      if (use_compute) {
         radv_meta_buffer_to_image_cs(cmd_buffer, &buf_bsurf, &img_bsurf, &img_offset_el, &img_extent_el);
      } else {
         radv_gfx_copy_memory_to_image(cmd_buffer, &buf_bsurf, &img_bsurf, &img_offset_el, &img_extent_el);
      }

      /* Once we've done the blit, all of the actual information about
       * the image is embedded in the command buffer so we can just
       * increment the offset directly in the image effectively
       * re-binding it to different backing memory.
       */
      buf_bsurf.offset += buf_layout.image_stride_B;
      img_bsurf.layer++;
      if (image->vk.image_type == VK_IMAGE_TYPE_3D)
         slice_3d++;
      else
         slice_array++;
   }

   if (use_compute) {
      radv_fixup_copy_dst_htile_metadata(cmd_buffer, image, region->imageLayout, &region->imageSubresource,
                                         &region->imageOffset, &region->imageExtent, false);
   }
}

/* Big endian GFX8: image memory holds what the GPU decodes, little-endian elements.
 * Application memory holds host-order elements, so copies between them swap each element.
 */
static unsigned
radv_native_copy_element_width(const struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image,
                               VkImageAspectFlags aspect_mask)
{
   const struct radv_physical_device *pdev = radv_device_physical(radv_cmd_buffer_device(cmd_buffer));
   if (!UTIL_ARCH_BIG_ENDIAN || pdev->info.gfx_level != GFX8 || cmd_buffer->qf == RADV_QUEUE_TRANSFER)
      return 1;

   const VkFormat format = radv_get_aspect_format((struct radv_image *)image, aspect_mask);
   if (image->plane_count > 1) {
      const unsigned plane = radv_plane_from_aspect(aspect_mask);
      return image->native_plane_width[plane] ? image->native_plane_width[plane] : radv_native_format_width(format);
   }
   if (vk_format_is_depth_or_stencil(image->vk.format))
      return radv_native_format_width(format);
   return image->native_width;
}

/* Returns the byte range of a region and whether it has bytes the copy skips. */
static uint64_t
radv_native_copy_size(const struct radv_image *image, const VkDeviceMemoryImageCopyKHR *region, bool *has_gaps)
{
   const VkFormat format = radv_get_aspect_format((struct radv_image *)image, region->imageSubresource.aspectMask);
   const uint64_t bs = vk_format_get_blocksize(format);
   const VkExtent3D buffer_extent = {region->addressRowLength ? region->addressRowLength : region->imageExtent.width,
                                     region->addressImageHeight ? region->addressImageHeight
                                                                : region->imageExtent.height,
                                     1};
   const VkExtent3D buf_el = vk_image_extent_to_elements(&image->vk, buffer_extent);
   const VkExtent3D img_el = vk_image_extent_to_elements(&image->vk, region->imageExtent);
   const uint64_t slices = MAX2(img_el.depth, vk_image_subresource_layer_count(&image->vk, &region->imageSubresource));

   *has_gaps = buf_el.width != img_el.width || (slices > 1 && buf_el.height != img_el.height);
   return (slices - 1) * buf_el.height * buf_el.width * bs + (uint64_t)(img_el.height - 1) * buf_el.width * bs +
          img_el.width * bs;
}

static void
radv_be_copy_memory_to_image(struct radv_cmd_buffer *cmd_buffer, VkAddressCopyFlagsKHR src_copy_flags,
                             struct radv_image *image, const VkDeviceMemoryImageCopyKHR *region, bool use_compute)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const unsigned width = radv_native_copy_element_width(cmd_buffer, image, region->imageSubresource.aspectMask);
   if (width == 1) {
      gfx_or_compute_copy_memory_to_image(cmd_buffer, src_copy_flags, image, region, use_compute);
      return;
   }

   bool has_gaps;
   const uint64_t size = radv_native_copy_size(image, region, &has_gaps);
   struct radeon_winsys_bo *bo;
   unsigned offset;
   if (!radv_meta_alloc_native_scratch(cmd_buffer, size, &bo, &offset))
      return;
   radv_cs_add_buffer(device->ws, cmd_buffer->cs->b, bo);
   const uint64_t scratch_va = radv_buffer_get_va(bo) + offset;
   radv_meta_swap_native_elements(cmd_buffer, region->addressRange.address, scratch_va, size / width, width);

   VkDeviceMemoryImageCopyKHR copy = *region;
   copy.addressRange.address = scratch_va;
   copy.addressRange.size = size;
   gfx_or_compute_copy_memory_to_image(cmd_buffer, radv_get_copy_flags_from_bo(bo), image, &copy, use_compute);
}

static void
radv_be_copy_image_to_memory(struct radv_cmd_buffer *cmd_buffer, VkAddressCopyFlagsKHR dst_copy_flags,
                             struct radv_image *image, const VkDeviceMemoryImageCopyKHR *region)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const unsigned width = radv_native_copy_element_width(cmd_buffer, image, region->imageSubresource.aspectMask);
   if (width == 1) {
      compute_copy_image_to_memory(cmd_buffer, dst_copy_flags, image, region);
      return;
   }

   bool has_gaps;
   const uint64_t size = radv_native_copy_size(image, region, &has_gaps);
   struct radeon_winsys_bo *bo;
   unsigned offset;
   if (!radv_meta_alloc_native_scratch(cmd_buffer, size, &bo, &offset))
      return;
   radv_cs_add_buffer(device->ws, cmd_buffer->cs->b, bo);
   const uint64_t dst_va = region->addressRange.address;
   const uint64_t scratch_va = radv_buffer_get_va(bo) + offset;

   /* Bytes between rows must survive the final swap unchanged. */
   if (has_gaps)
      radv_meta_swap_native_elements(cmd_buffer, dst_va, scratch_va, size / width, width);

   VkDeviceMemoryImageCopyKHR copy = *region;
   copy.addressRange.address = scratch_va;
   copy.addressRange.size = size;
   compute_copy_image_to_memory(cmd_buffer, radv_get_copy_flags_from_bo(bo), image, &copy);

   radv_meta_swap_native_elements(cmd_buffer, scratch_va, dst_va, size / width, width);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_buffer, src_buffer, pCopyBufferToImageInfo->srcBuffer);
   VK_FROM_HANDLE(radv_image, dst_image, pCopyBufferToImageInfo->dstImage);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   const VkAddressCopyFlagsKHR src_copy_flags = radv_get_copy_flags_from_bo(src_buffer->bo);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_cs_add_buffer(device->ws, cs->b, src_buffer->bo);

   for (unsigned r = 0; r < pCopyBufferToImageInfo->regionCount; r++) {
      const VkBufferImageCopy2 *region = &pCopyBufferToImageInfo->pRegions[r];
      const VkImageAspectFlags aspect_mask = region->imageSubresource.aspectMask;
      const unsigned bind_idx = dst_image->disjoint ? radv_plane_from_aspect(aspect_mask) : 0;

      radv_cs_add_buffer(device->ws, cs->b, dst_image->bindings[bind_idx].bo);

      const VkDeviceMemoryImageCopyKHR copy = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
         .addressRange = vk_device_address_range(&src_buffer->vk, region->bufferOffset, VK_WHOLE_SIZE),
         .addressRowLength = region->bufferRowLength,
         .addressImageHeight = region->bufferImageHeight,
         .imageSubresource = region->imageSubresource,
         .imageLayout = pCopyBufferToImageInfo->dstImageLayout,
         .imageOffset = region->imageOffset,
         .imageExtent = region->imageExtent,
      };

      if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
         transfer_copy_memory_image(cmd_buffer, src_copy_flags, dst_image, &copy, true);
      } else {
         const bool use_compute = cmd_buffer->qf == RADV_QUEUE_COMPUTE || !radv_image_is_renderable(dst_image);
         radv_be_copy_memory_to_image(cmd_buffer, src_copy_flags, dst_image, &copy, use_compute);
      }
   }

   if (radv_is_format_emulated(pdev, dst_image->vk.format) && cmd_buffer->qf != RADV_QUEUE_TRANSFER) {
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH |
                                      radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, dst_image, NULL) |
                                      radv_dst_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_2_TRANSFER_READ_BIT, 0, dst_image, NULL);

      const enum util_format_layout format_layout = radv_format_description(dst_image->vk.format)->layout;
      for (unsigned r = 0; r < pCopyBufferToImageInfo->regionCount; r++) {
         if (format_layout == UTIL_FORMAT_LAYOUT_ASTC) {
            radv_meta_decode_astc(cmd_buffer, dst_image, pCopyBufferToImageInfo->dstImageLayout,
                                  &pCopyBufferToImageInfo->pRegions[r].imageSubresource,
                                  pCopyBufferToImageInfo->pRegions[r].imageOffset,
                                  pCopyBufferToImageInfo->pRegions[r].imageExtent);
         } else {
            radv_meta_decode_etc(cmd_buffer, dst_image, pCopyBufferToImageInfo->dstImageLayout,
                                 &pCopyBufferToImageInfo->pRegions[r].imageSubresource,
                                 pCopyBufferToImageInfo->pRegions[r].imageOffset,
                                 pCopyBufferToImageInfo->pRegions[r].imageExtent);
         }
      }
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyMemoryToImageKHR(VkCommandBuffer commandBuffer, const VkCopyDeviceMemoryImageInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, dst_image, pCopyMemoryInfo->image);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   for (unsigned r = 0; r < pCopyMemoryInfo->regionCount; r++) {
      const VkDeviceMemoryImageCopyKHR *region = &pCopyMemoryInfo->pRegions[r];
      const VkImageAspectFlags aspect_mask = region->imageSubresource.aspectMask;
      const unsigned bind_idx = dst_image->disjoint ? radv_plane_from_aspect(aspect_mask) : 0;

      radv_cs_add_buffer(device->ws, cs->b, dst_image->bindings[bind_idx].bo);

      VkAddressCopyFlagsKHR copy_flags = radv_get_copy_flags_from_command_flags(region->addressFlags);

      if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
         transfer_copy_memory_image(cmd_buffer, copy_flags, dst_image, region, true);
      } else {
         const bool use_compute = cmd_buffer->qf == RADV_QUEUE_COMPUTE || !radv_image_is_renderable(dst_image);
         radv_be_copy_memory_to_image(cmd_buffer, copy_flags, dst_image, region, use_compute);
      }
   }

   if (radv_is_format_emulated(pdev, dst_image->vk.format) && cmd_buffer->qf != RADV_QUEUE_TRANSFER) {
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH |
                                      radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_TRANSFER_WRITE_BIT, 0, dst_image, NULL) |
                                      radv_dst_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_TRANSFER_READ_BIT, 0, dst_image, NULL);

      const enum util_format_layout format_layout = radv_format_description(dst_image->vk.format)->layout;
      for (unsigned r = 0; r < pCopyMemoryInfo->regionCount; r++) {
         const VkDeviceMemoryImageCopyKHR *region = &pCopyMemoryInfo->pRegions[r];

         if (format_layout == UTIL_FORMAT_LAYOUT_ASTC) {
            radv_meta_decode_astc(cmd_buffer, dst_image, region->imageLayout, &region->imageSubresource,
                                  region->imageOffset, region->imageExtent);
         } else {
            radv_meta_decode_etc(cmd_buffer, dst_image, region->imageLayout, &region->imageSubresource,
                                 region->imageOffset, region->imageExtent);
         }
      }
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

static void
compute_copy_image_to_memory(struct radv_cmd_buffer *cmd_buffer, VkAddressCopyFlagsKHR dst_copy_flags,
                             struct radv_image *image, const VkDeviceMemoryImageCopyKHR *region)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);

   /**
    * From the Vulkan 1.0.6 spec: 18.3 Copying Data Between Images
    *    extent is the size in texels of the source image to copy in width,
    *    height and depth. 1D images use only x and width. 2D images use x, y,
    *    width and height. 3D images use x, y, z, width, height and depth.
    *
    *
    * Also, convert the offsets and extent from units of texels to units of
    * blocks - which is the highest resolution accessible in this command.
    */
   const VkOffset3D img_offset_el = vk_image_offset_to_elements(&image->vk, region->imageOffset);
   const VkExtent3D bufferExtent = {
      .width = region->addressRowLength ? region->addressRowLength : region->imageExtent.width,
      .height = region->addressImageHeight ? region->addressImageHeight : region->imageExtent.height,
   };
   const VkExtent3D buf_extent_el = vk_image_extent_to_elements(&image->vk, bufferExtent);

   /* Start creating blit rect */
   const VkExtent3D img_extent_el = vk_image_extent_to_elements(&image->vk, region->imageExtent);

   /* Create blit surfaces */
   struct radv_meta_blit2d_surf img_info =
      radv_blit_surf_for_image_level_layer(image, region->imageLayout, &region->imageSubresource);

   if (!radv_is_buffer_format_supported(img_info.format, NULL)) {
      const uint32_t queue_mask = radv_image_queue_family_mask(image, cmd_buffer->qf, cmd_buffer->qf);
      const VkFormat raw_format = vk_format_for_size(vk_format_get_blocksize(img_info.format));

      if (!radv_dcc_formats_compatible(pdev->info.gfx_level, img_info.format, raw_format, NULL) &&
          radv_layout_dcc_compressed(device, image, region->imageSubresource.mipLevel, region->imageLayout,
                                     queue_mask)) {
         radv_describe_barrier_start(cmd_buffer, RGP_BARRIER_UNKNOWN_REASON);

         radv_decompress_dcc(cmd_buffer, image,
                             &(VkImageSubresourceRange){
                                .aspectMask = region->imageSubresource.aspectMask,
                                .baseMipLevel = region->imageSubresource.mipLevel,
                                .levelCount = 1,
                                .baseArrayLayer = region->imageSubresource.baseArrayLayer,
                                .layerCount = vk_image_subresource_layer_count(&image->vk, &region->imageSubresource),
                             });
         img_info.disable_compression = true;

         radv_describe_barrier_end(cmd_buffer);
      }

      img_info.format = raw_format;
   }

   struct radv_meta_blit2d_buffer buf_info = {
      .addr = region->addressRange.address,
      .size = region->addressRange.size,
      .format = img_info.format,
      .pitch = buf_extent_el.width,
      .copy_flags = dst_copy_flags,
   };

   if (image->vk.image_type == VK_IMAGE_TYPE_3D)
      img_info.layer = img_offset_el.z;
   /* Loop through each 3D or array slice */
   unsigned num_slices_3d = img_extent_el.depth;
   unsigned num_slices_array = vk_image_subresource_layer_count(&image->vk, &region->imageSubresource);
   unsigned slice_3d = 0;
   unsigned slice_array = 0;
   while (slice_3d < num_slices_3d && slice_array < num_slices_array) {
      /* Perform Blit */
      radv_meta_image_to_buffer(cmd_buffer, &img_info, &buf_info, &img_offset_el, &img_extent_el);

      buf_info.offset += buf_extent_el.width * buf_extent_el.height * img_info.bs;
      img_info.layer++;
      if (image->vk.image_type == VK_IMAGE_TYPE_3D)
         slice_3d++;
      else
         slice_array++;
   }
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 *pCopyImageToBufferInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, src_image, pCopyImageToBufferInfo->srcImage);
   VK_FROM_HANDLE(radv_buffer, dst_buffer, pCopyImageToBufferInfo->dstBuffer);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   const VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_bo(dst_buffer->bo);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_cs_add_buffer(device->ws, cs->b, dst_buffer->bo);

   for (unsigned r = 0; r < pCopyImageToBufferInfo->regionCount; r++) {
      const VkBufferImageCopy2 *region = &pCopyImageToBufferInfo->pRegions[r];
      const VkImageAspectFlags aspect_mask = region->imageSubresource.aspectMask;
      const unsigned bind_idx = src_image->disjoint ? radv_plane_from_aspect(aspect_mask) : 0;

      radv_cs_add_buffer(device->ws, cs->b, src_image->bindings[bind_idx].bo);

      VkDeviceMemoryImageCopyKHR copy = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
         .addressRange = vk_device_address_range(&dst_buffer->vk, region->bufferOffset, VK_WHOLE_SIZE),
         .addressFlags = dst_buffer->vk.address_flags,
         .addressRowLength = region->bufferRowLength,
         .addressImageHeight = region->bufferImageHeight,
         .imageSubresource = region->imageSubresource,
         .imageLayout = pCopyImageToBufferInfo->srcImageLayout,
         .imageOffset = region->imageOffset,
         .imageExtent = region->imageExtent,
      };

      if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
         transfer_copy_memory_image(cmd_buffer, dst_copy_flags, src_image, &copy, false);
      } else {
         radv_be_copy_image_to_memory(cmd_buffer, dst_copy_flags, src_image, &copy);
      }
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyImageToMemoryKHR(VkCommandBuffer commandBuffer, const VkCopyDeviceMemoryImageInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, src_image, pCopyMemoryInfo->image);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   for (unsigned r = 0; r < pCopyMemoryInfo->regionCount; r++) {
      const VkDeviceMemoryImageCopyKHR *region = &pCopyMemoryInfo->pRegions[r];
      const VkImageAspectFlags aspect_mask = region->imageSubresource.aspectMask;
      const unsigned bind_idx = src_image->disjoint ? radv_plane_from_aspect(aspect_mask) : 0;

      VkAddressCopyFlagsKHR copy_flags = radv_get_copy_flags_from_command_flags(region->addressFlags);

      radv_cs_add_buffer(device->ws, cs->b, src_image->bindings[bind_idx].bo);

      if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
         transfer_copy_memory_image(cmd_buffer, copy_flags, src_image, region, false);
      } else {
         radv_be_copy_image_to_memory(cmd_buffer, copy_flags, src_image, region);
      }
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

static void gfx_or_compute_copy_image(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image,
                                      VkImageLayout src_image_layout, struct radv_image *dst_image,
                                      VkImageLayout dst_image_layout, const VkImageCopy2 *region,
                                      const bool use_compute);

static void
transfer_copy_image(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image, VkImageLayout src_image_layout,
                    struct radv_image *dst_image, VkImageLayout dst_image_layout, const VkImageCopy2 *region)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;
   unsigned int dst_aspect_mask_remaining = region->dstSubresource.aspectMask;

   if (!radv_sdma_supports_image(device, src_image, false) || !radv_sdma_supports_image(device, dst_image, true)) {
      if (!radv_gang_init(cmd_buffer))
         return;

      if (radv_flush_gang_leader_semaphore(cmd_buffer)) {
         radv_wait_gang_leader(cmd_buffer);
         cmd_buffer->gang.flush_bits |= RADV_CMD_FLAG_INV_L2;
      }

      radv_gang_cache_flush(cmd_buffer);

      gfx_or_compute_copy_image(cmd_buffer, src_image, src_image_layout, dst_image, dst_image_layout, region, true);

      return;
   }

   if (cmd_buffer->gang.cs && radv_flush_gang_follower_semaphore(cmd_buffer))
      radv_wait_gang_follower(cmd_buffer);

   VkImageSubresourceLayers src_subresource = region->srcSubresource;
   VkImageSubresourceLayers dst_subresource = region->dstSubresource;

   const VkOffset3D dst_offset_el = vk_image_offset_to_elements(&dst_image->vk, region->dstOffset);
   const VkOffset3D src_offset_el = vk_image_offset_to_elements(&src_image->vk, region->srcOffset);
   const VkExtent3D img_extent_el = vk_image_extent_to_elements(&src_image->vk, region->extent);

   u_foreach_bit (b, region->srcSubresource.aspectMask) {
      src_subresource.aspectMask = BITFIELD_BIT(b);
      dst_subresource.aspectMask = BITFIELD_BIT(u_bit_scan(&dst_aspect_mask_remaining));

      const struct ac_sdma_surf src =
         radv_sdma_get_surf(cmd_buffer, src_image, src_image_layout, src_subresource, src_offset_el);
      const struct ac_sdma_surf dst =
         radv_sdma_get_surf(cmd_buffer, dst_image, dst_image_layout, dst_subresource, dst_offset_el);
      const VkExtent3D extent = radv_sdma_get_copy_extent(src_image, src_subresource, img_extent_el);

      if (radv_sdma_use_t2t_scanline_copy(device, &src, &dst, extent)) {
         if (!alloc_transfer_temp_bo(cmd_buffer))
            return;

         radv_sdma_copy_image_t2t_scanline(device, cs, &src, &dst, extent, cmd_buffer->transfer.copy_temp);
      } else {
         radv_sdma_copy_image(device, cs, &src, &dst, extent);
      }
   }
}

static VkFormat
radv_get_compat_color_ds_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
      return VK_FORMAT_R8_UINT;
      break;
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
      return VK_FORMAT_R16_UNORM;
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_UINT:
      return VK_FORMAT_R32_SFLOAT;
   default:
      UNREACHABLE("invalid color format for color to depth/stencil image copy.");
   }
}

static void
gfx_or_compute_copy_image(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image,
                          VkImageLayout src_image_layout, struct radv_image *dst_image, VkImageLayout dst_image_layout,
                          const VkImageCopy2 *region, const bool use_compute)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);

   /* From the Vulkan 1.0 spec:
    *
    *    vkCmdCopyImage can be used to copy image data between multisample images, but both images must have the same
    *    number of samples.
    */
   assert(src_image->vk.samples == dst_image->vk.samples);
   /* From the Vulkan 1.3 spec:
    *
    *    Multi-planar images can only be copied on a per-plane basis, and the subresources used in each region when
    *    copying to or from such images must specify only one plane, though different regions can specify different
    *    planes.
    */
   assert(src_image->plane_count == 1 || util_is_power_of_two_nonzero(region->srcSubresource.aspectMask));
   assert(dst_image->plane_count == 1 || util_is_power_of_two_nonzero(region->dstSubresource.aspectMask));

   if (use_compute) {
      radv_fixup_copy_dst_htile_metadata(cmd_buffer, dst_image, dst_image_layout, &region->dstSubresource,
                                         &region->dstOffset, &region->extent, true);
   }

   /* Create blit surfaces */
   struct radv_meta_blit2d_surf b_src =
      radv_blit_surf_for_image_level_layer(src_image, src_image_layout, &region->srcSubresource);

   struct radv_meta_blit2d_surf b_dst =
      radv_blit_surf_for_image_level_layer(dst_image, dst_image_layout, &region->dstSubresource);

   uint32_t dst_queue_mask = radv_image_queue_family_mask(dst_image, cmd_buffer->qf, cmd_buffer->qf);
   bool dst_compressed =
      radv_layout_dcc_compressed(device, dst_image, region->dstSubresource.mipLevel, dst_image_layout, dst_queue_mask);
   uint32_t src_queue_mask = radv_image_queue_family_mask(src_image, cmd_buffer->qf, cmd_buffer->qf);
   bool src_compressed =
      radv_layout_dcc_compressed(device, src_image, region->srcSubresource.mipLevel, src_image_layout, src_queue_mask);
   bool need_dcc_sign_reinterpret = false;

   if (!src_compressed ||
       (radv_dcc_formats_compatible(pdev->info.gfx_level, b_src.format, b_dst.format, &need_dcc_sign_reinterpret) &&
        !need_dcc_sign_reinterpret)) {
      b_src.format = b_dst.format;
   } else if (!dst_compressed) {
      b_dst.format = b_src.format;
   } else {
      radv_describe_barrier_start(cmd_buffer, RGP_BARRIER_UNKNOWN_REASON);

      radv_decompress_dcc(cmd_buffer, dst_image,
                          &(VkImageSubresourceRange){
                             .aspectMask = region->dstSubresource.aspectMask,
                             .baseMipLevel = region->dstSubresource.mipLevel,
                             .levelCount = 1,
                             .baseArrayLayer = region->dstSubresource.baseArrayLayer,
                             .layerCount = vk_image_subresource_layer_count(&dst_image->vk, &region->dstSubresource),
                          });
      b_dst.format = b_src.format;
      b_dst.disable_compression = true;

      radv_describe_barrier_end(cmd_buffer);
   }

   /* Select a compatible color format for color<->depth/stencil copies. */
   if (vk_format_is_color(src_image->vk.format) && vk_format_is_depth_or_stencil(dst_image->vk.format)) {
      b_src.format = radv_get_compat_color_ds_format(src_image->vk.format);
   } else if (vk_format_is_depth_or_stencil(src_image->vk.format) && vk_format_is_color(dst_image->vk.format)) {
      b_dst.format = radv_get_compat_color_ds_format(dst_image->vk.format);
   }

   /**
    * From the Vulkan 1.0.6 spec: 18.4 Copying Data Between Buffers and Images
    *    imageExtent is the size in texels of the image to copy in width, height
    *    and depth. 1D images use only x and width. 2D images use x, y, width
    *    and height. 3D images use x, y, z, width, height and depth.
    *
    * Also, convert the offsets and extent from units of texels to units of
    * blocks - which is the highest resolution accessible in this command.
    */
   const VkOffset3D dst_offset_el = vk_image_offset_to_elements(&dst_image->vk, region->dstOffset);
   const VkOffset3D src_offset_el = vk_image_offset_to_elements(&src_image->vk, region->srcOffset);

   /*
    * From Vulkan 1.0.68, "Copying Data Between Images":
    *    "When copying between compressed and uncompressed formats
    *     the extent members represent the texel dimensions of the
    *     source image and not the destination."
    * However, we must use the destination image type to avoid
    * clamping depth when copying multiple layers of a 2D image to
    * a 3D image.
    */
   const VkExtent3D img_extent_el = vk_image_extent_to_elements(&src_image->vk, region->extent);

   unsigned num_slices = vk_image_subresource_layer_count(&src_image->vk, &region->srcSubresource);

   if (src_image->vk.image_type == VK_IMAGE_TYPE_3D) {
      b_src.layer = src_offset_el.z;
      num_slices = img_extent_el.depth;
   }

   if (dst_image->vk.image_type == VK_IMAGE_TYPE_3D)
      b_dst.layer = dst_offset_el.z;

   for (unsigned slice = 0; slice < num_slices; slice++) {
      /* Perform Blit */
      if (use_compute) {
         radv_meta_image_to_image_cs(cmd_buffer, &b_src, &b_dst, &src_offset_el, &dst_offset_el, &img_extent_el);
      } else {
         if (radv_can_use_fmask_copy(cmd_buffer, b_src.image, b_dst.image, &src_offset_el, &dst_offset_el,
                                     &img_extent_el)) {
            radv_fmask_copy(cmd_buffer, &b_src, &b_dst);
         } else {
            radv_gfx_copy_image(cmd_buffer, &b_src, &b_dst, &src_offset_el, &dst_offset_el, &img_extent_el);
         }
      }

      b_src.layer++;
      b_dst.layer++;
   }

   if (use_compute) {
      radv_fixup_copy_dst_htile_metadata(cmd_buffer, dst_image, dst_image_layout, &region->dstSubresource,
                                         &region->dstOffset, &region->extent, false);
   }
}

/* Big endian: Vulkan copies between compatible formats preserve the bytes the host would see.
 * Image memory holds little-endian elements of each format's own width, so when the widths
 * differ a raw copy would expose the other format's byte order. Going through host order
 * gives the same result as a copy through application buffers. Returns false when the
 * region does not need this.
 */
static bool
radv_native_copy_image_through_host_order(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image,
                                          VkImageLayout src_layout, struct radv_image *dst_image,
                                          VkImageLayout dst_layout, const VkImageCopy2 *region, bool use_compute)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);

   const VkImageAspectFlags zs = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   if (src_image->vk.samples > 1 || ((region->srcSubresource.aspectMask | region->dstSubresource.aspectMask) & zs))
      return false;

   const unsigned src_width = radv_native_copy_element_width(cmd_buffer, src_image, region->srcSubresource.aspectMask);
   const unsigned dst_width = radv_native_copy_element_width(cmd_buffer, dst_image, region->dstSubresource.aspectMask);
   if (src_width == dst_width)
      return false;

   const VkFormat src_format = radv_get_aspect_format(src_image, region->srcSubresource.aspectMask);
   const VkFormat dst_format = radv_get_aspect_format(dst_image, region->dstSubresource.aspectMask);
   const unsigned layers = vk_image_subresource_layer_count(&src_image->vk, &region->srcSubresource);
   const unsigned slices = MAX2(region->extent.depth, layers);
   const VkExtent3D elements = {DIV_ROUND_UP(region->extent.width, vk_format_get_blockwidth(src_format)),
                                DIV_ROUND_UP(region->extent.height, vk_format_get_blockheight(src_format)), 1};
   const uint32_t bw = vk_format_get_blockwidth(dst_format);
   const uint32_t bh = vk_format_get_blockheight(dst_format);
   const uint64_t size = (uint64_t)elements.width * elements.height * slices * vk_format_get_blocksize(src_format);

   struct radeon_winsys_bo *from_bo, *to_bo;
   unsigned from_offset, to_offset;
   if (!radv_meta_alloc_native_scratch(cmd_buffer, size, &from_bo, &from_offset) ||
       !radv_meta_alloc_native_scratch(cmd_buffer, size, &to_bo, &to_offset))
      return true;

   /* Both allocations must stay valid: a new upload BO may have replaced the first. */
   radv_cs_add_buffer(device->ws, cmd_buffer->cs->b, from_bo);
   radv_cs_add_buffer(device->ws, cmd_buffer->cs->b, to_bo);
   const uint64_t from_va = radv_buffer_get_va(from_bo) + from_offset;
   const uint64_t to_va = radv_buffer_get_va(to_bo) + to_offset;

   VkDeviceMemoryImageCopyKHR src_region = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
      .addressRange = {.address = from_va, .size = size},
      .imageSubresource = region->srcSubresource,
      .imageLayout = src_layout,
      .imageOffset = region->srcOffset,
      .imageExtent = {region->extent.width, region->extent.height, 1},
   };
   VkDeviceMemoryImageCopyKHR dst_region = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
      .imageSubresource = region->dstSubresource,
      .imageLayout = dst_layout,
      .imageOffset = region->dstOffset,
      .imageExtent = {elements.width * bw, elements.height * bh, 1},
   };
   if (src_image->vk.image_type == VK_IMAGE_TYPE_3D) {
      src_region.imageExtent.depth = slices;
      src_region.imageSubresource.layerCount = 1;
   } else {
      src_region.imageSubresource.layerCount = slices;
   }
   if (dst_image->vk.image_type == VK_IMAGE_TYPE_3D) {
      dst_region.imageExtent.depth = slices;
      dst_region.imageSubresource.layerCount = 1;
   } else {
      dst_region.imageSubresource.layerCount = slices;
   }

   compute_copy_image_to_memory(cmd_buffer, radv_get_copy_flags_from_bo(from_bo), src_image, &src_region);

   /* to_va ends up holding the destination's little-endian elements. */
   uint64_t host_va = from_va;
   if (src_width > 1) {
      radv_meta_swap_native_elements(cmd_buffer, from_va, to_va, size / src_width, src_width);
      host_va = to_va;
   }
   uint64_t final_va = host_va;
   if (dst_width > 1) {
      final_va = host_va == from_va ? to_va : from_va;
      radv_meta_swap_native_elements(cmd_buffer, host_va, final_va, size / dst_width, dst_width);
   }

   dst_region.addressRange.address = final_va;
   dst_region.addressRange.size = size;
   gfx_or_compute_copy_memory_to_image(cmd_buffer, radv_get_copy_flags_from_bo(final_va == to_va ? to_bo : from_bo),
                                       dst_image, &dst_region, use_compute);
   return true;
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyImage2(VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, src_image, pCopyImageInfo->srcImage);
   VK_FROM_HANDLE(radv_image, dst_image, pCopyImageInfo->dstImage);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   for (unsigned r = 0; r < pCopyImageInfo->regionCount; r++) {
      const VkImageCopy2 *region = &pCopyImageInfo->pRegions[r];
      const VkImageAspectFlags src_aspect_mask = region->srcSubresource.aspectMask;
      const unsigned src_bind_idx = src_image->disjoint ? radv_plane_from_aspect(src_aspect_mask) : 0;
      const VkImageAspectFlags dst_aspect_mask = region->dstSubresource.aspectMask;
      const unsigned dst_bind_idx = dst_image->disjoint ? radv_plane_from_aspect(dst_aspect_mask) : 0;

      radv_cs_add_buffer(device->ws, cs->b, src_image->bindings[src_bind_idx].bo);
      radv_cs_add_buffer(device->ws, cs->b, dst_image->bindings[dst_bind_idx].bo);

      if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
         transfer_copy_image(cmd_buffer, src_image, pCopyImageInfo->srcImageLayout, dst_image,
                             pCopyImageInfo->dstImageLayout, region);
      } else {
         const bool use_compute = cmd_buffer->qf == RADV_QUEUE_COMPUTE || !radv_image_is_renderable(dst_image);
         if (!radv_native_copy_image_through_host_order(cmd_buffer, src_image, pCopyImageInfo->srcImageLayout,
                                                        dst_image, pCopyImageInfo->dstImageLayout, region,
                                                        use_compute))
            gfx_or_compute_copy_image(cmd_buffer, src_image, pCopyImageInfo->srcImageLayout, dst_image,
                                      pCopyImageInfo->dstImageLayout, region, use_compute);
      }
   }

   if (radv_is_format_emulated(pdev, dst_image->vk.format) && cmd_buffer->qf != RADV_QUEUE_TRANSFER) {
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH |
                                      radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, dst_image, NULL) |
                                      radv_dst_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_2_TRANSFER_READ_BIT, 0, dst_image, NULL);

      const enum util_format_layout format_layout = radv_format_description(dst_image->vk.format)->layout;
      for (unsigned r = 0; r < pCopyImageInfo->regionCount; r++) {
         VkExtent3D dst_extent = pCopyImageInfo->pRegions[r].extent;

         /* The Vulken spec 1.4.347 says:
          *
          * "VUID-VkCopyImageInfo2-srcImage-09247
          *  If the VkFormat of each of srcImage and dstImage is a compressed image format, the
          *  formats must have the same texel block extent"
          */
         if (vk_format_is_compressed(src_image->vk.format) != vk_format_is_compressed(dst_image->vk.format)) {
            dst_extent.width = dst_extent.width / vk_format_get_blockwidth(src_image->vk.format) *
                               vk_format_get_blockwidth(dst_image->vk.format);
            dst_extent.height = dst_extent.height / vk_format_get_blockheight(src_image->vk.format) *
                                vk_format_get_blockheight(dst_image->vk.format);
         }
         if (format_layout == UTIL_FORMAT_LAYOUT_ASTC) {
            radv_meta_decode_astc(cmd_buffer, dst_image, pCopyImageInfo->dstImageLayout,
                                  &pCopyImageInfo->pRegions[r].dstSubresource, pCopyImageInfo->pRegions[r].dstOffset,
                                  dst_extent);
         } else {
            radv_meta_decode_etc(cmd_buffer, dst_image, pCopyImageInfo->dstImageLayout,
                                 &pCopyImageInfo->pRegions[r].dstSubresource, pCopyImageInfo->pRegions[r].dstOffset,
                                 dst_extent);
         }
      }
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyMemoryIndirectKHR(VkCommandBuffer commandBuffer, const VkCopyMemoryIndirectInfoKHR *pCopyMemoryIndirectInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);

   if (!pCopyMemoryIndirectInfo->copyCount)
      return;

   assert(!(pCopyMemoryIndirectInfo->copyAddressRange.address & 3));
   assert(!(pCopyMemoryIndirectInfo->copyAddressRange.stride & 3));
   assert(pCopyMemoryIndirectInfo->copyAddressRange.stride >= sizeof(VkCopyMemoryIndirectCommandKHR));
   assert(pCopyMemoryIndirectInfo->copyCount <=
          (pCopyMemoryIndirectInfo->copyAddressRange.size / pCopyMemoryIndirectInfo->copyAddressRange.stride));

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_compute_copy_memory_indirect(cmd_buffer, pCopyMemoryIndirectInfo);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyMemoryToImageIndirectKHR(VkCommandBuffer commandBuffer,
                                     const VkCopyMemoryToImageIndirectInfoKHR *pCopyMemoryToImageIndirectInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, dst_image, pCopyMemoryToImageIndirectInfo->dstImage);
   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);

   if (!pCopyMemoryToImageIndirectInfo->copyCount)
      return;

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   const bool use_compute = cmd_buffer->qf == RADV_QUEUE_COMPUTE || !radv_image_is_renderable(dst_image);
   if (use_compute) {
      radv_compute_copy_memory_to_image_indirect(cmd_buffer, pCopyMemoryToImageIndirectInfo);
   } else {
      radv_gfx_copy_memory_to_image_indirect(cmd_buffer, pCopyMemoryToImageIndirectInfo);
   }

   if (radv_is_format_emulated(pdev, dst_image->vk.format)) {
      cmd_buffer->state.flush_bits |= (use_compute ? RADV_CMD_FLAG_CS_PARTIAL_FLUSH : RADV_CMD_FLAG_PS_PARTIAL_FLUSH) |
                                      radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_TRANSFER_WRITE_BIT, 0, dst_image, NULL) |
                                      radv_dst_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                            VK_ACCESS_TRANSFER_READ_BIT, 0, dst_image, NULL);

      const enum util_format_layout format_layout = radv_format_description(dst_image->vk.format)->layout;
      assert(format_layout == UTIL_FORMAT_LAYOUT_ETC);

      radv_meta_decode_etc_indirect(cmd_buffer, pCopyMemoryToImageIndirectInfo);
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}
