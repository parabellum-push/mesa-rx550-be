/* Based on anv:
 * Copyright © 2015 Intel Corporation
 *
 * Copyright © 2016 Red Hat Inc.
 * Copyright © 2025 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir/nir_builder.h"
#include "nir/radv_meta_nir.h"
#include "tools/radv_debug_hang.h"
#include "radv_cp_dma.h"
#include "radv_meta.h"
#include "radv_meta_native_indices.h"
#include "util/u_endian.h"
#include "radv_sdma.h"
#include "radv_tracepoints.h"
#include "vk_shader_module.h"

#include "radv_cs.h"

#include "vk_command_pool.h"

struct fill_constants {
   uint64_t addr;
   uint32_t max_offset;
   uint32_t data;
};

struct radv_fill_memory_key {
   enum radv_meta_object_key_type type;
   bool use_16B_copy;
};

static VkResult
get_fill_memory_pipeline(struct radv_device *device, uint64_t size, VkPipeline *pipeline_out,
                         VkPipelineLayout *layout_out)
{
   const bool use_16B_copy = size >= 16;
   struct radv_fill_memory_key key;
   VkResult result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_FILL_MEMORY;
   key.use_16B_copy = use_16B_copy;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct fill_constants),
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_fill_memory_shader(use_16B_copy ? 16 : 4);

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

struct copy_constants {
   uint64_t src_addr;
   uint64_t dst_addr;
   uint32_t max_offset;
};

struct radv_copy_memory_key {
   enum radv_meta_object_key_type type;
   bool use_16B_copy;
};

static bool
radv_is_copy_memory_4B_aligned(uint64_t src_va, uint64_t dst_va, uint64_t size)
{
   return !(size & 3) && !(src_va & 3) && !(dst_va & 3);
}

static VkResult
get_copy_memory_pipeline(struct radv_device *device, uint64_t src_va, uint64_t dst_va, uint64_t size,
                         VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   const bool use_16B_copy = size >= 16 && radv_is_copy_memory_4B_aligned(src_va, dst_va, size);
   struct radv_copy_memory_key key;
   VkResult result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_COPY_MEMORY;
   key.use_16B_copy = use_16B_copy;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct copy_constants),
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_copy_memory_shader(use_16B_copy ? 16 : 1);

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
radv_compute_fill_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, uint32_t data)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_fill_memory_pipeline(device, size, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   assert(size <= UINT32_MAX);

   struct fill_constants fill_consts = {
      .addr = radv_meta_pc_u64(va),
      .data = data,
   };
   uint32_t dim_x;

   if (size >= 16) {
      fill_consts.max_offset = size - 16;
      dim_x = DIV_ROUND_UP(size, 16);
   } else {
      fill_consts.max_offset = size - 4;
      dim_x = DIV_ROUND_UP(size, 4);
   }

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fill_consts), &fill_consts);

   radv_unaligned_dispatch(cmd_buffer, dim_x, 1, 1);
}

static void
radv_compute_copy_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint64_t dst_va, uint64_t size)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const bool use_16B_copy = size >= 16 && radv_is_copy_memory_4B_aligned(src_va, dst_va, size);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_copy_memory_pipeline(device, src_va, dst_va, size, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_utrace_begin_compute_copy_memory(cmd_buffer, size);

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   assert(size <= UINT32_MAX);

   struct copy_constants copy_consts = {
      .src_addr = radv_meta_pc_u64(src_va),
      .dst_addr = radv_meta_pc_u64(dst_va),
   };
   uint32_t dim_x;

   if (use_16B_copy) {
      copy_consts.max_offset = size - 16;
      dim_x = DIV_ROUND_UP(size, 16);
   } else {
      copy_consts.max_offset = size;
      dim_x = size;
   }

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(copy_consts), &copy_consts);

   radv_unaligned_dispatch(cmd_buffer, dim_x, 1, 1);

   radv_utrace_end_compute_copy_memory(cmd_buffer);
}

static bool
radv_prefer_compute_or_cp_dma(const struct radv_device *device, uint64_t size, VkAddressCopyFlagsKHR src_copy_flags,
                              VkAddressCopyFlagsKHR dst_copy_flags)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   bool use_compute = size >= RADV_BUFFER_OPS_CS_THRESHOLD;

   if (pdev->info.gfx_level >= GFX10 && pdev->info.has_dedicated_vram) {
      if (!(src_copy_flags & VK_ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR) ||
          !(dst_copy_flags & VK_ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR)) {
         /* Prefer CP DMA for GTT on dGPUS due to slow PCIe. */
         use_compute = false;
      }
   }

   return use_compute;
}

static bool
radv_is_compute_required(const struct radv_device *device, VkAddressCopyFlagsKHR src_copy_flags,
                         VkAddressCopyFlagsKHR dst_copy_flags)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   /* Use compute when CP DMA doesn't support sparse. */
   return !pdev->info.cp_dma_supports_sparse &&
          ((src_copy_flags & VK_ADDRESS_COPY_SPARSE_BIT_KHR) || (dst_copy_flags & VK_ADDRESS_COPY_SPARSE_BIT_KHR));
}

static uint32_t
radv_fill_memory_internal(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image, uint64_t va,
                          uint64_t size, uint32_t value, VkAddressCopyFlagsKHR copy_flags)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const bool use_compute = radv_is_compute_required(device, copy_flags, copy_flags) ||
                            radv_prefer_compute_or_cp_dma(device, size, copy_flags, copy_flags);
   uint32_t flush_bits = 0;

   assert(!(va & 3));
   assert(!(size & 3));

   if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
      radv_sdma_fill_memory(device, cmd_buffer->cs, va, size, value);
   } else if (use_compute) {
      radv_compute_fill_memory(cmd_buffer, va, size, value);

      flush_bits = RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                   radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_WRITE_BIT, 0, image, NULL);
   } else if (size)
      radv_cp_dma_fill_memory(cmd_buffer, va, size, value);

   return flush_bits;
}

uint32_t
radv_fill_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, uint32_t value,
                 VkAddressCopyFlagsKHR copy_flags)
{
   return radv_fill_memory_internal(cmd_buffer, NULL, va, size, value, copy_flags);
}

uint32_t
radv_fill_image(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image, uint64_t offset, uint64_t size,
                uint32_t value)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint64_t va = image->bindings[0].addr + offset;
   struct radeon_winsys_bo *bo = image->bindings[0].bo;
   const VkAddressCopyFlagsKHR copy_flags = radv_get_copy_flags_from_bo(bo);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   radv_cs_add_buffer(device->ws, cs->b, bo);

   return radv_fill_memory_internal(cmd_buffer, image, va, size, value, copy_flags);
}

uint32_t
radv_fill_buffer(struct radv_cmd_buffer *cmd_buffer, struct radeon_winsys_bo *bo, uint64_t va, uint64_t size,
                 uint32_t value)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const VkAddressCopyFlagsKHR copy_flags = radv_get_copy_flags_from_bo(bo);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   radv_cs_add_buffer(device->ws, cs->b, bo);

   return radv_fill_memory(cmd_buffer, va, size, value, copy_flags);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize fillSize,
                   uint32_t data)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_buffer, dst_buffer, dstBuffer);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   fillSize = vk_buffer_range(&dst_buffer->vk, dstOffset, fillSize) & ~3ull;

   /* Vulkan specifies this word's byte pattern in host order. */
   if (UTIL_ARCH_BIG_ENDIAN)
      data = util_bswap32(data);

   radv_fill_buffer(cmd_buffer, dst_buffer->bo, vk_buffer_address(&dst_buffer->vk, dstOffset), fillSize, data);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdFillMemoryKHR(VkCommandBuffer commandBuffer, const VkDeviceAddressRangeKHR *pDstRange,
                      VkAddressCommandFlagsKHR dstFlags, uint32_t data)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_command_flags(dstFlags);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   /* Vulkan specifies this word's byte pattern in host order. */
   if (UTIL_ARCH_BIG_ENDIAN)
      data = util_bswap32(data);
   radv_fill_memory(cmd_buffer, pDstRange->address, pDstRange->size, data, dst_copy_flags);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

void
radv_copy_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint64_t dst_va, uint64_t size,
                 VkAddressCopyFlagsKHR src_copy_flags, VkAddressCopyFlagsKHR dst_copy_flags)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const bool use_compute = radv_is_compute_required(device, src_copy_flags, dst_copy_flags) ||
                            (radv_is_copy_memory_4B_aligned(src_va, dst_va, size) &&
                             radv_prefer_compute_or_cp_dma(device, size, src_copy_flags, dst_copy_flags));

   if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
      radv_sdma_copy_memory(device, cmd_buffer->cs, src_va, dst_va, size,
                            cmd_buffer->vk.pool->flags & VK_COMMAND_POOL_CREATE_PROTECTED_BIT);
   } else if (use_compute) {
      radv_compute_copy_memory(cmd_buffer, src_va, dst_va, size);
   } else if (size) {
      radv_cp_dma_copy_memory(cmd_buffer, src_va, dst_va, size);
   }
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *pCopyBufferInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_buffer, src_buffer, pCopyBufferInfo->srcBuffer);
   VK_FROM_HANDLE(radv_buffer, dst_buffer, pCopyBufferInfo->dstBuffer);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   const VkAddressCopyFlagsKHR src_copy_flags = radv_get_copy_flags_from_bo(src_buffer->bo);
   const VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_bo(dst_buffer->bo);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_cs_add_buffer(device->ws, cs->b, src_buffer->bo);
   radv_cs_add_buffer(device->ws, cs->b, dst_buffer->bo);

   for (unsigned r = 0; r < pCopyBufferInfo->regionCount; r++) {
      const VkBufferCopy2 *region = &pCopyBufferInfo->pRegions[r];
      const uint64_t src_va = vk_buffer_address(&src_buffer->vk, region->srcOffset);
      const uint64_t dst_va = vk_buffer_address(&dst_buffer->vk, region->dstOffset);

      radv_copy_memory(cmd_buffer, src_va, dst_va, region->size, src_copy_flags, dst_copy_flags);
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyMemoryKHR(VkCommandBuffer commandBuffer, const VkCopyDeviceMemoryInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   for (unsigned r = 0; r < pCopyMemoryInfo->regionCount; r++) {
      const VkDeviceMemoryCopyKHR *region = &pCopyMemoryInfo->pRegions[r];

      VkAddressCopyFlagsKHR src_copy_flags = radv_get_copy_flags_from_command_flags(region->srcFlags);
      VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_command_flags(region->dstFlags);

      radv_copy_memory(cmd_buffer, region->srcRange.address, region->dstRange.address, region->srcRange.size,
                       src_copy_flags, dst_copy_flags);
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

static void
radv_update_memory_cp_internal(struct radv_cmd_buffer *cmd_buffer, uint64_t va, const void *data, uint64_t size,
                               bool raw_bytes);

void
radv_update_memory_cp(struct radv_cmd_buffer *cmd_buffer, uint64_t va, const void *data, uint64_t size)
{
   radv_update_memory_cp_internal(cmd_buffer, va, data, size, false);
}

/* raw_bytes: opaque application bytes, preserved through command serialization on big endian. */
static void
radv_update_memory_cp_internal(struct radv_cmd_buffer *cmd_buffer, uint64_t va, const void *data, uint64_t size,
                               bool raw_bytes)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   uint64_t words = size / 4;
   bool mec = radv_cmd_buffer_uses_mec(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   assert(size < RADV_BUFFER_UPDATE_THRESHOLD);

   radv_emit_cache_flush(cmd_buffer);
   radeon_check_space(device->ws, cs->b, words + 4);

   uint32_t le_words[RADV_BUFFER_UPDATE_THRESHOLD / 4];
   if (raw_bytes && UTIL_ARCH_BIG_ENDIAN) {
      for (uint64_t i = 0; i < words; ++i) {
         uint32_t value;
         memcpy(&value, (const uint8_t *)data + i * 4, sizeof(value));
         le_words[i] = util_le32_to_cpu(value);
      }
      data = le_words;
   }
   ac_emit_cp_write_data(cs->b, V_371_MICRO_ENGINE, mec ? V_371_MEMORY : V_371_MEM_GRBM, va, words, data, false);

   if (radv_device_fault_detection_enabled(device))
      radv_cmd_buffer_trace_emit(cmd_buffer);
}

static void
radv_update_memory_internal(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, const void *data,
                            VkAddressCopyFlagsKHR dst_copy_flags, bool raw_bytes)
{
   assert(!(size & 3));
   assert(!(va & 3));

   if (!size)
      return;

   if (size < RADV_BUFFER_UPDATE_THRESHOLD && cmd_buffer->qf != RADV_QUEUE_TRANSFER) {
      radv_update_memory_cp_internal(cmd_buffer, va, data, size, raw_bytes);
   } else {
      uint32_t buf_offset;

      radv_cmd_buffer_upload_data(cmd_buffer, size, data, &buf_offset);

      const VkAddressCopyFlagsKHR src_copy_flags = radv_get_copy_flags_from_bo(cmd_buffer->upload.upload_bo);
      const uint64_t src_va = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + buf_offset;

      radv_copy_memory(cmd_buffer, src_va, va, size, src_copy_flags, dst_copy_flags);
   }
}

void
radv_update_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, const void *data,
                   VkAddressCopyFlagsKHR dst_copy_flags)
{
   radv_update_memory_internal(cmd_buffer, va, size, data, dst_copy_flags, false);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize,
                     const void *pData)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_buffer, dst_buffer, dstBuffer);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint64_t dst_va = vk_buffer_address(&dst_buffer->vk, dstOffset);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   const VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_bo(dst_buffer->bo);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_cs_add_buffer(device->ws, cs->b, dst_buffer->bo);

   radv_update_memory_internal(cmd_buffer, dst_va, dataSize, pData, dst_copy_flags, true);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdUpdateMemoryKHR(VkCommandBuffer commandBuffer, const VkDeviceAddressRangeKHR *pDstRange,
                        VkAddressCommandFlagsKHR dstFlags, VkDeviceSize dataSize, const void *pData)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);

   VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_command_flags(dstFlags);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_update_memory_internal(cmd_buffer, pDstRange->address, dataSize, pData, dst_copy_flags, true);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

/* Big endian GFX8: swap host-order elements (2, 4 or 8 bytes) into the GPU's little-endian
 * layout, for example index buffers, which GFX8's VGT cannot byte-swap.
 */
struct radv_swap_elements_key {
   enum radv_meta_object_key_type type;
};

static nir_shader *
build_swap_elements_shader(void)
{
   nir_builder b = radv_meta_nir_init_shader(MESA_SHADER_COMPUTE, "meta_be_swap_elements");
   b.shader->info.workgroup_size[0] = 64;

   nir_def *pconst = nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *count = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 16, .range = 4);
   nir_def *width = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 20, .range = 4);
   nir_def *src_addr = nir_pack_64_2x32(&b, nir_channels(&b, pconst, 0b0011));
   nir_def *dst_addr = nir_pack_64_2x32(&b, nir_channels(&b, pconst, 0b1100));

   nir_def *global_id =
      nir_iadd(&b, nir_imul_imm(&b, nir_channel(&b, nir_load_workgroup_id(&b), 0), b.shader->info.workgroup_size[0]),
               nir_load_local_invocation_index(&b));

   nir_if *valid = nir_push_if(&b, nir_ult(&b, global_id, count));
   nir_def *offset = nir_u2u64(&b, nir_imul(&b, global_id, width));
   nir_def *src = nir_iadd(&b, src_addr, offset);
   nir_def *dst = nir_iadd(&b, dst_addr, offset);
   nir_if *narrow = nir_push_if(&b, nir_ieq_imm(&b, width, 2));
   {
      nir_def *v16 = nir_load_global(&b, 1, 16, src, .align_mul = 2);
      v16 = nir_ior(&b, nir_ushr_imm(&b, v16, 8), nir_ishl_imm(&b, v16, 8));
      nir_store_global(&b, v16, dst, .align_mul = 2);
   }
   nir_push_else(&b, narrow);
   {
      nir_def *v32 = nir_load_global(&b, 1, 32, src, .align_mul = 4);
      v32 = nir_ior(&b, nir_ior(&b, nir_ushr_imm(&b, v32, 24), nir_iand_imm(&b, nir_ushr_imm(&b, v32, 8), 0xff00)),
                    nir_ior(&b, nir_iand_imm(&b, nir_ishl_imm(&b, v32, 8), 0xff0000), nir_ishl_imm(&b, v32, 24)));
      nir_store_global(&b, v32, dst, .align_mul = 4);
      /* 64-bit elements also exchange their two words. */
      nir_if *wide = nir_push_if(&b, nir_ieq_imm(&b, width, 8));
      {
         nir_def *hi = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, src, 4), .align_mul = 4);
         hi = nir_ior(&b, nir_ior(&b, nir_ushr_imm(&b, hi, 24), nir_iand_imm(&b, nir_ushr_imm(&b, hi, 8), 0xff00)),
                      nir_ior(&b, nir_iand_imm(&b, nir_ishl_imm(&b, hi, 8), 0xff0000), nir_ishl_imm(&b, hi, 24)));
         nir_store_global(&b, hi, dst, .align_mul = 4);
         nir_store_global(&b, v32, nir_iadd_imm(&b, dst, 4), .align_mul = 4);
      }
      nir_pop_if(&b, wide);
   }
   nir_pop_if(&b, narrow);
   nir_pop_if(&b, valid);

   return b.shader;
}

static VkResult
get_swap_elements_pipeline(struct radv_device *device, VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   struct radv_swap_elements_key key;
   VkResult result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_BE_SWAP_ELEMENTS;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 24,
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = build_swap_elements_shader();

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
radv_meta_swap_native_elements(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint64_t dst_va, uint32_t count,
                               unsigned width)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   VkPipelineLayout layout;
   VkPipeline pipeline;

   assert(width == 2 || width == 4 || width == 8);
   assert(count > 0);

   VkResult result = get_swap_elements_pipeline(device, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   /* This also runs from draw calls, outside any meta operation. */
   const bool nested = cmd_buffer->state.meta.inside_meta_op;
   const bool suspend = !cmd_buffer->state.cond_render.suspended;
   if (suspend)
      radv_suspend_conditional_rendering(cmd_buffer);
   if (!nested)
      radv_meta_begin(cmd_buffer);

   radv_cp_dma_wait_for_idle(cmd_buffer);
   /* Extend the application's index-read dependency to this shader read. */
   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_PS_PARTIAL_FLUSH |
                                   RADV_CMD_FLAG_INV_VCACHE | RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_L2;

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);
   /* Explicit numeric dwords avoid the host's uint64 member order. */
   const uint32_t constants[6] = {src_va, src_va >> 32, dst_va, dst_va >> 32, count, width};
   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);
   radv_unaligned_dispatch(cmd_buffer, count, 1, 1);

   /* GFX8's index fetch is a non-L2 client. Publish the generated bytes before graphics
    * state emission and the indexed draw packet.
    */
   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_WB_L2 | RADV_CMD_FLAG_INV_VCACHE;

   if (!nested)
      radv_meta_end(cmd_buffer);
   if (suspend)
      radv_resume_conditional_rendering(cmd_buffer);
}

bool
radv_meta_alloc_native_scratch(struct radv_cmd_buffer *cmd_buffer, uint64_t size, struct radeon_winsys_bo **bo,
                               unsigned *offset)
{
   if (size > UINT32_MAX - 3) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   void *mapped;
   if (!radv_cmd_buffer_upload_alloc(cmd_buffer, align(size, 4), offset, &mapped))
      return false;
   *bo = cmd_buffer->upload.upload_bo;
   return true;
}

bool
radv_meta_convert_native_indices(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint32_t count,
                                 unsigned width, uint64_t *dst_va)
{
   struct radeon_winsys_bo *bo;
   unsigned offset;
   if (!radv_meta_alloc_native_scratch(cmd_buffer, (uint64_t)count * width, &bo, &offset))
      return false;
   *dst_va = radv_buffer_get_va(bo) + offset;
   radv_meta_swap_native_elements(cmd_buffer, src_va, *dst_va, count, width);
   return true;
}
