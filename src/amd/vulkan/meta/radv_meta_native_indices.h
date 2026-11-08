/* SPDX-License-Identifier: MIT */
#ifndef RADV_META_NATIVE_INDICES_H
#define RADV_META_NATIVE_INDICES_H
#include <stdbool.h>
#include <stdint.h>
struct radv_cmd_buffer;
struct radeon_winsys_bo;
void radv_meta_swap_native_elements(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint64_t dst_va,
                                    uint32_t count, unsigned width);
bool radv_meta_alloc_native_scratch(struct radv_cmd_buffer *cmd_buffer, uint64_t size,
                                    struct radeon_winsys_bo **bo, unsigned *offset);
bool radv_meta_convert_native_indices(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va,
                                      uint32_t count, unsigned width, uint64_t *dst_va);
#endif
