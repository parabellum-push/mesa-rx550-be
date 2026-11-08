/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * Based on u_format.h which is:
 * Copyright 2009-2010 VMware, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_FORMATS_H
#define RADV_FORMATS_H

#include "util/u_endian.h"
#include <assert.h>
#include <util/macros.h>
#include <vulkan/vulkan.h>

#include "amd_family.h"

#include "vk_format.h"

static inline enum pipe_format
radv_format_to_pipe_format(VkFormat vkformat)
{
   switch (vkformat) {
   case VK_FORMAT_R10X6_UNORM_PACK16:
   case VK_FORMAT_R12X4_UNORM_PACK16:
      return PIPE_FORMAT_R16_UNORM;
   case VK_FORMAT_R10X6G10X6_UNORM_2PACK16:
   case VK_FORMAT_R12X4G12X4_UNORM_2PACK16:
      return PIPE_FORMAT_R16G16_UNORM;
   case VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16:
      return PIPE_FORMAT_NONE;
   default:
      return vk_format_to_pipe_format(vkformat);
   }
}

static inline const struct util_format_description *
radv_format_description(VkFormat format)
{
   return util_format_description(radv_format_to_pipe_format(format));
}

/**
 * Return the index of the first non-void channel
 * -1 if no non-void channels
 */
static inline int
vk_format_get_first_non_void_channel(VkFormat format)
{
   return util_format_get_first_non_void_channel(radv_format_to_pipe_format(format));
}

static inline enum pipe_swizzle
radv_swizzle_conv(VkComponentSwizzle component, const unsigned char chan[4], VkComponentSwizzle vk_swiz)
{
   if (vk_swiz == VK_COMPONENT_SWIZZLE_IDENTITY)
      vk_swiz = component;
   switch (vk_swiz) {
   case VK_COMPONENT_SWIZZLE_ZERO:
      return PIPE_SWIZZLE_0;
   case VK_COMPONENT_SWIZZLE_ONE:
      return PIPE_SWIZZLE_1;
   case VK_COMPONENT_SWIZZLE_R:
   case VK_COMPONENT_SWIZZLE_G:
   case VK_COMPONENT_SWIZZLE_B:
   case VK_COMPONENT_SWIZZLE_A:
      return (enum pipe_swizzle)chan[vk_swiz - VK_COMPONENT_SWIZZLE_R];
   default:
      UNREACHABLE("Illegal swizzle");
   }
}

static inline void
vk_format_compose_swizzles(const VkComponentMapping *mapping, const unsigned char swz[4], enum pipe_swizzle dst[4])
{
   dst[0] = radv_swizzle_conv(VK_COMPONENT_SWIZZLE_R, swz, mapping->r);
   dst[1] = radv_swizzle_conv(VK_COMPONENT_SWIZZLE_G, swz, mapping->g);
   dst[2] = radv_swizzle_conv(VK_COMPONENT_SWIZZLE_B, swz, mapping->b);
   dst[3] = radv_swizzle_conv(VK_COMPONENT_SWIZZLE_A, swz, mapping->a);
}

static inline bool
vk_format_is_subsampled(VkFormat format)
{
   return util_format_is_subsampled_422(radv_format_to_pipe_format(format));
}

static inline bool
vk_format_is_96bit(VkFormat format)
{
   return format == VK_FORMAT_R32G32B32_UINT || format == VK_FORMAT_R32G32B32_SINT ||
          format == VK_FORMAT_R32G32B32_SFLOAT;
}

static inline VkFormat
vk_format_no_srgb(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_SRGB:
      return VK_FORMAT_R8_UNORM;
   case VK_FORMAT_R8G8_SRGB:
      return VK_FORMAT_R8G8_UNORM;
   case VK_FORMAT_R8G8B8_SRGB:
      return VK_FORMAT_R8G8B8_UNORM;
   case VK_FORMAT_B8G8R8_SRGB:
      return VK_FORMAT_B8G8R8_UNORM;
   case VK_FORMAT_R8G8B8A8_SRGB:
      return VK_FORMAT_R8G8B8A8_UNORM;
   case VK_FORMAT_B8G8R8A8_SRGB:
      return VK_FORMAT_B8G8R8A8_UNORM;
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
      return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
   case VK_FORMAT_BC2_SRGB_BLOCK:
      return VK_FORMAT_BC2_UNORM_BLOCK;
   case VK_FORMAT_BC3_SRGB_BLOCK:
      return VK_FORMAT_BC3_UNORM_BLOCK;
   case VK_FORMAT_BC7_SRGB_BLOCK:
      return VK_FORMAT_BC7_UNORM_BLOCK;
   case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
      return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
   case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
      return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
   case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
      return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
   default:
      assert(!vk_format_is_srgb(format));
      return format;
   }
}

struct radv_physical_device;

uint32_t radv_translate_buffer_numformat(const struct util_format_description *desc, int first_non_void);

uint32_t radv_translate_tex_dataformat(const struct radv_physical_device *pdev,
                                       const struct util_format_description *desc, int first_non_void);

uint32_t radv_translate_tex_numformat(const struct util_format_description *desc, int first_non_void);

bool radv_is_atomic_format_supported(VkFormat format);

bool radv_is_storage_image_format_supported(const struct radv_physical_device *dev, VkFormat format);

bool radv_is_buffer_format_supported(VkFormat format, bool *scaled);

bool radv_is_colorbuffer_format_supported(const struct radv_physical_device *pdev, VkFormat format);

bool radv_is_format_emulated(const struct radv_physical_device *pdev, VkFormat format);

bool radv_format_pack_clear_color(VkFormat format, uint32_t clear_vals[2], VkClearColorValue *value);

bool radv_dcc_formats_compatible(enum amd_gfx_level gfx_level, VkFormat format1, VkFormat format2,
                                 bool *sign_reinterpret);

const struct util_format_description *radv_format_hw_description(const struct util_format_description *desc,
                                                                 struct util_format_description *storage);

uint32_t radv_get_cb_format(enum amd_gfx_level gfx_level, enum pipe_format format);

unsigned radv_translate_colorswap(enum amd_gfx_level gfx_level, VkFormat format, bool do_endian_swap);

/* The hardware applies sRGB to the first three bytes in memory, so storing packed
 * A8B8G8R8 as bytes A,B,G,R with a swizzle would convert alpha instead of red. On
 * big-endian hosts images of this family are stored as little-endian words, which
 * is the R8G8B8A8 hardware layout. All views of an image must agree, so this covers
 * the whole family. Buffers keep host order and do not use this.
 */
static inline VkFormat
radv_native_image_format(VkFormat format)
{
   if (!UTIL_ARCH_BIG_ENDIAN)
      return format;
   switch (format) {
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return VK_FORMAT_R8G8B8A8_UNORM;
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32: return VK_FORMAT_R8G8B8A8_SNORM;
   case VK_FORMAT_A8B8G8R8_USCALED_PACK32: return VK_FORMAT_R8G8B8A8_USCALED;
   case VK_FORMAT_A8B8G8R8_SSCALED_PACK32: return VK_FORMAT_R8G8B8A8_SSCALED;
   case VK_FORMAT_A8B8G8R8_UINT_PACK32: return VK_FORMAT_R8G8B8A8_UINT;
   case VK_FORMAT_A8B8G8R8_SINT_PACK32: return VK_FORMAT_R8G8B8A8_SINT;
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_R8G8B8A8_SRGB;
   default: return format;
   }
}

/* The size of the little-endian elements a format's image memory is made of. */
unsigned radv_native_format_width(VkFormat format);

#endif /* RADV_FORMATS_H */
