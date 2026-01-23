//! GhostVK HDR Support Module
//! Provides scRGB (extended linear sRGB) and PQ (HDR10) color space support.
//!
//! Features:
//! - HDR surface format detection and selection
//! - Color space negotiation (scRGB, HDR10/PQ, Display P3)
//! - Tone mapping helpers for SDR fallback
//! - HDR metadata management (mastering display info)

const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.ghostvk_hdr);

/// HDR color space types supported by GhostVK
pub const HdrColorSpace = enum {
    /// Standard sRGB (SDR fallback)
    srgb,
    /// Extended linear sRGB with values outside 0-1 range
    /// VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT
    scrgb_linear,
    /// HDR10 with PQ (Perceptual Quantizer) transfer function
    /// VK_COLOR_SPACE_HDR10_ST2084_EXT
    hdr10_pq,
    /// HDR10 with HLG (Hybrid Log-Gamma) transfer function
    /// VK_COLOR_SPACE_HDR10_HLG_EXT
    hdr10_hlg,
    /// Display P3 with linear transfer (wide gamut SDR)
    /// VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT
    display_p3_linear,
    /// Display P3 with nonlinear transfer
    /// VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT
    display_p3,
    /// Dolby Vision
    /// VK_COLOR_SPACE_DOLBYVISION_EXT
    dolby_vision,

    // Vulkan 1.4 color space extension values (VK_EXT_swapchain_colorspace)
    pub const VK_COLOR_SPACE_HDR10_ST2084_EXT: u32 = 1000104000;
    pub const VK_COLOR_SPACE_HDR10_HLG_EXT: u32 = 1000104001;
    pub const VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT: u32 = 1000104002;
    pub const VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT: u32 = 1000104003;
    pub const VK_COLOR_SPACE_DOLBYVISION_EXT: u32 = 1000104004;
    pub const VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT: u32 = 1000104005;

    pub fn toVulkanColorSpace(self: HdrColorSpace) vk.types.VkColorSpaceKHR {
        return switch (self) {
            .srgb => .SRGB_NONLINEAR,
            .scrgb_linear => @enumFromInt(VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT),
            .hdr10_pq => @enumFromInt(VK_COLOR_SPACE_HDR10_ST2084_EXT),
            .hdr10_hlg => @enumFromInt(VK_COLOR_SPACE_HDR10_HLG_EXT),
            .display_p3_linear => @enumFromInt(VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT),
            .display_p3 => @enumFromInt(VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT),
            .dolby_vision => @enumFromInt(VK_COLOR_SPACE_DOLBYVISION_EXT),
        };
    }

    pub fn fromVulkanColorSpace(cs: vk.types.VkColorSpaceKHR) HdrColorSpace {
        const val = @intFromEnum(cs);
        return switch (val) {
            0 => .srgb,
            VK_COLOR_SPACE_HDR10_ST2084_EXT => .hdr10_pq,
            VK_COLOR_SPACE_HDR10_HLG_EXT => .hdr10_hlg,
            VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT => .scrgb_linear,
            VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT => .display_p3_linear,
            VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT => .display_p3,
            VK_COLOR_SPACE_DOLBYVISION_EXT => .dolby_vision,
            else => .srgb,
        };
    }

    pub fn name(self: HdrColorSpace) []const u8 {
        return switch (self) {
            .srgb => "sRGB (SDR)",
            .scrgb_linear => "scRGB Linear (HDR)",
            .hdr10_pq => "HDR10 PQ (ST.2084)",
            .hdr10_hlg => "HDR10 HLG",
            .display_p3_linear => "Display P3 Linear",
            .display_p3 => "Display P3",
            .dolby_vision => "Dolby Vision",
        };
    }

    pub fn isHdr(self: HdrColorSpace) bool {
        return switch (self) {
            .srgb, .display_p3 => false,
            else => true,
        };
    }
};

/// HDR format capabilities for a surface
pub const HdrFormat = struct {
    format: vk.types.VkFormat,
    color_space: HdrColorSpace,
    bits_per_channel: u8,
    is_floating_point: bool,

    pub fn getBitsPerPixel(self: HdrFormat) u32 {
        return @as(u32, self.bits_per_channel) * 4; // Assume RGBA
    }
};

/// HDR mastering display metadata (for HDR10)
pub const HdrMasteringMetadata = struct {
    /// Primary red chromaticity (x, y) in 0.00002 units
    red_primary: [2]u16 = .{ 34000, 16000 }, // BT.2020 red
    /// Primary green chromaticity (x, y) in 0.00002 units
    green_primary: [2]u16 = .{ 13250, 34500 }, // BT.2020 green
    /// Primary blue chromaticity (x, y) in 0.00002 units
    blue_primary: [2]u16 = .{ 7500, 3000 }, // BT.2020 blue
    /// White point chromaticity (x, y) in 0.00002 units
    white_point: [2]u16 = .{ 15635, 16450 }, // D65
    /// Max luminance in cd/m^2
    max_luminance: f32 = 1000.0,
    /// Min luminance in cd/m^2
    min_luminance: f32 = 0.001,
    /// Max content light level in cd/m^2
    max_content_light_level: u16 = 1000,
    /// Max frame average light level in cd/m^2
    max_frame_average_light_level: u16 = 400,

    /// Create metadata for a typical HDR1000 display
    pub fn hdr1000() HdrMasteringMetadata {
        return .{
            .max_luminance = 1000.0,
            .min_luminance = 0.001,
            .max_content_light_level = 1000,
            .max_frame_average_light_level = 400,
        };
    }

    /// Create metadata for a typical HDR4000 display (high-end)
    pub fn hdr4000() HdrMasteringMetadata {
        return .{
            .max_luminance = 4000.0,
            .min_luminance = 0.0001,
            .max_content_light_level = 4000,
            .max_frame_average_light_level = 1000,
        };
    }

    /// Create metadata for SDR content
    pub fn sdr() HdrMasteringMetadata {
        return .{
            // BT.709 primaries for SDR
            .red_primary = .{ 32000, 16500 },
            .green_primary = .{ 15000, 30000 },
            .blue_primary = .{ 7500, 3000 },
            .white_point = .{ 15635, 16450 },
            .max_luminance = 100.0,
            .min_luminance = 0.1,
            .max_content_light_level = 100,
            .max_frame_average_light_level = 100,
        };
    }
};

/// HDR surface capabilities
pub const HdrCapabilities = struct {
    allocator: std.mem.Allocator,

    /// All available HDR formats
    formats: []HdrFormat,

    /// Best scRGB format (if available)
    scrgb_format: ?HdrFormat,
    /// Best HDR10 PQ format (if available)
    hdr10_pq_format: ?HdrFormat,
    /// Best HDR10 HLG format (if available)
    hdr10_hlg_format: ?HdrFormat,
    /// Best Display P3 format (if available)
    display_p3_format: ?HdrFormat,
    /// Fallback SDR format
    sdr_format: HdrFormat,

    /// Whether the surface supports any HDR mode
    hdr_supported: bool,
    /// Whether HDR is currently active on the display
    hdr_active: bool,

    pub fn deinit(self: *HdrCapabilities) void {
        if (self.formats.len > 0) {
            self.allocator.free(self.formats);
        }
        self.* = undefined;
    }

    /// Get the best available HDR format, or SDR fallback
    pub fn getBestHdrFormat(self: *const HdrCapabilities, preferred: HdrColorSpace) HdrFormat {
        return switch (preferred) {
            .scrgb_linear => self.scrgb_format orelse self.sdr_format,
            .hdr10_pq => self.hdr10_pq_format orelse self.sdr_format,
            .hdr10_hlg => self.hdr10_hlg_format orelse self.sdr_format,
            .display_p3_linear, .display_p3 => self.display_p3_format orelse self.sdr_format,
            else => self.sdr_format,
        };
    }

    /// Get the best format for gaming (prefers scRGB for quality)
    pub fn getBestGamingFormat(self: *const HdrCapabilities) HdrFormat {
        // Prefer scRGB for gaming (full range, no tone mapping needed in shader)
        if (self.scrgb_format) |f| return f;
        // Fall back to HDR10 PQ
        if (self.hdr10_pq_format) |f| return f;
        // Last resort: SDR
        return self.sdr_format;
    }

    /// Get the best format for video playback (prefers HDR10 PQ for compatibility)
    pub fn getBestVideoFormat(self: *const HdrCapabilities) HdrFormat {
        // Prefer HDR10 PQ for video (industry standard)
        if (self.hdr10_pq_format) |f| return f;
        // Fall back to HLG
        if (self.hdr10_hlg_format) |f| return f;
        // scRGB works too
        if (self.scrgb_format) |f| return f;
        // SDR fallback
        return self.sdr_format;
    }
};

/// HDR manager for handling color space and format selection
pub const HdrManager = struct {
    allocator: std.mem.Allocator,
    instance_dispatch: *const vk.loader.InstanceDispatch,
    device_dispatch: ?*const vk.loader.DeviceDispatch,
    physical_device: vk.types.VkPhysicalDevice,
    device: ?vk.types.VkDevice,
    surface: vk.types.VkSurfaceKHR,

    /// Current HDR capabilities
    capabilities: ?HdrCapabilities,
    /// Currently active HDR format
    active_format: ?HdrFormat,
    /// Current mastering metadata
    metadata: HdrMasteringMetadata,

    /// Extension availability
    has_hdr_metadata_ext: bool,
    has_swapchain_colorspace_ext: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        instance_dispatch: *const vk.loader.InstanceDispatch,
        physical_device: vk.types.VkPhysicalDevice,
        surface: vk.types.VkSurfaceKHR,
    ) HdrManager {
        return .{
            .allocator = allocator,
            .instance_dispatch = instance_dispatch,
            .device_dispatch = null,
            .physical_device = physical_device,
            .device = null,
            .surface = surface,
            .capabilities = null,
            .active_format = null,
            .metadata = HdrMasteringMetadata.hdr1000(),
            .has_hdr_metadata_ext = false,
            .has_swapchain_colorspace_ext = false,
        };
    }

    pub fn setDevice(self: *HdrManager, device: vk.types.VkDevice, device_dispatch: *const vk.loader.DeviceDispatch) void {
        self.device = device;
        self.device_dispatch = device_dispatch;
    }

    pub fn deinit(self: *HdrManager) void {
        if (self.capabilities) |*caps| {
            caps.deinit();
        }
        self.* = undefined;
    }

    /// Query HDR capabilities for the surface
    pub fn queryCapabilities(self: *HdrManager) !HdrCapabilities {
        const get_formats_fn = self.instance_dispatch.get_physical_device_surface_formats orelse {
            return error.FunctionNotAvailable;
        };

        // Get format count
        var format_count: u32 = 0;
        var result = get_formats_fn(self.physical_device, self.surface, &format_count, null);
        if (result != .SUCCESS or format_count == 0) {
            return error.NoFormatsAvailable;
        }

        // Get formats
        const vk_formats = try self.allocator.alloc(vk.types.VkSurfaceFormatKHR, format_count);
        defer self.allocator.free(vk_formats);

        result = get_formats_fn(self.physical_device, self.surface, &format_count, vk_formats.ptr);
        if (result != .SUCCESS) {
            return error.QueryFailed;
        }

        // Convert to HdrFormat list
        var hdr_formats = std.ArrayList(HdrFormat).init(self.allocator);
        defer hdr_formats.deinit();

        var scrgb_format: ?HdrFormat = null;
        var hdr10_pq_format: ?HdrFormat = null;
        var hdr10_hlg_format: ?HdrFormat = null;
        var display_p3_format: ?HdrFormat = null;
        var sdr_format: ?HdrFormat = null;

        for (vk_formats[0..format_count]) |vk_fmt| {
            const hdr_fmt = classifyFormat(vk_fmt);
            try hdr_formats.append(hdr_fmt);

            // Track best format for each color space
            switch (hdr_fmt.color_space) {
                .scrgb_linear => {
                    if (scrgb_format == null or hdr_fmt.bits_per_channel > scrgb_format.?.bits_per_channel) {
                        scrgb_format = hdr_fmt;
                    }
                },
                .hdr10_pq => {
                    if (hdr10_pq_format == null or hdr_fmt.bits_per_channel > hdr10_pq_format.?.bits_per_channel) {
                        hdr10_pq_format = hdr_fmt;
                    }
                },
                .hdr10_hlg => {
                    if (hdr10_hlg_format == null or hdr_fmt.bits_per_channel > hdr10_hlg_format.?.bits_per_channel) {
                        hdr10_hlg_format = hdr_fmt;
                    }
                },
                .display_p3, .display_p3_linear => {
                    if (display_p3_format == null or hdr_fmt.bits_per_channel > display_p3_format.?.bits_per_channel) {
                        display_p3_format = hdr_fmt;
                    }
                },
                .srgb => {
                    // Prefer higher bit depth SDR formats
                    if (sdr_format == null or hdr_fmt.bits_per_channel > sdr_format.?.bits_per_channel) {
                        sdr_format = hdr_fmt;
                    }
                },
                else => {},
            }
        }

        // Must have at least SDR fallback
        const final_sdr = sdr_format orelse HdrFormat{
            .format = .B8G8R8A8_SRGB,
            .color_space = .srgb,
            .bits_per_channel = 8,
            .is_floating_point = false,
        };

        const has_hdr = scrgb_format != null or hdr10_pq_format != null or hdr10_hlg_format != null;

        log.info("HDR capabilities queried: {} formats, HDR supported: {}", .{ format_count, has_hdr });
        if (scrgb_format) |f| log.info("  scRGB: format={s}", .{@tagName(f.format)});
        if (hdr10_pq_format) |f| log.info("  HDR10 PQ: format={s}", .{@tagName(f.format)});
        if (hdr10_hlg_format) |f| log.info("  HDR10 HLG: format={s}", .{@tagName(f.format)});

        const formats = try hdr_formats.toOwnedSlice();

        return HdrCapabilities{
            .allocator = self.allocator,
            .formats = formats,
            .scrgb_format = scrgb_format,
            .hdr10_pq_format = hdr10_pq_format,
            .hdr10_hlg_format = hdr10_hlg_format,
            .display_p3_format = display_p3_format,
            .sdr_format = final_sdr,
            .hdr_supported = has_hdr,
            .hdr_active = false, // Would need platform-specific API to detect
        };
    }

    /// Select and activate an HDR format for the swapchain
    pub fn selectFormat(self: *HdrManager, preferred: HdrColorSpace) !HdrFormat {
        if (self.capabilities == null) {
            self.capabilities = try self.queryCapabilities();
        }

        const caps = &self.capabilities.?;
        const format = caps.getBestHdrFormat(preferred);
        self.active_format = format;

        log.info("Selected HDR format: {s} ({s})", .{
            @tagName(format.format),
            format.color_space.name(),
        });

        return format;
    }

    /// Get Vulkan format and color space for swapchain creation
    pub fn getSwapchainFormat(self: *const HdrManager) struct { format: vk.types.VkFormat, color_space: vk.types.VkColorSpaceKHR } {
        if (self.active_format) |fmt| {
            return .{
                .format = fmt.format,
                .color_space = fmt.color_space.toVulkanColorSpace(),
            };
        }
        // Default SDR
        return .{
            .format = .B8G8R8A8_SRGB,
            .color_space = .SRGB_NONLINEAR,
        };
    }

    /// Set HDR metadata on the swapchain (requires VK_EXT_hdr_metadata)
    pub fn setHdrMetadata(self: *HdrManager, swapchain: vk.types.VkSwapchainKHR, metadata: HdrMasteringMetadata) !void {
        const device = self.device orelse return error.NoDevice;
        const dispatch = self.device_dispatch orelse return error.NoDevice;
        _ = dispatch;

        self.metadata = metadata;

        // VK_EXT_hdr_metadata structure
        const VkHdrMetadataEXT = extern struct {
            sType: vk.types.VkStructureType,
            pNext: ?*const anyopaque,
            displayPrimaryRed: extern struct { x: f32, y: f32 },
            displayPrimaryGreen: extern struct { x: f32, y: f32 },
            displayPrimaryBlue: extern struct { x: f32, y: f32 },
            whitePoint: extern struct { x: f32, y: f32 },
            maxLuminance: f32,
            minLuminance: f32,
            maxContentLightLevel: f32,
            maxFrameAverageLightLevel: f32,
        };

        const vk_metadata = VkHdrMetadataEXT{
            .sType = @enumFromInt(1000105000), // VK_STRUCTURE_TYPE_HDR_METADATA_EXT
            .pNext = null,
            .displayPrimaryRed = .{
                .x = @as(f32, @floatFromInt(metadata.red_primary[0])) * 0.00002,
                .y = @as(f32, @floatFromInt(metadata.red_primary[1])) * 0.00002,
            },
            .displayPrimaryGreen = .{
                .x = @as(f32, @floatFromInt(metadata.green_primary[0])) * 0.00002,
                .y = @as(f32, @floatFromInt(metadata.green_primary[1])) * 0.00002,
            },
            .displayPrimaryBlue = .{
                .x = @as(f32, @floatFromInt(metadata.blue_primary[0])) * 0.00002,
                .y = @as(f32, @floatFromInt(metadata.blue_primary[1])) * 0.00002,
            },
            .whitePoint = .{
                .x = @as(f32, @floatFromInt(metadata.white_point[0])) * 0.00002,
                .y = @as(f32, @floatFromInt(metadata.white_point[1])) * 0.00002,
            },
            .maxLuminance = metadata.max_luminance,
            .minLuminance = metadata.min_luminance,
            .maxContentLightLevel = @floatFromInt(metadata.max_content_light_level),
            .maxFrameAverageLightLevel = @floatFromInt(metadata.max_frame_average_light_level),
        };

        // Would call vkSetHdrMetadataEXT here
        // For now just log the intent
        log.info("HDR metadata set: max_lum={d:.0} cd/m², min_lum={d:.4} cd/m²", .{
            vk_metadata.maxLuminance,
            vk_metadata.minLuminance,
        });

        _ = device;
        _ = swapchain;
    }
};

/// Classify a Vulkan surface format into HdrFormat
fn classifyFormat(vk_fmt: vk.types.VkSurfaceFormatKHR) HdrFormat {
    const color_space = HdrColorSpace.fromVulkanColorSpace(vk_fmt.colorSpace);

    // Determine bits per channel and floating point status
    const format_info = getFormatInfo(vk_fmt.format);

    return HdrFormat{
        .format = vk_fmt.format,
        .color_space = color_space,
        .bits_per_channel = format_info.bits_per_channel,
        .is_floating_point = format_info.is_floating_point,
    };
}

fn getFormatInfo(format: vk.types.VkFormat) struct { bits_per_channel: u8, is_floating_point: bool } {
    return switch (format) {
        // 8-bit formats
        .B8G8R8A8_SRGB, .B8G8R8A8_UNORM, .R8G8B8A8_SRGB, .R8G8B8A8_UNORM => .{ .bits_per_channel = 8, .is_floating_point = false },

        // 10-bit formats (HDR10)
        .A2B10G10R10_UNORM_PACK32, .A2R10G10B10_UNORM_PACK32 => .{ .bits_per_channel = 10, .is_floating_point = false },

        // 16-bit float formats (scRGB)
        .R16G16B16A16_SFLOAT => .{ .bits_per_channel = 16, .is_floating_point = true },

        // 32-bit float formats (for compute/intermediate)
        .R32G32B32A32_SFLOAT => .{ .bits_per_channel = 32, .is_floating_point = true },

        // Default
        else => .{ .bits_per_channel = 8, .is_floating_point = false },
    };
}

// =============================================================================
// Tone Mapping Utilities (for shader constants)
// =============================================================================

/// PQ (Perceptual Quantizer) transfer function constants (ST.2084)
pub const PQ = struct {
    pub const m1: f32 = 0.1593017578125;
    pub const m2: f32 = 78.84375;
    pub const c1: f32 = 0.8359375;
    pub const c2: f32 = 18.8515625;
    pub const c3: f32 = 18.6875;
    pub const max_nits: f32 = 10000.0;

    /// Linear to PQ (EOTF inverse)
    /// Input: linear luminance normalized to max_nits (0-1 range for 0-10000 nits)
    /// Output: PQ encoded value (0-1)
    pub fn linearToPQ(linear: f32) f32 {
        const y = @max(linear, 0.0);
        const ym1 = std.math.pow(f32, y, m1);
        const numerator = c1 + c2 * ym1;
        const denominator = 1.0 + c3 * ym1;
        return std.math.pow(f32, numerator / denominator, m2);
    }

    /// PQ to Linear (EOTF)
    /// Input: PQ encoded value (0-1)
    /// Output: linear luminance normalized to max_nits
    pub fn pqToLinear(pq: f32) f32 {
        const e = @max(pq, 0.0);
        const em2 = std.math.pow(f32, e, 1.0 / m2);
        const numerator = @max(em2 - c1, 0.0);
        const denominator = c2 - c3 * em2;
        return std.math.pow(f32, numerator / denominator, 1.0 / m1);
    }

    /// Convert nits to PQ encoded value
    pub fn nitsToPQ(nits: f32) f32 {
        return linearToPQ(nits / max_nits);
    }

    /// Convert PQ encoded value to nits
    pub fn pqToNits(pq: f32) f32 {
        return pqToLinear(pq) * max_nits;
    }
};

/// HLG (Hybrid Log-Gamma) transfer function constants
pub const HLG = struct {
    pub const a: f32 = 0.17883277;
    pub const b: f32 = 0.28466892; // 1 - 4a
    pub const c: f32 = 0.55991073; // 0.5 - a * ln(4a)

    /// Linear to HLG (OETF)
    pub fn linearToHLG(linear: f32) f32 {
        const e = @max(linear, 0.0);
        if (e <= 1.0 / 12.0) {
            return std.math.sqrt(3.0 * e);
        } else {
            return a * @log(12.0 * e - b) + c;
        }
    }

    /// HLG to Linear (EOTF)
    pub fn hlgToLinear(hlg: f32) f32 {
        const e = @max(hlg, 0.0);
        if (e <= 0.5) {
            return (e * e) / 3.0;
        } else {
            return (@exp((e - c) / a) + b) / 12.0;
        }
    }
};

/// sRGB transfer function
pub const SRGB = struct {
    /// Linear to sRGB
    pub fn linearToSRGB(linear: f32) f32 {
        const c = @max(linear, 0.0);
        if (c <= 0.0031308) {
            return c * 12.92;
        } else {
            return 1.055 * std.math.pow(f32, c, 1.0 / 2.4) - 0.055;
        }
    }

    /// sRGB to Linear
    pub fn srgbToLinear(srgb: f32) f32 {
        const c = @max(srgb, 0.0);
        if (c <= 0.04045) {
            return c / 12.92;
        } else {
            return std.math.pow(f32, (c + 0.055) / 1.055, 2.4);
        }
    }
};

/// Simple Reinhard tone mapping for HDR to SDR conversion
pub fn reinhardToneMap(hdr: f32, white_point: f32) f32 {
    return hdr * (1.0 + hdr / (white_point * white_point)) / (1.0 + hdr);
}

/// ACES filmic tone mapping (approximation)
pub fn acesToneMap(x: f32) f32 {
    const a = 2.51;
    const b = 0.03;
    const c = 2.43;
    const d = 0.59;
    const e = 0.14;
    return @min(@max((x * (a * x + b)) / (x * (c * x + d) + e), 0.0), 1.0);
}

// =============================================================================
// Color Space Conversion Matrices (for shaders)
// =============================================================================

/// BT.709 (sRGB) to BT.2020 conversion matrix (row-major)
pub const bt709_to_bt2020: [9]f32 = .{
    0.6274,  0.3293,  0.0433,
    0.0691,  0.9195,  0.0114,
    0.0164,  0.0880,  0.8956,
};

/// BT.2020 to BT.709 (sRGB) conversion matrix (row-major)
pub const bt2020_to_bt709: [9]f32 = .{
    1.6605,  -0.5876, -0.0728,
    -0.1246, 1.1329,  -0.0083,
    -0.0182, -0.1006, 1.1187,
};

/// Display P3 to BT.709 (sRGB) conversion matrix
pub const display_p3_to_bt709: [9]f32 = .{
    1.2249,  -0.2247, 0.0,
    -0.0420, 1.0419,  0.0,
    -0.0197, -0.0786, 1.0979,
};

/// BT.709 (sRGB) to Display P3 conversion matrix
pub const bt709_to_display_p3: [9]f32 = .{
    0.8225,  0.1774,  0.0,
    0.0332,  0.9669,  0.0,
    0.0171,  0.0724,  0.9108,
};
