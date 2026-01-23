//! GhostVK core library: Vulkan bootstrap & diagnostics layer.
//! Phase 1: Project Bootstrap & Foundation
//!
//! Modules:
//! - memory: VMA-style GPU memory allocator with pools and sub-allocation
//! - command_pool: Efficient command buffer management with recycling
//! - pipeline_cache: Pipeline cache with disk persistence
//! - hdr: HDR color space support (scRGB, HDR10/PQ)

pub const std_options = struct {
    pub const log_level = .debug;
};

const std = @import("std");
const vk = @import("vulkan");
const wl = @import("wayland.zig");

// Export sub-modules
pub const memory = @import("memory.zig");
pub const command_pool = @import("command_pool.zig");
pub const pipeline_cache = @import("pipeline_cache.zig");
pub const hdr = @import("hdr.zig");
pub const render = @import("render.zig");
pub const frame_pacer = @import("frame_pacer.zig");

const log = std.log.scoped(.ghostvk);

const validation_layer_name: [:0]const u8 = "VK_LAYER_KHRONOS_validation";
const debug_utils_ext_name: [:0]const u8 = "VK_EXT_debug_utils";
const surface_ext_name: [:0]const u8 = "VK_KHR_surface";
const wayland_surface_ext_name: [:0]const u8 = "VK_KHR_wayland_surface";
const swapchain_ext_name: [:0]const u8 = "VK_KHR_swapchain";

pub const Error = error{
    MissingValidationLayer,
    VulkanInstanceCreationFailed,
    DebugMessengerCreationFailed,
    NoVulkanDevicesFound,
    QueueFamiliesIncomplete,
    VulkanDeviceCreationFailed,
    VulkanCallFailed,
    MissingDebugExtension,
    WaylandConnectionFailed,
    WaylandCompositorNotFound,
    WaylandSurfaceCreationFailed,
    VulkanSurfaceCreationFailed,
    OutOfMemory,
    InitializationFailed,
} || vk.errors.Error;

pub const InitOptions = struct {
    enable_validation: bool = true,
    application_name: [:0]const u8 = "GhostVK",
    application_version: u32 = vk.types.makeApiVersion(0, 1, 0),
    engine_name: [:0]const u8 = "GhostVK",
    engine_version: u32 = vk.types.makeApiVersion(0, 1, 0),
    /// Prefer HDR output if available (HDR10 ST2084 or scRGB)
    prefer_hdr: bool = true,
    /// Preferred HDR color space (defaults to HDR10 PQ)
    preferred_hdr_colorspace: hdr.HdrColorSpace = .hdr10_pq,
    /// Enable frame pacing with VRR awareness
    enable_frame_pacing: bool = true,
    /// Target FPS for frame pacing (0 = unlimited, synced to VRR max)
    target_fps: u32 = 0,
    /// Frame pacing mode
    pacing_mode: frame_pacer.PacingMode = .hybrid,
};

const QueueFamilies = struct {
    graphics: ?u32 = null,
    compute: ?u32 = null,
    transfer: ?u32 = null,

    fn isComplete(self: @This()) bool {
        return self.graphics != null;
    }
};

pub const GhostVK = struct {
    allocator: std.mem.Allocator,
    options: InitOptions,
    validation_enabled: bool = false,

    loader: vk.loader.Loader,
    global_dispatch: *const vk.loader.GlobalDispatch,
    instance_dispatch: ?vk.loader.InstanceDispatch = null,
    device_dispatch: ?vk.loader.DeviceDispatch = null,

    instance: ?vk.types.VkInstance = null,
    debug_messenger: ?vk.types.VkDebugUtilsMessengerEXT = null,
    physical_device: ?vk.types.VkPhysicalDevice = null,
    device: ?vk.types.VkDevice = null,

    graphics_queue: ?vk.types.VkQueue = null,
    compute_queue: ?vk.types.VkQueue = null,
    transfer_queue: ?vk.types.VkQueue = null,

    graphics_queue_family: u32 = 0,
    compute_queue_family: u32 = 0,
    transfer_queue_family: u32 = 0,

    // Wayland
    wl_display: ?*wl.wl_display = null,
    wl_compositor: ?*wl.wl_compositor = null,
    wl_surface: ?*wl.wl_surface = null,

    // Vulkan surface
    surface: ?vk.types.VkSurfaceKHR = null,

    // Swapchain
    swapchain: ?vk.types.VkSwapchainKHR = null,
    swapchain_images: []vk.types.VkImage = &[_]vk.types.VkImage{},
    swapchain_image_views: []vk.types.VkImageView = &[_]vk.types.VkImageView{},
    swapchain_format: vk.types.VkFormat = .UNDEFINED,
    swapchain_colorspace: hdr.HdrColorSpace = .srgb,
    swapchain_extent: vk.types.VkExtent2D = .{ .width = 0, .height = 0 },
    last_presented_image: u32 = 0,

    // Phase 2: Command buffers and synchronization
    command_pool: ?vk.types.VkCommandPool = null,
    command_buffers: []vk.types.VkCommandBuffer = &[_]vk.types.VkCommandBuffer{},

    // Synchronization primitives
    // Per-image semaphores for present (prevents reuse issues with swapchain)
    render_finished_semaphores: []vk.types.VkSemaphore = &[_]vk.types.VkSemaphore{},

    // Frame fences (per frame in flight)
    in_flight_fences: []vk.types.VkFence = &[_]vk.types.VkFence{},

    // Image-in-flight fences (tracks which frame is using which image)
    image_in_flight_fences: []?vk.types.VkFence = &[_]?vk.types.VkFence{},

    // Acquire fence and per-frame acquire semaphores
    acquire_fence: ?vk.types.VkFence = null,
    acquire_semaphores: []vk.types.VkSemaphore = &[_]vk.types.VkSemaphore{}, // Per-frame (cycles with current_frame)

    frames_in_flight: u32 = 2,
    current_frame: u32 = 0,
    frame_count: u64 = 0,
    acquire_semaphore_index: u32 = 0, // Cycles through acquire semaphore pool
    present_semaphore_index: u32 = 0, // Cycles through present semaphore pool

    // Render pipeline
    render_pipeline: ?render.RenderPipeline = null,
    start_time: std.time.Instant = undefined,

    // Frame pacing (VRR-aware)
    pacer: ?frame_pacer.FramePacer = null,

    pub fn init(allocator: std.mem.Allocator, options: InitOptions) Error!GhostVK {
        log.info("Bootstrapping GhostVK runtime (validation requested: {})", .{options.enable_validation});

        var loader = vk.loader.Loader.init(allocator, .{}) catch |err| {
            log.err("Failed to initialize Vulkan loader: {}", .{err});
            return Error.VulkanInstanceCreationFailed;
        };
        errdefer loader.deinit();
        log.info("Vulkan loader initialized", .{});

        const global_dispatch = loader.global() catch |err| {
            log.err("Failed to get global dispatch: {}", .{err});
            return Error.VulkanInstanceCreationFailed;
        };
        log.info("Global dispatch loaded", .{});

        var self = GhostVK{
            .allocator = allocator,
            .options = options,
            .loader = loader,
            .global_dispatch = global_dispatch,
        };

        var enabled_layers = try self.prepareValidationLayers();
        defer enabled_layers.deinit();

        var enabled_extensions = try self.prepareInstanceExtensions();
        defer enabled_extensions.deinit();

        try self.createInstance(enabled_layers.items, enabled_extensions.items);

        if (self.validation_enabled) {
            try self.setupDebugMessenger();
        }

        try self.selectPhysicalDevice();

        // Initialize Wayland and surface
        try self.initWayland();
        errdefer self.cleanupWayland();

        try self.createVulkanSurface();
        errdefer self.cleanupVulkanSurface();

        try self.createLogicalDevice();

        try self.createSwapchain();
        errdefer self.cleanupSwapchain();

        try self.createCommandPool();
        errdefer self.cleanupCommandPool();

        try self.createCommandBuffers();
        errdefer self.cleanupCommandBuffers();

        try self.createSyncPrimitives();
        errdefer self.cleanupSyncPrimitives();

        // Create render pipeline
        self.render_pipeline = render.RenderPipeline.init(
            allocator,
            self.device.?,
            &self.device_dispatch.?,
            self.swapchain_format,
            self.swapchain_extent,
            self.swapchain_image_views,
        ) catch |err| {
            log.err("Failed to create render pipeline: {}", .{err});
            return Error.VulkanCallFailed;
        };
        errdefer if (self.render_pipeline) |*rp| rp.deinit();

        // Initialize timing
        self.start_time = std.time.Instant.now() catch return Error.InitializationFailed;

        // Initialize frame pacer with VRR awareness
        if (options.enable_frame_pacing) {
            if (frame_pacer.FramePacer.init(allocator, .{
                .target_fps = options.target_fps,
                .mode = options.pacing_mode,
            })) |p| {
                self.pacer = p;
                var pacer_ptr = &self.pacer.?;
                const stats = pacer_ptr.getStats();
                if (stats.vrr_enabled) {
                    log.info("Frame pacer enabled with VRR: {}-{} Hz", .{ stats.vrr_range[0], stats.vrr_range[1] });
                } else {
                    log.info("Frame pacer enabled (VRR not detected)", .{});
                }
            } else |err| {
                log.warn("Failed to initialize frame pacer: {} (continuing without pacing)", .{err});
            }
        }

        log.info("GhostVK initialization complete", .{});
        return self;
    }

    pub fn deinit(self: *GhostVK) void {
        // Clean up frame pacer first (logs stats)
        if (self.pacer) |*p| {
            p.deinit();
            self.pacer = null;
        }

        if (self.device) |device| {
            if (self.device_dispatch) |dispatch| {
                // Wait for all queues to be completely idle before any cleanup
                // This ensures all pending operations (including overlay layers) complete
                if (self.graphics_queue) |q| {
                    const r = dispatch.queue_wait_idle(q);
                    if (r != .SUCCESS) log.warn("Graphics queue wait failed during shutdown: {}", .{r});
                }
                if (self.compute_queue) |q| {
                    const r = dispatch.queue_wait_idle(q);
                    if (r != .SUCCESS) log.warn("Compute queue wait failed during shutdown: {}", .{r});
                }
                if (self.transfer_queue) |q| {
                    const r = dispatch.queue_wait_idle(q);
                    if (r != .SUCCESS) log.warn("Transfer queue wait failed during shutdown: {}", .{r});
                }

                // Destroy render pipeline first (depends on swapchain image views)
                if (self.render_pipeline) |*rp| {
                    rp.deinit();
                    self.render_pipeline = null;
                }

                self.cleanupSyncPrimitives();
                self.cleanupCommandBuffers();
                self.cleanupCommandPool();
                self.cleanupSwapchain();

                // Final queue waits before device destruction
                if (self.graphics_queue) |q| {
                    const r = dispatch.queue_wait_idle(q);
                    if (r != .SUCCESS) log.warn("Final graphics queue wait failed: {}", .{r});
                }

                dispatch.destroy_device(device, null);
            }
            self.device = null;
        }

        self.cleanupVulkanSurface();

        if (self.debug_messenger) |messenger| {
            if (self.instance) |instance| {
                const get_proc = self.loader.getInstanceProcAddr() orelse return;
                const destroy_raw = get_proc(instance, "vkDestroyDebugUtilsMessengerEXT");
                if (destroy_raw) |raw| {
                    const destroy_fn: vk.types.PFN_vkDestroyDebugUtilsMessengerEXT = @ptrCast(raw);
                    if (destroy_fn) |func| {
                        func(instance, messenger, null);
                    }
                }
            }
            self.debug_messenger = null;
        }

        if (self.instance) |instance| {
            if (self.instance_dispatch) |dispatch| {
                dispatch.destroy_instance(instance, null);
            }
            self.instance = null;
        }

        self.cleanupWayland();

        self.loader.deinit();
        log.info("GhostVK shutdown complete", .{});
    }

    fn prepareValidationLayers(self: *GhostVK) Error!std.array_list.Managed([*:0]const u8) {
        var layers = std.array_list.Managed([*:0]const u8).init(self.allocator);
        errdefer layers.deinit();

        const validation_available = try self.enumerateLayers();

        if (self.options.enable_validation) {
            if (!validation_available) {
                log.err("Validation layer {s} missing", .{validation_layer_name});
                return Error.MissingValidationLayer;
            }
            try layers.append(validation_layer_name.ptr);
            self.validation_enabled = true;
            log.info("Enabled validation layer {s}", .{validation_layer_name});
        } else {
            log.info("Validation layers disabled via options", .{});
        }

        return layers;
    }

    fn prepareInstanceExtensions(self: *GhostVK) Error!std.array_list.Managed([*:0]const u8) {
        var extensions = std.array_list.Managed([*:0]const u8).init(self.allocator);
        errdefer extensions.deinit();

        try self.logInstanceExtensions();

        // VK_KHR_surface (required for any surface operations)
        if (!try self.hasInstanceExtension(surface_ext_name)) {
            log.err("Required instance extension {s} not available", .{surface_ext_name});
            return Error.MissingDebugExtension;
        }
        try extensions.append(surface_ext_name.ptr);

        // VK_KHR_wayland_surface (Wayland platform support)
        if (!try self.hasInstanceExtension(wayland_surface_ext_name)) {
            log.err("Required instance extension {s} not available", .{wayland_surface_ext_name});
            return Error.MissingDebugExtension;
        }
        try extensions.append(wayland_surface_ext_name.ptr);

        if (self.validation_enabled) {
            if (!try self.hasInstanceExtension(debug_utils_ext_name)) {
                log.err("Required instance extension {s} not available", .{debug_utils_ext_name});
                return Error.MissingDebugExtension;
            }
            try extensions.append(debug_utils_ext_name.ptr);
        }

        return extensions;
    }

    fn createInstance(self: *GhostVK, enabled_layers: []const [*:0]const u8, enabled_extensions: []const [*:0]const u8) Error!void {
        const app_info = vk.types.VkApplicationInfo{
            .sType = .APPLICATION_INFO,
            .pApplicationName = self.options.application_name.ptr,
            .applicationVersion = self.options.application_version,
            .pEngineName = self.options.engine_name.ptr,
            .engineVersion = self.options.engine_version,
            .apiVersion = vk.types.makeApiVersion(1, 3, 0),
            .pNext = null,
        };

        var debug_create_info: vk.types.VkDebugUtilsMessengerCreateInfoEXT = undefined;
        var create_info = vk.types.VkInstanceCreateInfo{
            .sType = .INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = @intCast(enabled_layers.len),
            .ppEnabledLayerNames = if (enabled_layers.len > 0) enabled_layers.ptr else null,
            .enabledExtensionCount = @intCast(enabled_extensions.len),
            .ppEnabledExtensionNames = if (enabled_extensions.len > 0) enabled_extensions.ptr else null,
            .flags = 0,
            .pNext = null,
        };

        if (self.validation_enabled) {
            debug_create_info = self.debugMessengerCreateInfo();
            create_info.pNext = @ptrCast(&debug_create_info);
        }

        var instance: vk.types.VkInstance = undefined;
        const result = self.global_dispatch.create_instance(&create_info, null, &instance);
        if (result != .SUCCESS) {
            log.err("vkCreateInstance failed with {s}", .{@tagName(result)});
            return Error.VulkanInstanceCreationFailed;
        }

        self.instance = instance;
        self.instance_dispatch = try self.loader.instanceDispatch(instance);
        log.info("Vulkan instance created", .{});
    }

    fn setupDebugMessenger(self: *GhostVK) Error!void {
        const instance = self.instance orelse return Error.VulkanInstanceCreationFailed;
        const get_proc = self.loader.getInstanceProcAddr() orelse return Error.DebugMessengerCreationFailed;

        const create_raw = get_proc(instance, "vkCreateDebugUtilsMessengerEXT");
        if (create_raw == null) {
            log.err("vkCreateDebugUtilsMessengerEXT not found", .{});
            return Error.DebugMessengerCreationFailed;
        }

        const create_fn: vk.types.PFN_vkCreateDebugUtilsMessengerEXT = @ptrCast(create_raw.?);
        var create_info = self.debugMessengerCreateInfo();

        var messenger: vk.types.VkDebugUtilsMessengerEXT = undefined;
        const result = if (create_fn) |func| func(instance, &create_info, null, &messenger) else return Error.DebugMessengerCreationFailed;
        if (result != .SUCCESS) {
            log.err("vkCreateDebugUtilsMessengerEXT failed with {s}", .{@tagName(result)});
            return Error.DebugMessengerCreationFailed;
        }

        self.debug_messenger = messenger;
        log.info("Debug messenger initialized", .{});
    }

    fn selectPhysicalDevice(self: *GhostVK) Error!void {
        const instance = self.instance orelse return Error.VulkanInstanceCreationFailed;
        const dispatch = self.instance_dispatch orelse return Error.VulkanInstanceCreationFailed;

        var device_count: u32 = 0;
        var result = dispatch.enumerate_physical_devices(instance, &device_count, null);
        if (result != .SUCCESS and result != .INCOMPLETE) {
            log.err("Failed to enumerate physical devices: {}", .{result});
            return Error.NoVulkanDevicesFound;
        }
        if (device_count == 0) {
            return Error.NoVulkanDevicesFound;
        }

        const devices = try self.allocator.alloc(vk.types.VkPhysicalDevice, device_count);
        defer self.allocator.free(devices);

        result = dispatch.enumerate_physical_devices(instance, &device_count, devices.ptr);
        if (result != .SUCCESS and result != .INCOMPLETE) {
            log.err("Failed to get physical devices: {}", .{result});
            return Error.NoVulkanDevicesFound;
        }

        var selected: ?vk.types.VkPhysicalDevice = null;
        var selected_families: QueueFamilies = .{};
        for (devices[0..device_count]) |device| {
            const families = try self.findQueueFamilies(device);
            if (!families.isComplete()) continue;

            var properties: vk.types.VkPhysicalDeviceProperties = undefined;
            dispatch.get_physical_device_properties(device, &properties);

            if (properties.deviceType == .DISCRETE_GPU) {
                selected = device;
                selected_families = families;
                break;
            }

            if (selected == null) {
                selected = device;
                selected_families = families;
            }
        }

        if (selected == null) {
            return Error.NoVulkanDevicesFound;
        }

        self.physical_device = selected.?;
        self.graphics_queue_family = selected_families.graphics.?;
        self.compute_queue_family = selected_families.compute.?;
        self.transfer_queue_family = selected_families.transfer.?;

        var properties: vk.types.VkPhysicalDeviceProperties = undefined;
        dispatch.get_physical_device_properties(self.physical_device.?, &properties);
        self.printDeviceProperties(properties, self.physical_device.?);
    }

    fn createLogicalDevice(self: *GhostVK) Error!void {
        const physical_device = self.physical_device orelse return Error.NoVulkanDevicesFound;
        const dispatch = self.instance_dispatch orelse return Error.VulkanInstanceCreationFailed;

        const families = [_]u32{ self.graphics_queue_family, self.compute_queue_family, self.transfer_queue_family };

        var unique_families: [3]u32 = undefined;
        var unique_count: usize = 0;
        for (families) |family| {
            var exists = false;
            for (unique_families[0..unique_count]) |existing| {
                if (existing == family) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                unique_families[unique_count] = family;
                unique_count += 1;
            }
        }

        const queue_priority = [_]f32{1.0};
        var queue_infos: [3]vk.types.VkDeviceQueueCreateInfo = undefined;
        for (unique_families[0..unique_count], 0..) |family, i| {
            queue_infos[i] = vk.types.VkDeviceQueueCreateInfo{
                .sType = .DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = family,
                .queueCount = 1,
                .pQueuePriorities = &queue_priority,
                .flags = 0,
                .pNext = null,
            };
        }

        var device_features = std.mem.zeroes(vk.types.VkPhysicalDeviceFeatures);

        const device_extensions = [_][*:0]const u8{swapchain_ext_name.ptr};

        const device_create_info = vk.types.VkDeviceCreateInfo{
            .sType = .DEVICE_CREATE_INFO,
            .queueCreateInfoCount = @intCast(unique_count),
            .pQueueCreateInfos = &queue_infos,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = device_extensions.len,
            .ppEnabledExtensionNames = &device_extensions,
            .pEnabledFeatures = &device_features,
            .flags = 0,
            .pNext = null,
        };

        var device: vk.types.VkDevice = undefined;
        const result = dispatch.create_device(physical_device, &device_create_info, null, &device);
        if (result != .SUCCESS) {
            log.err("vkCreateDevice failed with {s}", .{@tagName(result)});
            return Error.VulkanDeviceCreationFailed;
        }

        self.device = device;
        self.device_dispatch = try self.loader.deviceDispatch(device);

        const dev_dispatch = self.device_dispatch.?;
        var gfx_queue: vk.types.VkQueue = undefined;
        var cmp_queue: vk.types.VkQueue = undefined;
        var xfr_queue: vk.types.VkQueue = undefined;
        dev_dispatch.get_device_queue(device, self.graphics_queue_family, 0, &gfx_queue);
        dev_dispatch.get_device_queue(device, self.compute_queue_family, 0, &cmp_queue);
        dev_dispatch.get_device_queue(device, self.transfer_queue_family, 0, &xfr_queue);
        self.graphics_queue = gfx_queue;
        self.compute_queue = cmp_queue;
        self.transfer_queue = xfr_queue;

        log.info("Logical device created (graphics family={}, compute family={}, transfer family={})", .{
            self.graphics_queue_family,
            self.compute_queue_family,
            self.transfer_queue_family,
        });
    }

    fn enumerateLayers(self: *GhostVK) !bool {
        var layer_count: u32 = 0;
        var result = self.global_dispatch.enumerate_instance_layer_properties(&layer_count, null);
        if (result != .SUCCESS and result != .INCOMPLETE) {
            log.warn("Failed to enumerate instance layers: {}", .{result});
            return false;
        }
        if (layer_count == 0) {
            log.warn("No Vulkan instance layers reported", .{});
            return false;
        }

        const layers = try self.allocator.alloc(vk.types.VkLayerProperties, layer_count);
        defer self.allocator.free(layers);

        result = self.global_dispatch.enumerate_instance_layer_properties(&layer_count, layers.ptr);
        if (result != .SUCCESS and result != .INCOMPLETE) {
            log.warn("Failed to get instance layers: {}", .{result});
            return false;
        }

        log.info("Instance layers ({}):", .{layer_count});
        var has_validation = false;
        for (layers[0..layer_count]) |layer| {
            const name = std.mem.sliceTo(&layer.layerName, 0);
            const description = std.mem.sliceTo(&layer.description, 0);
            log.info("  {s} — {s} (spec {}.{}.{}, impl {})", .{
                name,
                description,
                (layer.specVersion >> 22) & 0x3FF,
                (layer.specVersion >> 12) & 0x3FF,
                layer.specVersion & 0xFFF,
                layer.implementationVersion,
            });
            if (!has_validation and std.mem.eql(u8, name, validation_layer_name)) {
                has_validation = true;
            }
        }

        return has_validation;
    }

    fn logInstanceExtensions(self: *GhostVK) !void {
        var extension_count: u32 = 0;
        var result = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, null);
        if (result != .SUCCESS and result != .INCOMPLETE) {
            log.warn("Failed to enumerate instance extensions: {}", .{result});
            return;
        }
        if (extension_count == 0) {
            log.warn("No Vulkan instance extensions reported", .{});
            return;
        }

        const extensions = try self.allocator.alloc(vk.types.VkExtensionProperties, extension_count);
        defer self.allocator.free(extensions);

        result = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, extensions.ptr);
        if (result != .SUCCESS and result != .INCOMPLETE) {
            log.warn("Failed to get instance extensions: {}", .{result});
            return;
        }

        log.info("Instance extensions ({}):", .{extension_count});
        for (extensions[0..extension_count]) |ext| {
            const name = std.mem.sliceTo(&ext.extensionName, 0);
            log.info("  {s} (spec {}.{}.{})", .{
                name,
                (ext.specVersion >> 22) & 0x3FF,
                (ext.specVersion >> 12) & 0x3FF,
                ext.specVersion & 0xFFF,
            });
        }
    }

    fn hasInstanceExtension(self: *GhostVK, name: [:0]const u8) !bool {
        var extension_count: u32 = 0;
        var result = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, null);
        if (result != .SUCCESS and result != .INCOMPLETE) return false;
        if (extension_count == 0) return false;

        const extensions = try self.allocator.alloc(vk.types.VkExtensionProperties, extension_count);
        defer self.allocator.free(extensions);

        result = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, extensions.ptr);
        if (result != .SUCCESS and result != .INCOMPLETE) return false;

        for (extensions[0..extension_count]) |ext| {
            const ext_name = std.mem.sliceTo(&ext.extensionName, 0);
            if (std.mem.eql(u8, ext_name, name)) {
                return true;
            }
        }
        return false;
    }

    fn findQueueFamilies(self: *GhostVK, device: vk.types.VkPhysicalDevice) Error!QueueFamilies {
        const dispatch = self.instance_dispatch orelse return Error.VulkanInstanceCreationFailed;

        var family_count: u32 = 0;
        dispatch.get_physical_device_queue_family_properties(device, &family_count, null);
        if (family_count == 0) {
            return Error.QueueFamiliesIncomplete;
        }

        const families = try self.allocator.alloc(vk.types.VkQueueFamilyProperties, family_count);
        defer self.allocator.free(families);

        dispatch.get_physical_device_queue_family_properties(device, &family_count, families.ptr);

        var result = QueueFamilies{};
        for (families[0..family_count], 0..) |family, idx_usize| {
            const idx: u32 = @intCast(idx_usize);
            if (family.queueCount == 0) continue;

            const GRAPHICS_BIT: u32 = 0x00000001;
            const COMPUTE_BIT: u32 = 0x00000002;
            const TRANSFER_BIT: u32 = 0x00000004;

            if ((family.queueFlags & GRAPHICS_BIT) != 0) {
                if (result.graphics == null) {
                    result.graphics = idx;
                }
            }

            if ((family.queueFlags & COMPUTE_BIT) != 0) {
                if (result.compute == null or (family.queueFlags & GRAPHICS_BIT) == 0) {
                    result.compute = idx;
                }
            }

            if ((family.queueFlags & TRANSFER_BIT) != 0) {
                if (result.transfer == null or ((family.queueFlags & GRAPHICS_BIT) == 0 and (family.queueFlags & COMPUTE_BIT) == 0)) {
                    result.transfer = idx;
                }
            }
        }

        if (result.graphics == null) {
            return Error.QueueFamiliesIncomplete;
        }

        if (result.compute == null) result.compute = result.graphics;
        if (result.transfer == null) result.transfer = result.graphics;

        return result;
    }

    fn printDeviceProperties(self: *GhostVK, properties: vk.types.VkPhysicalDeviceProperties, device: vk.types.VkPhysicalDevice) void {
        const dispatch = self.instance_dispatch orelse return;
        const name = std.mem.sliceTo(&properties.deviceName, 0);
        log.info("Using physical device: {s}", .{name});
        log.info("  Type: {s}", .{deviceTypeName(properties.deviceType)});
        log.info("  API Version: {}.{}.{}, Driver: {}.{}.{}, Vendor: 0x{x}, Device: 0x{x}", .{
            (properties.apiVersion >> 22) & 0x3FF,
            (properties.apiVersion >> 12) & 0x3FF,
            properties.apiVersion & 0xFFF,
            (properties.driverVersion >> 22) & 0x3FF,
            (properties.driverVersion >> 12) & 0x3FF,
            properties.driverVersion & 0xFFF,
            properties.vendorID,
            properties.deviceID,
        });

        var features: vk.types.VkPhysicalDeviceFeatures = undefined;
        dispatch.get_physical_device_features(device, &features);
        log.info("  Features: geometryShader={}, samplerAnisotropy={}, shaderInt64={}", .{
            features.geometryShader != 0,
            features.samplerAnisotropy != 0,
            features.shaderInt64 != 0,
        });

        log.info("  Limits: maxImageDimension2D={}, maxUniformBufferRange={}, maxPushConstants={} bytes", .{
            properties.limits.maxImageDimension2D,
            properties.limits.maxUniformBufferRange,
            properties.limits.maxPushConstantsSize,
        });
    }

    fn debugMessengerCreateInfo(self: *GhostVK) vk.types.VkDebugUtilsMessengerCreateInfoEXT {
        _ = self;
        const VERBOSE: u32 = 0x00000001;
        const WARNING: u32 = 0x00000100;
        const ERROR: u32 = 0x00001000;
        const GENERAL: u32 = 0x00000001;
        const VALIDATION: u32 = 0x00000002;
        const PERFORMANCE: u32 = 0x00000004;

        return vk.types.VkDebugUtilsMessengerCreateInfoEXT{
            .sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = null,
            .flags = 0,
            .messageSeverity = VERBOSE | WARNING | ERROR,
            .messageType = GENERAL | VALIDATION | PERFORMANCE,
            .pfnUserCallback = debugCallback,
            .pUserData = null,
        };
    }

    fn initWayland(self: *GhostVK) Error!void {
        // Connect to Wayland display
        const display = wl.wl_display_connect(null) orelse {
            log.err("Failed to connect to Wayland display", .{});
            return Error.WaylandConnectionFailed;
        };
        self.wl_display = display;

        // Get registry
        const registry = wl.wl_display_get_registry(display) orelse return Error.WaylandConnectionFailed;

        // Set up registry listener
        const Registry = struct {
            compositor: ?*wl.wl_compositor = null,

            fn registryGlobal(
                data: ?*anyopaque,
                reg: ?*wl.wl_registry,
                name: u32,
                interface: [*c]const u8,
                version: u32,
            ) callconv(.c) void {
                const ctx: *@This() = @ptrCast(@alignCast(data));
                const iface_str = std.mem.span(interface);

                log.debug("Wayland global: {s} (name={}, version={})", .{ iface_str, name, version });

                if (std.mem.eql(u8, iface_str, "wl_compositor")) {
                    log.debug("Binding wl_compositor...", .{});
                    const comp_proxy = wl.wl_registry_bind(reg, name, @ptrCast(wl.wl_compositor_interface()), version);
                    ctx.compositor = @ptrCast(comp_proxy);
                    log.debug("Compositor bound: {any}", .{comp_proxy});
                }
            }

            fn registryGlobalRemove(_: ?*anyopaque, _: ?*wl.wl_registry, _: u32) callconv(.c) void {}
        };

        var registry_data = Registry{};
        const listener = wl.wl_registry_listener{
            .global = Registry.registryGlobal,
            .global_remove = Registry.registryGlobalRemove,
        };

        const add_result = wl.wl_registry_add_listener(registry, &listener, &registry_data);
        if (add_result < 0) {
            log.err("Failed to add registry listener: {}", .{add_result});
            return Error.WaylandConnectionFailed;
        }

        log.debug("Registry listener added, dispatching...", .{});
        const dispatch_result = wl.wl_display_dispatch(display);
        log.debug("First dispatch result: {}", .{dispatch_result});

        const roundtrip_result = wl.wl_display_roundtrip(display);
        log.debug("Roundtrip result: {}", .{roundtrip_result});

        if (registry_data.compositor) |comp| {
            self.wl_compositor = comp;
        } else {
            log.err("Wayland compositor not found", .{});
            return Error.WaylandCompositorNotFound;
        }

        // Create Wayland surface
        const surface = wl.wl_compositor_create_surface(self.wl_compositor.?) orelse {
            log.err("Failed to create Wayland surface", .{});
            return Error.WaylandSurfaceCreationFailed;
        };
        self.wl_surface = surface;

        log.info("Wayland initialized successfully", .{});
    }

    fn cleanupWayland(self: *GhostVK) void {
        // Destroy Wayland resources in reverse order of creation
        if (self.wl_surface) |surface| {
            wl.wl_surface_destroy(surface);
            self.wl_surface = null;
        }

        if (self.wl_compositor) |compositor| {
            wl.wl_compositor_destroy(compositor);
            self.wl_compositor = null;
        }

        if (self.wl_display) |display| {
            // Flush and roundtrip to ensure all pending operations complete
            // This helps avoid crashes in overlay layers (MangoHUD, etc.)
            _ = wl.wl_display_flush(display);
            _ = wl.wl_display_roundtrip(display);
            wl.wl_display_disconnect(display);
            self.wl_display = null;
        }
    }

    fn createVulkanSurface(self: *GhostVK) Error!void {
        const instance = self.instance orelse return Error.VulkanInstanceCreationFailed;
        const get_proc = self.loader.getInstanceProcAddr() orelse return Error.VulkanSurfaceCreationFailed;

        // Get vkCreateWaylandSurfaceKHR function
        const create_raw = get_proc(instance, "vkCreateWaylandSurfaceKHR");
        if (create_raw == null) {
            log.err("vkCreateWaylandSurfaceKHR not found", .{});
            return Error.VulkanSurfaceCreationFailed;
        }

        const PFN_vkCreateWaylandSurfaceKHR = *const fn (
            instance: vk.types.VkInstance,
            pCreateInfo: *const anyopaque,
            pAllocator: ?*const vk.types.VkAllocationCallbacks,
            pSurface: *vk.types.VkSurfaceKHR,
        ) callconv(.c) vk.types.VkResult;

        const create_fn: PFN_vkCreateWaylandSurfaceKHR = @ptrCast(create_raw.?);

        // Create surface info struct
        const VkWaylandSurfaceCreateInfoKHR = extern struct {
            sType: vk.types.VkStructureType,
            pNext: ?*const anyopaque,
            flags: u32,
            display: *wl.wl_display,
            surface: *wl.wl_surface,
        };

        const create_info = VkWaylandSurfaceCreateInfoKHR{
            .sType = @enumFromInt(1000006000), // VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR
            .pNext = null,
            .flags = 0,
            .display = self.wl_display.?,
            .surface = self.wl_surface.?,
        };

        var surface: vk.types.VkSurfaceKHR = undefined;
        const result = create_fn(instance, @ptrCast(&create_info), null, &surface);
        if (result != .SUCCESS) {
            log.err("vkCreateWaylandSurfaceKHR failed with {s}", .{@tagName(result)});
            return Error.VulkanSurfaceCreationFailed;
        }

        self.surface = surface;
        log.info("Vulkan surface created", .{});
    }

    fn cleanupVulkanSurface(self: *GhostVK) void {
        if (self.surface) |surface| {
            if (self.instance_dispatch) |dispatch| {
                if (dispatch.destroy_surface) |destroy_fn| {
                    destroy_fn(self.instance.?, surface, null);
                }
            }
            self.surface = null;
        }
    }

    fn createSwapchain(self: *GhostVK) Error!void {
        const surface = self.surface orelse return Error.VulkanSurfaceCreationFailed;
        const physical_device = self.physical_device orelse return Error.NoVulkanDevicesFound;
        const device = self.device orelse return Error.VulkanDeviceCreationFailed;
        const instance_dispatch = self.instance_dispatch orelse return Error.VulkanInstanceCreationFailed;
        const device_dispatch = self.device_dispatch orelse return Error.VulkanDeviceCreationFailed;

        // Query surface capabilities
        const get_caps_fn = instance_dispatch.get_physical_device_surface_capabilities orelse return Error.VulkanCallFailed;
        var surface_caps: vk.types.VkSurfaceCapabilitiesKHR = undefined;
        var result = get_caps_fn(physical_device, surface, &surface_caps);
        if (result != .SUCCESS) {
            log.err("Failed to get surface capabilities: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        log.info("Surface capabilities:", .{});
        log.info("  Image count: min={}, max={}", .{ surface_caps.minImageCount, surface_caps.maxImageCount });
        log.info("  Current extent: {}x{}", .{ surface_caps.currentExtent.width, surface_caps.currentExtent.height });
        log.info("  Min extent: {}x{}", .{ surface_caps.minImageExtent.width, surface_caps.minImageExtent.height });
        log.info("  Max extent: {}x{}", .{ surface_caps.maxImageExtent.width, surface_caps.maxImageExtent.height });

        // Query surface formats
        const get_formats_fn = instance_dispatch.get_physical_device_surface_formats orelse return Error.VulkanCallFailed;
        var format_count: u32 = 0;
        result = get_formats_fn(physical_device, surface, &format_count, null);
        if (result != .SUCCESS or format_count == 0) {
            log.err("No surface formats available", .{});
            return Error.VulkanCallFailed;
        }

        const formats = try self.allocator.alloc(vk.types.VkSurfaceFormatKHR, format_count);
        defer self.allocator.free(formats);
        result = get_formats_fn(physical_device, surface, &format_count, formats.ptr);
        if (result != .SUCCESS) {
            log.err("Failed to get surface formats: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        log.info("Surface formats ({}): ", .{format_count});
        for (formats) |fmt| {
            log.info("  format={} colorspace={}", .{ @intFromEnum(fmt.format), @intFromEnum(fmt.colorSpace) });
        }

        // Query present modes
        const get_present_modes_fn = instance_dispatch.get_physical_device_surface_present_modes orelse return Error.VulkanCallFailed;
        var present_mode_count: u32 = 0;
        result = get_present_modes_fn(physical_device, surface, &present_mode_count, null);
        if (result != .SUCCESS or present_mode_count == 0) {
            log.err("No present modes available", .{});
            return Error.VulkanCallFailed;
        }

        const present_modes = try self.allocator.alloc(vk.types.VkPresentModeKHR, present_mode_count);
        defer self.allocator.free(present_modes);
        result = get_present_modes_fn(physical_device, surface, &present_mode_count, present_modes.ptr);
        if (result != .SUCCESS) {
            log.err("Failed to get present modes: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        log.info("Present modes ({}):", .{present_mode_count});
        for (present_modes) |mode| {
            log.info("  mode={}", .{@intFromEnum(mode)});
        }

        // Choose format: prefer HDR if requested, otherwise BGRA8_SRGB with SRGB_NONLINEAR
        var chosen_format = formats[0];
        var chosen_hdr_colorspace: hdr.HdrColorSpace = .srgb;

        if (self.options.prefer_hdr) {
            // HDR format selection priority based on preferred colorspace
            const preferred_vk_colorspace = self.options.preferred_hdr_colorspace.toVulkanColorSpace();

            // HDR format values (from Vulkan spec)
            const VK_FORMAT_A2B10G10R10_UNORM_PACK32: vk.types.VkFormat = @enumFromInt(64);
            const VK_FORMAT_A2R10G10B10_UNORM_PACK32: vk.types.VkFormat = @enumFromInt(58);
            const VK_FORMAT_R16G16B16A16_SFLOAT: vk.types.VkFormat = @enumFromInt(97);

            // First pass: try to find the exact preferred HDR format
            for (formats) |fmt| {
                if (fmt.colorSpace == preferred_vk_colorspace) {
                    // Check for compatible formats for this color space
                    const is_hdr10_format = (fmt.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 or
                        fmt.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
                    const is_scrgb_format = (fmt.format == VK_FORMAT_R16G16B16A16_SFLOAT);

                    const preferred_is_hdr10 = (self.options.preferred_hdr_colorspace == .hdr10_pq or
                        self.options.preferred_hdr_colorspace == .hdr10_hlg);

                    if ((preferred_is_hdr10 and is_hdr10_format) or
                        (self.options.preferred_hdr_colorspace == .scrgb_linear and is_scrgb_format))
                    {
                        chosen_format = fmt;
                        chosen_hdr_colorspace = self.options.preferred_hdr_colorspace;
                        log.info("Found preferred HDR format: {s}", .{self.options.preferred_hdr_colorspace.name()});
                        break;
                    }
                }
            }

            // Second pass: if preferred not found, try any HDR format
            if (chosen_hdr_colorspace == .srgb) {
                // Try HDR10 PQ first (most common)
                const hdr10_pq_cs: vk.types.VkColorSpaceKHR = @enumFromInt(1000104000);
                for (formats) |fmt| {
                    if (fmt.colorSpace == hdr10_pq_cs and
                        (fmt.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 or fmt.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32))
                    {
                        chosen_format = fmt;
                        chosen_hdr_colorspace = .hdr10_pq;
                        log.info("Found HDR10 PQ fallback format", .{});
                        break;
                    }
                }
            }

            // Try scRGB if HDR10 not available
            if (chosen_hdr_colorspace == .srgb) {
                const scrgb_cs: vk.types.VkColorSpaceKHR = @enumFromInt(1000104002);
                for (formats) |fmt| {
                    if (fmt.colorSpace == scrgb_cs and fmt.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
                        chosen_format = fmt;
                        chosen_hdr_colorspace = .scrgb_linear;
                        log.info("Found scRGB Linear fallback format", .{});
                        break;
                    }
                }
            }
        }

        // SDR fallback if HDR not found or not requested
        if (chosen_hdr_colorspace == .srgb) {
            for (formats) |fmt| {
                if (fmt.format == .B8G8R8A8_SRGB and fmt.colorSpace == .SRGB_NONLINEAR) {
                    chosen_format = fmt;
                    break;
                }
            }
        }

        self.swapchain_colorspace = chosen_hdr_colorspace;
        log.info("Chosen format: {s} with colorspace {s} ({s})", .{
            @tagName(chosen_format.format),
            @tagName(chosen_format.colorSpace),
            chosen_hdr_colorspace.name(),
        });

        // Choose present mode: prefer MAILBOX (triple buffering), fallback FIFO
        var chosen_present_mode: vk.types.VkPresentModeKHR = .FIFO;
        for (present_modes) |mode| {
            if (mode == .MAILBOX) {
                chosen_present_mode = mode;
                break;
            }
        }
        log.info("Chosen present mode: {s}", .{@tagName(chosen_present_mode)});

        // Choose image count: prefer triple buffering
        var image_count = surface_caps.minImageCount + 1;
        if (surface_caps.maxImageCount > 0 and image_count > surface_caps.maxImageCount) {
            image_count = surface_caps.maxImageCount;
        }
        log.info("Swapchain image count: {}", .{image_count});

        // Choose extent
        const extent = if (surface_caps.currentExtent.width != 0xFFFFFFFF)
            surface_caps.currentExtent
        else
            vk.types.VkExtent2D{
                .width = @max(surface_caps.minImageExtent.width, @min(surface_caps.maxImageExtent.width, 2560)),
                .height = @max(surface_caps.minImageExtent.height, @min(surface_caps.maxImageExtent.height, 1440)),
            };
        log.info("Swapchain extent: {}x{}", .{ extent.width, extent.height });

        // Create swapchain
        const create_fn = device_dispatch.create_swapchain orelse return Error.VulkanCallFailed;
        const swapchain_create_info = vk.types.VkSwapchainCreateInfoKHR{
            .sType = .SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = null,
            .flags = 0,
            .surface = surface,
            .minImageCount = image_count,
            .imageFormat = chosen_format.format,
            .imageColorSpace = chosen_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = vk.types.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.types.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .imageSharingMode = .EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .preTransform = surface_caps.currentTransform,
            .compositeAlpha = .OPAQUE,
            .presentMode = chosen_present_mode,
            .clipped = 1,
            .oldSwapchain = null,
        };

        var swapchain: vk.types.VkSwapchainKHR = undefined;
        result = create_fn(device, &swapchain_create_info, null, &swapchain);
        if (result != .SUCCESS) {
            log.err("Failed to create swapchain: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }
        self.swapchain = swapchain;
        self.swapchain_format = chosen_format.format;
        self.swapchain_extent = extent;

        // Get swapchain images
        const get_images_fn = device_dispatch.get_swapchain_images orelse return Error.VulkanCallFailed;
        var actual_image_count: u32 = 0;
        result = get_images_fn(device, swapchain, &actual_image_count, null);
        if (result != .SUCCESS) {
            log.err("Failed to query swapchain image count: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        const images = try self.allocator.alloc(vk.types.VkImage, actual_image_count);
        result = get_images_fn(device, swapchain, &actual_image_count, images.ptr);
        if (result != .SUCCESS) {
            self.allocator.free(images);
            log.err("Failed to get swapchain images: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }
        self.swapchain_images = images;

        // Create image views
        const image_views = try self.allocator.alloc(vk.types.VkImageView, actual_image_count);
        errdefer self.allocator.free(image_views);

        for (images, 0..) |image, i| {
            const view_create_info = vk.types.VkImageViewCreateInfo{
                .sType = .IMAGE_VIEW_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .image = image,
                .viewType = .@"2D",
                .format = chosen_format.format,
                .components = .{
                    .r = .IDENTITY,
                    .g = .IDENTITY,
                    .b = .IDENTITY,
                    .a = .IDENTITY,
                },
                .subresourceRange = .{
                    .aspectMask = vk.types.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

            var image_view: vk.types.VkImageView = undefined;
            result = device_dispatch.create_image_view(device, &view_create_info, null, &image_view);
            if (result != .SUCCESS) {
                // Clean up any views we've already created
                for (image_views[0..i]) |view| {
                    device_dispatch.destroy_image_view(device, view, null);
                }
                self.allocator.free(image_views);
                log.err("Failed to create image view {}: {s}", .{ i, @tagName(result) });
                return Error.VulkanCallFailed;
            }
            image_views[i] = image_view;
        }
        self.swapchain_image_views = image_views;

        log.info("Swapchain created: {} images, {}x{}, format={s}", .{
            actual_image_count,
            extent.width,
            extent.height,
            @tagName(chosen_format.format),
        });
    }

    fn cleanupSwapchain(self: *GhostVK) void {
        if (self.device_dispatch) |dispatch| {
            const device = self.device orelse return;

            // Destroy image views
            for (self.swapchain_image_views) |view| {
                dispatch.destroy_image_view(device, view, null);
            }
            if (self.swapchain_image_views.len > 0) {
                self.allocator.free(self.swapchain_image_views);
                self.swapchain_image_views = &[_]vk.types.VkImageView{};
            }

            // Free swapchain images (not destroyed, just the array)
            if (self.swapchain_images.len > 0) {
                self.allocator.free(self.swapchain_images);
                self.swapchain_images = &[_]vk.types.VkImage{};
            }

            // Destroy swapchain
            if (self.swapchain) |swapchain| {
                if (dispatch.destroy_swapchain) |destroy_fn| {
                    destroy_fn(device, swapchain, null);
                }
                self.swapchain = null;
            }
        }
    }

    /// Recreate swapchain after out-of-date or window resize
    /// This waits for device idle, cleans up old resources, and creates new swapchain
    pub fn recreateSwapchain(self: *GhostVK) Error!void {
        _ = self.device orelse return Error.VulkanDeviceCreationFailed;
        const dispatch = self.device_dispatch orelse return Error.VulkanDeviceCreationFailed;

        log.info("Recreating swapchain...", .{});

        // Wait for graphics queue to finish all operations
        const queue = self.graphics_queue orelse return Error.VulkanDeviceCreationFailed;
        const result = dispatch.queue_wait_idle(queue);
        if (result != .SUCCESS) {
            log.err("Failed to wait for queue idle: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Clean up old swapchain resources
        self.cleanupSwapchain();

        // Recreate swapchain with potentially new dimensions
        try self.createSwapchain();

        // Recreate image-in-flight fences array for new swapchain image count
        if (self.image_in_flight_fences.len > 0) {
            self.allocator.free(self.image_in_flight_fences);
        }
        const image_count = self.swapchain_images.len;
        self.image_in_flight_fences = try self.allocator.alloc(?vk.types.VkFence, image_count);
        for (self.image_in_flight_fences) |*f| {
            f.* = null;
        }

        // Recreate render pipeline framebuffers
        if (self.render_pipeline) |*rp| {
            rp.recreateFramebuffers(self.swapchain_extent, self.swapchain_image_views) catch |err| {
                log.err("Failed to recreate framebuffers: {}", .{err});
                return Error.VulkanCallFailed;
            };
        }

        log.info("Swapchain recreated: {}x{}", .{ self.swapchain_extent.width, self.swapchain_extent.height });
    }

    // ========================================================================
    // Phase 2: Command Pool, Command Buffers, and Synchronization
    // ========================================================================

    fn createCommandPool(self: *GhostVK) Error!void {
        const device = self.device orelse return Error.VulkanDeviceCreationFailed;
        const dispatch = self.device_dispatch orelse return Error.VulkanDeviceCreationFailed;

        const pool_info = vk.types.VkCommandPoolCreateInfo{
            .sType = .COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.types.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = self.graphics_queue_family,
        };

        var pool: vk.types.VkCommandPool = undefined;
        const result = dispatch.create_command_pool(device, &pool_info, null, &pool);
        if (result != .SUCCESS) {
            log.err("Failed to create command pool: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        self.command_pool = pool;
        log.info("Command pool created", .{});
    }

    fn cleanupCommandPool(self: *GhostVK) void {
        if (self.command_pool) |pool| {
            if (self.device_dispatch) |dispatch| {
                if (self.device) |device| {
                    dispatch.destroy_command_pool(device, pool, null);
                }
            }
            self.command_pool = null;
        }
    }

    fn createCommandBuffers(self: *GhostVK) Error!void {
        const device = self.device orelse return Error.VulkanDeviceCreationFailed;
        const dispatch = self.device_dispatch orelse return Error.VulkanDeviceCreationFailed;
        const pool = self.command_pool orelse return Error.VulkanCallFailed;

        const command_buffers = try self.allocator.alloc(vk.types.VkCommandBuffer, self.frames_in_flight);
        errdefer self.allocator.free(command_buffers);

        const alloc_info = vk.types.VkCommandBufferAllocateInfo{
            .sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = pool,
            .level = .PRIMARY,
            .commandBufferCount = self.frames_in_flight,
        };

        const result = dispatch.allocate_command_buffers(device, &alloc_info, @ptrCast(command_buffers.ptr));
        if (result != .SUCCESS) {
            self.allocator.free(command_buffers);
            log.err("Failed to allocate command buffers: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        self.command_buffers = command_buffers;
        log.info("Created {} command buffers", .{self.frames_in_flight});
    }

    fn cleanupCommandBuffers(self: *GhostVK) void {
        if (self.command_buffers.len > 0) {
            if (self.device_dispatch) |dispatch| {
                if (self.device) |device| {
                    if (self.command_pool) |pool| {
                        dispatch.free_command_buffers(device, pool, @intCast(self.command_buffers.len), @ptrCast(self.command_buffers.ptr));
                    }
                }
            }
            self.allocator.free(self.command_buffers);
            self.command_buffers = &[_]vk.types.VkCommandBuffer{};
        }
    }

    fn createSyncPrimitives(self: *GhostVK) Error!void {
        const device = self.device orelse return Error.VulkanDeviceCreationFailed;
        const dispatch = self.device_dispatch orelse return Error.VulkanDeviceCreationFailed;

        // Large semaphore pool for present operations
        // Need enough to never reuse before present consumes them
        // At high frame rates (7000+ FPS), present can lag significantly
        // NOTE: Binary semaphores have fundamental limitations at extreme FPS
        //       Phase 4 will introduce timeline semaphores for proper solution
        const swapchain_image_count = self.swapchain_images.len;
        const present_pool_size = 512; // Very large pool for extreme frame rates (7000+ FPS)
        const render_finished = try self.allocator.alloc(vk.types.VkSemaphore, present_pool_size);
        errdefer self.allocator.free(render_finished);

        // Per-frame fences
        const fences = try self.allocator.alloc(vk.types.VkFence, self.frames_in_flight);
        errdefer self.allocator.free(fences);

        const semaphore_info = vk.types.VkSemaphoreCreateInfo{
            .sType = .SEMAPHORE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };

        const fence_info = vk.types.VkFenceCreateInfo{
            .sType = .FENCE_CREATE_INFO,
            .pNext = null,
            .flags = vk.types.VK_FENCE_CREATE_SIGNALED_BIT, // Start signaled so first frame doesn't wait
        };

        // Create pool of render_finished semaphores for present
        var i: u32 = 0;
        while (i < present_pool_size) : (i += 1) {
            const result = dispatch.create_semaphore(device, &semaphore_info, null, &render_finished[i]);
            if (result != .SUCCESS) {
                log.err("Failed to create render_finished semaphore {}: {s}", .{ i, @tagName(result) });
                return Error.VulkanCallFailed;
            }
        }

        // Create per-frame fences
        i = 0;
        while (i < self.frames_in_flight) : (i += 1) {
            const result = dispatch.create_fence(device, &fence_info, null, &fences[i]);
            if (result != .SUCCESS) {
                log.err("Failed to create fence {}: {s}", .{ i, @tagName(result) });
                return Error.VulkanCallFailed;
            }
        }

        // Create acquire fence (start signaled so first wait succeeds)
        const acquire_fence_info = vk.types.VkFenceCreateInfo{
            .sType = .FENCE_CREATE_INFO,
            .pNext = null,
            .flags = vk.types.VK_FENCE_CREATE_SIGNALED_BIT,
        };
        var acquire_fence: vk.types.VkFence = undefined;
        const acquire_result = dispatch.create_fence(device, &acquire_fence_info, null, &acquire_fence);
        if (acquire_result != .SUCCESS) {
            log.err("Failed to create acquire fence: {s}", .{@tagName(acquire_result)});
            return Error.VulkanCallFailed;
        }

        // Create large semaphore pool for acquire operations
        // Need enough to never reuse before consumed
        const acquire_pool_size = 512; // Match present pool size
        const acquire_sems = try self.allocator.alloc(vk.types.VkSemaphore, acquire_pool_size);
        errdefer self.allocator.free(acquire_sems);

        i = 0;
        while (i < acquire_pool_size) : (i += 1) {
            const result = dispatch.create_semaphore(device, &semaphore_info, null, &acquire_sems[i]);
            if (result != .SUCCESS) {
                log.err("Failed to create acquire semaphore {}: {s}", .{ i, @tagName(result) });
                return Error.VulkanCallFailed;
            }
        }

        // Create image-in-flight tracking (initially null - no image in use)
        const image_fences = try self.allocator.alloc(?vk.types.VkFence, swapchain_image_count);
        errdefer self.allocator.free(image_fences);
        for (image_fences) |*img_fence| {
            img_fence.* = null;
        }

        self.render_finished_semaphores = render_finished;
        self.in_flight_fences = fences;
        self.image_in_flight_fences = image_fences;
        self.acquire_fence = acquire_fence;
        self.acquire_semaphores = acquire_sems;

        log.info("Created synchronization primitives: {} present sems (pool), {} per-frame fences, {} acquire sems (pool)", .{ present_pool_size, self.frames_in_flight, acquire_pool_size });
    }

    fn cleanupSyncPrimitives(self: *GhostVK) void {
        if (self.device_dispatch) |dispatch| {
            if (self.device) |device| {
                for (self.render_finished_semaphores) |sem| {
                    dispatch.destroy_semaphore(device, sem, null);
                }
                for (self.in_flight_fences) |fence| {
                    dispatch.destroy_fence(device, fence, null);
                }
                for (self.acquire_semaphores) |sem| {
                    dispatch.destroy_semaphore(device, sem, null);
                }
                if (self.acquire_fence) |fence| {
                    dispatch.destroy_fence(device, fence, null);
                }
            }
        }

        if (self.render_finished_semaphores.len > 0) {
            self.allocator.free(self.render_finished_semaphores);
            self.render_finished_semaphores = &[_]vk.types.VkSemaphore{};
        }
        if (self.in_flight_fences.len > 0) {
            self.allocator.free(self.in_flight_fences);
            self.in_flight_fences = &[_]vk.types.VkFence{};
        }
        if (self.acquire_semaphores.len > 0) {
            self.allocator.free(self.acquire_semaphores);
            self.acquire_semaphores = &[_]vk.types.VkSemaphore{};
        }
        if (self.image_in_flight_fences.len > 0) {
            self.allocator.free(self.image_in_flight_fences);
            self.image_in_flight_fences = &[_]?vk.types.VkFence{};
        }
    }

    // ========================================================================
    // Phase 2: Render Loop
    // ========================================================================

    pub fn drawFrame(self: *GhostVK) Error!bool {
        const device = self.device orelse return Error.VulkanDeviceCreationFailed;
        const dispatch = self.device_dispatch orelse return Error.VulkanDeviceCreationFailed;
        const swapchain = self.swapchain orelse return Error.VulkanCallFailed;

        // Begin frame timing
        if (self.pacer) |*p| {
            p.beginFrame();
        }

        // Wait for the previous frame to finish
        const fence = self.in_flight_fences[self.current_frame];
        var result = dispatch.wait_for_fences(device, 1, &fence, 1, std.math.maxInt(u64));
        if (result != .SUCCESS) {
            log.err("Failed to wait for fence: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Acquire next image using acquire fence
        const acquire_fn = dispatch.acquire_next_image orelse return Error.VulkanCallFailed;
        var image_index: u32 = undefined;

        // Wait and reset acquire fence before use
        const acq_fence = self.acquire_fence.?;
        result = dispatch.wait_for_fences(device, 1, &acq_fence, 1, std.math.maxInt(u64));
        if (result != .SUCCESS) {
            log.err("Failed to wait for acquire fence: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        result = dispatch.reset_fences(device, 1, &acq_fence);
        if (result != .SUCCESS) {
            log.err("Failed to reset acquire fence: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Acquire image with semaphore
        // Use semaphore from pool (cycles through large pool to prevent reuse)
        const acquire_sem = self.acquire_semaphores[self.acquire_semaphore_index];
        self.acquire_semaphore_index = (self.acquire_semaphore_index + 1) % @as(u32, @intCast(self.acquire_semaphores.len));

        result = acquire_fn(
            device,
            swapchain,
            std.math.maxInt(u64),
            acquire_sem,
            acq_fence, // Still provide fence for added sync
            &image_index,
        );

        // Handle swapchain out of date
        if (result == .ERROR_OUT_OF_DATE_KHR) {
            log.warn("Swapchain out of date, needs recreation", .{});
            return false;
        } else if (result != .SUCCESS and result != .SUBOPTIMAL_KHR) {
            log.err("Failed to acquire next image: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Wait for this image to finish being presented (if it was in use)
        if (self.image_in_flight_fences[image_index]) |image_fence| {
            const wait_result = dispatch.wait_for_fences(device, 1, &image_fence, 1, std.math.maxInt(u64));
            if (wait_result != .SUCCESS and wait_result != .TIMEOUT) {
                log.warn("Image fence wait returned: {}", .{wait_result});
            }
        }

        // Mark this image as now being used by current frame
        self.image_in_flight_fences[image_index] = fence;

        // Track the last acquired image index for external integrations
        self.last_presented_image = image_index;

        // Reset frame fence before submitting
        result = dispatch.reset_fences(device, 1, &fence);
        if (result != .SUCCESS) {
            log.err("Failed to reset fence: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Record command buffer
        try self.recordCommandBuffer(self.command_buffers[self.current_frame], image_index);

        // Submit: wait on acquire semaphore, signal present semaphore
        const present_sem = self.render_finished_semaphores[self.present_semaphore_index];
        self.present_semaphore_index = (self.present_semaphore_index + 1) % @as(u32, @intCast(self.render_finished_semaphores.len));

        const wait_semaphores = [_]vk.types.VkSemaphore{acquire_sem};
        const wait_stages = [_]vk.types.VkPipelineStageFlags{vk.types.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const signal_semaphores = [_]vk.types.VkSemaphore{present_sem};
        const command_buffers = [_]vk.types.VkCommandBuffer{self.command_buffers[self.current_frame]};

        const submit_info = vk.types.VkSubmitInfo{
            .sType = .SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffers,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphores,
        };

        result = dispatch.queue_submit(self.graphics_queue.?, 1, @ptrCast(&submit_info), fence);
        if (result != .SUCCESS) {
            log.err("Failed to submit draw command buffer: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Present
        const queue_present_fn = dispatch.queue_present orelse return Error.VulkanCallFailed;
        const swapchains = [_]vk.types.VkSwapchainKHR{swapchain};
        const image_indices = [_]u32{image_index};

        const present_info = vk.types.VkPresentInfoKHR{
            .sType = .PRESENT_INFO_KHR,
            .pNext = null,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &signal_semaphores,
            .swapchainCount = 1,
            .pSwapchains = &swapchains,
            .pImageIndices = &image_indices,
            .pResults = null,
        };

        result = queue_present_fn(self.graphics_queue.?, &present_info);
        if (result == .ERROR_OUT_OF_DATE_KHR or result == .SUBOPTIMAL_KHR) {
            log.warn("Swapchain suboptimal or out of date after present", .{});
            return false; // Signal caller to recreate swapchain
        } else if (result != .SUCCESS) {
            log.err("Failed to present: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Advance to next frame
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
        self.frame_count += 1;

        // End frame timing and apply pacing
        if (self.pacer) |*p| {
            p.endFrame();
        }

        return true; // Frame rendered successfully
    }

    fn recordCommandBuffer(self: *GhostVK, command_buffer: vk.types.VkCommandBuffer, image_index: u32) Error!void {
        const dispatch = self.device_dispatch orelse return Error.VulkanDeviceCreationFailed;

        // Begin recording (implicitly resets with VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
        const begin_info = vk.types.VkCommandBufferBeginInfo{
            .sType = .COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        };

        var result = dispatch.begin_command_buffer(command_buffer, &begin_info);
        if (result != .SUCCESS) {
            log.err("Failed to begin command buffer: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }

        // Use render pipeline if available, otherwise fallback to clear
        if (self.render_pipeline) |*rp| {
            // Calculate animation time
            const now = std.time.Instant.now() catch self.start_time;
            const elapsed_ns = now.since(self.start_time);
            const time_sec: f32 = @as(f32, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

            const push_constants = render.PushConstants{
                .time = time_sec,
            };

            rp.recordDrawCommands(command_buffer, image_index, push_constants);
        } else {
            // Fallback: clear to purple
            const image = self.swapchain_images[image_index];
            var barrier = vk.types.VkImageMemoryBarrier{
                .sType = .IMAGE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = 0,
                .dstAccessMask = vk.types.VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = .UNDEFINED,
                .newLayout = .TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = vk.types.VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = vk.types.VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange = .{
                    .aspectMask = vk.types.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

            dispatch.cmd_pipeline_barrier(
                command_buffer,
                vk.types.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.types.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                null,
                0,
                null,
                1,
                @ptrCast(&barrier),
            );

            const clear_color = vk.types.VkClearColorValue{
                .float32 = [_]f32{ 0.5, 0.0, 0.5, 1.0 },
            };

            const range = vk.types.VkImageSubresourceRange{
                .aspectMask = vk.types.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };

            dispatch.cmd_clear_color_image(
                command_buffer,
                image,
                .TRANSFER_DST_OPTIMAL,
                &clear_color,
                1,
                &range,
            );

            barrier.srcAccessMask = vk.types.VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = .TRANSFER_DST_OPTIMAL;
            barrier.newLayout = .PRESENT_SRC_KHR;

            dispatch.cmd_pipeline_barrier(
                command_buffer,
                vk.types.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.types.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0,
                null,
                0,
                null,
                1,
                @ptrCast(&barrier),
            );
        }

        // End recording
        result = dispatch.end_command_buffer(command_buffer);
        if (result != .SUCCESS) {
            log.err("Failed to end command buffer: {s}", .{@tagName(result)});
            return Error.VulkanCallFailed;
        }
    }

    /// Check if HDR output is currently active
    pub fn isHdrActive(self: *const GhostVK) bool {
        return self.swapchain_colorspace.isHdr();
    }

    /// Get the current swapchain color space
    pub fn getColorspace(self: *const GhostVK) hdr.HdrColorSpace {
        return self.swapchain_colorspace;
    }

    /// Get the HDR color space name
    pub fn getColorspaceName(self: *const GhostVK) []const u8 {
        return self.swapchain_colorspace.name();
    }

    /// Set target FPS for frame pacing (0 = unlimited)
    pub fn setTargetFps(self: *GhostVK, fps: u32) void {
        if (self.pacer) |*p| {
            p.setTargetFps(fps);
        }
    }

    /// Get frame pacing statistics
    pub fn getFramePacerStats(self: *const GhostVK) ?frame_pacer.FramePacerStats {
        if (self.pacer) |*p| {
            return p.getStats();
        }
        return null;
    }

    /// Check if VRR is active
    pub fn isVrrActive(self: *const GhostVK) bool {
        if (self.pacer) |*p| {
            return p.config.vrr_enabled;
        }
        return false;
    }

    /// Get average FPS from frame pacer
    pub fn getAverageFps(self: *const GhostVK) f64 {
        if (self.pacer) |*p| {
            return p.getAverageFps();
        }
        return 0.0;
    }
};

fn deviceTypeName(device_type: vk.types.VkPhysicalDeviceType) []const u8 {
    return switch (device_type) {
        .OTHER => "Other",
        .INTEGRATED_GPU => "Integrated GPU",
        .DISCRETE_GPU => "Discrete GPU",
        .VIRTUAL_GPU => "Virtual GPU",
        .CPU => "CPU",
    };
}

fn debugCallback(
    message_severity: vk.types.VkDebugUtilsMessageSeverityFlagBitsEXT,
    message_type: vk.types.VkDebugUtilsMessageTypeFlagsEXT,
    callback_data: ?*const vk.types.VkDebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque,
) callconv(.c) vk.types.VkBool32 {
    const message = if (callback_data) |data| blk: {
        if (data.pMessage) |ptr| {
            break :blk std.mem.span(ptr);
        } else {
            break :blk "<no message>";
        }
    } else "<no message>";

    const severity_str = debugMessageSeverityString(message_severity);
    const type_str = debugMessageTypeString(message_type);

    const ERROR: u32 = 0x00001000;
    const WARNING: u32 = 0x00000100;
    const INFO: u32 = 0x00000010;

    const sev_val: u32 = @intFromEnum(message_severity);
    if (sev_val >= ERROR) {
        log.err("[VK][{s}][{s}] {s}", .{ severity_str, type_str, message });
    } else if (sev_val >= WARNING) {
        log.warn("[VK][{s}][{s}] {s}", .{ severity_str, type_str, message });
    } else if (sev_val >= INFO) {
        log.info("[VK][{s}][{s}] {s}", .{ severity_str, type_str, message });
    } else {
        log.debug("[VK][{s}][{s}] {s}", .{ severity_str, type_str, message });
    }

    return 0; // VK_FALSE
}

fn debugMessageTypeString(message_type: vk.types.VkDebugUtilsMessageTypeFlagsEXT) []const u8 {
    const GENERAL: u32 = 0x00000001;
    const VALIDATION: u32 = 0x00000002;
    const PERFORMANCE: u32 = 0x00000004;

    const general = (message_type & GENERAL) != 0;
    const validation = (message_type & VALIDATION) != 0;
    const performance = (message_type & PERFORMANCE) != 0;

    if (general and validation and performance) return "general|validation|performance";
    if (validation and performance) return "validation|performance";
    if (general and performance) return "general|performance";
    if (general and validation) return "general|validation";
    if (general) return "general";
    if (validation) return "validation";
    if (performance) return "performance";
    return "unknown";
}

fn debugMessageSeverityString(severity: vk.types.VkDebugUtilsMessageSeverityFlagBitsEXT) []const u8 {
    const VERBOSE: u32 = 0x00000001;
    const INFO: u32 = 0x00000010;
    const WARNING: u32 = 0x00000100;
    const ERROR: u32 = 0x00001000;

    const sev_val: u32 = @intFromEnum(severity);
    if (sev_val >= ERROR) return "error";
    if (sev_val >= WARNING) return "warning";
    if (sev_val >= INFO) return "info";
    if (sev_val >= VERBOSE) return "verbose";
    return "unknown";
}
