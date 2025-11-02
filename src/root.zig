//! GhostVK core library: Vulkan bootstrap & diagnostics layer.
//! Phase 1: Project Bootstrap & Foundation

pub const std_options = struct {
    pub const log_level = .debug;
};

const std = @import("std");
const vk = @import("vulkan");
const wl = @import("wayland.zig");

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
} || vk.errors.Error;

pub const InitOptions = struct {
    enable_validation: bool = true,
    application_name: [:0]const u8 = "GhostVK",
    application_version: u32 = vk.types.makeApiVersion(0, 1, 0),
    engine_name: [:0]const u8 = "GhostVK",
    engine_version: u32 = vk.types.makeApiVersion(0, 1, 0),
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
    swapchain_extent: vk.types.VkExtent2D = .{ .width = 0, .height = 0 },

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

        log.info("GhostVK initialization complete", .{});
        return self;
    }

    pub fn deinit(self: *GhostVK) void {
        if (self.device) |device| {
            if (self.device_dispatch) |dispatch| {
                _ = dispatch.queue_wait_idle(self.graphics_queue.?);
                self.cleanupSwapchain();
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
        _ = dispatch.enumerate_physical_devices(instance, &device_count, null);
        if (device_count == 0) {
            return Error.NoVulkanDevicesFound;
        }

        const devices = try self.allocator.alloc(vk.types.VkPhysicalDevice, device_count);
        defer self.allocator.free(devices);

        _ = dispatch.enumerate_physical_devices(instance, &device_count, devices.ptr);

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
        _ = self.global_dispatch.enumerate_instance_layer_properties(&layer_count, null);
        if (layer_count == 0) {
            log.warn("No Vulkan instance layers reported", .{});
            return false;
        }

        const layers = try self.allocator.alloc(vk.types.VkLayerProperties, layer_count);
        defer self.allocator.free(layers);

        _ = self.global_dispatch.enumerate_instance_layer_properties(&layer_count, layers.ptr);

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
        _ = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, null);
        if (extension_count == 0) {
            log.warn("No Vulkan instance extensions reported", .{});
            return;
        }

        const extensions = try self.allocator.alloc(vk.types.VkExtensionProperties, extension_count);
        defer self.allocator.free(extensions);

        _ = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, extensions.ptr);

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
        _ = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, null);
        if (extension_count == 0) return false;

        const extensions = try self.allocator.alloc(vk.types.VkExtensionProperties, extension_count);
        defer self.allocator.free(extensions);

        _ = self.global_dispatch.enumerate_instance_extension_properties(null, &extension_count, extensions.ptr);

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
        if (self.wl_display) |display| {
            wl.wl_display_disconnect(display);
            self.wl_display = null;
            self.wl_compositor = null;
            self.wl_surface = null;
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

        // Choose format: prefer BGRA8_SRGB with SRGB_NONLINEAR
        var chosen_format = formats[0];
        for (formats) |fmt| {
            if (fmt.format == .B8G8R8A8_SRGB and fmt.colorSpace == .SRGB_NONLINEAR) {
                chosen_format = fmt;
                break;
            }
        }
        log.info("Chosen format: {s} with colorspace {s}", .{ @tagName(chosen_format.format), @tagName(chosen_format.colorSpace) });

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
            .imageUsage = vk.types.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
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
