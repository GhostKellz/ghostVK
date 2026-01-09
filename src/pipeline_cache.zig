//! GhostVK Pipeline Cache - Fast pipeline creation and caching
//!
//! Vulkan pipeline cache management with:
//! - Persistent cache storage (disk)
//! - Automatic cache loading/saving
//! - Pipeline creation helpers
//! - Shader module management
//! - Compute pipeline support

const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.ghostvk_pipeline);

/// Shader stage info for pipeline creation
pub const ShaderStage = struct {
    stage: vk.types.VkShaderStageFlagBits,
    module: vk.types.VkShaderModule,
    entry_point: [:0]const u8 = "main",
    specialization: ?*const vk.types.VkSpecializationInfo = null,
};

/// Graphics pipeline configuration
pub const GraphicsPipelineConfig = struct {
    // Shaders
    vertex_shader: ?vk.types.VkShaderModule = null,
    fragment_shader: ?vk.types.VkShaderModule = null,
    geometry_shader: ?vk.types.VkShaderModule = null,
    tessellation_control: ?vk.types.VkShaderModule = null,
    tessellation_eval: ?vk.types.VkShaderModule = null,

    // Vertex input
    vertex_bindings: []const vk.types.VkVertexInputBindingDescription = &.{},
    vertex_attributes: []const vk.types.VkVertexInputAttributeDescription = &.{},

    // Input assembly
    topology: vk.types.VkPrimitiveTopology = .TRIANGLE_LIST,
    primitive_restart: bool = false,

    // Rasterization
    polygon_mode: vk.types.VkPolygonMode = .FILL,
    cull_mode: vk.types.VkCullModeFlags = vk.types.VK_CULL_MODE_BACK_BIT,
    front_face: vk.types.VkFrontFace = .COUNTER_CLOCKWISE,
    line_width: f32 = 1.0,

    // Depth/stencil
    depth_test: bool = true,
    depth_write: bool = true,
    depth_compare: vk.types.VkCompareOp = .LESS,
    stencil_test: bool = false,

    // Multisampling
    samples: vk.types.VkSampleCountFlagBits = .@"1",

    // Color blending
    blend_enable: bool = false,
    src_color_blend: vk.types.VkBlendFactor = .SRC_ALPHA,
    dst_color_blend: vk.types.VkBlendFactor = .ONE_MINUS_SRC_ALPHA,
    color_blend_op: vk.types.VkBlendOp = .ADD,
    src_alpha_blend: vk.types.VkBlendFactor = .ONE,
    dst_alpha_blend: vk.types.VkBlendFactor = .ZERO,
    alpha_blend_op: vk.types.VkBlendOp = .ADD,
    color_write_mask: vk.types.VkColorComponentFlags = vk.types.VK_COLOR_COMPONENT_R_BIT |
        vk.types.VK_COLOR_COMPONENT_G_BIT |
        vk.types.VK_COLOR_COMPONENT_B_BIT |
        vk.types.VK_COLOR_COMPONENT_A_BIT,

    // Dynamic state
    dynamic_states: []const vk.types.VkDynamicState = &.{
        .VIEWPORT,
        .SCISSOR,
    },

    // Layout
    layout: vk.types.VkPipelineLayout = undefined,
    render_pass: vk.types.VkRenderPass = undefined,
    subpass: u32 = 0,
};

/// Compute pipeline configuration
pub const ComputePipelineConfig = struct {
    shader: vk.types.VkShaderModule,
    entry_point: [:0]const u8 = "main",
    specialization: ?*const vk.types.VkSpecializationInfo = null,
    layout: vk.types.VkPipelineLayout,
};

/// Pipeline cache with disk persistence
pub const PipelineCache = struct {
    allocator: std.mem.Allocator,
    device: vk.types.VkDevice,
    device_dispatch: *const vk.loader.DeviceDispatch,

    // Vulkan pipeline cache
    vk_cache: vk.types.VkPipelineCache,

    // Cache file path
    cache_path: ?[]const u8,

    // Statistics
    stats: Stats,

    pub const Stats = struct {
        graphics_created: u64 = 0,
        compute_created: u64 = 0,
        cache_hits: u64 = 0,
        cache_misses: u64 = 0,
        shaders_loaded: u64 = 0,
    };

    pub const Error = error{
        CacheCreationFailed,
        PipelineCreationFailed,
        ShaderCreationFailed,
        LayoutCreationFailed,
        FileError,
        OutOfMemory,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        device: vk.types.VkDevice,
        device_dispatch: *const vk.loader.DeviceDispatch,
        cache_path: ?[]const u8,
    ) Error!PipelineCache {
        var initial_data: ?[]const u8 = null;
        var owned_path: ?[]const u8 = null;

        // Try to load existing cache from disk
        if (cache_path) |path| {
            owned_path = allocator.dupe(u8, path) catch null;

            const file = std.fs.cwd().openFile(path, .{}) catch null;
            if (file) |f| {
                defer f.close();
                const stat = f.stat() catch null;
                if (stat) |s| {
                    const data = allocator.alloc(u8, s.size) catch null;
                    if (data) |d| {
                        const read = f.readAll(d) catch 0;
                        if (read == s.size) {
                            initial_data = d;
                            log.info("Loaded pipeline cache: {} bytes from {s}", .{ s.size, path });
                        } else {
                            allocator.free(d);
                        }
                    }
                }
            }
        }
        defer if (initial_data) |d| allocator.free(@constCast(d));

        // Create Vulkan pipeline cache
        const cache_info = vk.types.VkPipelineCacheCreateInfo{
            .sType = .PIPELINE_CACHE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .initialDataSize = if (initial_data) |d| d.len else 0,
            .pInitialData = if (initial_data) |d| d.ptr else null,
        };

        var vk_cache: vk.types.VkPipelineCache = undefined;
        const result = device_dispatch.create_pipeline_cache(device, &cache_info, null, &vk_cache);
        if (result != .SUCCESS) {
            log.err("Failed to create pipeline cache: {s}", .{@tagName(result)});
            if (owned_path) |p| allocator.free(p);
            return Error.CacheCreationFailed;
        }

        log.info("Pipeline cache initialized", .{});

        return PipelineCache{
            .allocator = allocator,
            .device = device,
            .device_dispatch = device_dispatch,
            .vk_cache = vk_cache,
            .cache_path = owned_path,
            .stats = .{},
        };
    }

    pub fn deinit(self: *PipelineCache) void {
        // Save cache to disk before destroying
        self.save() catch |e| {
            log.warn("Failed to save pipeline cache: {}", .{e});
        };

        self.device_dispatch.destroy_pipeline_cache(self.device, self.vk_cache, null);

        if (self.cache_path) |path| {
            self.allocator.free(path);
        }

        log.info("Pipeline cache destroyed: {} graphics, {} compute, {} shaders", .{
            self.stats.graphics_created,
            self.stats.compute_created,
            self.stats.shaders_loaded,
        });
    }

    /// Save cache to disk
    pub fn save(self: *PipelineCache) Error!void {
        const path = self.cache_path orelse return;

        // Get cache data size
        var data_size: usize = 0;
        var result = self.device_dispatch.get_pipeline_cache_data(self.device, self.vk_cache, &data_size, null);
        if (result != .SUCCESS or data_size == 0) {
            return;
        }

        // Get cache data
        const data = self.allocator.alloc(u8, data_size) catch return Error.OutOfMemory;
        defer self.allocator.free(data);

        result = self.device_dispatch.get_pipeline_cache_data(self.device, self.vk_cache, &data_size, data.ptr);
        if (result != .SUCCESS) {
            return;
        }

        // Write to file
        const file = std.fs.cwd().createFile(path, .{}) catch return Error.FileError;
        defer file.close();

        file.writeAll(data) catch return Error.FileError;

        log.info("Saved pipeline cache: {} bytes to {s}", .{ data_size, path });
    }

    /// Create a shader module from SPIR-V bytecode
    pub fn createShaderModule(self: *PipelineCache, spirv: []const u8) Error!vk.types.VkShaderModule {
        if (spirv.len == 0 or spirv.len % 4 != 0) {
            return Error.ShaderCreationFailed;
        }

        const create_info = vk.types.VkShaderModuleCreateInfo{
            .sType = .SHADER_MODULE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .codeSize = spirv.len,
            .pCode = @ptrCast(@alignCast(spirv.ptr)),
        };

        var module: vk.types.VkShaderModule = undefined;
        const result = self.device_dispatch.create_shader_module(self.device, &create_info, null, &module);
        if (result != .SUCCESS) {
            log.err("Failed to create shader module: {s}", .{@tagName(result)});
            return Error.ShaderCreationFailed;
        }

        self.stats.shaders_loaded += 1;
        return module;
    }

    /// Load shader module from file
    pub fn loadShaderModule(self: *PipelineCache, path: []const u8) Error!vk.types.VkShaderModule {
        const file = std.fs.cwd().openFile(path, .{}) catch return Error.FileError;
        defer file.close();

        const stat = file.stat() catch return Error.FileError;
        const spirv = self.allocator.alloc(u8, stat.size) catch return Error.OutOfMemory;
        defer self.allocator.free(spirv);

        const read = file.readAll(spirv) catch return Error.FileError;
        if (read != stat.size) {
            return Error.FileError;
        }

        return self.createShaderModule(spirv);
    }

    /// Destroy a shader module
    pub fn destroyShaderModule(self: *PipelineCache, module: vk.types.VkShaderModule) void {
        self.device_dispatch.destroy_shader_module(self.device, module, null);
    }

    /// Create a graphics pipeline
    pub fn createGraphicsPipeline(self: *PipelineCache, config: GraphicsPipelineConfig) Error!vk.types.VkPipeline {
        // Build shader stages
        var stages: [5]vk.types.VkPipelineShaderStageCreateInfo = undefined;
        var stage_count: u32 = 0;

        if (config.vertex_shader) |vs| {
            stages[stage_count] = makeShaderStage(.VERTEX, vs, "main");
            stage_count += 1;
        }
        if (config.fragment_shader) |fs| {
            stages[stage_count] = makeShaderStage(.FRAGMENT, fs, "main");
            stage_count += 1;
        }
        if (config.geometry_shader) |gs| {
            stages[stage_count] = makeShaderStage(.GEOMETRY, gs, "main");
            stage_count += 1;
        }
        if (config.tessellation_control) |tc| {
            stages[stage_count] = makeShaderStage(.TESSELLATION_CONTROL, tc, "main");
            stage_count += 1;
        }
        if (config.tessellation_eval) |te| {
            stages[stage_count] = makeShaderStage(.TESSELLATION_EVALUATION, te, "main");
            stage_count += 1;
        }

        // Vertex input state
        const vertex_input = vk.types.VkPipelineVertexInputStateCreateInfo{
            .sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .vertexBindingDescriptionCount = @intCast(config.vertex_bindings.len),
            .pVertexBindingDescriptions = if (config.vertex_bindings.len > 0) config.vertex_bindings.ptr else null,
            .vertexAttributeDescriptionCount = @intCast(config.vertex_attributes.len),
            .pVertexAttributeDescriptions = if (config.vertex_attributes.len > 0) config.vertex_attributes.ptr else null,
        };

        // Input assembly
        const input_assembly = vk.types.VkPipelineInputAssemblyStateCreateInfo{
            .sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .topology = config.topology,
            .primitiveRestartEnable = if (config.primitive_restart) 1 else 0,
        };

        // Viewport state (dynamic)
        const viewport_state = vk.types.VkPipelineViewportStateCreateInfo{
            .sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = null, // Dynamic
            .scissorCount = 1,
            .pScissors = null, // Dynamic
        };

        // Rasterization
        const rasterization = vk.types.VkPipelineRasterizationStateCreateInfo{
            .sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .depthClampEnable = 0,
            .rasterizerDiscardEnable = 0,
            .polygonMode = config.polygon_mode,
            .cullMode = config.cull_mode,
            .frontFace = config.front_face,
            .depthBiasEnable = 0,
            .depthBiasConstantFactor = 0,
            .depthBiasClamp = 0,
            .depthBiasSlopeFactor = 0,
            .lineWidth = config.line_width,
        };

        // Multisampling
        const multisample = vk.types.VkPipelineMultisampleStateCreateInfo{
            .sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .rasterizationSamples = config.samples,
            .sampleShadingEnable = 0,
            .minSampleShading = 1.0,
            .pSampleMask = null,
            .alphaToCoverageEnable = 0,
            .alphaToOneEnable = 0,
        };

        // Depth/stencil
        const depth_stencil = vk.types.VkPipelineDepthStencilStateCreateInfo{
            .sType = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .depthTestEnable = if (config.depth_test) 1 else 0,
            .depthWriteEnable = if (config.depth_write) 1 else 0,
            .depthCompareOp = config.depth_compare,
            .depthBoundsTestEnable = 0,
            .stencilTestEnable = if (config.stencil_test) 1 else 0,
            .front = std.mem.zeroes(vk.types.VkStencilOpState),
            .back = std.mem.zeroes(vk.types.VkStencilOpState),
            .minDepthBounds = 0.0,
            .maxDepthBounds = 1.0,
        };

        // Color blending
        const blend_attachment = vk.types.VkPipelineColorBlendAttachmentState{
            .blendEnable = if (config.blend_enable) 1 else 0,
            .srcColorBlendFactor = config.src_color_blend,
            .dstColorBlendFactor = config.dst_color_blend,
            .colorBlendOp = config.color_blend_op,
            .srcAlphaBlendFactor = config.src_alpha_blend,
            .dstAlphaBlendFactor = config.dst_alpha_blend,
            .alphaBlendOp = config.alpha_blend_op,
            .colorWriteMask = config.color_write_mask,
        };

        const color_blend = vk.types.VkPipelineColorBlendStateCreateInfo{
            .sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .logicOpEnable = 0,
            .logicOp = .COPY,
            .attachmentCount = 1,
            .pAttachments = &blend_attachment,
            .blendConstants = [_]f32{ 0, 0, 0, 0 },
        };

        // Dynamic state
        const dynamic_state = vk.types.VkPipelineDynamicStateCreateInfo{
            .sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .dynamicStateCount = @intCast(config.dynamic_states.len),
            .pDynamicStates = if (config.dynamic_states.len > 0) config.dynamic_states.ptr else null,
        };

        // Create pipeline
        const pipeline_info = vk.types.VkGraphicsPipelineCreateInfo{
            .sType = .GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stageCount = stage_count,
            .pStages = &stages,
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pTessellationState = null,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blend,
            .pDynamicState = &dynamic_state,
            .layout = config.layout,
            .renderPass = config.render_pass,
            .subpass = config.subpass,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };

        var pipeline: vk.types.VkPipeline = undefined;
        const result = self.device_dispatch.create_graphics_pipelines(
            self.device,
            self.vk_cache,
            1,
            &pipeline_info,
            null,
            &pipeline,
        );

        if (result != .SUCCESS) {
            log.err("Failed to create graphics pipeline: {s}", .{@tagName(result)});
            return Error.PipelineCreationFailed;
        }

        self.stats.graphics_created += 1;
        log.debug("Created graphics pipeline", .{});

        return pipeline;
    }

    /// Create a compute pipeline
    pub fn createComputePipeline(self: *PipelineCache, config: ComputePipelineConfig) Error!vk.types.VkPipeline {
        const stage = vk.types.VkPipelineShaderStageCreateInfo{
            .sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = .COMPUTE,
            .module = config.shader,
            .pName = config.entry_point.ptr,
            .pSpecializationInfo = config.specialization,
        };

        const pipeline_info = vk.types.VkComputePipelineCreateInfo{
            .sType = .COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = stage,
            .layout = config.layout,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };

        var pipeline: vk.types.VkPipeline = undefined;
        const result = self.device_dispatch.create_compute_pipelines(
            self.device,
            self.vk_cache,
            1,
            &pipeline_info,
            null,
            &pipeline,
        );

        if (result != .SUCCESS) {
            log.err("Failed to create compute pipeline: {s}", .{@tagName(result)});
            return Error.PipelineCreationFailed;
        }

        self.stats.compute_created += 1;
        log.debug("Created compute pipeline", .{});

        return pipeline;
    }

    /// Destroy a pipeline
    pub fn destroyPipeline(self: *PipelineCache, pipeline: vk.types.VkPipeline) void {
        self.device_dispatch.destroy_pipeline(self.device, pipeline, null);
    }

    /// Create a pipeline layout
    pub fn createPipelineLayout(
        self: *PipelineCache,
        set_layouts: []const vk.types.VkDescriptorSetLayout,
        push_constant_ranges: []const vk.types.VkPushConstantRange,
    ) Error!vk.types.VkPipelineLayout {
        const layout_info = vk.types.VkPipelineLayoutCreateInfo{
            .sType = .PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .setLayoutCount = @intCast(set_layouts.len),
            .pSetLayouts = if (set_layouts.len > 0) set_layouts.ptr else null,
            .pushConstantRangeCount = @intCast(push_constant_ranges.len),
            .pPushConstantRanges = if (push_constant_ranges.len > 0) push_constant_ranges.ptr else null,
        };

        var layout: vk.types.VkPipelineLayout = undefined;
        const result = self.device_dispatch.create_pipeline_layout(self.device, &layout_info, null, &layout);
        if (result != .SUCCESS) {
            return Error.LayoutCreationFailed;
        }

        return layout;
    }

    /// Destroy a pipeline layout
    pub fn destroyPipelineLayout(self: *PipelineCache, layout: vk.types.VkPipelineLayout) void {
        self.device_dispatch.destroy_pipeline_layout(self.device, layout, null);
    }

    /// Get statistics
    pub fn getStats(self: *const PipelineCache) Stats {
        return self.stats;
    }

    fn makeShaderStage(stage: vk.types.VkShaderStageFlagBits, module: vk.types.VkShaderModule, entry: [:0]const u8) vk.types.VkPipelineShaderStageCreateInfo {
        return vk.types.VkPipelineShaderStageCreateInfo{
            .sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = stage,
            .module = module,
            .pName = entry.ptr,
            .pSpecializationInfo = null,
        };
    }
};
