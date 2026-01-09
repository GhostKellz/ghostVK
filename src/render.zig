//! Render pipeline for ghostVK
//!
//! Provides:
//! - Shader module loading from embedded SPIR-V
//! - Render pass creation
//! - Framebuffer management
//! - Graphics pipeline with push constants
//! - Draw command recording

const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.render);

// Embedded SPIR-V shaders
const vert_spv = @embedFile("triangle.vert.spv");
const frag_spv = @embedFile("triangle.frag.spv");

pub const PushConstants = extern struct {
    time: f32,
    padding: [3]f32 = .{ 0, 0, 0 },
};

pub const RenderPipeline = struct {
    allocator: std.mem.Allocator,
    device: vk.types.VkDevice,
    dispatch: *const vk.loader.DeviceDispatch,

    // Shader modules
    vert_shader: ?vk.types.VkShaderModule = null,
    frag_shader: ?vk.types.VkShaderModule = null,

    // Render pass
    render_pass: ?vk.types.VkRenderPass = null,

    // Pipeline
    pipeline_cache: ?vk.types.VkPipelineCache = null,
    pipeline_layout: ?vk.types.VkPipelineLayout = null,
    pipeline: ?vk.types.VkPipeline = null,

    // Framebuffers (one per swapchain image)
    framebuffers: []vk.types.VkFramebuffer = &[_]vk.types.VkFramebuffer{},

    // Swapchain info (stored for recreation)
    swapchain_format: vk.types.VkFormat = .UNDEFINED,
    swapchain_extent: vk.types.VkExtent2D = .{ .width = 0, .height = 0 },

    pub fn init(
        allocator: std.mem.Allocator,
        device: vk.types.VkDevice,
        dispatch: *const vk.loader.DeviceDispatch,
        swapchain_format: vk.types.VkFormat,
        swapchain_extent: vk.types.VkExtent2D,
        swapchain_image_views: []const vk.types.VkImageView,
    ) !RenderPipeline {
        var self = RenderPipeline{
            .allocator = allocator,
            .device = device,
            .dispatch = dispatch,
            .swapchain_format = swapchain_format,
            .swapchain_extent = swapchain_extent,
        };

        errdefer self.deinit();

        try self.createShaderModules();
        try self.createRenderPass();
        try self.createPipelineCache();
        try self.createPipelineLayout();
        try self.createPipeline();
        try self.createFramebuffers(swapchain_image_views);

        log.info("Render pipeline initialized: {}x{}, {} framebuffers", .{
            swapchain_extent.width,
            swapchain_extent.height,
            swapchain_image_views.len,
        });

        return self;
    }

    pub fn deinit(self: *RenderPipeline) void {
        // Destroy framebuffers
        for (self.framebuffers) |fb| {
            self.dispatch.destroy_framebuffer(self.device, fb, null);
        }
        if (self.framebuffers.len > 0) {
            self.allocator.free(self.framebuffers);
            self.framebuffers = &[_]vk.types.VkFramebuffer{};
        }

        // Destroy pipeline
        if (self.pipeline) |p| {
            self.dispatch.destroy_pipeline(self.device, p, null);
            self.pipeline = null;
        }

        // Destroy pipeline layout
        if (self.pipeline_layout) |pl| {
            self.dispatch.destroy_pipeline_layout(self.device, pl, null);
            self.pipeline_layout = null;
        }

        // Destroy pipeline cache
        if (self.pipeline_cache) |pc| {
            self.dispatch.destroy_pipeline_cache(self.device, pc, null);
            self.pipeline_cache = null;
        }

        // Destroy render pass
        if (self.render_pass) |rp| {
            self.dispatch.destroy_render_pass(self.device, rp, null);
            self.render_pass = null;
        }

        // Destroy shader modules
        if (self.vert_shader) |s| {
            self.dispatch.destroy_shader_module(self.device, s, null);
            self.vert_shader = null;
        }
        if (self.frag_shader) |s| {
            self.dispatch.destroy_shader_module(self.device, s, null);
            self.frag_shader = null;
        }

        log.info("Render pipeline destroyed", .{});
    }

    fn createShaderModules(self: *RenderPipeline) !void {
        // Vertex shader
        const vert_info = vk.types.VkShaderModuleCreateInfo{
            .code_size = vert_spv.len,
            .p_code = @ptrCast(@alignCast(vert_spv.ptr)),
        };

        var vert_module: vk.types.VkShaderModule = undefined;
        var result = self.dispatch.create_shader_module(self.device, &vert_info, null, &vert_module);
        if (result != .SUCCESS) {
            log.err("Failed to create vertex shader module: {s}", .{@tagName(result)});
            return error.ShaderCreationFailed;
        }
        self.vert_shader = vert_module;

        // Fragment shader
        const frag_info = vk.types.VkShaderModuleCreateInfo{
            .code_size = frag_spv.len,
            .p_code = @ptrCast(@alignCast(frag_spv.ptr)),
        };

        var frag_module: vk.types.VkShaderModule = undefined;
        result = self.dispatch.create_shader_module(self.device, &frag_info, null, &frag_module);
        if (result != .SUCCESS) {
            log.err("Failed to create fragment shader module: {s}", .{@tagName(result)});
            return error.ShaderCreationFailed;
        }
        self.frag_shader = frag_module;

        log.info("Shader modules created (vert: {} bytes, frag: {} bytes)", .{ vert_spv.len, frag_spv.len });
    }

    fn createRenderPass(self: *RenderPipeline) !void {
        const color_attachment = vk.types.VkAttachmentDescription{
            .format = self.swapchain_format,
            .samples = .@"1",
            .loadOp = .CLEAR,
            .storeOp = .STORE,
            .stencilLoadOp = .DONT_CARE,
            .stencilStoreOp = .DONT_CARE,
            .initialLayout = .UNDEFINED,
            .finalLayout = .PRESENT_SRC_KHR,
        };

        const color_attachment_ref = vk.types.VkAttachmentReference{
            .attachment = 0,
            .layout = .COLOR_ATTACHMENT_OPTIMAL,
        };

        const subpass = vk.types.VkSubpassDescription{
            .pipelineBindPoint = .GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = @ptrCast(&color_attachment_ref),
        };

        // VK_SUBPASS_EXTERNAL = ~0U (max u32)
        const VK_SUBPASS_EXTERNAL: u32 = 0xFFFFFFFF;
        const dependency = vk.types.VkSubpassDependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk.types.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask = vk.types.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = vk.types.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        const render_pass_info = vk.types.VkRenderPassCreateInfo{
            .attachmentCount = 1,
            .pAttachments = @ptrCast(&color_attachment),
            .subpassCount = 1,
            .pSubpasses = @ptrCast(&subpass),
            .dependencyCount = 1,
            .pDependencies = @ptrCast(&dependency),
        };

        var render_pass: vk.types.VkRenderPass = undefined;
        const result = self.dispatch.create_render_pass(self.device, &render_pass_info, null, &render_pass);
        if (result != .SUCCESS) {
            log.err("Failed to create render pass: {s}", .{@tagName(result)});
            return error.RenderPassCreationFailed;
        }

        self.render_pass = render_pass;
        log.info("Render pass created", .{});
    }

    fn createPipelineCache(self: *RenderPipeline) !void {
        const cache_info = vk.types.VkPipelineCacheCreateInfo{};

        var cache: vk.types.VkPipelineCache = undefined;
        const result = self.dispatch.create_pipeline_cache(self.device, &cache_info, null, &cache);
        if (result != .SUCCESS) {
            log.err("Failed to create pipeline cache: {s}", .{@tagName(result)});
            return error.PipelineCacheCreationFailed;
        }

        self.pipeline_cache = cache;
        log.info("Pipeline cache created", .{});
    }

    fn createPipelineLayout(self: *RenderPipeline) !void {
        const push_constant_range = vk.types.VkPushConstantRange{
            .stageFlags = vk.types.VK_SHADER_STAGE_VERTEX_BIT | vk.types.VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = @sizeOf(PushConstants),
        };

        const layout_info = vk.types.VkPipelineLayoutCreateInfo{
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_constant_range),
        };

        var layout: vk.types.VkPipelineLayout = undefined;
        const result = self.dispatch.create_pipeline_layout(self.device, &layout_info, null, &layout);
        if (result != .SUCCESS) {
            log.err("Failed to create pipeline layout: {s}", .{@tagName(result)});
            return error.PipelineLayoutCreationFailed;
        }

        self.pipeline_layout = layout;
        log.info("Pipeline layout created (push constants: {} bytes)", .{@sizeOf(PushConstants)});
    }

    fn createPipeline(self: *RenderPipeline) !void {
        // Shader stages
        const vert_stage = vk.types.VkPipelineShaderStageCreateInfo{
            .stage = .VERTEX_BIT,
            .module = self.vert_shader.?,
            .p_name = "main",
        };

        const frag_stage = vk.types.VkPipelineShaderStageCreateInfo{
            .stage = .FRAGMENT_BIT,
            .module = self.frag_shader.?,
            .p_name = "main",
        };

        const shader_stages = [_]vk.types.VkPipelineShaderStageCreateInfo{ vert_stage, frag_stage };

        // Vertex input (none - generated in shader)
        const vertex_input = vk.types.VkPipelineVertexInputStateCreateInfo{};

        // Input assembly
        const input_assembly = vk.types.VkPipelineInputAssemblyStateCreateInfo{
            .topology = .TRIANGLE_LIST,
        };

        // Viewport (dynamic state)
        const viewport = vk.types.VkViewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.swapchain_extent.width),
            .height = @floatFromInt(self.swapchain_extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };

        const scissor = vk.types.VkRect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain_extent,
        };

        const viewport_state = vk.types.VkPipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .p_viewports = @ptrCast(&viewport),
            .scissor_count = 1,
            .p_scissors = @ptrCast(&scissor),
        };

        // Rasterization
        const rasterization = vk.types.VkPipelineRasterizationStateCreateInfo{
            .polygon_mode = .FILL,
            .cull_mode = 0, // VK_CULL_MODE_NONE
            .front_face = .CLOCKWISE,
            .line_width = 1.0,
        };

        // Multisampling
        const multisampling = vk.types.VkPipelineMultisampleStateCreateInfo{
            .rasterization_samples = .@"1",
        };

        // Color blending
        const color_blend_attachment = vk.types.VkPipelineColorBlendAttachmentState{
            .blend_enable = 0,
            .src_color_blend_factor = .ONE,
            .dst_color_blend_factor = .ZERO,
            .color_blend_op = .ADD,
            .src_alpha_blend_factor = .ONE,
            .dst_alpha_blend_factor = .ZERO,
            .alpha_blend_op = .ADD,
            .color_write_mask = 0xF, // RGBA
        };

        const color_blending = vk.types.VkPipelineColorBlendStateCreateInfo{
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_blend_attachment),
        };

        // Dynamic state (viewport and scissor)
        const dynamic_states = [_]vk.types.VkDynamicState{ .VIEWPORT, .SCISSOR };
        const dynamic_state = vk.types.VkPipelineDynamicStateCreateInfo{
            .dynamic_state_count = dynamic_states.len,
            .p_dynamic_states = &dynamic_states,
        };

        // Create pipeline
        const pipeline_info = vk.types.VkGraphicsPipelineCreateInfo{
            .stage_count = shader_stages.len,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &vertex_input,
            .p_input_assembly_state = &input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterization,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state,
            .layout = self.pipeline_layout.?,
            .render_pass = self.render_pass.?,
            .subpass = 0,
        };

        var pipeline: vk.types.VkPipeline = undefined;
        const result = self.dispatch.create_graphics_pipelines(
            self.device,
            self.pipeline_cache.?,
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&pipeline),
        );
        if (result != .SUCCESS) {
            log.err("Failed to create graphics pipeline: {s}", .{@tagName(result)});
            return error.PipelineCreationFailed;
        }

        self.pipeline = pipeline;
        log.info("Graphics pipeline created", .{});
    }

    fn createFramebuffers(self: *RenderPipeline, image_views: []const vk.types.VkImageView) !void {
        const framebuffers = try self.allocator.alloc(vk.types.VkFramebuffer, image_views.len);
        errdefer self.allocator.free(framebuffers);

        for (image_views, 0..) |view, i| {
            const attachments = [_]vk.types.VkImageView{view};

            const fb_info = vk.types.VkFramebufferCreateInfo{
                .renderPass = self.render_pass.?,
                .attachmentCount = 1,
                .pAttachments = &attachments,
                .width = self.swapchain_extent.width,
                .height = self.swapchain_extent.height,
                .layers = 1,
            };

            const result = self.dispatch.create_framebuffer(self.device, &fb_info, null, &framebuffers[i]);
            if (result != .SUCCESS) {
                // Cleanup already-created framebuffers
                for (framebuffers[0..i]) |fb| {
                    self.dispatch.destroy_framebuffer(self.device, fb, null);
                }
                self.allocator.free(framebuffers);
                log.err("Failed to create framebuffer {}: {s}", .{ i, @tagName(result) });
                return error.FramebufferCreationFailed;
            }
        }

        self.framebuffers = framebuffers;
        log.info("Created {} framebuffers", .{framebuffers.len});
    }

    /// Recreate framebuffers for new swapchain
    pub fn recreateFramebuffers(
        self: *RenderPipeline,
        new_extent: vk.types.VkExtent2D,
        new_image_views: []const vk.types.VkImageView,
    ) !void {
        // Destroy old framebuffers
        for (self.framebuffers) |fb| {
            self.dispatch.destroy_framebuffer(self.device, fb, null);
        }
        if (self.framebuffers.len > 0) {
            self.allocator.free(self.framebuffers);
            self.framebuffers = &[_]vk.types.VkFramebuffer{};
        }

        self.swapchain_extent = new_extent;
        try self.createFramebuffers(new_image_views);

        log.info("Framebuffers recreated for {}x{}", .{ new_extent.width, new_extent.height });
    }

    /// Record draw commands into a command buffer
    pub fn recordDrawCommands(
        self: *RenderPipeline,
        cmd: vk.types.VkCommandBuffer,
        image_index: u32,
        push_constants: PushConstants,
    ) void {
        // Begin render pass
        const clear_color = vk.types.VkClearValue{
            .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } }, // Black background
        };

        const render_pass_begin = vk.types.VkRenderPassBeginInfo{
            .renderPass = self.render_pass.?,
            .framebuffer = self.framebuffers[image_index],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swapchain_extent,
            },
            .clearValueCount = 1,
            .pClearValues = @ptrCast(&clear_color),
        };

        self.dispatch.cmd_begin_render_pass(cmd, &render_pass_begin, .INLINE);

        // Bind pipeline
        self.dispatch.cmd_bind_pipeline(cmd, .GRAPHICS, self.pipeline.?);

        // Set dynamic viewport
        const viewport = vk.types.VkViewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.swapchain_extent.width),
            .height = @floatFromInt(self.swapchain_extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };
        self.dispatch.cmd_set_viewport(cmd, 0, 1, @ptrCast(&viewport));

        // Set dynamic scissor
        const scissor = vk.types.VkRect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain_extent,
        };
        self.dispatch.cmd_set_scissor(cmd, 0, 1, @ptrCast(&scissor));

        // Push constants
        self.dispatch.cmd_push_constants(
            cmd,
            self.pipeline_layout.?,
            vk.types.VK_SHADER_STAGE_VERTEX_BIT | vk.types.VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            @sizeOf(PushConstants),
            @ptrCast(&push_constants),
        );

        // Draw fullscreen triangle (3 vertices, generated in shader)
        self.dispatch.cmd_draw(cmd, 3, 1, 0, 0);

        // End render pass
        self.dispatch.cmd_end_render_pass(cmd);
    }
};
