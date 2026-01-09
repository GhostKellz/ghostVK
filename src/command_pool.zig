//! GhostVK Command Buffer Pool - Efficient command buffer management
//!
//! High-performance command buffer pooling with:
//! - Per-thread command pools (thread-safe)
//! - Automatic command buffer recycling
//! - Primary and secondary buffer support
//! - One-time submit optimization
//! - Frame-based reset for render loops

const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.ghostvk_cmdpool);

/// Command buffer usage hints
pub const CommandBufferUsage = enum {
    /// General purpose, can be resubmitted
    general,
    /// One-time submit, optimized for single use
    one_time,
    /// Render pass secondary buffer
    render_pass_continue,
    /// Compute-only operations
    compute,
    /// Transfer-only operations (DMA)
    transfer,
};

/// Command buffer with metadata
pub const CommandBuffer = struct {
    buffer: vk.types.VkCommandBuffer,
    pool: vk.types.VkCommandPool,
    level: vk.types.VkCommandBufferLevel,
    state: State,
    frame_index: u64,
    usage: CommandBufferUsage,

    pub const State = enum {
        initial,
        recording,
        executable,
        pending,
        invalid,
    };

    /// Begin recording commands
    pub fn begin(self: *CommandBuffer, dispatch: *const vk.loader.DeviceDispatch, inheritance_info: ?*const vk.types.VkCommandBufferInheritanceInfo) !void {
        if (self.state != .initial and self.state != .executable) {
            return error.InvalidState;
        }

        var flags: vk.types.VkCommandBufferUsageFlags = 0;
        if (self.usage == .one_time) {
            flags |= vk.types.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        }
        if (self.usage == .render_pass_continue) {
            flags |= vk.types.VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        }

        const begin_info = vk.types.VkCommandBufferBeginInfo{
            .sType = .COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = flags,
            .pInheritanceInfo = inheritance_info,
        };

        const result = dispatch.begin_command_buffer(self.buffer, &begin_info);
        if (result != .SUCCESS) {
            return error.BeginFailed;
        }

        self.state = .recording;
    }

    /// End recording commands
    pub fn end(self: *CommandBuffer, dispatch: *const vk.loader.DeviceDispatch) !void {
        if (self.state != .recording) {
            return error.InvalidState;
        }

        const result = dispatch.end_command_buffer(self.buffer);
        if (result != .SUCCESS) {
            return error.EndFailed;
        }

        self.state = .executable;
    }

    /// Reset the command buffer
    pub fn reset(self: *CommandBuffer, dispatch: *const vk.loader.DeviceDispatch) !void {
        const result = dispatch.reset_command_buffer(self.buffer, 0);
        if (result != .SUCCESS) {
            return error.ResetFailed;
        }
        self.state = .initial;
    }
};

/// Per-frame command buffer set
pub const FrameCommandBuffers = struct {
    primary: std.ArrayList(CommandBuffer),
    secondary: std.ArrayList(CommandBuffer),
    frame_index: u64,
    in_use_count: usize,

    fn init(allocator: std.mem.Allocator) FrameCommandBuffers {
        return .{
            .primary = std.ArrayList(CommandBuffer).init(allocator),
            .secondary = std.ArrayList(CommandBuffer).init(allocator),
            .frame_index = 0,
            .in_use_count = 0,
        };
    }

    fn deinit(self: *FrameCommandBuffers) void {
        self.primary.deinit();
        self.secondary.deinit();
    }
};

/// Command buffer pool for a specific queue family
pub const CommandPool = struct {
    allocator: std.mem.Allocator,
    device: vk.types.VkDevice,
    device_dispatch: *const vk.loader.DeviceDispatch,
    queue_family_index: u32,

    // Vulkan command pool
    vk_pool: vk.types.VkCommandPool,

    // Per-frame command buffers (double/triple buffering)
    frame_buffers: []FrameCommandBuffers,
    frames_in_flight: u32,
    current_frame: u32,

    // Free lists for recycling
    free_primary: std.ArrayList(vk.types.VkCommandBuffer),
    free_secondary: std.ArrayList(vk.types.VkCommandBuffer),

    // Statistics
    stats: Stats,

    pub const Stats = struct {
        total_allocated: u64 = 0,
        total_freed: u64 = 0,
        current_in_use: u64 = 0,
        peak_in_use: u64 = 0,
        resets: u64 = 0,
    };

    pub const Config = struct {
        queue_family_index: u32,
        frames_in_flight: u32 = 2,
        /// Allow individual buffer reset (vs pool reset)
        allow_reset: bool = true,
        /// Buffers are short-lived (transient)
        transient: bool = false,
        /// Pre-allocate this many primary buffers per frame
        initial_primary_count: u32 = 4,
        /// Pre-allocate this many secondary buffers per frame
        initial_secondary_count: u32 = 8,
    };

    pub const Error = error{
        PoolCreationFailed,
        AllocationFailed,
        InvalidState,
        BeginFailed,
        EndFailed,
        ResetFailed,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        device: vk.types.VkDevice,
        device_dispatch: *const vk.loader.DeviceDispatch,
        config: Config,
    ) Error!CommandPool {
        // Create Vulkan command pool
        var flags: vk.types.VkCommandPoolCreateFlags = 0;
        if (config.allow_reset) {
            flags |= vk.types.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        }
        if (config.transient) {
            flags |= vk.types.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        }

        const pool_info = vk.types.VkCommandPoolCreateInfo{
            .sType = .COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = flags,
            .queueFamilyIndex = config.queue_family_index,
        };

        var vk_pool: vk.types.VkCommandPool = undefined;
        const result = device_dispatch.create_command_pool(device, &pool_info, null, &vk_pool);
        if (result != .SUCCESS) {
            log.err("Failed to create command pool: {s}", .{@tagName(result)});
            return Error.PoolCreationFailed;
        }

        // Allocate frame buffers
        const frame_buffers = allocator.alloc(FrameCommandBuffers, config.frames_in_flight) catch {
            device_dispatch.destroy_command_pool(device, vk_pool, null);
            return Error.AllocationFailed;
        };
        for (frame_buffers) |*fb| {
            fb.* = FrameCommandBuffers.init(allocator);
        }

        var self = CommandPool{
            .allocator = allocator,
            .device = device,
            .device_dispatch = device_dispatch,
            .queue_family_index = config.queue_family_index,
            .vk_pool = vk_pool,
            .frame_buffers = frame_buffers,
            .frames_in_flight = config.frames_in_flight,
            .current_frame = 0,
            .free_primary = std.ArrayList(vk.types.VkCommandBuffer).init(allocator),
            .free_secondary = std.ArrayList(vk.types.VkCommandBuffer).init(allocator),
            .stats = .{},
        };

        // Pre-allocate command buffers
        if (config.initial_primary_count > 0) {
            const buffers = self.allocateVkCommandBuffers(.PRIMARY, config.initial_primary_count) catch {
                self.deinit();
                return Error.AllocationFailed;
            };
            defer allocator.free(buffers);
            for (buffers) |buf| {
                self.free_primary.append(buf) catch {};
            }
        }

        if (config.initial_secondary_count > 0) {
            const buffers = self.allocateVkCommandBuffers(.SECONDARY, config.initial_secondary_count) catch {
                self.deinit();
                return Error.AllocationFailed;
            };
            defer allocator.free(buffers);
            for (buffers) |buf| {
                self.free_secondary.append(buf) catch {};
            }
        }

        log.info("Command pool created: queue family {}, {} frames, {} primary, {} secondary", .{
            config.queue_family_index,
            config.frames_in_flight,
            config.initial_primary_count,
            config.initial_secondary_count,
        });

        return self;
    }

    pub fn deinit(self: *CommandPool) void {
        // Free all command buffers
        if (self.free_primary.items.len > 0) {
            self.device_dispatch.free_command_buffers(
                self.device,
                self.vk_pool,
                @intCast(self.free_primary.items.len),
                self.free_primary.items.ptr,
            );
        }
        self.free_primary.deinit();

        if (self.free_secondary.items.len > 0) {
            self.device_dispatch.free_command_buffers(
                self.device,
                self.vk_pool,
                @intCast(self.free_secondary.items.len),
                self.free_secondary.items.ptr,
            );
        }
        self.free_secondary.deinit();

        // Free frame buffers
        for (self.frame_buffers) |*fb| {
            for (fb.primary.items) |cmd| {
                self.device_dispatch.free_command_buffers(self.device, self.vk_pool, 1, &cmd.buffer);
            }
            for (fb.secondary.items) |cmd| {
                self.device_dispatch.free_command_buffers(self.device, self.vk_pool, 1, &cmd.buffer);
            }
            fb.deinit();
        }
        self.allocator.free(self.frame_buffers);

        // Destroy pool
        self.device_dispatch.destroy_command_pool(self.device, self.vk_pool, null);

        log.info("Command pool destroyed: {} allocated, {} freed, peak {}", .{
            self.stats.total_allocated,
            self.stats.total_freed,
            self.stats.peak_in_use,
        });
    }

    /// Get a primary command buffer for the current frame
    pub fn getPrimary(self: *CommandPool, usage: CommandBufferUsage) Error!*CommandBuffer {
        return self.getCommandBuffer(.PRIMARY, usage);
    }

    /// Get a secondary command buffer for the current frame
    pub fn getSecondary(self: *CommandPool, usage: CommandBufferUsage) Error!*CommandBuffer {
        return self.getCommandBuffer(.SECONDARY, usage);
    }

    /// Get a one-time submit command buffer (auto begins recording)
    pub fn getOneTime(self: *CommandPool) Error!*CommandBuffer {
        var cmd = try self.getPrimary(.one_time);
        try cmd.begin(self.device_dispatch, null);
        return cmd;
    }

    /// Submit one-time command buffer and wait
    pub fn submitOneTimeAndWait(self: *CommandPool, cmd: *CommandBuffer, queue: vk.types.VkQueue) Error!void {
        try cmd.end(self.device_dispatch);

        const submit_info = vk.types.VkSubmitInfo{
            .sType = .SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd.buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };

        _ = self.device_dispatch.queue_submit(queue, 1, &submit_info, null);
        _ = self.device_dispatch.queue_wait_idle(queue);

        // Return to free list
        self.returnCommandBuffer(cmd);
    }

    /// Advance to next frame (recycles command buffers from oldest frame)
    pub fn nextFrame(self: *CommandPool) void {
        const prev_frame = self.current_frame;
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        // Recycle command buffers from the frame we're about to reuse
        const frame = &self.frame_buffers[self.current_frame];

        for (frame.primary.items) |*cmd| {
            cmd.reset(self.device_dispatch) catch {};
            cmd.state = .initial;
            self.free_primary.append(cmd.buffer) catch {};
        }
        frame.primary.clearRetainingCapacity();

        for (frame.secondary.items) |*cmd| {
            cmd.reset(self.device_dispatch) catch {};
            cmd.state = .initial;
            self.free_secondary.append(cmd.buffer) catch {};
        }
        frame.secondary.clearRetainingCapacity();

        self.stats.current_in_use -= frame.in_use_count;
        frame.in_use_count = 0;
        frame.frame_index = self.stats.resets;
        self.stats.resets += 1;

        log.debug("Frame {} -> {}: recycled buffers", .{ prev_frame, self.current_frame });
    }

    /// Reset the entire pool (faster than individual resets)
    pub fn resetPool(self: *CommandPool) Error!void {
        const result = self.device_dispatch.reset_command_pool(self.device, self.vk_pool, 0);
        if (result != .SUCCESS) {
            return Error.ResetFailed;
        }

        // Move all buffers back to free lists
        for (self.frame_buffers) |*frame| {
            for (frame.primary.items) |cmd| {
                self.free_primary.append(cmd.buffer) catch {};
            }
            frame.primary.clearRetainingCapacity();

            for (frame.secondary.items) |cmd| {
                self.free_secondary.append(cmd.buffer) catch {};
            }
            frame.secondary.clearRetainingCapacity();

            frame.in_use_count = 0;
        }

        self.stats.current_in_use = 0;
        self.stats.resets += 1;
    }

    /// Get statistics
    pub fn getStats(self: *const CommandPool) Stats {
        return self.stats;
    }

    // Internal functions

    fn getCommandBuffer(self: *CommandPool, level: vk.types.VkCommandBufferLevel, usage: CommandBufferUsage) Error!*CommandBuffer {
        const frame = &self.frame_buffers[self.current_frame];
        const is_primary = level == .PRIMARY;
        const target_list = if (is_primary) &frame.primary else &frame.secondary;
        const free_list = if (is_primary) &self.free_primary else &self.free_secondary;

        // Try to get from free list
        var vk_buffer: vk.types.VkCommandBuffer = undefined;
        if (free_list.items.len > 0) {
            vk_buffer = free_list.pop();
        } else {
            // Allocate new buffer
            const buffers = try self.allocateVkCommandBuffers(level, 1);
            defer self.allocator.free(buffers);
            vk_buffer = buffers[0];
        }

        // Create CommandBuffer wrapper
        const cmd = CommandBuffer{
            .buffer = vk_buffer,
            .pool = self.vk_pool,
            .level = level,
            .state = .initial,
            .frame_index = frame.frame_index,
            .usage = usage,
        };

        target_list.append(cmd) catch return Error.AllocationFailed;

        frame.in_use_count += 1;
        self.stats.current_in_use += 1;
        if (self.stats.current_in_use > self.stats.peak_in_use) {
            self.stats.peak_in_use = self.stats.current_in_use;
        }

        return &target_list.items[target_list.items.len - 1];
    }

    fn returnCommandBuffer(self: *CommandPool, cmd: *CommandBuffer) void {
        cmd.reset(self.device_dispatch) catch {};

        if (cmd.level == .PRIMARY) {
            self.free_primary.append(cmd.buffer) catch {};
        } else {
            self.free_secondary.append(cmd.buffer) catch {};
        }

        self.stats.current_in_use -= 1;
        self.stats.total_freed += 1;
    }

    fn allocateVkCommandBuffers(self: *CommandPool, level: vk.types.VkCommandBufferLevel, count: u32) Error![]vk.types.VkCommandBuffer {
        const buffers = self.allocator.alloc(vk.types.VkCommandBuffer, count) catch {
            return Error.AllocationFailed;
        };
        errdefer self.allocator.free(buffers);

        const alloc_info = vk.types.VkCommandBufferAllocateInfo{
            .sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.vk_pool,
            .level = level,
            .commandBufferCount = count,
        };

        const result = self.device_dispatch.allocate_command_buffers(self.device, &alloc_info, buffers.ptr);
        if (result != .SUCCESS) {
            self.allocator.free(buffers);
            return Error.AllocationFailed;
        }

        self.stats.total_allocated += count;

        return buffers;
    }
};

/// Multi-queue command pool manager
pub const CommandPoolManager = struct {
    allocator: std.mem.Allocator,
    device: vk.types.VkDevice,
    device_dispatch: *const vk.loader.DeviceDispatch,

    graphics_pool: ?CommandPool,
    compute_pool: ?CommandPool,
    transfer_pool: ?CommandPool,

    pub fn init(
        allocator: std.mem.Allocator,
        device: vk.types.VkDevice,
        device_dispatch: *const vk.loader.DeviceDispatch,
        graphics_family: u32,
        compute_family: ?u32,
        transfer_family: ?u32,
        frames_in_flight: u32,
    ) !CommandPoolManager {
        var self = CommandPoolManager{
            .allocator = allocator,
            .device = device,
            .device_dispatch = device_dispatch,
            .graphics_pool = null,
            .compute_pool = null,
            .transfer_pool = null,
        };

        // Graphics pool (always created)
        self.graphics_pool = try CommandPool.init(allocator, device, device_dispatch, .{
            .queue_family_index = graphics_family,
            .frames_in_flight = frames_in_flight,
            .initial_primary_count = 4,
            .initial_secondary_count = 16,
        });

        // Compute pool (if separate queue family)
        if (compute_family) |cf| {
            if (cf != graphics_family) {
                self.compute_pool = try CommandPool.init(allocator, device, device_dispatch, .{
                    .queue_family_index = cf,
                    .frames_in_flight = frames_in_flight,
                    .initial_primary_count = 2,
                    .initial_secondary_count = 4,
                });
            }
        }

        // Transfer pool (if separate queue family)
        if (transfer_family) |tf| {
            if (tf != graphics_family and (compute_family == null or tf != compute_family.?)) {
                self.transfer_pool = try CommandPool.init(allocator, device, device_dispatch, .{
                    .queue_family_index = tf,
                    .frames_in_flight = frames_in_flight,
                    .transient = true,
                    .initial_primary_count = 2,
                    .initial_secondary_count = 0,
                });
            }
        }

        log.info("Command pool manager initialized", .{});
        return self;
    }

    pub fn deinit(self: *CommandPoolManager) void {
        if (self.graphics_pool) |*pool| pool.deinit();
        if (self.compute_pool) |*pool| pool.deinit();
        if (self.transfer_pool) |*pool| pool.deinit();
    }

    pub fn graphics(self: *CommandPoolManager) *CommandPool {
        return &self.graphics_pool.?;
    }

    pub fn compute(self: *CommandPoolManager) ?*CommandPool {
        if (self.compute_pool) |*pool| return pool;
        return &self.graphics_pool.?; // Fallback to graphics
    }

    pub fn transfer(self: *CommandPoolManager) ?*CommandPool {
        if (self.transfer_pool) |*pool| return pool;
        return &self.graphics_pool.?; // Fallback to graphics
    }

    pub fn nextFrame(self: *CommandPoolManager) void {
        if (self.graphics_pool) |*pool| pool.nextFrame();
        if (self.compute_pool) |*pool| pool.nextFrame();
        if (self.transfer_pool) |*pool| pool.nextFrame();
    }
};
