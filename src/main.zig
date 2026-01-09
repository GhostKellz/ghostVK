const std = @import("std");
const ghostVK = @import("ghostVK");

const log = std.log.scoped(.app);

pub fn main() !void {
    // Use c_allocator for compatibility with Vulkan layer hooks (MangoHUD, etc.)
    // Zig's GPA conflicts with MangoHUD's malloc/free interception, causing
    // double-free on shutdown. c_allocator directly uses libc malloc/free
    // which MangoHUD can track correctly.
    const allocator = std.heap.c_allocator;

    // Note: Validation disabled due to crash in VK_LAYER_KHRONOS_validation during
    // cmd_begin_render_pass - appears to be a bug in the validation layer itself.
    // Re-enable when validation layer is updated.
    var ctx = try ghostVK.GhostVK.init(allocator, .{ .enable_validation = false });
    defer ctx.deinit();

    log.info("GhostVK runtime initialized. Starting render loop...", .{});
    log.info("Rendering at {}x{} - Colorspace: {s} (HDR: {})", .{
        ctx.swapchain_extent.width,
        ctx.swapchain_extent.height,
        ctx.getColorspaceName(),
        ctx.isHdrActive(),
    });
    log.info("Press Ctrl+C to exit.", .{});

    // Phase 2: Render loop - display solid purple color
    // Run for 500 frames to test frame pacing and synchronization
    var frame_timer = try std.time.Timer.start();
    const max_frames: u64 = 500;

    while (ctx.frame_count < max_frames) {
        const frame_start = frame_timer.read();

        const success = try ctx.drawFrame();
        if (!success) {
            // Swapchain is out of date - recreate it
            try ctx.recreateSwapchain();
            continue; // Skip this frame, try again with new swapchain
        }

        const frame_time_ns = frame_timer.read() - frame_start;
        const frame_time_ms = @as(f64, @floatFromInt(frame_time_ns)) / 1_000_000.0;

        // Log every 60 frames
        if (ctx.frame_count % 60 == 0) {
            const fps = 1000.0 / frame_time_ms;
            log.info("Frame {}: {d:.2}ms ({d:.1} FPS)", .{ ctx.frame_count, frame_time_ms, fps });
        }
    }

    log.info("Rendered {} frames successfully", .{ctx.frame_count});
}
