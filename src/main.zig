const std = @import("std");
const ghostVK = @import("ghostVK");

const log = std.log.scoped(.app);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var ctx = try ghostVK.GhostVK.init(gpa.allocator(), .{});
    defer ctx.deinit();

    log.info("GhostVK runtime initialized. Starting render loop...", .{});
    log.info("Rendering purple screen at {}x{}", .{ ctx.swapchain_extent.width, ctx.swapchain_extent.height });
    log.info("Press Ctrl+C to exit.", .{});

    // Phase 2: Render loop - display solid purple color
    // Run for 500 frames to test frame pacing and synchronization
    var frame_timer = try std.time.Timer.start();
    const max_frames: u64 = 500;

    while (ctx.frame_count < max_frames) {
        const frame_start = frame_timer.read();

        const success = try ctx.drawFrame();
        if (!success) {
            log.warn("Swapchain needs recreation (not implemented yet)", .{});
            break;
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
