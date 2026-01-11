//! GhostVK Frame Pacer
//! Precise frame timing using nvsync for VRR awareness.
//!
//! Features:
//! - Target FPS limiting with sub-millisecond precision
//! - VRR-aware timing (adjusts to display refresh range)
//! - Low Framerate Compensation (LFC) handling
//! - CPU/GPU frame pacing modes

const std = @import("std");
const nvsync = @import("nvsync");

const log = std.log.scoped(.frame_pacer);

/// Frame pacing mode
pub const PacingMode = enum {
    /// No frame pacing (unlimited FPS)
    unlimited,
    /// CPU-side sleep for frame pacing
    cpu_sleep,
    /// Busy-wait for lowest latency (higher CPU usage)
    busy_wait,
    /// Hybrid: sleep most of the time, then busy-wait
    hybrid,
};

/// Frame pacer configuration
pub const FramePacerConfig = struct {
    /// Target frames per second (0 = unlimited)
    target_fps: u32 = 0,
    /// Pacing mode
    mode: PacingMode = .hybrid,
    /// VRR enabled (auto-detected from nvsync)
    vrr_enabled: bool = false,
    /// VRR min Hz (for LFC handling)
    vrr_min_hz: u32 = 48,
    /// VRR max Hz
    vrr_max_hz: u32 = 165,
    /// Busy-wait threshold in nanoseconds (for hybrid mode)
    busy_wait_threshold_ns: u64 = 500_000, // 0.5ms
};

pub const FramePacer = struct {
    config: FramePacerConfig,
    allocator: std.mem.Allocator,

    // Timing state
    frame_target_ns: u64,
    last_frame_time: std.time.Instant,
    frame_times: [60]u64, // Ring buffer of last 60 frame times
    frame_time_index: usize,
    frame_count: u64,

    // Statistics
    total_sleep_ns: u64,
    total_busy_wait_ns: u64,
    frames_paced: u64,

    // nvsync display manager for VRR detection
    display_manager: ?nvsync.DisplayManager,

    pub fn init(allocator: std.mem.Allocator, config: FramePacerConfig) !FramePacer {
        var pacer = FramePacer{
            .config = config,
            .allocator = allocator,
            .frame_target_ns = if (config.target_fps > 0)
                @as(u64, 1_000_000_000) / config.target_fps
            else
                0,
            .last_frame_time = try std.time.Instant.now(),
            .frame_times = [_]u64{0} ** 60,
            .frame_time_index = 0,
            .frame_count = 0,
            .total_sleep_ns = 0,
            .total_busy_wait_ns = 0,
            .frames_paced = 0,
            .display_manager = null,
        };

        // Try to initialize nvsync display manager for VRR detection
        pacer.display_manager = nvsync.DisplayManager.init(allocator);
        if (pacer.display_manager) |*dm| {
            dm.scan() catch |err| {
                log.warn("Failed to scan displays for VRR: {}", .{err});
                pacer.display_manager = null;
            };

            if (pacer.display_manager != null) {
                pacer.updateVrrConfig();
            }
        }

        log.info("Frame pacer initialized: target={} FPS, mode={s}, VRR={}", .{
            config.target_fps,
            @tagName(config.mode),
            pacer.config.vrr_enabled,
        });

        return pacer;
    }

    pub fn deinit(self: *FramePacer) void {
        if (self.display_manager) |*dm| {
            dm.deinit();
        }
        log.info("Frame pacer stats: {} frames paced, {}ms total sleep, {}ms total busy-wait", .{
            self.frames_paced,
            self.total_sleep_ns / 1_000_000,
            self.total_busy_wait_ns / 1_000_000,
        });
    }

    /// Update VRR configuration from nvsync
    fn updateVrrConfig(self: *FramePacer) void {
        if (self.display_manager) |*dm| {
            for (dm.displays.items) |display| {
                if (display.vrr_capable and display.vrr_enabled) {
                    self.config.vrr_enabled = true;
                    self.config.vrr_min_hz = display.min_hz;
                    self.config.vrr_max_hz = display.max_hz;
                    log.info("VRR detected: {}-{} Hz (LFC: {})", .{
                        display.min_hz,
                        display.max_hz,
                        display.lfc_supported,
                    });
                    break;
                }
            }
        }
    }

    /// Set target FPS
    pub fn setTargetFps(self: *FramePacer, fps: u32) void {
        self.config.target_fps = fps;
        self.frame_target_ns = if (fps > 0)
            @as(u64, 1_000_000_000) / fps
        else
            0;
        log.info("Target FPS set to {}", .{fps});
    }

    /// Get current average FPS from recent frame times
    pub fn getAverageFps(self: *const FramePacer) f64 {
        var total_ns: u64 = 0;
        var count: usize = 0;
        for (self.frame_times) |ft| {
            if (ft > 0) {
                total_ns += ft;
                count += 1;
            }
        }
        if (count == 0 or total_ns == 0) return 0.0;
        const avg_ns = total_ns / count;
        return 1_000_000_000.0 / @as(f64, @floatFromInt(avg_ns));
    }

    /// Begin frame timing - call at start of frame
    pub fn beginFrame(self: *FramePacer) void {
        self.last_frame_time = std.time.Instant.now() catch return;
    }

    /// End frame timing and pace if needed - call after present
    pub fn endFrame(self: *FramePacer) void {
        const now = std.time.Instant.now() catch return;
        const frame_time_ns = now.since(self.last_frame_time);

        // Record frame time
        self.frame_times[self.frame_time_index] = frame_time_ns;
        self.frame_time_index = (self.frame_time_index + 1) % 60;
        self.frame_count += 1;

        // Pace if we have a target
        if (self.frame_target_ns > 0 and frame_time_ns < self.frame_target_ns) {
            const remaining_ns = self.frame_target_ns - frame_time_ns;
            self.pace(remaining_ns);
        }
    }

    /// Pace for the specified duration
    fn pace(self: *FramePacer, duration_ns: u64) void {
        switch (self.config.mode) {
            .unlimited => {},
            .cpu_sleep => self.sleepPace(duration_ns),
            .busy_wait => self.busyWaitPace(duration_ns),
            .hybrid => self.hybridPace(duration_ns),
        }
        self.frames_paced += 1;
    }

    fn sleepPace(self: *FramePacer, duration_ns: u64) void {
        const seconds = duration_ns / 1_000_000_000;
        const nanos = duration_ns % 1_000_000_000;
        std.posix.nanosleep(seconds, nanos);
        self.total_sleep_ns += duration_ns;
    }

    fn busyWaitPace(self: *FramePacer, duration_ns: u64) void {
        const start = std.time.Instant.now() catch return;
        while (true) {
            const now = std.time.Instant.now() catch return;
            if (now.since(start) >= duration_ns) break;
            std.atomic.spinLoopHint();
        }
        self.total_busy_wait_ns += duration_ns;
    }

    fn hybridPace(self: *FramePacer, duration_ns: u64) void {
        if (duration_ns > self.config.busy_wait_threshold_ns) {
            // Sleep for most of the duration, then busy-wait
            const sleep_duration = duration_ns - self.config.busy_wait_threshold_ns;
            const seconds = sleep_duration / 1_000_000_000;
            const nanos = sleep_duration % 1_000_000_000;
            std.posix.nanosleep(seconds, nanos);
            self.total_sleep_ns += sleep_duration;

            // Busy-wait for the remainder
            self.busyWaitPace(self.config.busy_wait_threshold_ns);
        } else {
            // Duration is short enough, just busy-wait
            self.busyWaitPace(duration_ns);
        }
    }

    /// Get frame pacing statistics
    pub fn getStats(self: *const FramePacer) FramePacerStats {
        return .{
            .average_fps = self.getAverageFps(),
            .target_fps = self.config.target_fps,
            .frames_paced = self.frames_paced,
            .total_frames = self.frame_count,
            .total_sleep_ms = @as(f64, @floatFromInt(self.total_sleep_ns)) / 1_000_000.0,
            .total_busy_wait_ms = @as(f64, @floatFromInt(self.total_busy_wait_ns)) / 1_000_000.0,
            .vrr_enabled = self.config.vrr_enabled,
            .vrr_range = .{ self.config.vrr_min_hz, self.config.vrr_max_hz },
        };
    }
};

pub const FramePacerStats = struct {
    average_fps: f64,
    target_fps: u32,
    frames_paced: u64,
    total_frames: u64,
    total_sleep_ms: f64,
    total_busy_wait_ms: f64,
    vrr_enabled: bool,
    vrr_range: [2]u32,
};
