//! GhostVK Frame Pacer
//! Precise frame timing using nvsync for VRR awareness.
//!
//! Features:
//! - Target FPS limiting with sub-millisecond precision
//! - VRR-aware timing (adjusts to display refresh range)
//! - Low Framerate Compensation (LFC) handling
//! - CPU/GPU frame pacing modes
//! - VK_EXT_present_timing feedback (NVIDIA 595+)

const std = @import("std");
const nvsync = @import("nvsync");

const log = std.log.scoped(.frame_pacer);

/// Get monotonic time in nanoseconds using Linux clock_gettime
fn getMonotonicNs() i128 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

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
    /// Adaptive: uses VK_EXT_present_timing feedback for optimal pacing
    adaptive,
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
    /// Display refresh duration in nanoseconds (from VK_EXT_present_timing)
    display_refresh_ns: u64 = 0,
    /// VK_EXT_present_timing available
    present_timing_available: bool = false,
    /// Display in VRR mode (from VK_EXT_present_timing)
    present_timing_vrr: bool = false,
};

pub const FramePacer = struct {
    config: FramePacerConfig,
    allocator: std.mem.Allocator,

    // Timing state
    frame_target_ns: u64,
    last_frame_time: i128, // Monotonic timestamp in nanoseconds
    frame_times: [60]u64, // Ring buffer of last 60 frame times
    frame_time_index: usize,
    frame_count: u64,
    frame_time_sum: u64, // Rolling sum of valid frame times for O(1) average
    valid_frame_count: usize, // Count of valid (>0) frame times in ring buffer

    // Statistics
    total_sleep_ns: u64,
    total_busy_wait_ns: u64,
    frames_paced: u64,

    // nvsync display manager for VRR detection
    display_manager: ?nvsync.DisplayManager,

    // VK_EXT_present_timing feedback
    present_timing_deltas: [16]i64, // Signed deltas between target and actual present
    present_timing_index: usize,
    adaptive_offset_ns: i64, // Adjustment based on present timing feedback

    pub fn init(allocator: std.mem.Allocator, config: FramePacerConfig) FramePacer {
        var pacer = FramePacer{
            .config = config,
            .allocator = allocator,
            .frame_target_ns = if (config.target_fps > 0)
                @as(u64, 1_000_000_000) / config.target_fps
            else
                0,
            .last_frame_time = getMonotonicNs(),
            .frame_times = [_]u64{0} ** 60,
            .frame_time_index = 0,
            .frame_count = 0,
            .frame_time_sum = 0,
            .valid_frame_count = 0,
            .total_sleep_ns = 0,
            .total_busy_wait_ns = 0,
            .frames_paced = 0,
            .display_manager = null,
            .present_timing_deltas = [_]i64{0} ** 16,
            .present_timing_index = 0,
            .adaptive_offset_ns = 0,
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

        // Auto-select adaptive mode if present timing is available and not explicitly set
        if (config.present_timing_available and config.mode == .hybrid) {
            pacer.config.mode = .adaptive;
            log.info("VK_EXT_present_timing available, using adaptive pacing mode", .{});
        }

        log.info("Frame pacer initialized: target={} FPS, mode={s}, VRR={}, present_timing={}", .{
            config.target_fps,
            @tagName(pacer.config.mode),
            pacer.config.vrr_enabled,
            pacer.config.present_timing_available,
        });

        if (config.present_timing_available and config.display_refresh_ns > 0) {
            const refresh_hz: f64 = 1_000_000_000.0 / @as(f64, @floatFromInt(config.display_refresh_ns));
            log.info("Display refresh: {d:.2} Hz ({} ns), VRR mode: {}", .{
                refresh_hz,
                config.display_refresh_ns,
                config.present_timing_vrr,
            });
        }

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
        if (self.config.present_timing_available) {
            log.info("Present timing adaptive offset: {} ns", .{self.adaptive_offset_ns});
        }
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

    /// Update present timing configuration (called when swapchain timing properties change)
    pub fn updatePresentTiming(self: *FramePacer, refresh_ns: u64, vrr_mode: bool) void {
        self.config.display_refresh_ns = refresh_ns;
        self.config.present_timing_vrr = vrr_mode;
        self.config.present_timing_available = (refresh_ns > 0);

        if (refresh_ns > 0) {
            const refresh_hz: f64 = 1_000_000_000.0 / @as(f64, @floatFromInt(refresh_ns));
            log.info("Present timing updated: {d:.2} Hz, VRR: {}", .{ refresh_hz, vrr_mode });

            // If VRR mode and no target set, auto-configure based on max refresh
            if (vrr_mode and self.config.target_fps == 0 and self.config.vrr_max_hz > 0) {
                // In VRR mode with present timing, we can be more aggressive
                self.config.mode = .adaptive;
            }
        }
    }

    /// Record actual present time delta for adaptive pacing
    /// delta_ns: difference between when we wanted to present vs when it actually happened
    pub fn recordPresentTimingFeedback(self: *FramePacer, actual_present_ns: u64, target_present_ns: u64) void {
        if (!self.config.present_timing_available) return;

        const delta: i64 = @as(i64, @intCast(actual_present_ns)) - @as(i64, @intCast(target_present_ns));
        self.present_timing_deltas[self.present_timing_index] = delta;
        self.present_timing_index = (self.present_timing_index + 1) % self.present_timing_deltas.len;

        // Update adaptive offset based on recent deltas
        self.updateAdaptiveOffset();
    }

    /// Update adaptive offset based on accumulated present timing deltas
    fn updateAdaptiveOffset(self: *FramePacer) void {
        var total: i64 = 0;
        var count: i64 = 0;

        for (self.present_timing_deltas) |delta| {
            if (delta != 0) {
                total += delta;
                count += 1;
            }
        }

        if (count > 4) {
            // Use weighted average to adjust pacing
            // If we're consistently presenting late (positive delta), sleep less
            // If we're consistently presenting early (negative delta), sleep more
            const avg_delta = @divTrunc(total, count);

            // Smooth the adjustment (don't overcorrect)
            const adjustment = @divTrunc(avg_delta, 4);
            self.adaptive_offset_ns = std.math.clamp(
                self.adaptive_offset_ns - adjustment,
                -500_000, // Max 0.5ms early
                500_000, // Max 0.5ms late adjustment
            );
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

    /// Get current average FPS from recent frame times (O(1) using rolling sum)
    pub fn getAverageFps(self: *const FramePacer) f64 {
        if (self.valid_frame_count == 0 or self.frame_time_sum == 0) return 0.0;
        const avg_ns = self.frame_time_sum / self.valid_frame_count;
        return 1_000_000_000.0 / @as(f64, @floatFromInt(avg_ns));
    }

    /// Begin frame timing - call at start of frame
    pub fn beginFrame(self: *FramePacer) void {
        self.last_frame_time = getMonotonicNs();
    }

    /// End frame timing and pace if needed - call after present
    pub fn endFrame(self: *FramePacer) void {
        const now = getMonotonicNs();
        const frame_time_ns: u64 = @intCast(now - self.last_frame_time);

        // Update rolling sum: subtract old value, add new value
        const old_value = self.frame_times[self.frame_time_index];
        if (old_value > 0) {
            self.frame_time_sum -= old_value;
            self.valid_frame_count -= 1;
        }
        self.frame_time_sum += frame_time_ns;
        self.valid_frame_count += 1;

        // Record frame time in ring buffer
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
            .adaptive => self.adaptivePace(duration_ns),
        }
        self.frames_paced += 1;
    }

    fn sleepPace(self: *FramePacer, duration_ns: u64) void {
        const seconds = duration_ns / 1_000_000_000;
        const nanos = duration_ns % 1_000_000_000;
        const ts = std.c.timespec{
            .sec = @intCast(seconds),
            .nsec = @intCast(nanos),
        };
        _ = std.c.nanosleep(&ts, null);
        self.total_sleep_ns += duration_ns;
    }

    fn busyWaitPace(self: *FramePacer, duration_ns: u64) void {
        const start = getMonotonicNs();
        while (true) {
            const now = getMonotonicNs();
            const elapsed: u64 = @intCast(now - start);
            if (elapsed >= duration_ns) break;
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
            const ts = std.c.timespec{
                .sec = @intCast(seconds),
                .nsec = @intCast(nanos),
            };
            _ = std.c.nanosleep(&ts, null);
            self.total_sleep_ns += sleep_duration;

            // Busy-wait for the remainder
            self.busyWaitPace(self.config.busy_wait_threshold_ns);
        } else {
            // Duration is short enough, just busy-wait
            self.busyWaitPace(duration_ns);
        }
    }

    /// Adaptive pacing using VK_EXT_present_timing feedback
    fn adaptivePace(self: *FramePacer, duration_ns: u64) void {
        // Apply adaptive offset based on present timing feedback
        const adjusted_duration: i64 = @as(i64, @intCast(duration_ns)) + self.adaptive_offset_ns;

        if (adjusted_duration <= 0) {
            // No sleep needed (we're behind schedule)
            return;
        }

        const actual_duration: u64 = @intCast(adjusted_duration);

        // Use hybrid approach with adaptive offset
        if (actual_duration > self.config.busy_wait_threshold_ns) {
            const sleep_duration = actual_duration - self.config.busy_wait_threshold_ns;
            const seconds = sleep_duration / 1_000_000_000;
            const nanos = sleep_duration % 1_000_000_000;
            const ts = std.c.timespec{
                .sec = @intCast(seconds),
                .nsec = @intCast(nanos),
            };
            _ = std.c.nanosleep(&ts, null);
            self.total_sleep_ns += sleep_duration;

            self.busyWaitPace(self.config.busy_wait_threshold_ns);
        } else {
            self.busyWaitPace(actual_duration);
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
            .present_timing_available = self.config.present_timing_available,
            .adaptive_offset_ns = self.adaptive_offset_ns,
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
    /// VK_EXT_present_timing available (NVIDIA 595+)
    present_timing_available: bool,
    /// Adaptive offset based on present timing feedback (ns)
    adaptive_offset_ns: i64,
};
