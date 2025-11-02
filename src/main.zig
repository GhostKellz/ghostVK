const std = @import("std");
const ghostVK = @import("ghostVK");

const log = std.log.scoped(.app);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var ctx = try ghostVK.GhostVK.init(gpa.allocator(), .{});
    defer ctx.deinit();

    log.info("GhostVK runtime initialized. Press Ctrl+C to exit.", .{});
}
