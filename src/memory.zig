//! GhostVK Memory Allocator - VMA-style GPU memory management
//!
//! High-performance memory allocator for Vulkan with:
//! - Automatic memory type selection
//! - Memory pools for reduced allocation overhead
//! - Sub-allocation within large memory blocks
//! - Dedicated allocations for large resources
//! - Staging buffer management for uploads
//! - Memory budget tracking

const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.ghostvk_memory);

/// Memory usage hints for automatic type selection
pub const MemoryUsage = enum {
    /// GPU-only memory, fastest for GPU access (DEVICE_LOCAL)
    gpu_only,
    /// CPU-to-GPU uploads, mappable (HOST_VISIBLE | HOST_COHERENT)
    cpu_to_gpu,
    /// GPU-to-CPU readback (HOST_VISIBLE | HOST_CACHED)
    gpu_to_cpu,
    /// CPU-only staging (HOST_VISIBLE | HOST_COHERENT)
    cpu_only,
    /// Let allocator decide based on resource requirements
    auto,
};

/// Allocation flags for fine-grained control
pub const AllocationFlags = packed struct {
    /// Create dedicated allocation (bypass sub-allocation)
    dedicated: bool = false,
    /// Memory can be mapped for CPU access
    mapped: bool = false,
    /// Prefer device local even if slower
    prefer_device: bool = false,
    /// Allow memory to be in host memory if device is full
    allow_host_fallback: bool = true,
    _padding: u4 = 0,
};

/// Memory block for sub-allocation
const MemoryBlock = struct {
    memory: vk.types.VkDeviceMemory,
    size: vk.types.VkDeviceSize,
    memory_type_index: u32,
    mapped_ptr: ?[*]u8 = null,
    allocations: std.ArrayList(SubAllocation),
    free_size: vk.types.VkDeviceSize,

    const SubAllocation = struct {
        offset: vk.types.VkDeviceSize,
        size: vk.types.VkDeviceSize,
        alignment: vk.types.VkDeviceSize,
        in_use: bool,
        id: u64,
    };

    fn init(allocator: std.mem.Allocator, memory: vk.types.VkDeviceMemory, size: vk.types.VkDeviceSize, memory_type_index: u32) MemoryBlock {
        return .{
            .memory = memory,
            .size = size,
            .memory_type_index = memory_type_index,
            .allocations = std.ArrayList(SubAllocation).init(allocator),
            .free_size = size,
        };
    }

    fn deinit(self: *MemoryBlock) void {
        self.allocations.deinit();
    }

    /// Try to sub-allocate from this block
    fn allocate(self: *MemoryBlock, size: vk.types.VkDeviceSize, alignment: vk.types.VkDeviceSize, id: u64) ?vk.types.VkDeviceSize {
        if (self.free_size < size) return null;

        // First-fit allocation with alignment
        var offset: vk.types.VkDeviceSize = 0;

        // Sort allocations by offset for proper gap finding
        for (self.allocations.items) |alloc| {
            if (!alloc.in_use) continue;

            // Align the current offset
            const aligned_offset = alignUp(offset, alignment);

            // Check if there's a gap before this allocation
            if (aligned_offset + size <= alloc.offset) {
                // Found a gap!
                self.allocations.append(.{
                    .offset = aligned_offset,
                    .size = size,
                    .alignment = alignment,
                    .in_use = true,
                    .id = id,
                }) catch return null;
                self.free_size -= size;
                return aligned_offset;
            }

            // Move past this allocation
            offset = alloc.offset + alloc.size;
        }

        // Try at the end
        const aligned_offset = alignUp(offset, alignment);
        if (aligned_offset + size <= self.size) {
            self.allocations.append(.{
                .offset = aligned_offset,
                .size = size,
                .alignment = alignment,
                .in_use = true,
                .id = id,
            }) catch return null;
            self.free_size -= size;
            return aligned_offset;
        }

        return null;
    }

    /// Free a sub-allocation by ID
    fn free(self: *MemoryBlock, id: u64) bool {
        for (self.allocations.items) |*alloc| {
            if (alloc.id == id and alloc.in_use) {
                alloc.in_use = false;
                self.free_size += alloc.size;
                return true;
            }
        }
        return false;
    }
};

/// Memory pool for a specific memory type
const MemoryPool = struct {
    memory_type_index: u32,
    blocks: std.ArrayList(MemoryBlock),
    block_size: vk.types.VkDeviceSize,
    total_allocated: vk.types.VkDeviceSize,
    total_used: vk.types.VkDeviceSize,

    fn init(allocator: std.mem.Allocator, memory_type_index: u32, block_size: vk.types.VkDeviceSize) MemoryPool {
        return .{
            .memory_type_index = memory_type_index,
            .blocks = std.ArrayList(MemoryBlock).init(allocator),
            .block_size = block_size,
            .total_allocated = 0,
            .total_used = 0,
        };
    }

    fn deinit(self: *MemoryPool) void {
        for (self.blocks.items) |*block| {
            block.deinit();
        }
        self.blocks.deinit();
    }
};

/// Allocation result
pub const Allocation = struct {
    memory: vk.types.VkDeviceMemory,
    offset: vk.types.VkDeviceSize,
    size: vk.types.VkDeviceSize,
    mapped_ptr: ?[*]u8,
    id: u64,
    is_dedicated: bool,
    memory_type_index: u32,
};

/// Buffer with allocation
pub const Buffer = struct {
    buffer: vk.types.VkBuffer,
    allocation: Allocation,
    size: vk.types.VkDeviceSize,
    usage: vk.types.VkBufferUsageFlags,
};

/// Image with allocation
pub const Image = struct {
    image: vk.types.VkImage,
    allocation: Allocation,
    extent: vk.types.VkExtent3D,
    format: vk.types.VkFormat,
    usage: vk.types.VkImageUsageFlags,
};

/// Memory statistics
pub const MemoryStats = struct {
    total_allocated: vk.types.VkDeviceSize,
    total_used: vk.types.VkDeviceSize,
    allocation_count: u64,
    block_count: u64,
    dedicated_count: u64,

    pub fn utilizationPercent(self: MemoryStats) f32 {
        if (self.total_allocated == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_used)) / @as(f32, @floatFromInt(self.total_allocated)) * 100.0;
    }
};

/// GhostVK Memory Allocator
pub const MemoryAllocator = struct {
    allocator: std.mem.Allocator,
    device: vk.types.VkDevice,
    physical_device: vk.types.VkPhysicalDevice,
    device_dispatch: *const vk.loader.DeviceDispatch,
    instance_dispatch: *const vk.loader.InstanceDispatch,

    // Memory properties
    memory_properties: vk.types.VkPhysicalDeviceMemoryProperties,

    // Memory pools per memory type
    pools: [32]?MemoryPool,

    // Dedicated allocations (large resources)
    dedicated_allocations: std.ArrayList(DedicatedAllocation),

    // Configuration
    config: Config,

    // Statistics
    next_allocation_id: u64,
    stats: MemoryStats,

    const DedicatedAllocation = struct {
        memory: vk.types.VkDeviceMemory,
        size: vk.types.VkDeviceSize,
        memory_type_index: u32,
        id: u64,
        mapped_ptr: ?[*]u8,
    };

    pub const Config = struct {
        /// Default block size for pools (256MB)
        default_block_size: vk.types.VkDeviceSize = 256 * 1024 * 1024,
        /// Minimum allocation size for dedicated allocation (64MB)
        dedicated_allocation_threshold: vk.types.VkDeviceSize = 64 * 1024 * 1024,
        /// Maximum number of blocks per pool
        max_blocks_per_pool: u32 = 16,
        /// Enable memory budget tracking
        track_budget: bool = true,
    };

    pub const Error = error{
        OutOfDeviceMemory,
        OutOfHostMemory,
        NoSuitableMemoryType,
        AllocationFailed,
        MappingFailed,
        InvalidAllocation,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        device: vk.types.VkDevice,
        physical_device: vk.types.VkPhysicalDevice,
        device_dispatch: *const vk.loader.DeviceDispatch,
        instance_dispatch: *const vk.loader.InstanceDispatch,
        config: Config,
    ) MemoryAllocator {
        var self = MemoryAllocator{
            .allocator = allocator,
            .device = device,
            .physical_device = physical_device,
            .device_dispatch = device_dispatch,
            .instance_dispatch = instance_dispatch,
            .memory_properties = undefined,
            .pools = [_]?MemoryPool{null} ** 32,
            .dedicated_allocations = std.ArrayList(DedicatedAllocation).init(allocator),
            .config = config,
            .next_allocation_id = 1,
            .stats = .{
                .total_allocated = 0,
                .total_used = 0,
                .allocation_count = 0,
                .block_count = 0,
                .dedicated_count = 0,
            },
        };

        // Query memory properties
        instance_dispatch.get_physical_device_memory_properties(physical_device, &self.memory_properties);

        log.info("Memory allocator initialized: {} memory types, {} heaps", .{
            self.memory_properties.memoryTypeCount,
            self.memory_properties.memoryHeapCount,
        });

        // Log memory heaps
        for (0..self.memory_properties.memoryHeapCount) |i| {
            const heap = self.memory_properties.memoryHeaps[i];
            const is_device_local = (heap.flags & vk.types.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            log.info("  Heap {}: {} MB {s}", .{
                i,
                heap.size / (1024 * 1024),
                if (is_device_local) "(device local)" else "(host)",
            });
        }

        return self;
    }

    pub fn deinit(self: *MemoryAllocator) void {
        // Free all dedicated allocations
        for (self.dedicated_allocations.items) |alloc| {
            self.device_dispatch.free_memory(self.device, alloc.memory, null);
        }
        self.dedicated_allocations.deinit();

        // Free all pool blocks
        for (&self.pools) |*pool_opt| {
            if (pool_opt.*) |*pool| {
                for (pool.blocks.items) |block| {
                    if (block.mapped_ptr != null) {
                        self.device_dispatch.unmap_memory(self.device, block.memory);
                    }
                    self.device_dispatch.free_memory(self.device, block.memory, null);
                }
                pool.deinit();
                pool_opt.* = null;
            }
        }

        log.info("Memory allocator destroyed: {} total allocations", .{self.stats.allocation_count});
    }

    /// Allocate memory for a buffer
    pub fn createBuffer(
        self: *MemoryAllocator,
        size: vk.types.VkDeviceSize,
        usage: vk.types.VkBufferUsageFlags,
        memory_usage: MemoryUsage,
        flags: AllocationFlags,
    ) Error!Buffer {
        // Create buffer
        const buffer_info = vk.types.VkBufferCreateInfo{
            .sType = .BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = size,
            .usage = usage,
            .sharingMode = .EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        var buffer: vk.types.VkBuffer = undefined;
        const result = self.device_dispatch.create_buffer(self.device, &buffer_info, null, &buffer);
        if (result != .SUCCESS) {
            log.err("Failed to create buffer: {s}", .{@tagName(result)});
            return Error.AllocationFailed;
        }
        errdefer self.device_dispatch.destroy_buffer(self.device, buffer, null);

        // Get memory requirements
        var mem_requirements: vk.types.VkMemoryRequirements = undefined;
        self.device_dispatch.get_buffer_memory_requirements(self.device, buffer, &mem_requirements);

        // Allocate memory
        const allocation = try self.allocate(mem_requirements, memory_usage, flags);
        errdefer self.free(allocation) catch {};

        // Bind memory
        const bind_result = self.device_dispatch.bind_buffer_memory(
            self.device,
            buffer,
            allocation.memory,
            allocation.offset,
        );
        if (bind_result != .SUCCESS) {
            log.err("Failed to bind buffer memory: {s}", .{@tagName(bind_result)});
            return Error.AllocationFailed;
        }

        return Buffer{
            .buffer = buffer,
            .allocation = allocation,
            .size = size,
            .usage = usage,
        };
    }

    /// Destroy a buffer and free its memory
    pub fn destroyBuffer(self: *MemoryAllocator, buffer: *Buffer) void {
        self.device_dispatch.destroy_buffer(self.device, buffer.buffer, null);
        self.free(buffer.allocation) catch {};
        buffer.* = undefined;
    }

    /// Allocate memory for an image
    pub fn createImage(
        self: *MemoryAllocator,
        extent: vk.types.VkExtent3D,
        format: vk.types.VkFormat,
        usage: vk.types.VkImageUsageFlags,
        memory_usage: MemoryUsage,
        flags: AllocationFlags,
    ) Error!Image {
        // Create image
        const image_info = vk.types.VkImageCreateInfo{
            .sType = .IMAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .imageType = if (extent.depth > 1) .@"3D" else if (extent.height > 1) .@"2D" else .@"1D",
            .format = format,
            .extent = extent,
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = .@"1",
            .tiling = .OPTIMAL,
            .usage = usage,
            .sharingMode = .EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .initialLayout = .UNDEFINED,
        };

        var image: vk.types.VkImage = undefined;
        const result = self.device_dispatch.create_image(self.device, &image_info, null, &image);
        if (result != .SUCCESS) {
            log.err("Failed to create image: {s}", .{@tagName(result)});
            return Error.AllocationFailed;
        }
        errdefer self.device_dispatch.destroy_image(self.device, image, null);

        // Get memory requirements
        var mem_requirements: vk.types.VkMemoryRequirements = undefined;
        self.device_dispatch.get_image_memory_requirements(self.device, image, &mem_requirements);

        // Images typically need dedicated allocations
        var alloc_flags = flags;
        if (mem_requirements.size >= self.config.dedicated_allocation_threshold) {
            alloc_flags.dedicated = true;
        }

        // Allocate memory
        const allocation = try self.allocate(mem_requirements, memory_usage, alloc_flags);
        errdefer self.free(allocation) catch {};

        // Bind memory
        const bind_result = self.device_dispatch.bind_image_memory(
            self.device,
            image,
            allocation.memory,
            allocation.offset,
        );
        if (bind_result != .SUCCESS) {
            log.err("Failed to bind image memory: {s}", .{@tagName(bind_result)});
            return Error.AllocationFailed;
        }

        return Image{
            .image = image,
            .allocation = allocation,
            .extent = extent,
            .format = format,
            .usage = usage,
        };
    }

    /// Destroy an image and free its memory
    pub fn destroyImage(self: *MemoryAllocator, image: *Image) void {
        self.device_dispatch.destroy_image(self.device, image.image, null);
        self.free(image.allocation) catch {};
        image.* = undefined;
    }

    /// Create a staging buffer for uploads
    pub fn createStagingBuffer(self: *MemoryAllocator, size: vk.types.VkDeviceSize) Error!Buffer {
        return self.createBuffer(
            size,
            vk.types.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .cpu_only,
            .{ .mapped = true },
        );
    }

    /// Map memory for CPU access
    pub fn mapMemory(self: *MemoryAllocator, allocation: Allocation) Error!?[*]u8 {
        if (allocation.mapped_ptr) |ptr| {
            return ptr;
        }

        if (allocation.is_dedicated) {
            var mapped: ?*anyopaque = null;
            const result = self.device_dispatch.map_memory(
                self.device,
                allocation.memory,
                allocation.offset,
                allocation.size,
                0,
                &mapped,
            );
            if (result != .SUCCESS) {
                return Error.MappingFailed;
            }
            return @ptrCast(mapped);
        }

        // For pooled allocations, the block should already be mapped
        return null;
    }

    /// Flush mapped memory range (for non-coherent memory)
    pub fn flushMemory(self: *MemoryAllocator, allocation: Allocation, offset: vk.types.VkDeviceSize, size: vk.types.VkDeviceSize) void {
        const range = vk.types.VkMappedMemoryRange{
            .sType = .MAPPED_MEMORY_RANGE,
            .pNext = null,
            .memory = allocation.memory,
            .offset = allocation.offset + offset,
            .size = size,
        };
        _ = self.device_dispatch.flush_mapped_memory_ranges(self.device, 1, &range);
    }

    /// Raw memory allocation
    pub fn allocate(
        self: *MemoryAllocator,
        requirements: vk.types.VkMemoryRequirements,
        usage: MemoryUsage,
        flags: AllocationFlags,
    ) Error!Allocation {
        // Find suitable memory type
        const memory_type_index = try self.findMemoryType(requirements.memoryTypeBits, usage, flags);

        // Use dedicated allocation for large allocations or if requested
        if (flags.dedicated or requirements.size >= self.config.dedicated_allocation_threshold) {
            return self.allocateDedicated(requirements.size, memory_type_index, flags.mapped);
        }

        // Try to sub-allocate from existing pool
        return self.allocateFromPool(requirements.size, requirements.alignment, memory_type_index, flags.mapped);
    }

    /// Free an allocation
    pub fn free(self: *MemoryAllocator, allocation: Allocation) Error!void {
        if (allocation.is_dedicated) {
            // Find and remove dedicated allocation
            for (self.dedicated_allocations.items, 0..) |alloc, i| {
                if (alloc.id == allocation.id) {
                    if (alloc.mapped_ptr != null) {
                        self.device_dispatch.unmap_memory(self.device, alloc.memory);
                    }
                    self.device_dispatch.free_memory(self.device, alloc.memory, null);
                    _ = self.dedicated_allocations.swapRemove(i);

                    self.stats.total_allocated -= allocation.size;
                    self.stats.total_used -= allocation.size;
                    self.stats.dedicated_count -= 1;
                    return;
                }
            }
            return Error.InvalidAllocation;
        }

        // Free from pool
        if (self.pools[allocation.memory_type_index]) |*pool| {
            for (pool.blocks.items) |*block| {
                if (block.free(allocation.id)) {
                    self.stats.total_used -= allocation.size;
                    pool.total_used -= allocation.size;
                    return;
                }
            }
        }

        return Error.InvalidAllocation;
    }

    /// Get memory statistics
    pub fn getStats(self: *const MemoryAllocator) MemoryStats {
        return self.stats;
    }

    /// Print memory statistics
    pub fn printStats(self: *const MemoryAllocator) void {
        const stats = self.getStats();
        log.info("Memory Statistics:", .{});
        log.info("  Total Allocated: {} MB", .{stats.total_allocated / (1024 * 1024)});
        log.info("  Total Used:      {} MB", .{stats.total_used / (1024 * 1024)});
        log.info("  Utilization:     {d:.1}%", .{stats.utilizationPercent()});
        log.info("  Allocations:     {}", .{stats.allocation_count});
        log.info("  Blocks:          {}", .{stats.block_count});
        log.info("  Dedicated:       {}", .{stats.dedicated_count});
    }

    // Internal functions

    fn findMemoryType(self: *MemoryAllocator, type_filter: u32, usage: MemoryUsage, flags: AllocationFlags) Error!u32 {
        const required_properties: vk.types.VkMemoryPropertyFlags = switch (usage) {
            .gpu_only => vk.types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .cpu_to_gpu => vk.types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            .gpu_to_cpu => vk.types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.types.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            .cpu_only => vk.types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            .auto => 0,
        };

        const preferred_properties: vk.types.VkMemoryPropertyFlags = switch (usage) {
            .gpu_only => vk.types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .cpu_to_gpu => vk.types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | vk.types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            .gpu_to_cpu => vk.types.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            .cpu_only => 0,
            .auto => if (flags.prefer_device) vk.types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT else 0,
        };

        // First pass: try to find preferred type
        for (0..self.memory_properties.memoryTypeCount) |i| {
            const idx: u5 = @intCast(i);
            if ((type_filter & (@as(u32, 1) << idx)) != 0) {
                const props = self.memory_properties.memoryTypes[i].propertyFlags;
                if ((props & required_properties) == required_properties and
                    (props & preferred_properties) == preferred_properties)
                {
                    return @intCast(i);
                }
            }
        }

        // Second pass: just find required type
        for (0..self.memory_properties.memoryTypeCount) |i| {
            const idx: u5 = @intCast(i);
            if ((type_filter & (@as(u32, 1) << idx)) != 0) {
                const props = self.memory_properties.memoryTypes[i].propertyFlags;
                if ((props & required_properties) == required_properties) {
                    return @intCast(i);
                }
            }
        }

        // Fallback for auto usage
        if (usage == .auto) {
            for (0..self.memory_properties.memoryTypeCount) |i| {
                const idx: u5 = @intCast(i);
                if ((type_filter & (@as(u32, 1) << idx)) != 0) {
                    return @intCast(i);
                }
            }
        }

        return Error.NoSuitableMemoryType;
    }

    fn allocateDedicated(self: *MemoryAllocator, size: vk.types.VkDeviceSize, memory_type_index: u32, map: bool) Error!Allocation {
        const alloc_info = vk.types.VkMemoryAllocateInfo{
            .sType = .MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = size,
            .memoryTypeIndex = memory_type_index,
        };

        var memory: vk.types.VkDeviceMemory = undefined;
        const result = self.device_dispatch.allocate_memory(self.device, &alloc_info, null, &memory);
        if (result == .ERROR_OUT_OF_DEVICE_MEMORY) {
            return Error.OutOfDeviceMemory;
        } else if (result == .ERROR_OUT_OF_HOST_MEMORY) {
            return Error.OutOfHostMemory;
        } else if (result != .SUCCESS) {
            return Error.AllocationFailed;
        }

        var mapped_ptr: ?[*]u8 = null;
        if (map) {
            var mapped: ?*anyopaque = null;
            const map_result = self.device_dispatch.map_memory(self.device, memory, 0, size, 0, &mapped);
            if (map_result == .SUCCESS) {
                mapped_ptr = @ptrCast(mapped);
            }
        }

        const id = self.next_allocation_id;
        self.next_allocation_id += 1;

        self.dedicated_allocations.append(.{
            .memory = memory,
            .size = size,
            .memory_type_index = memory_type_index,
            .id = id,
            .mapped_ptr = mapped_ptr,
        }) catch {
            self.device_dispatch.free_memory(self.device, memory, null);
            return Error.AllocationFailed;
        };

        self.stats.total_allocated += size;
        self.stats.total_used += size;
        self.stats.allocation_count += 1;
        self.stats.dedicated_count += 1;

        log.debug("Dedicated allocation: {} KB, type {}", .{ size / 1024, memory_type_index });

        return Allocation{
            .memory = memory,
            .offset = 0,
            .size = size,
            .mapped_ptr = mapped_ptr,
            .id = id,
            .is_dedicated = true,
            .memory_type_index = memory_type_index,
        };
    }

    fn allocateFromPool(self: *MemoryAllocator, size: vk.types.VkDeviceSize, alignment: vk.types.VkDeviceSize, memory_type_index: u32, map: bool) Error!Allocation {
        // Get or create pool
        if (self.pools[memory_type_index] == null) {
            self.pools[memory_type_index] = MemoryPool.init(self.allocator, memory_type_index, self.config.default_block_size);
        }
        var pool = &self.pools[memory_type_index].?;

        // Try existing blocks
        for (pool.blocks.items) |*block| {
            if (block.allocate(size, alignment, self.next_allocation_id)) |offset| {
                const id = self.next_allocation_id;
                self.next_allocation_id += 1;

                self.stats.total_used += size;
                self.stats.allocation_count += 1;
                pool.total_used += size;

                var mapped_ptr: ?[*]u8 = null;
                if (map and block.mapped_ptr != null) {
                    mapped_ptr = block.mapped_ptr.? + offset;
                }

                return Allocation{
                    .memory = block.memory,
                    .offset = offset,
                    .size = size,
                    .mapped_ptr = mapped_ptr,
                    .id = id,
                    .is_dedicated = false,
                    .memory_type_index = memory_type_index,
                };
            }
        }

        // Need new block
        if (pool.blocks.items.len >= self.config.max_blocks_per_pool) {
            // Fall back to dedicated allocation
            return self.allocateDedicated(size, memory_type_index, map);
        }

        const block_size = @max(self.config.default_block_size, size);
        const new_block = try self.createBlock(memory_type_index, block_size, map);
        pool.blocks.append(new_block) catch return Error.AllocationFailed;
        pool.total_allocated += block_size;
        self.stats.total_allocated += block_size;
        self.stats.block_count += 1;

        // Allocate from new block
        var block = &pool.blocks.items[pool.blocks.items.len - 1];
        if (block.allocate(size, alignment, self.next_allocation_id)) |offset| {
            const id = self.next_allocation_id;
            self.next_allocation_id += 1;

            self.stats.total_used += size;
            self.stats.allocation_count += 1;
            pool.total_used += size;

            var mapped_ptr: ?[*]u8 = null;
            if (map and block.mapped_ptr != null) {
                mapped_ptr = block.mapped_ptr.? + offset;
            }

            return Allocation{
                .memory = block.memory,
                .offset = offset,
                .size = size,
                .mapped_ptr = mapped_ptr,
                .id = id,
                .is_dedicated = false,
                .memory_type_index = memory_type_index,
            };
        }

        return Error.AllocationFailed;
    }

    fn createBlock(self: *MemoryAllocator, memory_type_index: u32, size: vk.types.VkDeviceSize, map: bool) Error!MemoryBlock {
        const alloc_info = vk.types.VkMemoryAllocateInfo{
            .sType = .MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = size,
            .memoryTypeIndex = memory_type_index,
        };

        var memory: vk.types.VkDeviceMemory = undefined;
        const result = self.device_dispatch.allocate_memory(self.device, &alloc_info, null, &memory);
        if (result == .ERROR_OUT_OF_DEVICE_MEMORY) {
            return Error.OutOfDeviceMemory;
        } else if (result != .SUCCESS) {
            return Error.AllocationFailed;
        }

        var block = MemoryBlock.init(self.allocator, memory, size, memory_type_index);

        // Map if requested and memory type supports it
        if (map) {
            const props = self.memory_properties.memoryTypes[memory_type_index].propertyFlags;
            if ((props & vk.types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                var mapped: ?*anyopaque = null;
                const map_result = self.device_dispatch.map_memory(self.device, memory, 0, size, 0, &mapped);
                if (map_result == .SUCCESS) {
                    block.mapped_ptr = @ptrCast(mapped);
                }
            }
        }

        log.debug("Created memory block: {} MB, type {}", .{ size / (1024 * 1024), memory_type_index });

        return block;
    }
};

fn alignUp(value: vk.types.VkDeviceSize, alignment: vk.types.VkDeviceSize) vk.types.VkDeviceSize {
    return (value + alignment - 1) & ~(alignment - 1);
}

test "memory allocator alignment" {
    try std.testing.expectEqual(@as(u64, 256), alignUp(200, 256));
    try std.testing.expectEqual(@as(u64, 256), alignUp(256, 256));
    try std.testing.expectEqual(@as(u64, 512), alignUp(257, 256));
}
