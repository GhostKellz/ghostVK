# GhostVK

![Zig Version](https://img.shields.io/badge/Zig-0.16.0--dev-orange?style=flat&logo=zig&logoColor=white)
![Vulkan](https://img.shields.io/badge/Vulkan-1.3+-red?style=flat&logo=vulkan&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-blue?style=flat&logo=linux&logoColor=white)
![Wayland](https://img.shields.io/badge/Wayland-Native-lightblue?style=flat)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Open%20Driver-76B900?style=flat&logo=nvidia&logoColor=white)
![Proton](https://img.shields.io/badge/Proton-Compatible-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=flat)

> **⚠️ Experimental Project** - This project is under active development and not fully tested. Use at your own risk.

**A next-generation Vulkan runtime written in Zig, purpose-built for high-refresh-rate HDR gaming on Linux with NVIDIA GPUs.**

---

## Overview

GhostVK is a high-performance Vulkan runtime designed to eliminate the traditional pain points of Linux graphics programming. Built from the ground up in Zig, it targets the modern Linux graphics stack (Wayland, DRM/KMS, NVIDIA Open Kernel Modules) with a focus on delivering best-in-class performance for high-refresh-rate HDR displays.

### Why GhostVK?

- **Zero-compromise performance**: Target sub-1ms frame times at 1440p/360Hz
- **HDR-first design**: Native support for PQ/ST2084 EOTF and HDR metadata
- **NVIDIA-optimized**: Tailored for RTX 40-series GPUs with BAR1-aware allocators
- **Wayland-native**: Direct DRM/KMS integration, no XWayland overhead
- **Proton-compatible**: Outperform vkd3d-proton for D3D12→Vulkan translation
- **Memory-safe**: Written in Zig for compile-time safety without runtime cost

---

## Target Platform

### Hardware
- **GPU**: NVIDIA RTX 4090 (RTX 30/40-series recommended)
- **Display**: 1440p OLED, 240-360Hz, HDR support
- **Multi-monitor**: Validated with mixed refresh rates (270Hz + 360Hz)

### Software
- **OS**: Arch Linux (or similar rolling-release distro)
- **Desktop**: KDE Plasma 6 on Wayland
- **Driver**: NVIDIA Open Kernel Modules (560.35.03+)
- **Compositor**: Wayland-native (Hyprland also supported)

---

## Features

### Core Runtime
- **Vulkan 1.3+** with comprehensive extension support
- Unified memory allocator for VRAM, BAR1, and DMA-BUF
- Async command pool and queue scheduler
- Timeline semaphores and modern synchronization primitives
- Dynamic descriptor management with recycling

### Graphics Pipeline
- SPIR-V reflection and shader caching
- Pipeline variant hashing and hot-reload support
- Multi-vendor shader paths (NVIDIA NVPTX, AMD ROCm, Intel Xe)

### Display & Presentation
- Native Wayland surface creation
- Direct DRM/KMS overlay path (bypasses compositor when needed)
- Jitter-free frame pacing with precise timing
- VRR/Adaptive Sync for 240/270/360Hz displays
- HDR10 metadata and PQ tone mapping
- Present timing API for latency measurement

### Developer Tools
- Validation layers with detailed debug messenger
- Built-in performance overlay (`GHOSTVK_HUD=1`)
- CPU/GPU frame time logging to `/tmp/ghostvk.prof`
- Timestamp query infrastructure
- Telemetry hooks for external tools

---

## Architecture

```
ghostVK/
├── src/
│   ├── core/           # Vulkan instance, device, queue management
│   ├── memory/         # Allocators, staging buffers, DMA-BUF
│   ├── pipeline/       # Shader reflection, caching, hot-reload
│   ├── present/        # Swapchain, frame pacing, HDR
│   ├── sync/           # Semaphores, fences, timeline sync
│   └── wayland/        # Native Wayland/DRM integration
├── archive/
│   └── vkd3d-proton/   # Reference implementation (read-only)
├── examples/           # Sample applications
└── docs/               # Technical documentation
```

---

## Building

### Prerequisites

```bash
# Install Zig 0.16.0-dev or later
curl https://ziglang.org/download/0.16.0-dev.164+bc7955306/zig-linux-x86_64-0.16.0-dev.164+bc7955306.tar.xz | tar -xJ
export PATH="$PWD/zig-linux-x86_64-0.16.0-dev.164+bc7955306:$PATH"

# Install Vulkan SDK
sudo pacman -S vulkan-devel vulkan-validation-layers

# Install NVIDIA Open Kernel Modules
sudo pacman -S nvidia-open nvidia-utils
```

### Compile

```bash
# Debug build (with validation layers)
zig build

# Optimized release build
zig build -Doptimize=ReleaseFast

# Run
zig build run

# Run tests
zig build test
```

---

## Usage

### Basic Example

```zig
const ghost = @import("ghostVK");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Initialize GhostVK runtime
    var runtime = try ghost.Runtime.init(gpa.allocator(), .{
        .enable_validation = true,
        .hdr_mode = .hdr10_st2084,
        .target_refresh = 360,
    });
    defer runtime.deinit();

    // Create Wayland surface
    var surface = try runtime.createWaylandSurface(.{
        .width = 2560,
        .height = 1440,
        .vsync_mode = .mailbox,
    });
    defer surface.deinit();

    // Render loop
    while (runtime.pollEvents()) {
        const frame = try surface.acquireNextFrame();
        defer frame.present();

        // Your rendering code here
    }
}
```

### Environment Variables

- `GHOSTVK_HUD=1` - Enable performance overlay
- `GHOSTVK_LOG=debug` - Set log level
- `GHOSTVK_PROF=/tmp/perf.json` - Write profiling data

---

## Integration with Zeus

GhostVK is designed to integrate seamlessly with [Zeus](https://github.com/ghostkellz/zeus), a native Zig text rendering library. Zeus will use GhostVK as its Vulkan backend for maximum performance, providing a complete graphics stack optimized for high-refresh-rate displays.

---

## Benchmarks

*Coming soon: Comparative benchmarks against Mesa Vulkan, vkd3d-proton, and NVK.*

**Goals:**
- <1ms GPU frame time @ 1440p/360Hz
- Zero presentation jitter on KDE Wayland
- 15-20% faster than vkd3d-proton for D3D12 translation

---

## Roadmap

See [TODO.md](TODO.md) for the complete 7-phase development roadmap.

**Current Phase:** Phase 1 - Project Bootstrap

---

## Contributing

GhostVK is in active development. Contributions, bug reports, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **vkd3d-proton** - Reference implementation for D3D12→Vulkan translation
- **Mesa** - Vulkan layer reference
- **Zig Community** - For an incredible language and toolchain
- **NVIDIA** - For open-sourcing kernel modules

---

## Contact

Built by [@ghostkellz](https://github.com/ghostkellz)

**GhostVK is where Vulkan + Zig + NVIDIA Open finally stop fighting each other.**
