//! Wayland client bindings for ghostVK
//! Using @cImport for proper C bindings

const c = @cImport({
    @cInclude("wayland-client.h");
});

// Re-export everything from C
pub const wl_display = c.wl_display;
pub const wl_registry = c.wl_registry;
pub const wl_compositor = c.wl_compositor;
pub const wl_surface = c.wl_surface;
pub const wl_proxy = c.wl_proxy;
pub const wl_interface = c.wl_interface;
pub const wl_registry_listener = c.wl_registry_listener;

// Interface globals are C extern variables, not comptime-known
// Access them via inline functions that return pointers
pub inline fn wl_registry_interface() *const wl_interface {
    return &c.wl_registry_interface;
}

pub inline fn wl_compositor_interface() *const wl_interface {
    return &c.wl_compositor_interface;
}

pub inline fn wl_surface_interface() *const wl_interface {
    return &c.wl_surface_interface;
}

pub const wl_display_connect = c.wl_display_connect;
pub const wl_display_disconnect = c.wl_display_disconnect;
pub const wl_display_roundtrip = c.wl_display_roundtrip;
pub const wl_display_dispatch = c.wl_display_dispatch;
pub const wl_display_get_registry = c.wl_display_get_registry;

pub const wl_registry_add_listener = c.wl_registry_add_listener;
pub const wl_registry_bind = c.wl_registry_bind;

pub const wl_compositor_create_surface = c.wl_compositor_create_surface;
pub const wl_surface_destroy = c.wl_surface_destroy;
pub const wl_compositor_destroy = c.wl_compositor_destroy;
pub const wl_registry_destroy = c.wl_registry_destroy;
pub const wl_display_flush = c.wl_display_flush;
