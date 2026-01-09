#version 450

// Fullscreen triangle - no vertex input needed
// Uses vertex ID to generate positions

layout(location = 0) out vec3 fragColor;

// Push constants for animation/effects
layout(push_constant) uniform PushConstants {
    float time;
    float padding[3];
} pc;

void main() {
    // Generate fullscreen triangle vertices from gl_VertexIndex
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    // Gradient colors based on position + time
    vec3 colors[3] = vec3[](
        vec3(0.5 + 0.5 * sin(pc.time), 0.0, 0.5),           // Purple-ish
        vec3(0.0, 0.5 + 0.5 * sin(pc.time + 2.0), 0.5),     // Cyan-ish
        vec3(0.5, 0.0, 0.5 + 0.5 * sin(pc.time + 4.0))      // Magenta-ish
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
