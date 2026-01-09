#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

// Push constants for effects
layout(push_constant) uniform PushConstants {
    float time;
    float padding[3];
} pc;

void main() {
    // Add subtle noise/shimmer effect
    float shimmer = 0.05 * sin(gl_FragCoord.x * 0.1 + pc.time * 3.0)
                  * sin(gl_FragCoord.y * 0.1 + pc.time * 2.0);

    outColor = vec4(fragColor + shimmer, 1.0);
}
