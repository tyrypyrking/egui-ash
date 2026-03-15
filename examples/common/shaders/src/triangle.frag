#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() { outColor = vec4(pow(fragColor, vec3(2.2)), 1.0); }
