#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform UniformBufferObject {
  mat4 modelMatrix;
  mat4 viewMatrix;
  mat4 projMatrix;
}
ubo;

layout(location = 0) out vec3 fragWorldPosition;
layout(location = 1) out vec3 fragWorldNormal;

out gl_PerVertex { vec4 gl_Position; };

void main() {
  vec4 pos = ubo.modelMatrix * vec4(inPos, 1.0);
  fragWorldPosition = pos.xyz / pos.w;
  fragWorldNormal = transpose(inverse(mat3(ubo.modelMatrix))) * inNormal;
  gl_Position = ubo.projMatrix * ubo.viewMatrix * pos;
}
