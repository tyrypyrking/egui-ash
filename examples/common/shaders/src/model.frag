#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 0) out vec4 outColor;

void main() {
  vec3 baseColor = vec3(1.0, 0.5, 0.5);
  vec3 lightPos = vec3(2.0, 2.0, -2.0);
  vec3 L = normalize(lightPos - fragWorldPos.xyz);
  vec3 N = normalize(fragWorldNormal);
  float lambertian = max(dot(N, L), 0.0);
  vec3 ambient = vec3(0.1);
  outColor = vec4(pow(baseColor * lambertian + ambient, vec3(2.2)), 1.0);
}
