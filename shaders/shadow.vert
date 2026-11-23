#version 450

layout(location = 0) in vec3 in_position;

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_color;
	uvec4 light_counts;
	mat4 light_view_projection;
	vec4 shadow_params;
} scene;

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	mat4 normal_matrix;
	vec4 specular_color;
	float shininess;
} model;

void main() {
	gl_Position = scene.light_view_projection * model.model * vec4(in_position, 1.0);
}
