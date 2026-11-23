#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;

layout(location = 0) out vec4 out_color;

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

struct PointLight {
	vec4 position;
	vec4 color;
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;
	vec4 cone;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	mat4 normal_matrix;
	vec4 specular_color;
	float shininess;
} model;

layout(set = 0, binding = 3) uniform sampler2DShadow shadow_map;

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;

layout(push_constant) uniform Sampling {
	layout(offset = 16) float time;
	float speed;
	float modulation;
	uint flags;
} sampling;

vec3 sampleAlbedo() {
	if ((sampling.flags & 1u) == 0u) {
		return texture(albedo_texture, frag_uv).rgb;
	}

	const float t = sampling.time * sampling.speed;

	const vec2 scrolled = frag_uv + vec2(0.15 * t, 0.05 * sin(t));

	const float zoom = 1.5 + 0.5 * sin(0.7 * t);
	const vec2 zoomed = (frag_uv - 0.5) * zoom + 0.5;

	const float angle = 0.3 * t;
	const mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
	const vec2 rotated = rotation * (frag_uv - vec2(0.5, 0.6)) * 2.0 + 0.5;

	vec3 color = mix(texture(albedo_texture, scrolled).rgb,
	                 texture(albedo_texture, zoomed).rgb,
	                 0.5 + 0.5 * sin(t));
	color = mix(color, texture(albedo_texture, rotated).rgb, 0.3);

	const vec3 tint = 0.5 + 0.5 * cos(t + vec3(0.0, 2.094, 4.188));
	return color * mix(vec3(1.0), tint * 1.4, sampling.modulation);
}

vec3 blinnPhong(vec3 normal, vec3 view_direction, vec3 light_direction, vec3 radiance,
                vec3 albedo) {
	const vec3 half_vector = normalize(light_direction + view_direction);

	const float n_dot_l = max(dot(normal, light_direction), 0.0);
	const float n_dot_h = max(dot(normal, half_vector), 0.0);

	const vec3 diffuse = albedo * n_dot_l;
	const vec3 specular = n_dot_l > 0.0
		? model.specular_color.rgb * pow(n_dot_h, model.shininess)
		: vec3(0.0);

	return (diffuse + specular) * radiance;
}

float shadowFactor(vec3 world_position, vec3 normal, vec3 light_direction) {
	if (scene.shadow_params.y < 0.5) {
		return 1.0;
	}

	const vec4 light_clip = scene.light_view_projection * vec4(world_position, 1.0);
	const vec3 ndc = light_clip.xyz / light_clip.w;
	const vec2 uv = ndc.xy * 0.5 + 0.5;

	if (ndc.z > 1.0 || ndc.z < 0.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
		return 1.0;
	}

	const float cos_theta = clamp(dot(normal, light_direction), 0.0, 1.0);
	const float reference = ndc.z - (scene.shadow_params.z + scene.shadow_params.w * (1.0 - cos_theta));

	const float texel = scene.shadow_params.x;

	float lit = 0.0;
	for (int y = -1; y <= 1; ++y) {
		for (int x = -1; x <= 1; ++x) {
			lit += texture(shadow_map, vec3(uv + vec2(x, y) * texel, reference));
		}
	}

	return lit / 9.0;
}

void main() {
	const vec3 normal = normalize(frag_normal);
	const vec3 view_direction = normalize(scene.camera_position.xyz - frag_world_position);

	const vec3 albedo = sampleAlbedo();

	vec3 color = scene.ambient.rgb * albedo;

	const vec3 sun_direction = normalize(scene.light_direction.xyz);
	const float sun_visibility = shadowFactor(frag_world_position, normal, sun_direction);

	color += blinnPhong(normal, view_direction, sun_direction,
	                    scene.light_color.rgb * sun_visibility, albedo);

	for (uint i = 0; i < scene.light_counts.x; ++i) {
		const vec3 to_light = point_lights[i].position.xyz - frag_world_position;
		const float distance_squared = max(dot(to_light, to_light), 0.0001);
		const float attenuation = 1.0 / distance_squared;

		color += blinnPhong(normal, view_direction, to_light * inversesqrt(distance_squared),
		                    point_lights[i].color.rgb * attenuation, albedo);
	}

	for (uint i = 0; i < scene.light_counts.y; ++i) {
		const vec3 to_light = spot_lights[i].position.xyz - frag_world_position;
		const float distance_squared = max(dot(to_light, to_light), 0.0001);
		const vec3 light_direction = to_light * inversesqrt(distance_squared);

		const float cos_inner = spot_lights[i].cone.x;
		const float cos_outer = spot_lights[i].cone.y;
		const float cos_angle = dot(-light_direction, normalize(spot_lights[i].direction.xyz));
		const float cone = smoothstep(0.0, 1.0,
			clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0));

		const float attenuation = cone / distance_squared;

		color += blinnPhong(normal, view_direction, light_direction,
		                    spot_lights[i].color.rgb * attenuation, albedo);
	}

	out_color = vec4(color, 1.0);
}
