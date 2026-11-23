#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include <vulkan/vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "math.hpp"

namespace application {

namespace {

using graphics::internal::context;

struct Vertex {
	float position[3];
	float normal[3];
	float uv[2];
};

struct SceneUniforms {
	math::Mat4 view_projection;
	float camera_position[4];
	float ambient[4];
	float light_direction[4];
	float light_color[4];
	uint32_t light_counts[4];
	math::Mat4 light_view_projection;
	float shadow_params[4];
};

struct PointLight {
	float position[4];
	float color[4];
};

struct SpotLight {
	float position[4];
	float direction[4];
	float color[4];
	float cone[4];
};

constexpr uint32_t max_point_lights = 128;
constexpr uint32_t max_spot_lights = 128;

struct ModelUniforms {
	math::Mat4 model;
	math::Mat4 normal_matrix;
	float specular_color[4];
	float shininess;
	float padding[3];
};

constexpr math::Vec3 octahedron_positions[] = {
	{  1.0f,  0.0f,  0.0f },
	{ -1.0f,  0.0f,  0.0f },
	{  0.0f,  1.0f,  0.0f },
	{  0.0f, -1.0f,  0.0f },
	{  0.0f,  0.0f,  1.0f },
	{  0.0f,  0.0f, -1.0f },
};

constexpr uint32_t octahedron_faces[][3] = {
	{ 2, 4, 0 }, { 2, 1, 4 }, { 2, 5, 1 }, { 2, 0, 5 },
	{ 3, 0, 4 }, { 3, 4, 1 }, { 3, 1, 5 }, { 3, 5, 0 },
};

constexpr uint32_t face_count = sizeof(octahedron_faces) / sizeof(octahedron_faces[0]);
constexpr uint32_t vertex_count = face_count * 3;
constexpr uint32_t floor_vertex_count = 4;
constexpr uint32_t floor_index_count = 6;

constexpr uint32_t shadow_map_size = 2048;
constexpr VkFormat shadow_map_format = VK_FORMAT_D32_SFLOAT;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

Buffer vertex_buffer;
Buffer index_buffer;
Buffer scene_buffer;
Buffer point_light_buffer;
Buffer spot_light_buffer;
Buffer model_buffer;

VkDescriptorSetLayout vk_scene_set_layout;
VkDescriptorSetLayout vk_model_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_set;

VkDeviceSize model_stride;

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
};

struct TextureEntry {
	std::string name;
	Texture texture;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

std::vector<TextureEntry> textures;
VkSampler vk_sampler;
VkCommandPool vk_upload_pool;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;
VkPipeline vk_shadow_pipeline;

struct ShadowMap {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
};

ShadowMap shadow_map;
VkSampler vk_shadow_sampler;

struct Camera {
	math::Vec3 position = { 2.5f, 2.0f, 7.5f };
	float yaw = 10.0f;
	float pitch = -12.0f;
	float field_of_view = 60.0f;
	float move_speed = 4.0f;
};

struct DirectionalLight {
	bool auto_rotate = true;
	float rotation_speed = 20.0f;
	float azimuth = 30.0f;
	float elevation = 40.0f;
	math::Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 1.0f;
};

struct Material {
	math::Vec3 specular_color = { 1.0f, 1.0f, 1.0f };
	float shininess = 64.0f;
};

constexpr uint32_t max_octahedra = 3;
constexpr uint32_t quad_count = 3;
constexpr uint32_t first_quad = max_octahedra;
constexpr uint32_t max_objects = max_octahedra + quad_count;

struct Object {
	math::Vec3 position = { 0.0f, 0.0f, 0.0f };
	math::Vec3 rotation = { 20.0f, 30.0f, 0.0f };
	math::Vec3 scale = { 1.0f, 1.0f, 1.0f };
	math::Vec3 spin = { 0.0f, 0.0f, 0.0f };
	Material material;
	int texture = 0;
	bool animated = false;
	bool is_quad = false;
};

struct SamplingPush {
	float time;
	float speed;
	float modulation;
	uint32_t flags;
};

constexpr uint32_t sampling_animated_flag = 1;
constexpr uint32_t sampling_push_offset = 16;

Camera camera;
DirectionalLight light;
Object objects[max_objects] = {
	{ .rotation = { 20.0f, 30.0f, 0.0f }, .spin = { 0.0f, 25.0f, 0.0f }, .texture = 0 },
	{ .position = { -2.5f, -0.5f, -1.0f }, .rotation = { 0.0f, 60.0f, 15.0f },
	  .scale = { 0.6f, 0.6f, 0.6f }, .spin = { 0.0f, -40.0f, 0.0f }, .texture = 1 },
	{ .position = { 2.5f, -0.5f, -1.0f }, .rotation = { 45.0f, 0.0f, 0.0f },
	  .scale = { 0.6f, 0.6f, 0.6f }, .spin = { 30.0f, 0.0f, 20.0f }, .material = { .shininess = 16.0f },
	  .texture = 2, .animated = true },
	{ .position = { 0.0f, -1.6f, 0.0f }, .rotation = { 0.0f, 0.0f, 0.0f },
	  .scale = { 6.0f, 1.0f, 6.0f }, .material = { .specular_color = { 0.15f, 0.15f, 0.15f }, .shininess = 8.0f },
	  .texture = 4, .is_quad = true },
	{ .position = { 0.0f, 4.4f, -6.0f }, .rotation = { 90.0f, 0.0f, 0.0f },
	  .scale = { 6.0f, 1.0f, 6.0f }, .material = { .specular_color = { 0.1f, 0.1f, 0.1f }, .shininess = 8.0f },
	  .texture = 0, .is_quad = true },
	{ .position = { -6.0f, 4.4f, 0.0f }, .rotation = { 0.0f, 0.0f, -90.0f },
	  .scale = { 6.0f, 1.0f, 6.0f }, .material = { .specular_color = { 0.1f, 0.1f, 0.1f }, .shininess = 8.0f },
	  .texture = 0, .is_quad = true },
};

int object_count = 3;
int selected_object = 0;

struct PointLightSource {
	math::Vec3 position = { 0.0f, 2.0f, 2.0f };
	math::Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 10.0f;
};

std::vector<PointLightSource> point_lights = {
	{ .position = { -2.5f, 1.5f, 2.0f }, .color = { 1.0f, 0.3f, 0.2f }, .intensity = 12.0f },
	{ .position = { 2.5f, 1.0f, 2.0f }, .color = { 0.2f, 0.5f, 1.0f }, .intensity = 12.0f },
};

struct SpotLightSource {
	math::Vec3 position = { 0.0f, 4.0f, 0.0f };
	float azimuth = 0.0f;
	float elevation = -90.0f;
	math::Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 40.0f;
	float inner_angle = 15.0f;
	float outer_angle = 30.0f;
};

std::vector<SpotLightSource> spot_lights = {
	{ .position = { 0.0f, 4.0f, 1.0f }, .color = { 1.0f, 0.9f, 0.6f } },
};

struct ShadowSettings {
	bool enabled = true;
	float ortho_size = 12.0f;
	float distance = 18.0f;
	float near_plane = 1.0f;
	float far_plane = 40.0f;
	float depth_bias = 0.0015f;
	float slope_bias = 0.004f;
};

ShadowSettings shadow;

math::Vec3 ambient_color = { 0.1f, 0.1f, 0.12f };

bool spin_enabled = true;
double previous_time;
double current_time;
float sampling_speed = 1.0f;
float sampling_modulation = 0.6f;

math::Vec3 cameraForward() {
	const float yaw = math::radians(camera.yaw);
	const float pitch = math::radians(camera.pitch);

	return { std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch) };
}

math::Vec3 directionFromAngles(float azimuth_degrees, float elevation_degrees) {
	const float azimuth = math::radians(azimuth_degrees);
	const float elevation = math::radians(elevation_degrees);

	return { std::cos(elevation) * std::sin(azimuth), std::sin(elevation),
	         std::cos(elevation) * std::cos(azimuth) };
}

math::Vec3 lightDirection() {
	return directionFromAngles(light.azimuth, light.elevation);
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& result_buffer) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result = {};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &result_buffer.buffer, &result_buffer.allocation, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer of " << size << " bytes\n";
		return false;
	}

	result_buffer.mapped = result.pMappedData;

	return true;
}

template <typename F>
bool submitOneTime(F&& record) {
	const VkCommandBufferAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_upload_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	VkCommandBuffer command_buffer = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(context.device, &allocate, &command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to allocate upload command buffer\n";
		return false;
	}

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(command_buffer, &begin);
	record(command_buffer);
	vkEndCommandBuffer(command_buffer);

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &command_buffer,
	};

	const bool ok = vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
	vkQueueWaitIdle(context.graphics_queue);
	vkFreeCommandBuffers(context.device, vk_upload_pool, 1, &command_buffer);

	return ok;
}

bool createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                        Buffer& result_buffer) {
	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
		return false;
	}
	std::memcpy(staging.mapped, data, size);
	vmaFlushAllocation(context.allocator, staging.allocation, 0, size);

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &result_buffer.buffer, &result_buffer.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create device buffer of " << size << " bytes\n";
		vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);
		return false;
	}

	const bool ok = submitOneTime([&](VkCommandBuffer command_buffer) {
		const VkBufferCopy region = { .size = size };
		vkCmdCopyBuffer(command_buffer, staging.buffer, result_buffer.buffer, 1, &region);
	});

	vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);

	return ok;
}

void transitionImage(VkCommandBuffer command_buffer, VkImage image, VkImageLayout old_layout,
                     VkImageLayout new_layout, VkAccessFlags src_access, VkAccessFlags dst_access,
                     VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                     uint32_t base_mip, uint32_t mip_count) {
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = base_mip,
			.levelCount = mip_count,
			.layerCount = 1,
		},
	};

	vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool loadTexture(const char* path, Texture& texture) {
	int width = 0, height = 0, channels = 0;
	stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	if (pixels == nullptr) {
		std::cerr << "Failed to load texture " << path << '\n';
		return false;
	}

	const VkDeviceSize size = VkDeviceSize(width) * VkDeviceSize(height) * 4;
	const uint32_t mip_levels = uint32_t(std::floor(std::log2(double(std::max(width, height))))) + 1;

	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
		stbi_image_free(pixels);
		return false;
	}
	std::memcpy(staging.mapped, pixels, size);
	vmaFlushAllocation(context.allocator, staging.allocation, 0, size);
	stbi_image_free(pixels);

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_SRGB,
		.extent = { uint32_t(width), uint32_t(height), 1 },
		.mipLevels = mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		         VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateImage(context.allocator, &image_info, &allocation_info, &texture.image,
	                   &texture.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create image for " << path << '\n';
		vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);
		return false;
	}

	const bool ok = submitOneTime([&](VkCommandBuffer command_buffer) {
		transitionImage(command_buffer, texture.image, VK_IMAGE_LAYOUT_UNDEFINED,
		                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
		                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                0, mip_levels);

		const VkBufferImageCopy region = {
			.imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
			.imageExtent = { uint32_t(width), uint32_t(height), 1 },
		};
		vkCmdCopyBufferToImage(command_buffer, staging.buffer, texture.image,
		                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		int32_t mip_width = width;
		int32_t mip_height = height;

		for (uint32_t level = 1; level < mip_levels; ++level) {
			transitionImage(command_buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
			                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			                VK_PIPELINE_STAGE_TRANSFER_BIT, level - 1, 1);

			const int32_t next_width = std::max(mip_width / 2, 1);
			const int32_t next_height = std::max(mip_height / 2, 1);

			const VkImageBlit blit = {
				.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 },
				.srcOffsets = { { 0, 0, 0 }, { mip_width, mip_height, 1 } },
				.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 },
				.dstOffsets = { { 0, 0, 0 }, { next_width, next_height, 1 } },
			};
			vkCmdBlitImage(command_buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			               texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
			               VK_FILTER_LINEAR);

			transitionImage(command_buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
			                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, level - 1, 1);

			mip_width = next_width;
			mip_height = next_height;
		}

		transitionImage(command_buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, mip_levels - 1, 1);
	});

	vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);

	if (!ok) {
		return false;
	}

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_SRGB,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = mip_levels,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &view_info, nullptr, &texture.view) != VK_SUCCESS) {
		std::cerr << "Failed to create image view for " << path << '\n';
		return false;
	}

	return true;
}

bool createSampler() {
	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = VK_LOD_CLAMP_NONE,
	};

	if (vkCreateSampler(context.device, &sampler_info, nullptr, &vk_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create sampler\n";
		return false;
	}

	return true;
}

bool createShadowMap() {
	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = shadow_map_format,
		.extent = { shadow_map_size, shadow_map_size, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateImage(context.allocator, &image_info, &allocation_info, &shadow_map.image,
	                   &shadow_map.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow map image\n";
		return false;
	}

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = shadow_map.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = shadow_map_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.levelCount = 1,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &view_info, nullptr, &shadow_map.view) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow map view\n";
		return false;
	}

	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.maxLod = 0.0f,
	};

	if (vkCreateSampler(context.device, &sampler_info, nullptr, &vk_shadow_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow sampler\n";
		return false;
	}

	return true;
}

void barrierShadowMap(VkCommandBuffer command_buffer, VkImage image, VkImageLayout old_layout,
                      VkImageLayout new_layout, VkAccessFlags src_access, VkAccessFlags dst_access,
                      VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.levelCount = 1,
			.layerCount = 1,
		},
	};

	vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool loadShaderModule(const char* path, VkShaderModule& module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader " << path << '\n';
		return false;
	}

	const std::streamsize size = file.tellg();
	std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);

	const VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size_t(size),
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(context.device, &module_info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module for " << path << '\n';
		return false;
	}

	return true;
}

bool createPipelines() {
	VkShaderModule vertex_shader = VK_NULL_HANDLE;
	VkShaderModule fragment_shader = VK_NULL_HANDLE;

	if (!loadShaderModule("shaders/lit.vert.spv", vertex_shader) ||
	    !loadShaderModule("shaders/lit.frag.spv", fragment_shader)) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDescriptorSetLayout set_layouts[] = { vk_scene_set_layout, vk_model_set_layout };

	const VkPushConstantRange push_ranges[] = {
		{ .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = sampling_push_offset,
		  .size = sizeof(SamplingPush) },
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = sizeof(push_ranges) / sizeof(push_ranges[0]),
		.pPushConstantRanges = push_ranges,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline, nullptr, &vk_pipeline);

	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	VkShaderModule shadow_vertex_shader = VK_NULL_HANDLE;
	if (!loadShaderModule("shaders/shadow.vert.spv", shadow_vertex_shader)) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo shadow_stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = shadow_vertex_shader,
		.pName = "main",
	};

	const VkVertexInputAttributeDescription shadow_attribute = {
		.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		.offset = offsetof(Vertex, position),
	};

	const VkPipelineVertexInputStateCreateInfo shadow_vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 1,
		.pVertexAttributeDescriptions = &shadow_attribute,
	};

	VkPipelineRasterizationStateCreateInfo shadow_rasterization = rasterization;
	shadow_rasterization.cullMode = VK_CULL_MODE_NONE;

	const VkPipelineColorBlendStateCreateInfo shadow_color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	};

	const VkPipelineRenderingCreateInfo rendering_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.depthAttachmentFormat = shadow_map_format,
	};

	const VkGraphicsPipelineCreateInfo shadow_pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &rendering_info,
		.stageCount = 1,
		.pStages = &shadow_stage,
		.pVertexInputState = &shadow_vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &shadow_rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &shadow_color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = VK_NULL_HANDLE,
	};

	const VkResult shadow_result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                         &shadow_pipeline, nullptr,
	                                                         &vk_shadow_pipeline);

	vkDestroyShaderModule(context.device, shadow_vertex_shader, nullptr);

	if (shadow_result != VK_SUCCESS) {
		std::cerr << "Failed to create shadow pipeline\n";
		return false;
	}

	return true;
}

bool createDescriptors() {
	const VkShaderStageFlags all_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = all_stages,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = &vk_shadow_sampler,
		},
	};

	const VkDescriptorSetLayoutBinding model_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = all_stages,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(model_bindings) / sizeof(model_bindings[0]),
		.pBindings = model_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout, nullptr,
	                                &vk_scene_set_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(context.device, &model_layout, nullptr,
	                                &vk_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	const uint32_t texture_count = uint32_t(textures.size());

	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = texture_count },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = texture_count + 1 },
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + texture_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	std::vector<VkDescriptorSetLayout> layouts(1 + texture_count, vk_model_set_layout);
	layouts[0] = vk_scene_set_layout;
	std::vector<VkDescriptorSet> sets(1 + texture_count);

	const VkDescriptorSetAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = uint32_t(sets.size()),
		.pSetLayouts = layouts.data(),
	};

	if (vkAllocateDescriptorSets(context.device, &allocate, sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	vk_scene_set = sets[0];

	const VkDescriptorBufferInfo scene_infos[] = {
		{ .buffer = scene_buffer.buffer, .offset = 0, .range = sizeof(SceneUniforms) },
		{ .buffer = point_light_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
		{ .buffer = spot_light_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
	};

	const VkDescriptorBufferInfo model_info = {
		.buffer = model_buffer.buffer, .offset = 0, .range = sizeof(ModelUniforms),
	};

	std::vector<VkDescriptorImageInfo> image_infos(texture_count);
	std::vector<VkWriteDescriptorSet> writes;

	writes.push_back({
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_scene_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &scene_infos[0],
	});
	writes.push_back({
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_scene_set,
		.dstBinding = 1,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &scene_infos[1],
	});
	writes.push_back({
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_scene_set,
		.dstBinding = 2,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &scene_infos[2],
	});

	const VkDescriptorImageInfo shadow_image_info = {
		.sampler = VK_NULL_HANDLE,
		.imageView = shadow_map.view,
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
	};
	writes.push_back({
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_scene_set,
		.dstBinding = 3,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &shadow_image_info,
	});

	for (uint32_t i = 0; i < texture_count; ++i) {
		textures[i].set = sets[1 + i];

		image_infos[i] = {
			.sampler = vk_sampler,
			.imageView = textures[i].texture.view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		writes.push_back({
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = textures[i].set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &model_info,
		});
		writes.push_back({
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = textures[i].set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_infos[i],
		});
	}

	vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);

	return true;
}

}

bool initialize() {
	const VkCommandPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &pool_info, nullptr, &vk_upload_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create upload command pool\n";
		return false;
	}

	constexpr math::Vec3 face_uvs[3] = {
		{ 0.5f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f },
	};

	Vertex vertices[vertex_count + floor_vertex_count] = {};
	uint16_t indices[vertex_count + floor_index_count] = {};

	for (uint32_t f = 0; f < face_count; ++f) {
		const math::Vec3 a = octahedron_positions[octahedron_faces[f][0]];
		const math::Vec3 b = octahedron_positions[octahedron_faces[f][1]];
		const math::Vec3 c = octahedron_positions[octahedron_faces[f][2]];

		math::Vec3 normal = math::normalize(math::cross(b - a, c - a));

		if (math::dot(normal, a + b + c) < 0.0f) {
			normal = normal * -1.0f;
		}

		const math::Vec3 face[3] = { a, b, c };
		for (uint32_t i = 0; i < 3; ++i) {
			vertices[f * 3 + i] = { { face[i].x, face[i].y, face[i].z },
			                        { normal.x, normal.y, normal.z },
			                        { face_uvs[i].x, face_uvs[i].y } };
			indices[f * 3 + i] = uint16_t(f * 3 + i);
		}
	}

	const Vertex floor_vertices[floor_vertex_count] = {
		{ { -1.0f, 0.0f, -1.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
		{ { -1.0f, 0.0f,  1.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 4.0f } },
		{ {  1.0f, 0.0f,  1.0f }, { 0.0f, 1.0f, 0.0f }, { 4.0f, 4.0f } },
		{ {  1.0f, 0.0f, -1.0f }, { 0.0f, 1.0f, 0.0f }, { 4.0f, 0.0f } },
	};
	const uint16_t floor_indices[floor_index_count] = {
		vertex_count, vertex_count + 1, vertex_count + 2,
		vertex_count, vertex_count + 2, vertex_count + 3,
	};

	std::memcpy(vertices + vertex_count, floor_vertices, sizeof(floor_vertices));
	std::memcpy(indices + vertex_count, floor_indices, sizeof(floor_indices));

	if (!createDeviceBuffer(vertices, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createDeviceBuffer(indices, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
		return false;
	}

	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);

	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	model_stride = (sizeof(ModelUniforms) + alignment - 1) / alignment * alignment;

	if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) ||
	    !createBuffer(sizeof(PointLight) * max_point_lights,
	                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_light_buffer) ||
	    !createBuffer(sizeof(SpotLight) * max_spot_lights,
	                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_light_buffer) ||
	    !createBuffer(model_stride * max_objects, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer)) {
		return false;
	}

	if (!createSampler() || !createShadowMap()) {
		return false;
	}

	std::vector<std::filesystem::path> texture_paths;
	for (const auto& entry : std::filesystem::directory_iterator("textures")) {
		const std::string extension = entry.path().extension().string();
		if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
		    extension == ".bmp" || extension == ".tga") {
			texture_paths.push_back(entry.path());
		}
	}
	std::sort(texture_paths.begin(), texture_paths.end());

	if (texture_paths.empty()) {
		std::cerr << "No textures found in textures/\n";
		return false;
	}

	textures.resize(texture_paths.size());
	for (size_t i = 0; i < texture_paths.size(); ++i) {
		textures[i].name = texture_paths[i].stem().string();
		if (!loadTexture(texture_paths[i].string().c_str(), textures[i].texture)) {
			return false;
		}
	}

	for (Object& object : objects) {
		object.texture = std::min(object.texture, int(textures.size()) - 1);
	}

	return createDescriptors() && createPipelines();
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_shadow_pipeline, nullptr);
	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	vmaDestroyBuffer(context.allocator, model_buffer.buffer, model_buffer.allocation);
	vmaDestroyBuffer(context.allocator, spot_light_buffer.buffer, spot_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, point_light_buffer.buffer, point_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, scene_buffer.buffer, scene_buffer.allocation);
	vmaDestroyBuffer(context.allocator, index_buffer.buffer, index_buffer.allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer.buffer, vertex_buffer.allocation);

	vkDestroySampler(context.device, vk_shadow_sampler, nullptr);
	vkDestroyImageView(context.device, shadow_map.view, nullptr);
	vmaDestroyImage(context.allocator, shadow_map.image, shadow_map.allocation);

	vkDestroySampler(context.device, vk_sampler, nullptr);
	for (TextureEntry& entry : textures) {
		vkDestroyImageView(context.device, entry.texture.view, nullptr);
		vmaDestroyImage(context.allocator, entry.texture.image, entry.texture.allocation);
	}

	vkDestroyCommandPool(context.device, vk_upload_pool, nullptr);
}

void update(double time) {
	const float delta = previous_time > 0.0 ? float(time - previous_time) : 0.0f;
	previous_time = time;
	current_time = time;

	if (light.auto_rotate) {
		light.azimuth = std::fmod(light.azimuth + light.rotation_speed * delta + 540.0f, 360.0f) - 180.0f;
	}

	if (spin_enabled) {
		for (Object& object : objects) {
			object.rotation = object.rotation + object.spin * delta;
		}
	}

	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera.yaw += io.MouseDelta.x * 0.2f;
		camera.pitch -= io.MouseDelta.y * 0.2f;
		camera.pitch = std::fmax(-89.0f, std::fmin(89.0f, camera.pitch));
	}

	if (!io.WantCaptureKeyboard) {
		const math::Vec3 forward = cameraForward();
		const math::Vec3 right = math::normalize(math::cross(forward, { 0.0f, 1.0f, 0.0f }));
		const float step = camera.move_speed * delta;

		if (ImGui::IsKeyDown(ImGuiKey_W)) camera.position = camera.position + forward * step;
		if (ImGui::IsKeyDown(ImGuiKey_S)) camera.position = camera.position - forward * step;
		if (ImGui::IsKeyDown(ImGuiKey_D)) camera.position = camera.position + right * step;
		if (ImGui::IsKeyDown(ImGuiKey_A)) camera.position = camera.position - right * step;
		if (ImGui::IsKeyDown(ImGuiKey_E)) camera.position.y += step;
		if (ImGui::IsKeyDown(ImGuiKey_Q)) camera.position.y -= step;
	}

	ImGui::Begin("Lab 4: shadow mapping");
	ImGui::Text("%.1f FPS", double(io.Framerate));

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::TextWrapped("RMB drag: look, WASD/QE: move");
		ImGui::DragFloat3("Position", &camera.position.x, 0.05f);
		ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg");
		ImGui::SliderFloat("Field of view", &camera.field_of_view, 20.0f, 110.0f, "%.0f deg");
		ImGui::SliderFloat("Move speed", &camera.move_speed, 0.5f, 20.0f);
	}

	if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient", &ambient_color.x);
		ImGui::SeparatorText("Directional light");
		ImGui::Checkbox("Auto-rotate sun", &light.auto_rotate);
		ImGui::SliderFloat("Rotation speed", &light.rotation_speed, -90.0f, 90.0f, "%.0f deg/s");
		ImGui::SliderFloat("Azimuth", &light.azimuth, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("Elevation", &light.elevation, -90.0f, 90.0f, "%.0f deg");
		ImGui::ColorEdit3("Light color", &light.color.x);
		ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 5.0f);

		ImGui::SeparatorText("Point lights");

		ImGui::BeginDisabled(point_lights.size() >= max_point_lights);
		if (ImGui::Button("Add point light")) {
			point_lights.push_back({});
		}
		ImGui::EndDisabled();

		int remove_index = -1;
		for (size_t i = 0; i < point_lights.size(); ++i) {
			PointLightSource& source = point_lights[i];

			ImGui::PushID(int(i));
			if (ImGui::TreeNodeEx("Point light", ImGuiTreeNodeFlags_DefaultOpen, "Point light %zu", i + 1)) {
				ImGui::DragFloat3("Position", &source.position.x, 0.05f);
				ImGui::ColorEdit3("Color", &source.color.x);
				ImGui::DragFloat("Intensity", &source.intensity, 0.1f, 0.0f, 1000.0f);

				if (ImGui::Button("Remove")) {
					remove_index = int(i);
				}

				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_index >= 0) {
			point_lights.erase(point_lights.begin() + remove_index);
		}

		ImGui::SeparatorText("Spot lights");

		ImGui::BeginDisabled(spot_lights.size() >= max_spot_lights);
		if (ImGui::Button("Add spot light")) {
			spot_lights.push_back({});
		}
		ImGui::EndDisabled();

		remove_index = -1;
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			SpotLightSource& source = spot_lights[i];

			ImGui::PushID(int(i));
			if (ImGui::TreeNodeEx("Spot light", ImGuiTreeNodeFlags_DefaultOpen, "Spot light %zu", i + 1)) {
				ImGui::DragFloat3("Position", &source.position.x, 0.05f);
				ImGui::SliderFloat("Azimuth", &source.azimuth, -180.0f, 180.0f, "%.0f deg");
				ImGui::SliderFloat("Elevation", &source.elevation, -90.0f, 90.0f, "%.0f deg");
				ImGui::ColorEdit3("Color", &source.color.x);
				ImGui::DragFloat("Intensity", &source.intensity, 0.5f, 0.0f, 5000.0f);

				if (ImGui::SliderFloat("Inner angle", &source.inner_angle, 0.0f, 89.0f, "%.0f deg")) {
					source.outer_angle = std::fmax(source.outer_angle, source.inner_angle);
				}
				if (ImGui::SliderFloat("Outer angle", &source.outer_angle, 1.0f, 90.0f, "%.0f deg")) {
					source.inner_angle = std::fmin(source.inner_angle, source.outer_angle);
				}

				if (ImGui::Button("Remove")) {
					remove_index = int(i);
				}

				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_index >= 0) {
			spot_lights.erase(spot_lights.begin() + remove_index);
		}
	}

	if (ImGui::CollapsingHeader("Shadows", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Enable shadows", &shadow.enabled);
		ImGui::TextWrapped("Sun shadow map %ux%u, orthographic projection", shadow_map_size, shadow_map_size);
		ImGui::SliderFloat("Covered area (half-size)", &shadow.ortho_size, 1.0f, 30.0f);
		ImGui::SliderFloat("Sun distance", &shadow.distance, 1.0f, 50.0f);
		ImGui::SliderFloat("Near plane", &shadow.near_plane, 0.01f, 20.0f);
		ImGui::SliderFloat("Far plane", &shadow.far_plane, 1.0f, 100.0f);
		shadow.far_plane = std::fmax(shadow.far_plane, shadow.near_plane + 0.1f);
		ImGui::SliderFloat("Depth bias", &shadow.depth_bias, 0.0f, 0.02f, "%.4f");
		ImGui::SliderFloat("Slope bias", &shadow.slope_bias, 0.0f, 0.05f, "%.4f");
	}

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Spin objects", &spin_enabled);
		ImGui::SliderInt("Count", &object_count, 1, int(max_octahedra));
		if (selected_object < int(first_quad)) {
			selected_object = std::min(selected_object, object_count - 1);
		}
		ImGui::Combo("Selected", &selected_object, "Object 1\0Object 2\0Object 3\0Floor\0Back wall\0Left wall\0");

		Object& object = objects[selected_object];

		ImGui::SliderFloat3("Position##object", &object.position.x, -12.0f, 12.0f);
		ImGui::SliderFloat3("Rotation", &object.rotation.x, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat3("Scale", &object.scale.x, 0.1f, 12.0f);
		ImGui::SliderFloat3("Spin", &object.spin.x, -180.0f, 180.0f, "%.0f deg/s");

		if (ImGui::BeginCombo("Texture", textures[size_t(object.texture)].name.c_str())) {
			for (size_t i = 0; i < textures.size(); ++i) {
				if (ImGui::Selectable(textures[i].name.c_str(), int(i) == object.texture)) {
					object.texture = int(i);
				}
			}
			ImGui::EndCombo();
		}

		ImGui::SeparatorText("Material");
		ImGui::ColorEdit3("Specular color", &object.material.specular_color.x);
		ImGui::SliderFloat("Shininess", &object.material.shininess, 1.0f, 512.0f, "%.0f",
		                   ImGuiSliderFlags_Logarithmic);
	}

	ImGui::End();

	ImGui::SetNextWindowPos({ 330.0f, 60.0f }, ImGuiCond_FirstUseEver);
	ImGui::Begin("Sampling animation");

	for (int i = 0; i < object_count; ++i) {
		ImGui::PushID(i);
		ImGui::Checkbox(i == 0 ? "Object 1" : i == 1 ? "Object 2" : "Object 3", &objects[i].animated);
		ImGui::PopID();
		if (i + 1 < object_count) {
			ImGui::SameLine();
		}
	}
	ImGui::SliderFloat("Speed", &sampling_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Color modulation", &sampling_modulation, 0.0f, 1.0f);

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	const math::Mat4 view = math::lookAt(camera.position, camera.position + cameraForward(),
	                                     { 0.0f, 1.0f, 0.0f });
	const math::Mat4 projection =
		math::perspective(math::radians(camera.field_of_view), aspect, 0.1f, 100.0f);

	const math::Vec3 light_dir = lightDirection();

	const math::Vec3 sun_target = { 0.0f, 0.0f, 0.0f };
	const math::Vec3 sun_position = sun_target + light_dir * shadow.distance;
	const math::Vec3 sun_up = std::fabs(light_dir.y) > 0.99f ? math::Vec3{ 0.0f, 0.0f, 1.0f }
	                                                           : math::Vec3{ 0.0f, 1.0f, 0.0f };
	const math::Mat4 light_view = math::lookAt(sun_position, sun_target, sun_up);
	const math::Mat4 light_projection = math::ortho(-shadow.ortho_size, shadow.ortho_size,
	                                                -shadow.ortho_size, shadow.ortho_size,
	                                                shadow.near_plane, shadow.far_plane);

	const SceneUniforms scene = {
		projection * view,
		{ camera.position.x, camera.position.y, camera.position.z, 1.0f },
		{ ambient_color.x, ambient_color.y, ambient_color.z, 1.0f },
		{ light_dir.x, light_dir.y, light_dir.z, 0.0f },
		{ light.color.x * light.intensity, light.color.y * light.intensity,
		  light.color.z * light.intensity, 1.0f },
		{ uint32_t(point_lights.size()), uint32_t(spot_lights.size()), 0, 0 },
		light_projection * light_view,
		{ 1.0f / float(shadow_map_size), shadow.enabled ? 1.0f : 0.0f, shadow.depth_bias,
		  shadow.slope_bias },
	};

	std::memcpy(scene_buffer.mapped, &scene, sizeof(scene));

	PointLight* gpu_point_lights = static_cast<PointLight*>(point_light_buffer.mapped);
	for (size_t i = 0; i < point_lights.size(); ++i) {
		const PointLightSource& source = point_lights[i];

		gpu_point_lights[i] = {
			{ source.position.x, source.position.y, source.position.z, 1.0f },
			{ source.color.x * source.intensity, source.color.y * source.intensity,
			  source.color.z * source.intensity, 1.0f },
		};
	}

	SpotLight* gpu_spot_lights = static_cast<SpotLight*>(spot_light_buffer.mapped);
	for (size_t i = 0; i < spot_lights.size(); ++i) {
		const SpotLightSource& source = spot_lights[i];
		const math::Vec3 direction = directionFromAngles(source.azimuth, source.elevation);

		gpu_spot_lights[i] = {
			{ source.position.x, source.position.y, source.position.z, 1.0f },
			{ direction.x, direction.y, direction.z, 0.0f },
			{ source.color.x * source.intensity, source.color.y * source.intensity,
			  source.color.z * source.intensity, 1.0f },
			{ std::cos(math::radians(source.inner_angle)),
			  std::cos(math::radians(source.outer_angle)), 0.0f, 0.0f },
		};
	}

	int active[max_objects];
	int active_count = 0;
	for (int i = 0; i < object_count; ++i) {
		active[active_count++] = i;
	}
	for (uint32_t q = 0; q < quad_count; ++q) {
		active[active_count++] = int(first_quad + q);
	}

	for (int k = 0; k < active_count; ++k) {
		const int i = active[k];
		const Object& object = objects[i];

		const math::Mat4 rotation = math::rotate({ math::radians(object.rotation.x),
		                                           math::radians(object.rotation.y),
		                                           math::radians(object.rotation.z) });

		const math::Mat4 normal_matrix =
			rotation * math::scale({ 1.0f / object.scale.x, 1.0f / object.scale.y, 1.0f / object.scale.z });

		const ModelUniforms model = {
			math::translate(object.position) * rotation * math::scale(object.scale),
			normal_matrix,
			{ object.material.specular_color.x, object.material.specular_color.y,
			  object.material.specular_color.z, 1.0f },
			object.material.shininess,
		};

		std::memcpy(static_cast<char*>(model_buffer.mapped) + model_stride * VkDeviceSize(i),
		            &model, sizeof(model));
	}
	vmaFlushAllocation(context.allocator, scene_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, point_light_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, spot_light_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, model_buffer.allocation, 0, VK_WHOLE_SIZE);

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	barrierShadowMap(fd.command_buffer, shadow_map.image, VK_IMAGE_LAYOUT_UNDEFINED,
	                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 0,
	                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
	                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
	                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

	const VkRenderingAttachmentInfo shadow_attachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = shadow_map.view,
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = { .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderingInfo shadow_rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { .extent = { shadow_map_size, shadow_map_size } },
		.layerCount = 1,
		.pDepthAttachment = &shadow_attachment,
	};

	vkCmdBeginRendering(fd.command_buffer, &shadow_rendering);

	if (shadow.enabled) {
		const VkViewport shadow_viewport = {
			.width = float(shadow_map_size),
			.height = float(shadow_map_size),
			.maxDepth = 1.0f,
		};
		const VkRect2D shadow_scissor = { .extent = { shadow_map_size, shadow_map_size } };

		vkCmdSetViewport(fd.command_buffer, 0, 1, &shadow_viewport);
		vkCmdSetScissor(fd.command_buffer, 0, 1, &shadow_scissor);

		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_shadow_pipeline);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        vk_pipeline_layout, 0, 1, &vk_scene_set, 0, nullptr);

		const VkDeviceSize shadow_vertex_offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.buffer,
		                       &shadow_vertex_offset);
		vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

		for (int k = 0; k < active_count; ++k) {
			const int i = active[k];
			const uint32_t dynamic_offset = uint32_t(model_stride * VkDeviceSize(i));

			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			                        vk_pipeline_layout, 1, 1,
			                        &textures[size_t(objects[i].texture)].set, 1, &dynamic_offset);

			if (objects[i].is_quad) {
				vkCmdDrawIndexed(fd.command_buffer, floor_index_count, 1, vertex_count, 0, 0);
			} else {
				vkCmdDrawIndexed(fd.command_buffer, vertex_count, 1, 0, 0, 0);
			}
		}
	}

	vkCmdEndRendering(fd.command_buffer);

	barrierShadowMap(fd.command_buffer, shadow_map.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
	                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
	                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
	                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
	                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
	                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.05f, 0.06f, 0.09f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = { .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        vk_pipeline_layout, 0, 1, &vk_scene_set, 0, nullptr);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	for (int k = 0; k < active_count; ++k) {
		const int i = active[k];
		const uint32_t dynamic_offset = uint32_t(model_stride * VkDeviceSize(i));

		const Object& object = objects[i];

		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        vk_pipeline_layout, 1, 1, &textures[size_t(object.texture)].set,
		                        1, &dynamic_offset);

		const SamplingPush sampling = {
			float(current_time),
			sampling_speed,
			sampling_modulation,
			object.animated ? sampling_animated_flag : 0u,
		};
		vkCmdPushConstants(fd.command_buffer, vk_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
		                   sampling_push_offset, sizeof(sampling), &sampling);

		if (object.is_quad) {
			vkCmdDrawIndexed(fd.command_buffer, floor_index_count, 1, vertex_count, 0, 0);
		} else {
			vkCmdDrawIndexed(fd.command_buffer, vertex_count, 1, 0, 0, 0);
		}
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

}
