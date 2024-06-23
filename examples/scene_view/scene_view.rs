use ash::{vk, Device};
use glam::{Mat4, Vec3};
use gpu_allocator::vulkan::{Allocation, Allocator};
use std::{
    ffi::CString,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use crate::vkutils;

macro_rules! include_spirv {
    ($file:literal) => {{
        let bytes = include_bytes!($file);
        bytes
            .chunks_exact(4)
            .map(|x| x.try_into().unwrap())
            .map(match bytes[0] {
                0x03 => u32::from_le_bytes,
                0x07 => u32::from_be_bytes,
                _ => panic!("Unknown endianness"),
            })
            .collect::<Vec<u32>>()
    }};
}

#[repr(C)]
#[derive(Debug, Clone)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
}
impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(4 * 3),
        ]
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
struct UniformBufferObject {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
}

struct SceneViewInner {
    width: u32,
    height: u32,

    device: Arc<Device>,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    queue_family_index: u32,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<Allocation>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    color_images: Vec<vk::Image>,
    color_image_allocations: Vec<Allocation>,
    depth_images: Vec<vk::Image>,
    depth_image_allocations: Vec<Allocation>,
    color_image_views: Vec<vk::ImageView>,
    depth_image_views: Vec<vk::ImageView>,
    sampler: vk::Sampler,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<Allocation>,
    vertex_count: u32,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    size_changed: bool,

    image_registry: egui_ash::ImageRegistry,
    texture_ids: Vec<egui::TextureId>,

    rotate_y: f32,
    rotate_x: f32,
}
impl SceneViewInner {
    const IN_FLIGHT_FRAMES: usize = 2;

    fn create_uniform_buffers(
        device: &Device,
        allocator: &Mutex<Allocator>,
    ) -> (Vec<vk::Buffer>, Vec<Allocation>) {
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;
        let buffer_usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffers = (0..Self::IN_FLIGHT_FRAMES)
            .map(|_| unsafe {
                device
                    .create_buffer(&buffer_create_info, None)
                    .expect("Failed to create buffer")
            })
            .collect::<Vec<_>>();
        let buffer_memory_requirements =
            unsafe { device.get_buffer_memory_requirements(buffers[0]) };
        let buffer_alloc_info = gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Uniform Buffer",
            requirements: buffer_memory_requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        };
        let buffer_allocations = buffers
            .iter()
            .map(|_| {
                allocator
                    .lock()
                    .unwrap()
                    .allocate(&buffer_alloc_info)
                    .expect("Failed to allocate memory")
            })
            .collect::<Vec<_>>();
        for (&buffer, buffer_memory) in buffers.iter().zip(buffer_allocations.iter()) {
            unsafe {
                device
                    .bind_buffer_memory(buffer, buffer_memory.memory(), buffer_memory.offset())
                    .expect("Failed to bind buffer memory")
            }
        }

        (buffers, buffer_allocations)
    }

    fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(Self::IN_FLIGHT_FRAMES as u32);
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(Self::IN_FLIGHT_FRAMES as u32);
        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create descriptor pool")
        }
    }

    fn create_descriptor_set_layouts(device: &Device) -> Vec<vk::DescriptorSetLayout> {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&ubo_layout_binding));

        (0..Self::IN_FLIGHT_FRAMES)
            .map(|_| unsafe {
                device
                    .create_descriptor_set_layout(&ubo_layout_create_info, None)
                    .expect("Failed to create descriptor set layout")
            })
            .collect()
    }

    fn create_descriptor_sets(
        device: &Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        uniform_buffers: &Vec<vk::Buffer>,
    ) -> Vec<vk::DescriptorSet> {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(descriptor_set_layouts);
        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets")
        };
        for index in 0..descriptor_sets.len() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[index])
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[index])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));
            unsafe {
                device.update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
            }
        }

        descriptor_sets
    }

    fn create_render_pass(device: &Device) -> vk::RenderPass {
        let attachments = [
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];
        let color_reference = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let depth_reference = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_reference)
            .depth_stencil_attachment(&depth_reference)];
        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .expect("Failed to create render pass")
        }
    }

    fn create_framebuffers(
        device: &Device,
        allocator: &Mutex<Allocator>,
        render_pass: vk::RenderPass,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        queue_family_index: u32,
        width: u32,
        height: u32,
    ) -> (
        Vec<vk::Framebuffer>,
        Vec<vk::Image>,
        Vec<vk::Image>,
        Vec<Allocation>,
        Vec<Allocation>,
        Vec<vk::ImageView>,
        Vec<vk::ImageView>,
    ) {
        let mut framebuffers = vec![];
        let mut color_images = vec![];
        let mut depth_images = vec![];
        let mut color_image_allocations = vec![];
        let mut depth_image_allocations = vec![];
        let mut color_image_views = vec![];
        let mut depth_image_views = vec![];
        for _ in 0..Self::IN_FLIGHT_FRAMES {
            let mut attachments = vec![];

            let color_image_create_info = vk::ImageCreateInfo::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            let color_image = unsafe {
                device
                    .create_image(&color_image_create_info, None)
                    .expect("Failed to create image")
            };
            let color_allocation = allocator
                .lock()
                .unwrap()
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "Color Image",
                    requirements: unsafe { device.get_image_memory_requirements(color_image) },
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to allocate memory");
            unsafe {
                device
                    .bind_image_memory(
                        color_image,
                        color_allocation.memory(),
                        color_allocation.offset(),
                    )
                    .expect("Failed to bind image memory")
            };
            let color_attachment = unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(color_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R8G8B8A8_UNORM)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                        None,
                    )
                    .expect("Failed to create image view")
            };
            attachments.push(color_attachment);
            color_images.push(color_image);
            color_image_allocations.push(color_allocation);
            color_image_views.push(color_attachment);

            let depth_image_create_info = vk::ImageCreateInfo::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            let depth_image = unsafe {
                device
                    .create_image(&depth_image_create_info, None)
                    .expect("Failed to create image")
            };
            let depth_allocation = allocator
                .lock()
                .unwrap()
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "Depth Image",
                    requirements: unsafe { device.get_image_memory_requirements(depth_image) },
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to allocate memory");
            unsafe {
                device
                    .bind_image_memory(
                        depth_image,
                        depth_allocation.memory(),
                        depth_allocation.offset(),
                    )
                    .expect("Failed to bind image memory")
            };
            let depth_attachment = unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(depth_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::D32_SFLOAT)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                        None,
                    )
                    .expect("Failed to create depth image view")
            };
            attachments.push(depth_attachment);
            depth_images.push(depth_image);
            depth_image_allocations.push(depth_allocation);
            depth_image_views.push(depth_attachment);

            framebuffers.push(unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(attachments.as_slice())
                            .width(width)
                            .height(height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer")
            });

            let cmd = unsafe {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .expect("Failed to allocate command buffer")[0]
            };
            unsafe {
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .expect("Failed to begin command buffer");
            }
            vkutils::insert_image_memory_barrier(
                &device,
                &cmd,
                &color_image,
                queue_family_index,
                queue_family_index,
                vk::AccessFlags::NONE,
                vk::AccessFlags::SHADER_READ,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0u32)
                    .layer_count(1u32)
                    .base_mip_level(0u32)
                    .level_count(1u32),
            );
            unsafe {
                device
                    .end_command_buffer(cmd)
                    .expect("Failed to end command buffer");
                device
                    .queue_submit(
                        queue,
                        &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                        vk::Fence::null(),
                    )
                    .expect("Failed to submit queue");
                device.queue_wait_idle(queue).expect("Failed to wait queue");
                device.free_command_buffers(command_pool, &[cmd]);
            }
        }
        (
            framebuffers,
            color_images,
            depth_images,
            color_image_allocations,
            depth_image_allocations,
            color_image_views,
            depth_image_views,
        )
    }

    fn create_sampler(device: &Device) -> vk::Sampler {
        unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }
        .expect("Failed to create sampler")
    }

    fn create_graphics_pipeline(
        device: &Device,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        render_pass: vk::RenderPass,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_shader_module = {
            let spirv = include_spirv!("./shaders/spv/model.vert.spv");
            let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
            unsafe {
                device
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create shader module")
            }
        };
        let fragment_shader_module = {
            let spirv = include_spirv!("./shaders/spv/model.frag.spv");
            let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
            unsafe {
                device
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create shader module")
            }
        };
        let main_function_name = CString::new("main").unwrap();
        let pipeline_shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader_module)
                .name(&main_function_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader_module)
                .name(&main_function_name),
        ];
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts),
                    None,
                )
                .expect("Failed to create pipeline layout")
        };
        let vertex_input_binding = Vertex::get_binding_descriptions();
        let vertex_input_attribute = Vertex::get_attribute_descriptions();
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);
        let stencil_op = vk::StencilOpState::default()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS);
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(stencil_op)
            .back(stencil_op);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );
        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_input_attribute)
            .vertex_binding_descriptions(&vertex_input_binding);
        let multisample_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&pipeline_shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_create_info),
                    None,
                )
                .unwrap()[0]
        };
        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        (graphics_pipeline, pipeline_layout)
    }

    fn load_model_and_create_vertex_buffer(
        device: &Device,
        allocator: &Mutex<Allocator>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> (vk::Buffer, Allocation, u32) {
        let mut allocator = allocator.lock().unwrap();
        let vertices = {
            let model_obj = tobj::load_obj(
                "./examples/scene_view/assets/suzanne.obj",
                &tobj::LoadOptions {
                    single_index: true,
                    triangulate: true,
                    ignore_points: true,
                    ignore_lines: true,
                },
            )
            .expect("Failed to load model");
            let mut vertices = vec![];
            let (models, _) = model_obj;
            for m in models.iter() {
                let mesh = &m.mesh;

                for &i in mesh.indices.iter() {
                    let i = i as usize;
                    let vertex = Vertex {
                        position: Vec3::new(
                            mesh.positions[3 * i],
                            mesh.positions[3 * i + 1],
                            mesh.positions[3 * i + 2],
                        ),
                        normal: Vec3::new(
                            mesh.normals[3 * i],
                            mesh.normals[3 * i + 1],
                            mesh.normals[3 * i + 2],
                        ),
                    };
                    vertices.push(vertex);
                }
            }

            vertices
        };
        let vertex_buffer_size = vertices.len() as u64 * std::mem::size_of::<Vertex>() as u64;
        let temporary_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(vertex_buffer_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .expect("Failed to create buffer")
        };
        let temporary_buffer_memory_requirements =
            unsafe { device.get_buffer_memory_requirements(temporary_buffer) };
        let temporary_buffer_allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Temporary Vertex Buffer",
                requirements: temporary_buffer_memory_requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate memory");
        unsafe {
            device
                .bind_buffer_memory(
                    temporary_buffer,
                    temporary_buffer_allocation.memory(),
                    temporary_buffer_allocation.offset(),
                )
                .expect("Failed to bind buffer memory")
        }
        unsafe {
            let ptr = temporary_buffer_allocation.mapped_ptr().unwrap().as_ptr() as *mut Vertex;
            ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
        }

        let vertex_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(vertex_buffer_size)
                        .usage(
                            vk::BufferUsageFlags::TRANSFER_DST
                                | vk::BufferUsageFlags::VERTEX_BUFFER,
                        ),
                    None,
                )
                .expect("Failed to create buffer")
        };
        let vertex_buffer_memory_requirements =
            unsafe { device.get_buffer_memory_requirements(vertex_buffer) };
        let vertex_buffer_allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Vertex Buffer",
                requirements: vertex_buffer_memory_requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate memory");
        unsafe {
            device
                .bind_buffer_memory(
                    vertex_buffer,
                    vertex_buffer_allocation.memory(),
                    vertex_buffer_allocation.offset(),
                )
                .expect("Failed to bind buffer memory")
        }

        let cmd = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("Failed to allocate command buffer")[0]
        };

        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin command buffer");
            device.cmd_copy_buffer(
                cmd,
                temporary_buffer,
                vertex_buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(vertex_buffer_size)],
            );
            device
                .end_command_buffer(cmd)
                .expect("Failed to end command buffer");

            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    vk::Fence::null(),
                )
                .expect("Failed to submit queue");
            device.queue_wait_idle(queue).expect("Failed to wait queue");

            device.free_command_buffers(command_pool, &[cmd]);
        }

        allocator
            .free(temporary_buffer_allocation)
            .expect("Failed to free memory");
        unsafe {
            device.destroy_buffer(temporary_buffer, None);
        }

        (
            vertex_buffer,
            vertex_buffer_allocation,
            vertices.len() as u32,
        )
    }

    fn create_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
    ) -> Vec<vk::CommandBuffer> {
        unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(Self::IN_FLIGHT_FRAMES as u32),
                )
                .expect("Failed to allocate command buffers")
        }
    }

    fn create_sync_objects(device: &Device) -> Vec<vk::Fence> {
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut in_flight_fences = vec![];
        for _ in 0..Self::IN_FLIGHT_FRAMES {
            let fence = unsafe {
                device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create fence")
            };
            in_flight_fences.push(fence);
        }
        in_flight_fences
    }

    fn recreate_images(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle")
        };

        unsafe {
            let mut allocator = self.allocator.lock().unwrap();
            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &image_view in self.color_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            for &image_view in self.depth_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            for allocation in self.color_image_allocations.drain(..) {
                allocator.free(allocation).expect("Failed to free memory");
            }
            for allocation in self.depth_image_allocations.drain(..) {
                allocator.free(allocation).expect("Failed to free memory");
            }
            for image in self.color_images.drain(..) {
                self.device.destroy_image(image, None);
            }
            for image in self.depth_images.drain(..) {
                self.device.destroy_image(image, None);
            }
            for id in self.texture_ids.drain(..) {
                self.image_registry.unregister_user_texture(id);
            }
        }

        let (
            framebuffers,
            color_images,
            depth_images,
            color_image_allocations,
            depth_image_allocations,
            color_image_views,
            depth_image_views,
        ) = Self::create_framebuffers(
            &self.device,
            &self.allocator,
            self.render_pass,
            self.command_pool,
            self.queue,
            self.queue_family_index,
            self.width,
            self.height,
        );
        self.framebuffers = framebuffers;
        self.color_images = color_images;
        self.depth_images = depth_images;
        self.color_image_allocations = color_image_allocations;
        self.depth_image_allocations = depth_image_allocations;
        self.color_image_views = color_image_views;
        self.depth_image_views = depth_image_views;

        for &image_views in &self.color_image_views {
            let id = self
                .image_registry
                .register_user_texture(image_views, self.sampler);
            self.texture_ids.push(id);
        }

        self.current_frame = 0;
        self.size_changed = false;
    }

    fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        queue: vk::Queue,
        queue_family_index: u32,
        command_pool: vk::CommandPool,
        image_registry: egui_ash::ImageRegistry,
    ) -> Self {
        let width = 1;
        let height = 1;

        let (uniform_buffers, uniform_buffer_allocations) =
            Self::create_uniform_buffers(&device, &allocator);
        let descriptor_pool = Self::create_descriptor_pool(&device);
        let descriptor_set_layouts = Self::create_descriptor_set_layouts(&device);
        let descriptor_sets = Self::create_descriptor_sets(
            &device,
            descriptor_pool,
            &descriptor_set_layouts,
            &uniform_buffers,
        );
        let render_pass = Self::create_render_pass(&device);
        let (
            framebuffers,
            color_images,
            depth_images,
            color_image_allocations,
            depth_image_allocations,
            color_image_views,
            depth_image_views,
        ) = Self::create_framebuffers(
            &device,
            &allocator,
            render_pass,
            command_pool,
            queue,
            queue_family_index,
            width,
            height,
        );
        let sampler = Self::create_sampler(&device);
        let (pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let (vertex_buffer, vertex_buffer_allocation, vertex_count) =
            Self::load_model_and_create_vertex_buffer(&device, &allocator, command_pool, queue);
        let command_buffers = Self::create_command_buffers(&device, command_pool);
        let in_flight_fences = Self::create_sync_objects(&device);

        let mut texture_ids = vec![];
        for &image_views in &color_image_views {
            let id = image_registry.register_user_texture(image_views, sampler);
            texture_ids.push(id);
        }

        Self {
            width,
            height,

            device,
            allocator: ManuallyDrop::new(allocator),
            command_pool,
            queue,
            queue_family_index,

            uniform_buffers,
            uniform_buffer_allocations,
            descriptor_pool,
            descriptor_set_layouts,
            descriptor_sets,
            render_pass,
            framebuffers,
            color_images,
            depth_images,
            color_image_allocations,
            depth_image_allocations,
            color_image_views,
            depth_image_views,
            sampler,
            pipeline,
            pipeline_layout,
            vertex_buffer,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),
            vertex_count,
            command_buffers,
            in_flight_fences,
            current_frame: 0,
            size_changed: false,

            image_registry,
            texture_ids,

            rotate_y: 0.0,
            rotate_x: 0.0,
        }
    }

    fn render(&mut self) {
        if self.width == 0 || self.height == 0 {
            return;
        }

        if self.size_changed {
            self.recreate_images();
        }

        unsafe {
            self.device.wait_for_fences(
                std::slice::from_ref(&self.in_flight_fences[self.current_frame]),
                true,
                u64::MAX,
            )
        }
        .expect("Failed to wait for fences");

        unsafe {
            self.device.reset_fences(std::slice::from_ref(
                &self.in_flight_fences[self.current_frame],
            ))
        }
        .expect("Failed to reset fences");

        let ubo = UniformBufferObject {
            model: Mat4::from_rotation_x(self.rotate_x.to_radians())
                .mul_mat4(&Mat4::from_rotation_y(self.rotate_y.to_radians()))
                .to_cols_array(),
            view: Mat4::look_at_rh(
                Vec3::new(0.0, 0.0, -5.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
            )
            .to_cols_array(),
            proj: Mat4::perspective_rh(
                45.0_f32.to_radians(),
                self.width as f32 / self.height as f32,
                0.1,
                10.0,
            )
            .to_cols_array(),
        };
        unsafe {
            let ptr = self.uniform_buffer_allocations[self.current_frame]
                .mapped_ptr()
                .unwrap()
                .as_ptr() as *mut UniformBufferObject;
            ptr.copy_from_nonoverlapping([ubo].as_ptr(), 1);
        }

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(
                    self.command_buffers[self.current_frame],
                    &command_buffer_begin_info,
                )
                .expect("Failed to begin command buffer");

            vkutils::insert_image_memory_barrier(
                &self.device,
                &self.command_buffers[self.current_frame],
                &self.color_images[self.current_frame],
                self.queue_family_index,
                self.queue_family_index,
                vk::AccessFlags::NONE,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0u32)
                    .layer_count(1u32)
                    .base_mip_level(0u32)
                    .level_count(1u32),
            );

            self.device.cmd_begin_render_pass(
                self.command_buffers[self.current_frame],
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[self.current_frame])
                    .render_area(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D::default().x(0).y(0))
                            .extent(
                                vk::Extent2D::default()
                                    .width(self.width)
                                    .height(self.height),
                            ),
                    )
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.4, 0.3, 0.2, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ]),
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                self.command_buffers[self.current_frame],
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_set_viewport(
                self.command_buffers[self.current_frame],
                0,
                std::slice::from_ref(
                    &vk::Viewport::default()
                        .width(self.width as f32)
                        .height(self.height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0),
                ),
            );
            self.device.cmd_set_scissor(
                self.command_buffers[self.current_frame],
                0,
                std::slice::from_ref(
                    &vk::Rect2D::default()
                        .offset(vk::Offset2D::default())
                        .extent(
                            vk::Extent2D::default()
                                .width(self.width)
                                .height(self.height),
                        ),
                ),
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffers[self.current_frame],
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.current_frame]],
                &[],
            );

            self.device.cmd_bind_vertex_buffers(
                self.command_buffers[self.current_frame],
                0,
                &[self.vertex_buffer],
                &[0],
            );
            self.device.cmd_draw(
                self.command_buffers[self.current_frame],
                self.vertex_count,
                1,
                0,
                0,
            );

            self.device
                .cmd_end_render_pass(self.command_buffers[self.current_frame]);

            vkutils::insert_image_memory_barrier(
                &self.device,
                &self.command_buffers[self.current_frame],
                &self.color_images[self.current_frame],
                self.queue_family_index,
                self.queue_family_index,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0u32)
                    .layer_count(1u32)
                    .base_mip_level(0u32)
                    .level_count(1u32),
            );

            self.device
                .end_command_buffer(self.command_buffers[self.current_frame])
                .expect("Failed to end command buffer");
        }

        let buffers_to_submit = [self.command_buffers[self.current_frame]];
        let submit_info = vk::SubmitInfo::default().command_buffers(&buffers_to_submit);
        unsafe {
            self.device
                .queue_submit(
                    self.queue,
                    std::slice::from_ref(&submit_info),
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit queue");
        };

        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();
    }

    fn set_size(&mut self, size: egui::Vec2) {
        if self.width != size.x as u32 || self.height != size.y as u32 {
            self.width = size.x as u32;
            self.height = size.y as u32;
            self.size_changed = true;
        }
    }

    fn next_texture(&self) -> egui::TextureId {
        let next_frame = (self.current_frame + 1) % self.in_flight_fences.len();
        unsafe {
            self.device.wait_for_fences(
                std::slice::from_ref(&self.in_flight_fences[next_frame]),
                true,
                u64::MAX,
            )
        }
        .expect("Failed to wait for fences");
        self.texture_ids[next_frame]
    }

    fn destroy(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            let mut allocator = self.allocator.lock().unwrap();
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_buffer(self.vertex_buffer, None);
            if let Some(vertex_buffer_allocation) = self.vertex_buffer_allocation.take() {
                allocator
                    .free(vertex_buffer_allocation)
                    .expect("Failed to free memory");
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_sampler(self.sampler, None);
            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &color_image_view in self.color_image_views.iter() {
                self.device.destroy_image_view(color_image_view, None);
            }
            for &depth_image_view in self.depth_image_views.iter() {
                self.device.destroy_image_view(depth_image_view, None);
            }
            for color_image_allocation in self.color_image_allocations.drain(..) {
                allocator
                    .free(color_image_allocation)
                    .expect("Failed to free memory");
            }
            for depth_image_allocation in self.depth_image_allocations.drain(..) {
                allocator
                    .free(depth_image_allocation)
                    .expect("Failed to free memory");
            }
            for color_image in self.color_images.drain(..) {
                self.device.destroy_image(color_image, None);
            }
            for depth_image in self.depth_images.drain(..) {
                self.device.destroy_image(depth_image, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            for &descriptor_set_layout in self.descriptor_set_layouts.iter() {
                self.device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for &uniform_buffer in self.uniform_buffers.iter() {
                self.device.destroy_buffer(uniform_buffer, None);
            }
            for allocation in self.uniform_buffer_allocations.drain(..) {
                allocator.free(allocation).expect("Failed to free memory");
            }
        }
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }
    }
}

#[derive(Clone)]
pub struct SceneView {
    inner: Arc<Mutex<SceneViewInner>>,
}
impl SceneView {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        queue: vk::Queue,
        queue_family_index: u32,
        command_pool: vk::CommandPool,
        image_registry: egui_ash::ImageRegistry,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SceneViewInner::new(
                device,
                allocator,
                queue,
                queue_family_index,
                command_pool,
                image_registry,
            ))),
        }
    }

    pub fn render(&mut self) {
        self.inner.lock().unwrap().render();
    }

    pub fn destroy(&mut self) {
        self.inner.lock().unwrap().destroy();
    }
}
impl egui::Widget for &SceneView {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let mut inner = self.inner.lock().unwrap();
        let texture_id = inner.next_texture();
        let response = ui
            .with_layout(
                egui::Layout::top_down_justified(egui::Align::Center),
                |ui| {
                    let size = ui.available_size();
                    inner.set_size(size);
                    ui.image(egui::load::SizedTexture {
                        id: texture_id,
                        size,
                    })
                },
            )
            .response;
        let response = response.interact(egui::Sense::drag());
        if response.dragged() {
            inner.rotate_y = (inner.rotate_y - response.drag_delta().x + 180.0) % 360.0 - 180.0;
            inner.rotate_x = (inner.rotate_x - response.drag_delta().y).clamp(-90.0, 90.0);
        }
        response
    }
}
