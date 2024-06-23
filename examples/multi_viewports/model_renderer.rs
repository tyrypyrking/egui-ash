use ash::{
    khr::{surface, swapchain},
    vk, Device,
};
use egui_ash::EguiCommand;
use glam::{Mat4, Vec3};
use gpu_allocator::vulkan::{Allocation, Allocator};
use std::{
    ffi::CString,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

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

struct ModelRendererInner {
    width: u32,
    height: u32,

    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    surface_loader: Arc<surface::Instance>,
    swapchain_loader: Arc<swapchain::Device>,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,

    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<Allocation>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    depth_images_and_allocations: Vec<(vk::Image, Allocation)>,
    color_image_views: Vec<vk::ImageView>,
    depth_image_views: Vec<vk::ImageView>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<Allocation>,
    vertex_count: u32,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_fences: Vec<vk::Fence>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    current_frame: usize,
    dirty_swapchain: bool,
}
impl ModelRendererInner {
    fn create_swapchain(
        physical_device: vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        swapchain_loader: &swapchain::Device,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        width: u32,
        height: u32,
    ) -> (
        vk::SwapchainKHR,
        vk::SurfaceFormatKHR,
        vk::Extent2D,
        Vec<vk::Image>,
    ) {
        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .expect("Failed to get physical device surface formats")
        };
        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    || format.format == vk::Format::R8G8B8A8_UNORM
            })
            .unwrap_or(&surface_formats[0])
            .clone();
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Failed to get physical device surface capabilities")
        };
        let surface_extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width
                    .max(surface_capabilities.min_image_extent.width)
                    .min(surface_capabilities.max_image_extent.width),
                height: height
                    .max(surface_capabilities.min_image_extent.height)
                    .min(surface_capabilities.max_image_extent.height),
            }
        };

        let image_count = surface_capabilities.min_image_count + 1;
        let image_count = if surface_capabilities.max_image_count != 0 {
            image_count.min(surface_capabilities.max_image_count)
        } else {
            image_count
        };

        let image_sharing_mode = vk::SharingMode::EXCLUSIVE;
        let queue_family_indices = [queue_family_index];

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain")
        };

        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        (swapchain, surface_format, surface_extent, swapchain_images)
    }

    fn create_uniform_buffers(
        device: &Device,
        allocator: &Mutex<Allocator>,
        swapchain_count: usize,
    ) -> (Vec<vk::Buffer>, Vec<Allocation>) {
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;
        let buffer_usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffers = (0..swapchain_count)
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

    fn create_descriptor_pool(device: &Device, swapchain_count: usize) -> vk::DescriptorPool {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(swapchain_count as u32);
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(swapchain_count as u32);
        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create descriptor pool")
        }
    }

    fn create_descriptor_set_layouts(
        device: &Device,
        swapchain_count: usize,
    ) -> Vec<vk::DescriptorSetLayout> {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&ubo_layout_binding));

        (0..swapchain_count)
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

    fn create_render_pass(device: &Device, surface_format: vk::SurfaceFormatKHR) -> vk::RenderPass {
        let attachments = [
            vk::AttachmentDescription::default()
                .format(surface_format.format)
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
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
        swapchain_images: &[vk::Image],
    ) -> (
        Vec<vk::Framebuffer>,
        Vec<(vk::Image, Allocation)>,
        Vec<vk::ImageView>,
        Vec<vk::ImageView>,
    ) {
        let mut framebuffers = vec![];
        let mut depth_images_and_allocations = vec![];
        let mut color_image_views = vec![];
        let mut depth_image_views = vec![];
        for &image in swapchain_images.iter() {
            let mut attachments = vec![];

            let color_attachment = unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
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
                    width: extent.width,
                    height: extent.height,
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
            depth_image_views.push(depth_attachment);
            framebuffers.push(unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(attachments.as_slice())
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer")
            });
            depth_images_and_allocations.push((depth_image, depth_allocation));
        }
        (
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        )
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
                "./examples/multi_viewports/assets/suzanne.obj",
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
        swapchain_count: usize,
    ) -> Vec<vk::CommandBuffer> {
        unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(swapchain_count as u32),
                )
                .expect("Failed to allocate command buffers")
        }
    }

    fn create_sync_objects(
        device: &Device,
        swapchain_count: usize,
    ) -> (Vec<vk::Fence>, Vec<vk::Semaphore>, Vec<vk::Semaphore>) {
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut in_flight_fences = vec![];
        for _ in 0..swapchain_count {
            let fence = unsafe {
                device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create fence")
            };
            in_flight_fences.push(fence);
        }
        let mut image_available_semaphores = vec![];
        for _ in 0..swapchain_count {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let semaphore = unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create semaphore")
            };
            image_available_semaphores.push(semaphore);
        }
        let mut render_finished_semaphores = vec![];
        for _ in 0..swapchain_count {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let semaphore = unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create semaphore")
            };
            render_finished_semaphores.push(semaphore);
        }
        (
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
        )
    }

    fn recreate_swapchain(&mut self, width: u32, height: u32, egui_cmd: &mut EguiCommand) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle")
        };

        unsafe {
            let mut allocator = self.allocator.lock().unwrap();
            for &fence in self.in_flight_fences.iter() {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in self.image_available_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.render_finished_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &image_view in self.color_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            for &image_view in self.depth_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            for (image, allocation) in self.depth_images_and_allocations.drain(..) {
                self.device.destroy_image(image, None);
                allocator.free(allocation).expect("Failed to free memory");
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        self.width = width;
        self.height = height;

        let (swapchain, swapchain_images, surface_format, surface_extent) = {
            let surface_capabilities = unsafe {
                self.surface_loader
                    .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                    .expect("Failed to get physical device surface capabilities")
            };
            let surface_formats = unsafe {
                self.surface_loader
                    .get_physical_device_surface_formats(self.physical_device, self.surface)
                    .expect("Failed to get physical device surface formats")
            };

            let surface_format = surface_formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_UNORM
                        || surface_format.format == vk::Format::R8G8B8A8_UNORM
                })
                .unwrap_or(&surface_formats[0])
                .clone();

            let surface_present_mode = vk::PresentModeKHR::FIFO;

            let surface_extent = if surface_capabilities.current_extent.width != u32::MAX {
                surface_capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: self.width.clamp(
                        surface_capabilities.min_image_extent.width,
                        surface_capabilities.max_image_extent.width,
                    ),
                    height: self.height.clamp(
                        surface_capabilities.min_image_extent.height,
                        surface_capabilities.max_image_extent.height,
                    ),
                }
            };

            let image_count = surface_capabilities.min_image_count + 1;
            let image_count = if surface_capabilities.max_image_count != 0 {
                image_count.min(surface_capabilities.max_image_count)
            } else {
                image_count
            };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_extent)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(surface_present_mode)
                .image_array_layers(1)
                .clipped(true);
            let swapchain = unsafe {
                self.swapchain_loader
                    .create_swapchain(&swapchain_create_info, None)
                    .expect("Failed to create swapchain")
            };

            let swapchain_images = unsafe {
                self.swapchain_loader
                    .get_swapchain_images(swapchain)
                    .expect("Failed to get swapchain images")
            };

            (swapchain, swapchain_images, surface_format, surface_extent)
        };
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.surface_format = surface_format;
        self.surface_extent = surface_extent;

        egui_cmd.update_swapchain(egui_ash::SwapchainUpdateInfo {
            swapchain_images: self.swapchain_images.clone(),
            surface_format: self.surface_format.format,
            width: self.width,
            height: self.height,
        });

        self.render_pass = Self::create_render_pass(&self.device, self.surface_format);

        let (framebuffers, depth_images_and_allocations, color_image_views, depth_image_views) =
            Self::create_framebuffers(
                &self.device,
                &self.allocator,
                self.render_pass,
                self.surface_format,
                self.surface_extent,
                &self.swapchain_images,
            );
        self.framebuffers = framebuffers;
        self.depth_images_and_allocations = depth_images_and_allocations;
        self.color_image_views = color_image_views;
        self.depth_image_views = depth_image_views;

        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(&self.device, self.swapchain_images.len());
        self.in_flight_fences = in_flight_fences;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;

        self.current_frame = 0;
        self.dirty_swapchain = false;
    }

    fn new(
        physical_device: vk::PhysicalDevice,
        device: Arc<Device>,
        surface_loader: Arc<surface::Instance>,
        swapchain_loader: Arc<swapchain::Device>,
        allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
    ) -> Self {
        let (swapchain, surface_format, surface_extent, swapchain_images) = Self::create_swapchain(
            physical_device,
            &surface_loader,
            &swapchain_loader,
            surface,
            queue_family_index,
            width,
            height,
        );
        let (uniform_buffers, uniform_buffer_allocations) =
            Self::create_uniform_buffers(&device, &allocator, swapchain_images.len());
        let descriptor_pool = Self::create_descriptor_pool(&device, swapchain_images.len());
        let descriptor_set_layouts =
            Self::create_descriptor_set_layouts(&device, swapchain_images.len());
        let descriptor_sets = Self::create_descriptor_sets(
            &device,
            descriptor_pool,
            &descriptor_set_layouts,
            &uniform_buffers,
        );
        let render_pass = Self::create_render_pass(&device, surface_format);
        let (framebuffers, depth_images_and_allocations, color_image_views, depth_image_views) =
            Self::create_framebuffers(
                &device,
                &allocator,
                render_pass,
                surface_format,
                surface_extent,
                &swapchain_images,
            );
        let (pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let (vertex_buffer, vertex_buffer_allocation, vertex_count) =
            Self::load_model_and_create_vertex_buffer(&device, &allocator, command_pool, queue);
        let command_buffers =
            Self::create_command_buffers(&device, command_pool, swapchain_images.len());
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(&device, swapchain_images.len());

        Self {
            width,
            height,
            physical_device,
            device,
            surface_loader,
            swapchain_loader,
            allocator,
            surface,
            queue,
            swapchain,
            surface_format,
            surface_extent,
            swapchain_images,
            uniform_buffers,
            uniform_buffer_allocations,
            descriptor_pool,
            descriptor_set_layouts,
            descriptor_sets,
            render_pass,
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
            pipeline,
            pipeline_layout,
            vertex_buffer,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),
            vertex_count,
            command_buffers,
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
            current_frame: 0,
            dirty_swapchain: true,
        }
    }

    fn render(&mut self, width: u32, height: u32, mut egui_cmd: EguiCommand, rotate_y: f32) {
        if width == 0 || height == 0 {
            return;
        }

        if self.dirty_swapchain
            || width != self.width
            || height != self.height
            || egui_cmd.swapchain_recreate_required()
        {
            self.recreate_swapchain(width, height, &mut egui_cmd);
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

        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        };
        let index = match result {
            Ok((index, _)) => index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty_swapchain = true;
                return;
            }
            Err(_) => return,
        };

        let ubo = UniformBufferObject {
            model: Mat4::from_rotation_y(rotate_y.to_radians()).to_cols_array(),
            view: Mat4::look_at_rh(
                Vec3::new(0.0, 0.0, -5.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
            )
            .to_cols_array(),
            proj: Mat4::perspective_rh(
                45.0_f32.to_radians(),
                width as f32 / height as f32,
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

            self.device.cmd_begin_render_pass(
                self.command_buffers[self.current_frame],
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index])
                    .render_area(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D::default().x(0).y(0))
                            .extent(self.surface_extent),
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
                        .width(width as f32)
                        .height(height as f32)
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
                        .extent(self.surface_extent),
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

            egui_cmd.record(self.command_buffers[self.current_frame], index);

            self.device
                .end_command_buffer(self.command_buffers[self.current_frame])
                .expect("Failed to end command buffer");
        }

        let buffers_to_submit = [self.command_buffers[self.current_frame]];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&buffers_to_submit)
            .wait_semaphores(std::slice::from_ref(
                &self.image_available_semaphores[self.current_frame],
            ))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .signal_semaphores(std::slice::from_ref(
                &self.render_finished_semaphores[self.current_frame],
            ));
        unsafe {
            self.device
                .queue_submit(
                    self.queue,
                    std::slice::from_ref(&submit_info),
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit queue");
        };

        let image_indices = [index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(
                &self.render_finished_semaphores[self.current_frame],
            ))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(&image_indices);
        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };
        let is_dirty_swapchain = match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => false,
        };
        self.dirty_swapchain = is_dirty_swapchain;

        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();
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
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
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
            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &color_image_view in self.color_image_views.iter() {
                self.device.destroy_image_view(color_image_view, None);
            }
            for &depth_image_view in self.depth_image_views.iter() {
                self.device.destroy_image_view(depth_image_view, None);
            }
            for (depth_image, allocation) in self.depth_images_and_allocations.drain(..) {
                self.device.destroy_image(depth_image, None);
                allocator.free(allocation).expect("Failed to free memory");
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
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }
    }
}

#[derive(Clone)]
pub struct ModelRenderer {
    inner: Arc<Mutex<ModelRendererInner>>,
}
impl ModelRenderer {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        device: Arc<Device>,
        surface_loader: Arc<surface::Instance>,
        swapchain_loader: Arc<swapchain::Device>,
        allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModelRendererInner::new(
                physical_device,
                device,
                surface_loader,
                swapchain_loader,
                allocator,
                surface,
                queue_family_index,
                queue,
                command_pool,
                width,
                height,
            ))),
        }
    }

    pub fn render(&self, width: u32, height: u32, egu_cmd: EguiCommand, rotate_y: f32) {
        self.inner
            .lock()
            .unwrap()
            .render(width, height, egu_cmd, rotate_y);
    }

    pub fn destroy(&mut self) {
        self.inner.lock().unwrap().destroy();
    }
}
