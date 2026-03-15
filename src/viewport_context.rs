use anyhow::Result;
use ash::{vk, Device, Entry, Instance};
use bytemuck::bytes_of;
use egui_winit::winit;
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use std::ffi::CString;

use crate::allocator::{Allocation, AllocationCreateInfo, Allocator, MemoryLocation};
use crate::renderer::{ManagedTextures, SwapchainUpdateInfo, UserTextures};
use crate::utils;

pub(crate) struct ViewportContext<A: Allocator + 'static> {
    // === Swapchain / presentation ===
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    width: u32,
    height: u32,
    present_mode: vk::PresentModeKHR,
    dirty_flag: bool,

    // === Frame sync ===
    render_command_buffers: Vec<vk::CommandBuffer>,
    in_flight_fences: Vec<vk::Fence>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    current_frame: usize,

    // === Rendering pipeline ===
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    swapchain_image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,

    // === Vertex / index buffers (one set per swapchain image) ===
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_allocations: Vec<A::Allocation>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_allocations: Vec<A::Allocation>,

    // === Metadata ===
    scale_factor: f32,
    physical_width: u32,
    physical_height: u32,

    // === Shared Vulkan handles ===
    device: Device,
    #[allow(dead_code)]
    entry: Entry,
    #[allow(dead_code)]
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    allocator: A,
}

impl<A: Allocator + 'static> ViewportContext<A> {
    // size for vertex buffer which egui-ash uses
    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    // size for index buffer which egui-ash uses
    fn index_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    fn create_swapchain(
        width: u32,
        height: u32,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        present_mode: vk::PresentModeKHR,
    ) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        let surface_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        let surface_present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };

        // select swapchain format
        let surface_format = *surface_formats
            .iter()
            .find(|surface_format| {
                surface_format.format == vk::Format::B8G8R8A8_UNORM
                    && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surface_formats[0]);

        // select surface present mode
        let surface_present_mode = surface_present_modes
            .iter()
            .find(|&&mode| mode == present_mode)
            .unwrap_or(&vk::PresentModeKHR::FIFO);

        // calculate extent
        let surface_extent = if surface_capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: width.clamp(
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    surface_capabilities.min_image_extent.height,
                    surface_capabilities.max_image_extent.height,
                ),
            }
        } else {
            surface_capabilities.current_extent
        };

        // get image count
        let image_count = surface_capabilities.min_image_count + 1;
        let image_count = if surface_capabilities.max_image_count != 0 {
            image_count.min(surface_capabilities.max_image_count)
        } else {
            image_count
        };

        // create swapchain
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(*surface_present_mode)
            .image_array_layers(1)
            .clipped(true);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        // get swapchain images
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        Ok((
            swapchain,
            swapchain_images,
            surface_format.format,
            surface_extent,
        ))
    }

    fn create_render_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
        len: u32,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(len);
        let command_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }?;
        Ok(command_buffers)
    }

    fn create_sync_objects(
        device: &Device,
        len: u32,
    ) -> Result<(Vec<vk::Fence>, Vec<vk::Semaphore>, Vec<vk::Semaphore>)> {
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut in_flight_fences = vec![];
        for _ in 0..len {
            let fence = unsafe { device.create_fence(&fence_create_info, None)? };
            in_flight_fences.push(fence);
        }

        let mut image_available_semaphores = vec![];
        for _ in 0..len {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None)? };
            image_available_semaphores.push(semaphore);
        }
        let mut render_finished_semaphores = vec![];
        for _ in 0..len {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None)? };
            render_finished_semaphores.push(semaphore);
        }

        Ok((
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
        ))
    }

    fn create_render_pass(device: &Device, surface_format: vk::Format) -> vk::RenderPass {
        unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(
                        &vk::AttachmentDescription::default()
                            .format(surface_format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                    ))
                    .subpasses(std::slice::from_ref(
                        &vk::SubpassDescription::default()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(std::slice::from_ref(
                                &vk::AttachmentReference::default()
                                    .attachment(0)
                                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                            )),
                    ))
                    .dependencies(&[vk::SubpassDependency::default()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)]),
                None,
            )
        }
        .expect("Failed to create render pass.")
    }

    fn create_pipeline_layout(
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> vk::PipelineLayout {
        unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[descriptor_set_layout])
                    .push_constant_ranges(std::slice::from_ref(
                        &vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .offset(0)
                            .size(std::mem::size_of::<f32>() as u32 * 2),
                    )),
                None,
            )
        }
        .expect("Failed to create pipeline layout.")
    }

    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let attributes = [
            // position
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .offset(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT),
            // uv
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .offset(8)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT),
            // color
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .offset(16)
                .location(2)
                .format(vk::Format::R8G8B8A8_UNORM),
        ];

        let vertex_shader_module = {
            let bytes_code = include_bytes!("shaders/spv/vert.spv");
            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr().cast::<u32>(),
                ..Default::default()
            };
            unsafe { device.create_shader_module(&shader_module_create_info, None) }
                .expect("Failed to create vertex shader module.")
        };
        let fragment_shader_module = {
            let bytes_code = include_bytes!("shaders/spv/frag.spv");
            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr().cast::<u32>(),
                ..Default::default()
            };
            unsafe { device.create_shader_module(&shader_module_create_info, None) }
                .expect("Failed to create fragment shader module.")
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
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::ALWAYS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(vk::StencilOpState {
                compare_op: vk::CompareOp::ALWAYS,
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                ..Default::default()
            })
            .back(vk::StencilOpState {
                compare_op: vk::CompareOp::ALWAYS,
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                ..Default::default()
            });
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let multisample_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let pipeline = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(
                    &vk::GraphicsPipelineCreateInfo::default()
                        .stages(&pipeline_shader_stages)
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_attribute_descriptions(&attributes)
                                .vertex_binding_descriptions(std::slice::from_ref(
                                    &vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .input_rate(vk::VertexInputRate::VERTEX)
                                        .stride(
                                            4 * std::mem::size_of::<f32>() as u32
                                                + 4 * std::mem::size_of::<u8>() as u32,
                                        ),
                                )),
                        )
                        .input_assembly_state(&input_assembly_info)
                        .viewport_state(&viewport_info)
                        .rasterization_state(&rasterization_info)
                        .multisample_state(&multisample_info)
                        .depth_stencil_state(&depth_stencil_info)
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default().attachments(
                                std::slice::from_ref(
                                    &vk::PipelineColorBlendAttachmentState::default()
                                        .color_write_mask(
                                            vk::ColorComponentFlags::R
                                                | vk::ColorComponentFlags::G
                                                | vk::ColorComponentFlags::B
                                                | vk::ColorComponentFlags::A,
                                        )
                                        .blend_enable(true)
                                        .src_color_blend_factor(vk::BlendFactor::ONE)
                                        .dst_color_blend_factor(
                                            vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                        ),
                                ),
                            ),
                        )
                        .dynamic_state(&dynamic_state_info)
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0),
                ),
                None,
            )
        }
        .expect("Failed to create graphics pipeline.")[0];
        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }
        pipeline
    }

    fn create_framebuffers(
        device: &Device,
        swap_images: &[vk::Image],
        render_pass: vk::RenderPass,
        surface_format: vk::Format,
        width: u32,
        height: u32,
    ) -> (Vec<vk::Framebuffer>, Vec<vk::ImageView>) {
        let swapchain_image_views = swap_images
            .iter()
            .map(|swapchain_image| unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(*swapchain_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(surface_format)
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
                    .expect("Failed to create image view.")
            })
            .collect::<Vec<_>>();
        let framebuffers = swapchain_image_views
            .iter()
            .map(|&image_view| unsafe {
                let attachments = &[image_view];
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(attachments)
                            .width(width)
                            .height(height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer.")
            })
            .collect::<Vec<_>>();

        (framebuffers, swapchain_image_views)
    }

    fn create_vertex_index_buffers(
        device: &Device,
        swapchain_count: usize,
        allocator: &A,
    ) -> (Vec<vk::Buffer>, Vec<A::Allocation>, Vec<vk::Buffer>, Vec<A::Allocation>) {
        let mut vertex_buffers = vec![];
        let mut vertex_buffer_allocations = vec![];
        let mut index_buffers = vec![];
        let mut index_buffer_allocations = vec![];

        for _ in 0..swapchain_count {
            let vertex_buffer = unsafe {
                device
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .size(Self::vertex_buffer_size()),
                        None,
                    )
                    .expect("Failed to create vertex buffer.")
            };
            let vertex_buffer_requirements =
                unsafe { device.get_buffer_memory_requirements(vertex_buffer) };
            let vertex_buffer_allocation = allocator
                .allocate(A::AllocationCreateInfo::new(
                    Some("egui-ash vertex buffer"),
                    vertex_buffer_requirements,
                    MemoryLocation::cpu_to_gpu(),
                    true,
                ))
                .expect("Failed to create vertex buffer.");
            unsafe {
                device
                    .bind_buffer_memory(
                        vertex_buffer,
                        vertex_buffer_allocation.memory(),
                        vertex_buffer_allocation.offset(),
                    )
                    .expect("Failed to create vertex buffer.");
            }

            let index_buffer = unsafe {
                device
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .usage(vk::BufferUsageFlags::INDEX_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .size(Self::index_buffer_size()),
                        None,
                    )
                    .expect("Failed to create index buffer.")
            };
            let index_buffer_requirements =
                unsafe { device.get_buffer_memory_requirements(index_buffer) };
            let index_buffer_allocation = allocator
                .allocate(A::AllocationCreateInfo::new(
                    Some("egui-ash index buffer"),
                    index_buffer_requirements,
                    MemoryLocation::cpu_to_gpu(),
                    true,
                ))
                .expect("Failed to create index buffer.");
            unsafe {
                device
                    .bind_buffer_memory(
                        index_buffer,
                        index_buffer_allocation.memory(),
                        index_buffer_allocation.offset(),
                    )
                    .expect("Failed to create index buffer.");
            }

            vertex_buffers.push(vertex_buffer);
            vertex_buffer_allocations.push(vertex_buffer_allocation);
            index_buffers.push(index_buffer);
            index_buffer_allocations.push(index_buffer_allocation);
        }

        (vertex_buffers, vertex_buffer_allocations, index_buffers, index_buffer_allocations)
    }

    fn destroy_pipeline_resources(&mut self) {
        unsafe {
            for fb in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(fb, None);
            }
            for iv in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(iv, None);
            }
            for buf in self.vertex_buffers.drain(..) {
                self.device.destroy_buffer(buf, None);
            }
            for alloc in self.vertex_buffer_allocations.drain(..) {
                self.allocator
                    .free(alloc)
                    .expect("Failed to free vertex buffer allocation.");
            }
            for buf in self.index_buffers.drain(..) {
                self.device.destroy_buffer(buf, None);
            }
            for alloc in self.index_buffer_allocations.drain(..) {
                self.allocator
                    .free(alloc)
                    .expect("Failed to free index buffer allocation.");
            }
        }
    }

    fn build_pipeline_resources_from_swapchain(&mut self) {
        let (framebuffers, swapchain_image_views) = Self::create_framebuffers(
            &self.device,
            &self.swapchain_images,
            self.render_pass,
            self.swapchain_format,
            self.width,
            self.height,
        );
        let (vbs, vba, ibs, iba) = Self::create_vertex_index_buffers(
            &self.device,
            self.swapchain_images.len(),
            &self.allocator,
        );
        self.framebuffers = framebuffers;
        self.swapchain_image_views = swapchain_image_views;
        self.vertex_buffers = vbs;
        self.vertex_buffer_allocations = vba;
        self.index_buffers = ibs;
        self.index_buffer_allocations = iba;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create(
        entry: &Entry,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        window: &winit::window::Window,
        present_mode: vk::PresentModeKHR,
        descriptor_set_layout: vk::DescriptorSetLayout,
        allocator: A,
    ) -> Option<Self> {
        let width = window.inner_size().width;
        let height = window.inner_size().height;

        // if window is minimized, return None
        if width == 0 || height == 0 {
            return None;
        }

        // create surface
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window
                    .display_handle()
                    .expect("Unable to retrieve a display handle")
                    .as_raw(),
                window
                    .window_handle()
                    .expect("Unable to retrieve a window handle")
                    .as_raw(),
                None,
            )
            .expect("Failed to create surface")
        };

        // create swapchain
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            Self::create_swapchain(
                width,
                height,
                physical_device,
                surface,
                surface_loader,
                swapchain_loader,
                present_mode,
            )
            .expect("Failed to create swapchain");

        // create render command buffers
        let render_command_buffers = Self::create_render_command_buffers(
            &device,
            command_pool,
            swapchain_images.len() as u32,
        )
        .expect("Failed to create render command buffers");

        // create sync objects
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(&device, swapchain_images.len() as u32)
                .expect("Failed to create sync objects");

        // create render pass + pipeline
        let render_pass = Self::create_render_pass(&device, swapchain_format);
        let pipeline_layout = Self::create_pipeline_layout(&device, descriptor_set_layout);
        let pipeline = Self::create_pipeline(&device, render_pass, pipeline_layout);

        // create framebuffers and vertex/index buffers
        let (framebuffers, swapchain_image_views) = Self::create_framebuffers(
            &device,
            &swapchain_images,
            render_pass,
            swapchain_format,
            width,
            height,
        );
        let (vertex_buffers, vertex_buffer_allocations, index_buffers, index_buffer_allocations) =
            Self::create_vertex_index_buffers(&device, swapchain_images.len(), &allocator);

        Some(Self {
            surface,
            swapchain,
            swapchain_images,
            swapchain_format,
            swapchain_extent,
            width,
            height,
            present_mode,
            dirty_flag: false,

            render_command_buffers,
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
            current_frame: 0,

            render_pass,
            pipeline_layout,
            pipeline,
            swapchain_image_views,
            framebuffers,

            vertex_buffers,
            vertex_buffer_allocations,
            index_buffers,
            index_buffer_allocations,

            scale_factor: 1.0,
            physical_width: width,
            physical_height: height,

            device,
            entry: entry.clone(),
            instance: instance.clone(),
            physical_device,
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            queue,
            command_pool,
            descriptor_set_layout,
            allocator,
        })
    }

    /// Create a viewport context for the Handle path (no owned swapchain/surface).
    /// Pipeline state will be created lazily when update_pipeline is first called.
    pub(crate) fn new_for_handle_path(
        entry: Entry,
        instance: Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        surface_loader: ash::khr::surface::Instance,
        swapchain_loader: ash::khr::swapchain::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        allocator: A,
    ) -> Self {
        Self {
            surface: vk::SurfaceKHR::null(),
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: vec![],
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D::default(),
            width: 0,
            height: 0,
            present_mode: vk::PresentModeKHR::FIFO,
            dirty_flag: false,

            render_command_buffers: vec![],
            in_flight_fences: vec![],
            image_available_semaphores: vec![],
            render_finished_semaphores: vec![],
            current_frame: 0,

            render_pass: vk::RenderPass::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            swapchain_image_views: vec![],
            framebuffers: vec![],

            vertex_buffers: vec![],
            vertex_buffer_allocations: vec![],
            index_buffers: vec![],
            index_buffer_allocations: vec![],

            scale_factor: 0.0,
            physical_width: 0,
            physical_height: 0,

            device,
            entry,
            instance,
            physical_device,
            surface_loader,
            swapchain_loader,
            queue,
            command_pool,
            descriptor_set_layout,
            allocator,
        }
    }

    pub(crate) fn mark_dirty(&mut self) {
        self.dirty_flag = true;
    }

    pub(crate) fn swapchain_recreate_required(&self, current_scale: f32) -> bool {
        self.scale_factor != current_scale
    }

    pub(crate) fn recreate_if_dirty(&mut self, window: &winit::window::Window) {
        if !self.dirty_flag {
            return;
        }

        let width = window.inner_size().width;
        let height = window.inner_size().height;

        // if window is minimized, do nothing
        if width == 0 || height == 0 {
            return;
        }

        // wait device idle
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }

        // cleanup old sync objects, command buffers, swapchain
        unsafe {
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &cmd in &self.render_command_buffers {
                self.device
                    .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        // destroy pipeline resources (framebuffers, image views, vertex/index buffers)
        self.destroy_pipeline_resources();

        // create new swapchain
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            Self::create_swapchain(
                width,
                height,
                self.physical_device,
                self.surface,
                &self.surface_loader,
                &self.swapchain_loader,
                self.present_mode,
            )
            .expect("Failed to recreate swapchain");

        // create new command buffers
        let render_command_buffers = Self::create_render_command_buffers(
            &self.device,
            self.command_pool,
            swapchain_images.len() as u32,
        )
        .expect("Failed to create render command buffers");

        // create new sync objects
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(&self.device, swapchain_images.len() as u32)
                .expect("Failed to create sync objects");

        self.width = width;
        self.height = height;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;
        self.render_command_buffers = render_command_buffers;
        self.in_flight_fences = in_flight_fences;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;
        self.current_frame = 0;

        // rebuild pipeline resources from new swapchain
        self.build_pipeline_resources_from_swapchain();

        self.dirty_flag = false;
    }

    /// Update pipeline state from externally-provided swapchain info (Handle path).
    pub(crate) fn update_pipeline(&mut self, info: SwapchainUpdateInfo, scale_factor: f32, physical_size: winit::dpi::PhysicalSize<u32>) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }

        // destroy old pipeline resources
        self.destroy_pipeline_resources();

        // create render pass + pipeline only if first call
        if self.render_pass == vk::RenderPass::null() {
            self.render_pass = Self::create_render_pass(&self.device, info.surface_format);
            self.pipeline_layout =
                Self::create_pipeline_layout(&self.device, self.descriptor_set_layout);
            self.pipeline =
                Self::create_pipeline(&self.device, self.render_pass, self.pipeline_layout);
        }

        // create framebuffers and buffers from provided images
        let (framebuffers, swapchain_image_views) = Self::create_framebuffers(
            &self.device,
            &info.swapchain_images,
            self.render_pass,
            info.surface_format,
            info.width,
            info.height,
        );
        let (vertex_buffers, vertex_buffer_allocations, index_buffers, index_buffer_allocations) =
            Self::create_vertex_index_buffers(&self.device, info.swapchain_images.len(), &self.allocator);

        self.width = info.width;
        self.height = info.height;
        self.swapchain_format = info.surface_format;
        self.swapchain_images = info.swapchain_images;
        self.framebuffers = framebuffers;
        self.swapchain_image_views = swapchain_image_views;
        self.vertex_buffers = vertex_buffers;
        self.vertex_buffer_allocations = vertex_buffer_allocations;
        self.index_buffers = index_buffers;
        self.index_buffer_allocations = index_buffer_allocations;
        self.scale_factor = scale_factor;
        self.physical_width = physical_size.width;
        self.physical_height = physical_size.height;
    }

    /// Record egui draw commands into the given command buffer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_commands(
        &mut self,
        cmd: vk::CommandBuffer,
        index: usize,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: egui::TexturesDelta,
        scale: f32,
        size: winit::dpi::PhysicalSize<u32>,
        managed_textures: &mut ManagedTextures<A>,
        user_textures: &mut UserTextures,
    ) {
        // update textures
        managed_textures.update_textures(textures_delta);
        user_textures.update_textures();

        // get buffer ptrs
        let mut vertex_buffer_ptr = self.vertex_buffer_allocations[index]
            .mapped_ptr()
            .unwrap()
            .as_ptr()
            .cast::<u8>();
        let vertex_buffer_ptr_end =
            unsafe { vertex_buffer_ptr.add(Self::vertex_buffer_size() as usize) };
        let mut index_buffer_ptr = self.index_buffer_allocations[index]
            .mapped_ptr()
            .unwrap()
            .as_ptr()
            .cast::<u8>();
        let index_buffer_ptr_end =
            unsafe { index_buffer_ptr.add(Self::index_buffer_size() as usize) };

        // begin render pass
        unsafe {
            self.device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index])
                    .clear_values(&[])
                    .render_area(
                        vk::Rect2D::default().extent(
                            vk::Extent2D::default()
                                .width(self.width)
                                .height(self.height),
                        ),
                    ),
                vk::SubpassContents::INLINE,
            );
        }

        // bind resources
        let width_points = size.width as f32 / scale;
        let height_points = size.height as f32 / scale;
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_bind_vertex_buffers(
                cmd,
                0,
                &[self.vertex_buffers[index]],
                &[0],
            );
            self.device.cmd_bind_index_buffer(
                cmd,
                self.index_buffers[index],
                0,
                vk::IndexType::UINT32,
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes_of(&width_points),
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                4,
                bytes_of(&height_points),
            );
        }

        // render meshes
        let mut vertex_base = 0;
        let mut index_base = 0;
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_primitives
        {
            let mesh = match primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh,
                egui::epaint::Primitive::Callback(_) => todo!(),
            };
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            unsafe {
                match mesh.texture_id {
                    egui::TextureId::User(id) => {
                        if let Some(&descriptor_set) = user_textures.texture_desc_sets.get(&id) {
                            self.device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[descriptor_set],
                                &[],
                            );
                        } else {
                            log::error!(
                                "This UserTexture has already been unregistered: {:?}",
                                mesh.texture_id
                            );
                            continue;
                        }
                    }
                    egui::TextureId::Managed(_) => {
                        self.device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[*managed_textures
                                .texture_desc_sets
                                .get(&mesh.texture_id)
                                .unwrap()],
                            &[],
                        );
                    }
                }
            }

            let v_slice = &mesh.vertices;
            let v_size = std::mem::size_of::<egui::epaint::Vertex>();
            let v_copy_size = v_slice.len() * v_size;

            let i_slice = &mesh.indices;
            let i_size = std::mem::size_of::<u32>();
            let i_copy_size = i_slice.len() * i_size;

            let vertex_buffer_ptr_next = unsafe { vertex_buffer_ptr.add(v_copy_size) };
            let index_buffer_ptr_next = unsafe { index_buffer_ptr.add(i_copy_size) };

            if vertex_buffer_ptr_next >= vertex_buffer_ptr_end
                || index_buffer_ptr_next >= index_buffer_ptr_end
            {
                panic!("egui paint out of memory");
            }

            unsafe {
                vertex_buffer_ptr.copy_from(v_slice.as_ptr().cast::<u8>(), v_copy_size);
                index_buffer_ptr.copy_from(i_slice.as_ptr().cast::<u8>(), i_copy_size);
            }

            vertex_buffer_ptr = vertex_buffer_ptr_next;
            index_buffer_ptr = index_buffer_ptr_next;

            unsafe {
                let min = clip_rect.min;
                let min = egui::Pos2 {
                    x: min.x * scale,
                    y: min.y * scale,
                };
                let min = egui::Pos2 {
                    x: f32::clamp(min.x, 0.0, size.width as f32),
                    y: f32::clamp(min.y, 0.0, size.height as f32),
                };
                let max = clip_rect.max;
                let max = egui::Pos2 {
                    x: max.x * scale,
                    y: max.y * scale,
                };
                let max = egui::Pos2 {
                    x: f32::clamp(max.x, min.x, size.width as f32),
                    y: f32::clamp(max.y, min.y, size.height as f32),
                };
                self.device.cmd_set_scissor(
                    cmd,
                    0,
                    std::slice::from_ref(
                        &vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: min.x.round() as i32,
                                y: min.y.round() as i32,
                            })
                            .extent(vk::Extent2D {
                                width: (max.x.round() - min.x) as u32,
                                height: (max.y.round() - min.y) as u32,
                            }),
                    ),
                );
                self.device.cmd_set_viewport(
                    cmd,
                    0,
                    std::slice::from_ref(
                        &vk::Viewport::default()
                            .x(0.0)
                            .y(0.0)
                            .width(size.width as f32)
                            .height(size.height as f32)
                            .min_depth(0.0)
                            .max_depth(1.0),
                    ),
                );
                self.device.cmd_draw_indexed(
                    cmd,
                    mesh.indices.len() as u32,
                    1,
                    index_base,
                    vertex_base,
                    0,
                );
            }

            vertex_base += mesh.vertices.len() as i32;
            index_base += mesh.indices.len() as u32;
        }

        // end render pass
        unsafe {
            self.device.cmd_end_render_pass(cmd);
        }
    }

    /// Full auto-managed frame: acquire, barrier, record, submit, present.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn present_managed(
        &mut self,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        scale: f32,
        size: winit::dpi::PhysicalSize<u32>,
        managed_textures: &mut ManagedTextures<A>,
        user_textures: &mut UserTextures,
    ) -> Result<()> {
        // Wait for the resources at this index to be completed on the GPU
        unsafe {
            self.device.wait_for_fences(
                std::slice::from_ref(&self.in_flight_fences[self.current_frame]),
                true,
                u64::MAX,
            )?;
        }

        // check if we need to recreate pipeline resources due to scale factor change
        if self.swapchain_recreate_required(scale) {
            self.scale_factor = scale;
            self.physical_width = size.width;
            self.physical_height = size.height;
        }

        // acquire next image
        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        };
        let index = match result {
            Ok((index, _)) => index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty_flag = true;
                return Ok(());
            }
            Err(error) => return Err(anyhow::anyhow!(error)),
        };

        // clear command buffer
        let cmd = self.render_command_buffers[self.current_frame];
        unsafe {
            self.device.reset_command_buffer(
                cmd,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;
        }

        // begin command buffer
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device.begin_command_buffer(cmd, &command_buffer_begin_info)?;
        }

        // update swapchain if needed
        if self.dirty_flag {
            // rebuild pipeline resources (framebuffers + vertex/index buffers)
            unsafe {
                self.device.device_wait_idle().expect("device_wait_idle failed");
            }
            self.destroy_pipeline_resources();
            self.build_pipeline_resources_from_swapchain();
            self.dirty_flag = false;
        }

        // image layout barrier: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        utils::insert_image_memory_barrier(
            &self.device,
            cmd,
            self.swapchain_images[index],
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1),
        );

        // record egui draw commands
        self.record_commands(
            cmd,
            index,
            &clipped_primitives,
            textures_delta,
            scale,
            size,
            managed_textures,
            user_textures,
        );

        // end command buffer
        unsafe {
            self.device.end_command_buffer(cmd)?;
        }

        // reset fence
        unsafe {
            self.device.reset_fences(std::slice::from_ref(
                &self.in_flight_fences[self.current_frame],
            ))?;
        }

        // submit
        let buffers_to_submit = [cmd];
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
            self.device.queue_submit(
                self.queue,
                std::slice::from_ref(&submit_info),
                self.in_flight_fences[self.current_frame],
            )?;
        }

        // present
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
        let is_dirty = match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(error) => panic!("Failed to present queue. Cause: {error}"),
            _ => false,
        };
        if is_dirty {
            self.dirty_flag = true;
        }

        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();

        Ok(())
    }

    pub(crate) fn destroy(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            // destroy sync objects and command buffers
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &cmd in &self.render_command_buffers {
                self.device
                    .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
            }

            // destroy swapchain (if owned)
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            // destroy surface (if owned)
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
            }
        }

        // destroy pipeline resources (framebuffers, image views, vertex/index buffers)
        self.destroy_pipeline_resources();

        // destroy pipeline, pipeline layout, render pass
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.render_pass, None);
            }
        }
    }
}
