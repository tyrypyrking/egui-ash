//! Swapchain-based triangle renderer used by the `multi_viewports` example.
//!
//! Renders a coloured triangle that auto-rotates over time, demonstrating
//! custom Vulkan rendering in the main window while egui occupies a deferred
//! viewport in a second window.

use ash::{vk, Device};
use egui_ash::EguiCommand;
use glam::{Mat4, Vec3};
use gpu_allocator::vulkan::{Allocation, Allocator};
use std::{
    ffi::CString,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
    time::Instant,
};

use super::common::render::{
    create_command_buffers, create_descriptor_pool, create_descriptor_set_layouts,
    create_descriptor_sets, create_swapchain, create_sync_objects, create_uniform_buffers,
};

// ── include_spirv! ────────────────────────────────────────────────────────────
// Kept local so that `include_bytes!` resolves paths relative to this file.
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

// ── Triangle-specific vertex layout ──────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone)]
struct Vertex {
    position: Vec3,
    color: Vec3,
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

// ── Uniform buffer ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone)]
struct UniformBufferObject {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
}

// ── Inner state ───────────────────────────────────────────────────────────────

struct TriangleRendererInner {
    width: u32,
    height: u32,

    /// Set on the first rendered frame to track elapsed time for auto-rotation.
    start_time: Option<Instant>,

    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    surface_loader: Arc<ash::khr::surface::Instance>,
    swapchain_loader: Arc<ash::khr::swapchain::Device>,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,
    queue_family_index: u32,

    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<Allocation>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    /// Color-only render pass (no depth attachment for this renderer).
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    color_image_views: Vec<vk::ImageView>,

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

impl TriangleRendererInner {
    // ── Triangle-specific Vulkan setup ────────────────────────────────────────

    /// Color-only render pass without a depth attachment.
    fn create_render_pass(device: &Device, surface_format: vk::SurfaceFormatKHR) -> vk::RenderPass {
        let attachments = [vk::AttachmentDescription::default()
            .format(surface_format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let color_reference = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_reference)];
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
        unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses)
                        .dependencies(&dependencies),
                    None,
                )
                .expect("Failed to create render pass")
        }
    }

    /// Framebuffers without depth images (triangle doesn't need depth testing).
    fn create_framebuffers(
        device: &Device,
        render_pass: vk::RenderPass,
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
        swapchain_images: &[vk::Image],
    ) -> (Vec<vk::Framebuffer>, Vec<vk::ImageView>) {
        let mut framebuffers = Vec::with_capacity(swapchain_images.len());
        let mut color_image_views = Vec::with_capacity(swapchain_images.len());

        for &image in swapchain_images {
            let color_view = unsafe {
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
            let framebuffer = unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&[color_view])
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer")
            };
            color_image_views.push(color_view);
            framebuffers.push(framebuffer);
        }
        (framebuffers, color_image_views)
    }

    /// Graphics pipeline using the triangle vertex/fragment shaders.
    fn create_graphics_pipeline(
        device: &Device,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        render_pass: vk::RenderPass,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_spirv = include_spirv!("../common/shaders/spv/triangle.vert.spv");
        let frag_spirv = include_spirv!("../common/shaders/spv/triangle.frag.spv");
        let main = CString::new("main").unwrap();

        let vert_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&vert_spirv),
                    None,
                )
                .expect("Failed to create vertex shader module")
        };
        let frag_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&frag_spirv),
                    None,
                )
                .expect("Failed to create fragment shader module")
        };

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&main),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&main),
        ];
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(descriptor_set_layouts),
                    None,
                )
                .expect("Failed to create pipeline layout")
        };
        let vertex_binding = Vertex::get_binding_descriptions();
        let vertex_attributes = Vertex::get_attribute_descriptions();
        let stencil_op = vk::StencilOpState::default()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_attributes)
            .vertex_binding_descriptions(&vertex_binding);
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .stencil_test_enable(false)
            .front(stencil_op)
            .back(stencil_op);
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_create_info),
                    None,
                )
                .unwrap()[0]
        };
        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }
        (pipeline, pipeline_layout)
    }

    /// Upload a hardcoded RGB triangle to a GPU-local vertex buffer.
    fn create_vertex_buffer(
        device: &Device,
        allocator: &Mutex<Allocator>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> (vk::Buffer, Allocation, u32) {
        let vertices = [
            Vertex {
                position: Vec3::new(0.0, 1.0, 0.0),
                color: Vec3::new(0.0, 0.0, 1.0),
            },
            Vertex {
                position: Vec3::new(-1.0, -1.0, 0.0),
                color: Vec3::new(1.0, 0.0, 0.0),
            },
            Vertex {
                position: Vec3::new(1.0, -1.0, 0.0),
                color: Vec3::new(0.0, 1.0, 0.0),
            },
        ];
        let vertex_buffer_size = vertices.len() as u64 * std::mem::size_of::<Vertex>() as u64;

        let mut alloc = allocator.lock().unwrap();

        // Stage into a CPU-visible buffer then copy to GPU-only storage.
        let staging = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(vertex_buffer_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .expect("Failed to create staging buffer")
        };
        let staging_alloc = alloc
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Temporary Vertex Buffer",
                requirements: unsafe { device.get_buffer_memory_requirements(staging) },
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate staging memory");
        unsafe {
            device
                .bind_buffer_memory(staging, staging_alloc.memory(), staging_alloc.offset())
                .expect("Failed to bind staging memory");
            let ptr = staging_alloc.mapped_ptr().unwrap().as_ptr() as *mut Vertex;
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
                .expect("Failed to create vertex buffer")
        };
        let vertex_alloc = alloc
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Vertex Buffer",
                requirements: unsafe { device.get_buffer_memory_requirements(vertex_buffer) },
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate vertex buffer memory");
        unsafe {
            device
                .bind_buffer_memory(vertex_buffer, vertex_alloc.memory(), vertex_alloc.offset())
                .expect("Failed to bind vertex buffer memory");
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
                staging,
                vertex_buffer,
                &[vk::BufferCopy::default().size(vertex_buffer_size)],
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

        alloc
            .free(staging_alloc)
            .expect("Failed to free staging memory");
        unsafe { device.destroy_buffer(staging, None) };

        (vertex_buffer, vertex_alloc, vertices.len() as u32)
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    fn new(
        physical_device: vk::PhysicalDevice,
        device: Arc<Device>,
        surface_loader: Arc<ash::khr::surface::Instance>,
        swapchain_loader: Arc<ash::khr::swapchain::Device>,
        allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
    ) -> Self {
        let (swapchain, surface_format, surface_extent, swapchain_images) = create_swapchain(
            physical_device,
            &surface_loader,
            &swapchain_loader,
            surface,
            queue_family_index,
            width,
            height,
        );
        let (uniform_buffers, uniform_buffer_allocations) = create_uniform_buffers(
            &device,
            &*allocator,
            swapchain_images.len(),
            std::mem::size_of::<UniformBufferObject>() as u64,
        );
        let descriptor_pool = create_descriptor_pool(&device, swapchain_images.len());
        let descriptor_set_layouts = create_descriptor_set_layouts(&device, swapchain_images.len());
        let descriptor_sets = create_descriptor_sets(
            &device,
            descriptor_pool,
            &descriptor_set_layouts,
            &uniform_buffers,
        );
        let render_pass = Self::create_render_pass(&device, surface_format);
        let (framebuffers, color_image_views) = Self::create_framebuffers(
            &device,
            render_pass,
            surface_format,
            surface_extent,
            &swapchain_images,
        );
        let (pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let (vertex_buffer, vertex_buffer_allocation, vertex_count) =
            Self::create_vertex_buffer(&device, &**allocator, command_pool, queue);
        let command_buffers = create_command_buffers(&device, command_pool, swapchain_images.len());
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            create_sync_objects(&device, swapchain_images.len());

        Self {
            width,
            height,
            start_time: None,
            physical_device,
            device,
            surface_loader,
            swapchain_loader,
            allocator,
            surface,
            queue,
            queue_family_index,
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
            color_image_views,
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

    /// Rebuild the swapchain and all per-image resources after a resize or
    /// out-of-date notification.
    fn recreate_swapchain(&mut self, width: u32, height: u32, egui_cmd: &mut EguiCommand) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            for &sem in &self.image_available_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &sem in &self.render_finished_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &view in &self.color_image_views {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        self.width = width;
        self.height = height;

        let (swapchain, surface_format, surface_extent, swapchain_images) = create_swapchain(
            self.physical_device,
            &self.surface_loader,
            &self.swapchain_loader,
            self.surface,
            self.queue_family_index,
            width,
            height,
        );
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
        let (framebuffers, color_image_views) = Self::create_framebuffers(
            &self.device,
            self.render_pass,
            self.surface_format,
            self.surface_extent,
            &self.swapchain_images,
        );
        self.framebuffers = framebuffers;
        self.color_image_views = color_image_views;

        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            create_sync_objects(&self.device, self.swapchain_images.len());
        self.in_flight_fences = in_flight_fences;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;

        self.current_frame = 0;
        self.dirty_swapchain = false;
    }

    fn render(&mut self, width: u32, height: u32, mut egui_cmd: EguiCommand) {
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

        // Wait for the previous use of this frame's resources to complete.
        unsafe {
            self.device
                .wait_for_fences(
                    std::slice::from_ref(&self.in_flight_fences[self.current_frame]),
                    true,
                    u64::MAX,
                )
                .expect("Failed to wait for fence");
            self.device
                .reset_fences(std::slice::from_ref(
                    &self.in_flight_fences[self.current_frame],
                ))
                .expect("Failed to reset fence");
        }

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
                self.dirty_swapchain = true;
                return;
            }
            Err(_) => return,
        };

        // Derive rotation from elapsed time (45°/s).
        let rotate_y = {
            let t = self.start_time.get_or_insert_with(Instant::now);
            t.elapsed().as_secs_f32() * 45.0
        };

        // Upload MVP uniform buffer.
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

        let cb = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin command buffer");

            self.device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index])
                    .render_area(vk::Rect2D::default().extent(self.surface_extent))
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.5, 0.1, 0.1, 1.0],
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
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport::default()
                    .width(width as f32)
                    .height(height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            self.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D::default().extent(self.surface_extent)],
            );
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.current_frame]],
                &[],
            );
            self.device
                .cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer], &[0]);
            self.device.cmd_draw(cb, self.vertex_count, 1, 0, 0);

            self.device.cmd_end_render_pass(cb);
            egui_cmd.record(cb, index);

            self.device
                .end_command_buffer(cb)
                .expect("Failed to end command buffer");
        }

        unsafe {
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[cb])
                        .wait_semaphores(std::slice::from_ref(
                            &self.image_available_semaphores[self.current_frame],
                        ))
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .signal_semaphores(std::slice::from_ref(
                            &self.render_finished_semaphores[self.current_frame],
                        ))],
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit queue");
        }

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
        self.dirty_swapchain = match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(e) => panic!("Failed to present queue: {e}"),
            _ => false,
        };

        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();
    }

    /// Destroy all GPU resources.  Must be called before the Vulkan device is
    /// destroyed; `Drop` alone is **not** sufficient.
    fn destroy(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            let mut allocator = self.allocator.lock().unwrap();
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            for sem in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(sem, None);
            }
            for sem in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(sem, None);
            }
            self.device.destroy_buffer(self.vertex_buffer, None);
            if let Some(alloc) = self.vertex_buffer_allocation.take() {
                allocator.free(alloc).expect("Failed to free memory");
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &view in &self.color_image_views {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            for &layout in &self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for &buf in &self.uniform_buffers {
                self.device.destroy_buffer(buf, None);
            }
            for alloc in self.uniform_buffer_allocations.drain(..) {
                allocator.free(alloc).expect("Failed to free memory");
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// A swapchain-based renderer that draws an auto-rotating coloured triangle
/// and composites egui on top.
///
/// Cheaply clonable — inner state is reference-counted.
#[derive(Clone)]
pub struct TriangleRenderer {
    inner: Arc<Mutex<TriangleRendererInner>>,
}

impl TriangleRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        physical_device: vk::PhysicalDevice,
        device: Arc<Device>,
        surface_loader: Arc<ash::khr::surface::Instance>,
        swapchain_loader: Arc<ash::khr::swapchain::Device>,
        allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TriangleRendererInner::new(
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

    pub fn render(&self, width: u32, height: u32, egui_cmd: EguiCommand) {
        self.inner.lock().unwrap().render(width, height, egui_cmd);
    }

    /// Destroy all GPU resources.  Must be called before the Vulkan device is
    /// destroyed.
    pub fn destroy(&mut self) {
        self.inner.lock().unwrap().destroy();
    }
}
