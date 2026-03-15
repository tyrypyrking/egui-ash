//! Scene view widgets for offscreen render-to-texture model rendering.
//!
//! Two complementary egui widgets are provided:
//!
//! - [`SimpleSceneView`]: renders the Suzanne model with drag-to-rotate
//!   interaction, storing rotation state internally.
//! - [`SceneView`]: renders the Suzanne model driven by an external [`Scene`],
//!   which also carries lighting and material properties.
//!
//! Both widgets use double-buffered offscreen images registered as egui user
//! textures.  Call `render()` once per frame (inside
//! [`App::request_redraw`](egui_ash::App::request_redraw)) before the egui
//! pass, then add the widget with `ui.add(...)`.

use ash::{vk, Device};
use glam::{Mat4, Vec3};
use gpu_allocator::vulkan::{Allocation, Allocator};
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use crate::common::{
    render::{
        create_command_buffers, create_descriptor_pool, create_descriptor_set_layouts,
        create_descriptor_sets, create_graphics_pipeline, create_uniform_buffers,
        load_model_and_create_vertex_buffer,
    },
    scene::Scene,
    vkutils,
};

// ── Uniform buffer objects ────────────────────────────────────────────────────

/// MVP-only uniform buffer used by [`SimpleSceneView`].
#[repr(C)]
#[derive(Clone, Copy)]
struct MvpUbo {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
}

/// Full Phong-shading uniform buffer used by [`SceneView`].
#[repr(C)]
#[derive(Clone, Copy)]
struct PhongUbo {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
    diffuse_color: [f32; 3],
    _pad0: f32,
    specular_color: [f32; 3],
    shininess: f32,
    light_position: [f32; 3],
    light_intensity: f32,
    light_color: [f32; 3],
    _pad1: f32,
}

// ── RendererCore ──────────────────────────────────────────────────────────────

/// Shared Vulkan resources for double-buffered offscreen rendering.
///
/// Owns the render pass, per-frame framebuffers, depth images, sampler,
/// graphics pipeline, vertex buffer, command buffers, and synchronisation
/// objects.  Callers submit a frame via [`render`](RendererCore::render),
/// which accepts raw UBO bytes and a clear colour so that the two concrete
/// scene-view types can supply different uniform data without duplicating
/// Vulkan boilerplate.
struct RendererCore {
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
}

impl RendererCore {
    const IN_FLIGHT_FRAMES: usize = 2;

    // ── Construction helpers ──────────────────────────────────────────────────

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
        let color_ref = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref)
            .depth_stencil_attachment(&depth_ref)];
        unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses),
                    None,
                )
                .expect("Failed to create render pass")
        }
    }

    /// Create per-frame color + depth images, views, and framebuffers.
    ///
    /// Also submits a one-time command to transition each color image to
    /// `SHADER_READ_ONLY_OPTIMAL` so that egui can sample it before the first
    /// render pass has written to it.
    fn create_offscreen_frames(
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
        Vec<Allocation>,
        Vec<vk::ImageView>,
        Vec<vk::Image>,
        Vec<Allocation>,
        Vec<vk::ImageView>,
    ) {
        let cap = Self::IN_FLIGHT_FRAMES;
        let mut framebuffers = Vec::with_capacity(cap);
        let mut color_images = Vec::with_capacity(cap);
        let mut color_allocs = Vec::with_capacity(cap);
        let mut color_views = Vec::with_capacity(cap);
        let mut depth_images = Vec::with_capacity(cap);
        let mut depth_allocs = Vec::with_capacity(cap);
        let mut depth_views = Vec::with_capacity(cap);

        for _ in 0..cap {
            // ── Color image ──────────────────────────────────────────────────
            let color_image = unsafe {
                device
                    .create_image(
                        &vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(vk::Format::R8G8B8A8_UNORM)
                            .extent(vk::Extent3D { width, height, depth: 1 })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .usage(
                                vk::ImageUsageFlags::COLOR_ATTACHMENT
                                    | vk::ImageUsageFlags::SAMPLED,
                            )
                            .initial_layout(vk::ImageLayout::UNDEFINED),
                        None,
                    )
                    .expect("Failed to create color image")
            };
            let color_alloc = allocator
                .lock()
                .unwrap()
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "Color Image",
                    requirements: unsafe { device.get_image_memory_requirements(color_image) },
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to allocate color image memory");
            unsafe {
                device
                    .bind_image_memory(color_image, color_alloc.memory(), color_alloc.offset())
                    .expect("Failed to bind color image memory");
            }
            let color_view = unsafe {
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
                    .expect("Failed to create color image view")
            };

            // ── Depth image ──────────────────────────────────────────────────
            let depth_image = unsafe {
                device
                    .create_image(
                        &vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(vk::Format::D32_SFLOAT)
                            .extent(vk::Extent3D { width, height, depth: 1 })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                            .initial_layout(vk::ImageLayout::UNDEFINED),
                        None,
                    )
                    .expect("Failed to create depth image")
            };
            let depth_alloc = allocator
                .lock()
                .unwrap()
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "Depth Image",
                    requirements: unsafe { device.get_image_memory_requirements(depth_image) },
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to allocate depth image memory");
            unsafe {
                device
                    .bind_image_memory(depth_image, depth_alloc.memory(), depth_alloc.offset())
                    .expect("Failed to bind depth image memory");
            }
            let depth_view = unsafe {
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

            // ── Framebuffer ──────────────────────────────────────────────────
            let framebuffer = unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&[color_view, depth_view])
                            .width(width)
                            .height(height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer")
            };

            // ── Initial layout transition ────────────────────────────────────
            // Transition the color image to SHADER_READ_ONLY_OPTIMAL so egui
            // can sample it on the very first frame before a render pass runs.
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
                device,
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
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(0)
                    .level_count(1),
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

            color_images.push(color_image);
            color_allocs.push(color_alloc);
            color_views.push(color_view);
            depth_images.push(depth_image);
            depth_allocs.push(depth_alloc);
            depth_views.push(depth_view);
            framebuffers.push(framebuffer);
        }

        (
            framebuffers,
            color_images,
            color_allocs,
            color_views,
            depth_images,
            depth_allocs,
            depth_views,
        )
    }

    fn create_sampler(device: &Device) -> vk::Sampler {
        unsafe {
            device
                .create_sampler(
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
                .expect("Failed to create sampler")
        }
    }

    fn create_fences(device: &Device) -> Vec<vk::Fence> {
        (0..Self::IN_FLIGHT_FRAMES)
            .map(|_| unsafe {
                device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .expect("Failed to create fence")
            })
            .collect()
    }

    fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        queue: vk::Queue,
        queue_family_index: u32,
        command_pool: vk::CommandPool,
        image_registry: egui_ash::ImageRegistry,
        ubo_size: u64,
    ) -> Self {
        let width = 1;
        let height = 1;

        let (uniform_buffers, uniform_buffer_allocations) =
            create_uniform_buffers(&device, &allocator, Self::IN_FLIGHT_FRAMES, ubo_size);
        let descriptor_pool = create_descriptor_pool(&device, Self::IN_FLIGHT_FRAMES);
        let descriptor_set_layouts =
            create_descriptor_set_layouts(&device, Self::IN_FLIGHT_FRAMES);
        let descriptor_sets = create_descriptor_sets(
            &device,
            descriptor_pool,
            &descriptor_set_layouts,
            &uniform_buffers,
        );
        let render_pass = Self::create_render_pass(&device);
        let (
            framebuffers,
            color_images,
            color_image_allocations,
            color_image_views,
            depth_images,
            depth_image_allocations,
            depth_image_views,
        ) = Self::create_offscreen_frames(
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
            create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let (vertex_buffer, vertex_buffer_allocation, vertex_count) =
            load_model_and_create_vertex_buffer(&device, &allocator, command_pool, queue);
        let command_buffers =
            create_command_buffers(&device, command_pool, Self::IN_FLIGHT_FRAMES);
        let in_flight_fences = Self::create_fences(&device);

        let mut texture_ids = Vec::with_capacity(color_image_views.len());
        for &view in &color_image_views {
            texture_ids.push(image_registry.register_user_texture(view, sampler));
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
            color_image_allocations,
            depth_images,
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
        }
    }

    // ── Per-frame API ─────────────────────────────────────────────────────────

    /// Recreate all offscreen images and framebuffers after a resize.
    fn recreate_images(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");
            let mut allocator = self.allocator.lock().unwrap();
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &v in &self.color_image_views {
                self.device.destroy_image_view(v, None);
            }
            for &v in &self.depth_image_views {
                self.device.destroy_image_view(v, None);
            }
            for a in self.color_image_allocations.drain(..) {
                allocator.free(a).expect("Failed to free memory");
            }
            for a in self.depth_image_allocations.drain(..) {
                allocator.free(a).expect("Failed to free memory");
            }
            for img in self.color_images.drain(..) {
                self.device.destroy_image(img, None);
            }
            for img in self.depth_images.drain(..) {
                self.device.destroy_image(img, None);
            }
            for id in self.texture_ids.drain(..) {
                self.image_registry.unregister_user_texture(id);
            }
        }

        let (
            framebuffers,
            color_images,
            color_image_allocations,
            color_image_views,
            depth_images,
            depth_image_allocations,
            depth_image_views,
        ) = Self::create_offscreen_frames(
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
        self.color_image_allocations = color_image_allocations;
        self.color_image_views = color_image_views;
        self.depth_images = depth_images;
        self.depth_image_allocations = depth_image_allocations;
        self.depth_image_views = depth_image_views;

        for &view in &self.color_image_views {
            self.texture_ids
                .push(self.image_registry.register_user_texture(view, self.sampler));
        }

        self.current_frame = 0;
        self.size_changed = false;
    }

    /// Record and submit a full model draw for the current frame.
    ///
    /// `ubo_bytes` is written verbatim into the current frame's uniform buffer.
    /// `clear_color` is used as the background colour for the render pass.
    fn render(&mut self, ubo_bytes: &[u8], clear_color: [f32; 4]) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        if self.size_changed {
            self.recreate_images();
        }

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

        // Upload UBO data for this frame.
        unsafe {
            let dst = self.uniform_buffer_allocations[self.current_frame]
                .mapped_ptr()
                .unwrap()
                .as_ptr() as *mut u8;
            dst.copy_from_nonoverlapping(ubo_bytes.as_ptr(), ubo_bytes.len());
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

            // Transition color image: shader-read → color-attachment.
            vkutils::insert_image_memory_barrier(
                &self.device,
                &cb,
                &self.color_images[self.current_frame],
                self.queue_family_index,
                self.queue_family_index,
                vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(0)
                    .level_count(1),
            );

            self.device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[self.current_frame])
                    .render_area(
                        vk::Rect2D::default().extent(
                            vk::Extent2D::default()
                                .width(self.width)
                                .height(self.height),
                        ),
                    )
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue { float32: clear_color },
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
                    .width(self.width as f32)
                    .height(self.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            self.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D::default().extent(
                    vk::Extent2D::default()
                        .width(self.width)
                        .height(self.height),
                )],
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

            // Transition color image back: color-attachment → shader-read.
            vkutils::insert_image_memory_barrier(
                &self.device,
                &cb,
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
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(0)
                    .level_count(1),
            );

            self.device
                .end_command_buffer(cb)
                .expect("Failed to end command buffer");
        }

        unsafe {
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cb])],
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit queue");
        }

        self.current_frame = (self.current_frame + 1) % Self::IN_FLIGHT_FRAMES;
    }

    /// Update the render target dimensions.
    ///
    /// Images are recreated lazily at the start of the next
    /// [`render`](RendererCore::render) call.
    fn set_size(&mut self, size: egui::Vec2) {
        let w = size.x as u32;
        let h = size.y as u32;
        if self.width != w || self.height != h {
            self.width = w;
            self.height = h;
            self.size_changed = true;
        }
    }

    /// Return the egui `TextureId` that is safe to sample in the upcoming egui
    /// frame (the *next* slot in the double-buffer ring).
    fn next_texture(&self) -> egui::TextureId {
        let next = (self.current_frame + 1) % Self::IN_FLIGHT_FRAMES;
        unsafe {
            self.device
                .wait_for_fences(
                    std::slice::from_ref(&self.in_flight_fences[next]),
                    true,
                    u64::MAX,
                )
                .expect("Failed to wait for fence");
        }
        self.texture_ids[next]
    }

    /// Destroy all GPU resources.
    ///
    /// Must be called explicitly before the Vulkan device is destroyed;
    /// `Drop` alone is **not** sufficient.
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
            if let Some(a) = self.vertex_buffer_allocation.take() {
                allocator.free(a).expect("Failed to free vertex buffer");
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_sampler(self.sampler, None);
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &v in &self.color_image_views {
                self.device.destroy_image_view(v, None);
            }
            for &v in &self.depth_image_views {
                self.device.destroy_image_view(v, None);
            }
            for a in self.color_image_allocations.drain(..) {
                allocator.free(a).expect("Failed to free memory");
            }
            for a in self.depth_image_allocations.drain(..) {
                allocator.free(a).expect("Failed to free memory");
            }
            for img in self.color_images.drain(..) {
                self.device.destroy_image(img, None);
            }
            for img in self.depth_images.drain(..) {
                self.device.destroy_image(img, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            for &layout in &self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            for &buf in &self.uniform_buffers {
                self.device.destroy_buffer(buf, None);
            }
            for a in self.uniform_buffer_allocations.drain(..) {
                allocator.free(a).expect("Failed to free memory");
            }
        }
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }
    }
}

// ── Shared camera helpers ─────────────────────────────────────────────────────

/// Standard right-hand view matrix looking at the origin from –Z.
fn default_view() -> Mat4 {
    Mat4::look_at_rh(
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    )
}

/// Perspective projection with 45° FOV for the given viewport dimensions.
fn default_proj(width: u32, height: u32) -> Mat4 {
    Mat4::perspective_rh(
        45.0_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        10.0,
    )
}

// ── SimpleSceneView ───────────────────────────────────────────────────────────

struct SimpleSceneViewState {
    renderer: RendererCore,
    /// Y-axis rotation in degrees (yaw), kept in [-180, 180).
    rotate_y: f32,
    /// X-axis rotation in degrees (pitch), clamped to [-90, 90].
    rotate_x: f32,
}

/// An egui widget that renders the Suzanne model into an offscreen texture.
///
/// Rotation is stored internally; drag the panel to rotate the model.
/// Call [`render`](SimpleSceneView::render) once per frame before the egui
/// pass.
///
/// Cheaply clonable — inner state is reference-counted.
#[derive(Clone)]
pub struct SimpleSceneView {
    inner: Arc<Mutex<SimpleSceneViewState>>,
}

impl SimpleSceneView {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        queue: vk::Queue,
        queue_family_index: u32,
        command_pool: vk::CommandPool,
        image_registry: egui_ash::ImageRegistry,
    ) -> Self {
        let renderer = RendererCore::new(
            device,
            allocator,
            queue,
            queue_family_index,
            command_pool,
            image_registry,
            std::mem::size_of::<MvpUbo>() as u64,
        );
        Self {
            inner: Arc::new(Mutex::new(SimpleSceneViewState {
                renderer,
                rotate_y: 0.0,
                rotate_x: 0.0,
            })),
        }
    }

    /// Render one frame into the offscreen texture.
    pub fn render(&mut self) {
        let mut s = self.inner.lock().unwrap();
        let ubo = MvpUbo {
            model: Mat4::from_rotation_x(s.rotate_x.to_radians())
                .mul_mat4(&Mat4::from_rotation_y(s.rotate_y.to_radians()))
                .to_cols_array(),
            view: default_view().to_cols_array(),
            proj: default_proj(s.renderer.width, s.renderer.height).to_cols_array(),
        };
        let ubo_bytes = unsafe {
            std::slice::from_raw_parts(
                &ubo as *const MvpUbo as *const u8,
                std::mem::size_of::<MvpUbo>(),
            )
        };
        s.renderer.render(ubo_bytes, [0.4, 0.3, 0.2, 1.0]);
    }

    /// Destroy all GPU resources.  Must be called before the Vulkan device is
    /// destroyed.
    pub fn destroy(&mut self) {
        self.inner.lock().unwrap().renderer.destroy();
    }
}

impl egui::Widget for &SimpleSceneView {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let mut s = self.inner.lock().unwrap();
        let texture_id = s.renderer.next_texture();
        let response = ui
            .with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                let size = ui.available_size();
                s.renderer.set_size(size);
                ui.image(egui::load::SizedTexture { id: texture_id, size })
            })
            .response
            .interact(egui::Sense::drag());
        if response.dragged() {
            let delta = response.drag_delta();
            s.rotate_y = (s.rotate_y - delta.x + 180.0) % 360.0 - 180.0;
            s.rotate_x = (s.rotate_x - delta.y).clamp(-90.0, 90.0);
        }
        response
    }
}

// ── SceneView ─────────────────────────────────────────────────────────────────

struct SceneViewState {
    renderer: RendererCore,
    scene: Arc<Mutex<Scene>>,
}

/// An egui widget that renders the Suzanne model driven by an external
/// [`Scene`].
///
/// The [`Scene`] carries rotation, material, lighting, and background-colour
/// settings that are applied immediately the next time
/// [`render`](SceneView::render) is called.  Dragging the panel updates
/// `scene.suzanne.rotation_x/y`.
///
/// Cheaply clonable — inner state is reference-counted.
#[derive(Clone)]
pub struct SceneView {
    inner: Arc<Mutex<SceneViewState>>,
}

impl SceneView {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        queue: vk::Queue,
        queue_family_index: u32,
        command_pool: vk::CommandPool,
        image_registry: egui_ash::ImageRegistry,
        scene: Arc<Mutex<Scene>>,
    ) -> Self {
        let renderer = RendererCore::new(
            device,
            allocator,
            queue,
            queue_family_index,
            command_pool,
            image_registry,
            std::mem::size_of::<PhongUbo>() as u64,
        );
        Self {
            inner: Arc::new(Mutex::new(SceneViewState { renderer, scene })),
        }
    }

    /// Render one frame into the offscreen texture.
    pub fn render(&mut self) {
        let mut s = self.inner.lock().unwrap();
        let (ubo, clear_color) = {
            let scene = s.scene.lock().unwrap();
            let ubo = PhongUbo {
                model: Mat4::from_rotation_x(scene.suzanne.rotation_x.to_radians())
                    .mul_mat4(&Mat4::from_rotation_y(scene.suzanne.rotation_y.to_radians()))
                    .mul_mat4(&Mat4::from_rotation_z(scene.suzanne.rotation_z.to_radians()))
                    .to_cols_array(),
                view: default_view().to_cols_array(),
                proj: default_proj(s.renderer.width, s.renderer.height).to_cols_array(),
                diffuse_color: scene.suzanne.diffuse_color,
                _pad0: 0.0,
                specular_color: scene.suzanne.specular_color,
                shininess: scene.suzanne.shininess,
                light_position: scene.light.position,
                light_intensity: scene.light.intensity,
                light_color: scene.light.color,
                _pad1: 0.0,
            };
            let clear = [
                scene.background.color[0],
                scene.background.color[1],
                scene.background.color[2],
                1.0,
            ];
            (ubo, clear)
        };
        let ubo_bytes = unsafe {
            std::slice::from_raw_parts(
                &ubo as *const PhongUbo as *const u8,
                std::mem::size_of::<PhongUbo>(),
            )
        };
        s.renderer.render(ubo_bytes, clear_color);
    }

    /// Destroy all GPU resources.  Must be called before the Vulkan device is
    /// destroyed.
    pub fn destroy(&mut self) {
        self.inner.lock().unwrap().renderer.destroy();
    }
}

impl egui::Widget for &SceneView {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let mut s = self.inner.lock().unwrap();
        let texture_id = s.renderer.next_texture();
        let response = ui
            .with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                let size = ui.available_size();
                s.renderer.set_size(size);
                ui.image(egui::load::SizedTexture { id: texture_id, size })
            })
            .response
            .interact(egui::Sense::drag());
        if response.dragged() {
            let delta = response.drag_delta();
            let mut scene = s.scene.lock().unwrap();
            scene.suzanne.rotation_y =
                (scene.suzanne.rotation_y - delta.x + 180.0) % 360.0 - 180.0;
            scene.suzanne.rotation_x =
                (scene.suzanne.rotation_x - delta.y).clamp(-90.0, 90.0);
        }
        response
    }
}

impl egui::Widget for &mut SceneView {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        egui::Widget::ui(&*self, ui)
    }
}
