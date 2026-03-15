//! Swapchain-based model renderer for examples that draw the Suzanne mesh
//! directly to a window surface alongside egui.
//!
//! Used by the `egui_ash_vulkan` and `multi_viewports` examples.

use ash::{vk, Device};
use egui_ash::EguiCommand;
use glam::{Mat4, Vec3};
use gpu_allocator::vulkan::{Allocation, Allocator};
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use crate::common::render::{
    create_command_buffers, create_descriptor_pool, create_descriptor_set_layouts,
    create_descriptor_sets, create_framebuffers, create_graphics_pipeline, create_render_pass,
    create_swapchain, create_sync_objects, create_uniform_buffers,
    load_model_and_create_vertex_buffer, FrameBufferInfo, UniformBufferObject,
};

// ── Inner state ───────────────────────────────────────────────────────────────

struct ModelRendererInner {
    width: u32,
    height: u32,

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
    fn new(info: ModelRendererCreationInfo) -> Self {
        let ModelRendererCreationInfo {
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
        } = info;

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
            &allocator,
            swapchain_images.len(),
            std::mem::size_of::<UniformBufferObject>() as u64,
        );
        let descriptor_pool = create_descriptor_pool(&device, swapchain_images.len());
        let descriptor_set_layouts =
            create_descriptor_set_layouts(&device, swapchain_images.len());
        let descriptor_sets = create_descriptor_sets(
            &device,
            descriptor_pool,
            &descriptor_set_layouts,
            &uniform_buffers,
        );
        let render_pass = create_render_pass(&device, surface_format);
        let FrameBufferInfo {
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        } = create_framebuffers(
            &device,
            &allocator,
            render_pass,
            surface_format,
            surface_extent,
            &swapchain_images,
        );
        let (pipeline, pipeline_layout) =
            create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let (vertex_buffer, vertex_buffer_allocation, vertex_count) =
            load_model_and_create_vertex_buffer(&device, &allocator, command_pool, queue);
        let command_buffers =
            create_command_buffers(&device, command_pool, swapchain_images.len());
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            create_sync_objects(&device, swapchain_images.len());

        Self {
            width,
            height,
            physical_device,
            device,
            surface_loader,
            swapchain_loader,
            allocator: ManuallyDrop::new(allocator),
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

    /// Rebuild the swapchain and all per-image resources after a resize or
    /// out-of-date notification.
    fn recreate_swapchain(&mut self, width: u32, height: u32, egui_cmd: &mut EguiCommand) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            let mut allocator = self.allocator.lock().unwrap();
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
            for &view in &self.depth_image_views {
                self.device.destroy_image_view(view, None);
            }
            for (img, alloc) in self.depth_images_and_allocations.drain(..) {
                self.device.destroy_image(img, None);
                allocator.free(alloc).expect("Failed to free memory");
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
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

        self.render_pass = create_render_pass(&self.device, self.surface_format);

        let FrameBufferInfo {
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        } = create_framebuffers(
            &self.device,
            &self.allocator.as_ref(),
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
            create_sync_objects(&self.device, self.swapchain_images.len());
        self.in_flight_fences = in_flight_fences;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;

        self.current_frame = 0;
        self.dirty_swapchain = false;
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
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &view in &self.color_image_views {
                self.device.destroy_image_view(view, None);
            }
            for &view in &self.depth_image_views {
                self.device.destroy_image_view(view, None);
            }
            for (img, alloc) in self.depth_images_and_allocations.drain(..) {
                self.device.destroy_image(img, None);
                allocator.free(alloc).expect("Failed to free memory");
            }
            self.device.destroy_render_pass(self.render_pass, None);
            for &layout in &self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            for &buf in &self.uniform_buffers {
                self.device.destroy_buffer(buf, None);
            }
            for alloc in self.uniform_buffer_allocations.drain(..) {
                allocator.free(alloc).expect("Failed to free memory");
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parameters for constructing a [`ModelRenderer`].
pub struct ModelRendererCreationInfo {
    pub physical_device: vk::PhysicalDevice,
    /// The logical device.  Wrapped in `Arc` so it can be shared with the
    /// application (e.g. for cleanup in `Drop`).
    pub device: Arc<Device>,
    pub surface_loader: Arc<ash::khr::surface::Instance>,
    pub swapchain_loader: Arc<ash::khr::swapchain::Device>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub surface: vk::SurfaceKHR,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub width: u32,
    pub height: u32,
}

/// A swapchain-based renderer that draws the Suzanne model and composites egui
/// on top.
///
/// Cheaply clonable — inner state is reference-counted.
#[derive(Clone)]
pub struct ModelRenderer {
    inner: Arc<Mutex<ModelRendererInner>>,
}

impl ModelRenderer {
    pub fn new(info: ModelRendererCreationInfo) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModelRendererInner::new(info))),
        }
    }

    /// Render one frame.  `rotate_y` is the model's Y-axis rotation in degrees.
    pub fn render(&self, width: u32, height: u32, egui_cmd: EguiCommand, rotate_y: f32) {
        self.inner
            .lock()
            .unwrap()
            .render(width, height, egui_cmd, rotate_y);
    }

    /// Destroy all GPU resources.  Must be called before the Vulkan device is
    /// destroyed.
    pub fn destroy(&mut self) {
        self.inner.lock().unwrap().destroy();
    }
}
