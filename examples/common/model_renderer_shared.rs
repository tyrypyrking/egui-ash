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

struct SharedRendererInner {
    width: u32,
    height: u32,

    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    surface_loader: Arc<ash::khr::surface::Instance>,
    swapchain_loader: Arc<ash::khr::swapchain::Device>,
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
impl SharedRendererInner {
    pub fn recreate_swapchain(&mut self, width: u32, height: u32, egui_cmd: &mut EguiCommand) {
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
                .unwrap_or(&surface_formats[0]);

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

            (swapchain, swapchain_images, *surface_format, surface_extent)
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

    fn new(create_info: RendererSharedCreationInfo) -> Self {
        let RendererSharedCreationInfo {
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
        } = create_info;

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
            &allocator.as_ref(),
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
        let render_pass = create_render_pass(&device, surface_format);
        let FrameBufferInfo {
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        } = create_framebuffers(
            &device,
            &allocator.as_ref(),
            render_pass,
            surface_format,
            surface_extent,
            &swapchain_images,
        );
        let (pipeline, pipeline_layout) =
            create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let (vertex_buffer, vertex_buffer_allocation, vertex_count) =
            load_model_and_create_vertex_buffer(&device, &allocator.as_ref(), command_pool, queue);
        let command_buffers = create_command_buffers(&device, command_pool, swapchain_images.len());
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            create_sync_objects(&device, swapchain_images.len());

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
pub struct RendererShared {
    inner: Arc<Mutex<SharedRendererInner>>,
}

pub struct RendererSharedCreationInfo {
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub surface_loader: Arc<ash::khr::surface::Instance>,
    pub swapchain_loader: Arc<ash::khr::swapchain::Device>,
    pub allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    pub surface: vk::SurfaceKHR,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub width: u32,
    pub height: u32,
}

impl RendererShared {
    pub fn new(create_info: RendererSharedCreationInfo) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SharedRendererInner::new(create_info))),
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
