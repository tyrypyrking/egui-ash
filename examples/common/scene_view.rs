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
        load_model_and_create_vertex_buffer, UniformBufferObject,
    },
    vkutils,
};

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

    fn create_render_pass(device: &Device) -> vk::RenderPass {
        let attachments = [
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
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
                                    .level_count(1)
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
                                    .level_count(1)
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

        let (uniform_buffers, uniform_buffer_allocations) = create_uniform_buffers(
            &device,
            &allocator,
            Self::IN_FLIGHT_FRAMES,
            std::mem::size_of::<UniformBufferObject>() as u64,
        );
        let descriptor_pool = create_descriptor_pool(&device, Self::IN_FLIGHT_FRAMES);
        let descriptor_set_layouts = create_descriptor_set_layouts(&device, Self::IN_FLIGHT_FRAMES);
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
            create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let (vertex_buffer, vertex_buffer_allocation, vertex_count) =
            load_model_and_create_vertex_buffer(&device, &allocator, command_pool, queue);
        let command_buffers = create_command_buffers(&device, command_pool, Self::IN_FLIGHT_FRAMES);
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

            // Insert image layout transitions and memory barriers for the color and depth images.
            vkutils::insert_image_memory_barrier(
                &self.device,
                &self.command_buffers[self.current_frame],
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
                    .base_array_layer(0u32)
                    .layer_count(1u32)
                    .base_mip_level(0u32)
                    .level_count(1u32),
            );
            vkutils::insert_image_memory_barrier(
                &self.device,
                &self.command_buffers[self.current_frame],
                &self.depth_images[self.current_frame],
                self.queue_family_index,
                self.queue_family_index,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .layer_count(1u32)
                    .level_count(1u32),
            );

            self.device.cmd_begin_render_pass(
                self.command_buffers[self.current_frame],
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
                    &vk::Rect2D::default().extent(
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
