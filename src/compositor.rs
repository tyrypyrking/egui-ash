use ash::vk;
use bytemuck::bytes_of;
use std::collections::HashMap;
use std::ffi::CString;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return i;
        }
    }
    panic!("failed to find suitable memory type");
}

// ─────────────────────────────────────────────────────────────────────────────
// ManagedTexture
// ─────────────────────────────────────────────────────────────────────────────

/// Per-texture Vulkan resources for egui managed textures.
struct ManagedTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    descriptor_set: vk::DescriptorSet,
}

// ─────────────────────────────────────────────────────────────────────────────
// Compositor
// ─────────────────────────────────────────────────────────────────────────────

/// Size for vertex/index buffers (4 MiB each).
const BUFFER_SIZE: vk::DeviceSize = 4 * 1024 * 1024;

/// Reserved egui User texture ID for the engine viewport.
const ENGINE_VIEWPORT_USER_ID: u64 = u64::MAX;

/// The host-side compositing pipeline.
///
/// Manages swapchain, egui texture upload, render pass, pipeline,
/// per-frame command buffer recording, submission, and presentation.
pub(crate) struct Compositor {
    // Vulkan handles (cloned from caller)
    #[allow(dead_code)]
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    host_queue: vk::Queue,
    host_queue_family: u32,
    /// Optional mutex serialising queue access — see VulkanContext::queue_mutex.
    /// When `Some`, every queue_submit / queue_present acquires this first.
    queue_mutex: Option<std::sync::Arc<std::sync::Mutex<()>>>,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,

    // Swapchain
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_extent: vk::Extent2D,
    swapchain_format: vk::Format,
    present_mode: vk::PresentModeKHR,

    /// Extent used when the driver reports `current_extent == u32::MAX`
    /// (WM defers sizing to the client). The host updates this on every
    /// `WindowEvent::Resized` so subsequent swapchain recreation picks the
    /// real window size instead of a hard-coded fallback.
    fallback_extent: vk::Extent2D,

    // Sync (one set per frame-in-flight; count == swapchain image count)
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    // Render pass + framebuffers
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    // Command pool + buffers (one per frame-in-flight)
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // egui pipeline
    egui_pipeline_layout: vk::PipelineLayout,
    egui_pipeline: vk::Pipeline,
    egui_descriptor_pool: vk::DescriptorPool,
    egui_descriptor_set_layout: vk::DescriptorSetLayout,
    egui_sampler: vk::Sampler,

    // Engine viewport (egui User texture)
    engine_viewport_texture_id: egui::TextureId,
    engine_viewport_descriptor_set: vk::DescriptorSet,
    black_image: vk::Image,
    black_image_view: vk::ImageView,
    black_image_memory: vk::DeviceMemory,

    // egui managed textures
    managed_textures: HashMap<egui::TextureId, ManagedTexture>,

    // Vertex/index buffers (per frame-in-flight), HOST_VISIBLE|HOST_COHERENT
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_memory: Vec<vk::DeviceMemory>,
    vertex_buffer_ptrs: Vec<*mut u8>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_memory: Vec<vk::DeviceMemory>,
    index_buffer_ptrs: Vec<*mut u8>,

    // Cached memory properties
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

/// Outcome of a single `Compositor::render_frame` call.
///
/// Any `Ok(_)` value means the GPU submit succeeded and the requested
/// `signal_value` will eventually be signaled on the compositor timeline —
/// the caller must commit the value in the target pool. Only
/// `Err(ERROR_OUT_OF_DATE_KHR)` indicates that no submit occurred and the
/// caller must NOT commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameOutcome {
    /// Submit + present succeeded.
    Rendered,
    /// Submit succeeded, but the swapchain is stale (present returned
    /// `SUBOPTIMAL_KHR`/`ERROR_OUT_OF_DATE_KHR`, or the swapchain was flagged
    /// suboptimal). Caller should recreate the swapchain on the next frame.
    RenderedNeedsRebuild,
}

impl Compositor {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        host_queue: vk::Queue,
        host_queue_family: u32,
        surface: vk::SurfaceKHR,
        present_mode: vk::PresentModeKHR,
        queue_mutex: Option<std::sync::Arc<std::sync::Mutex<()>>>,
        fallback_extent: vk::Extent2D,
    ) -> Self {
        let surface_loader = ash::khr::surface::Instance::new(entry, instance);
        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);

        let mem_props = instance.get_physical_device_memory_properties(physical_device);

        // -- Descriptor set layout -----------------------------------------
        //
        // `UPDATE_AFTER_BIND` lets us mutate a descriptor set even when a
        // command buffer that already bound and drew with it is still in
        // the pending state. egui's managed-texture path does exactly that
        // every time the font atlas or a user texture is reuploaded — last
        // frame's CB is queued against a fence we haven't yet waited on,
        // and we re-point the set at a freshly created image.
        //
        // `UPDATE_UNUSED_WHILE_PENDING` alone is insufficient: the binding
        // is *used* (via cmd_bind_descriptor_sets + cmd_draw_indexed), not
        // merely present-but-unused. Only `UPDATE_AFTER_BIND` covers the
        // actively-bound case.
        //
        // Requirements (must be enabled at device creation — see VulkanContext docs):
        //   - `descriptorBindingSampledImageUpdateAfterBind` (VK 1.2)
        //   - `descriptorBindingUpdateUnusedWhilePending`    (VK 1.2)
        //   - layout flag `UPDATE_AFTER_BIND_POOL`
        //   - pool flag   `UPDATE_AFTER_BIND`
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .binding(0)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let binding_flags = [vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let egui_descriptor_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .bindings(&bindings)
                    .push_next(&mut flags_info),
                None,
            )
            .expect("Failed to create descriptor set layout");

        // -- Descriptor pool -----------------------------------------------
        let egui_descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(
                        vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                            | vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
                    )
                    .max_sets(1024)
                    .pool_sizes(std::slice::from_ref(
                        &vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024),
                    )),
                None,
            )
            .expect("Failed to create descriptor pool");

        // -- Sampler -------------------------------------------------------
        let egui_sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
            .expect("Failed to create sampler");

        // -- Swapchain -----------------------------------------------------
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            Self::create_swapchain_inner(
                physical_device,
                surface,
                &surface_loader,
                &swapchain_loader,
                present_mode,
                fallback_extent,
            );
        let swapchain_image_views =
            Self::create_image_views(device, &swapchain_images, swapchain_format);

        // -- Render pass ---------------------------------------------------
        let render_pass = Self::create_render_pass(device, swapchain_format);

        // -- Framebuffers --------------------------------------------------
        let framebuffers = Self::create_framebuffers(
            device,
            &swapchain_image_views,
            render_pass,
            swapchain_extent,
        );

        // -- Pipeline ------------------------------------------------------
        let egui_pipeline_layout = Self::create_pipeline_layout(device, egui_descriptor_set_layout);
        let egui_pipeline = Self::create_pipeline(device, render_pass, egui_pipeline_layout);

        // -- Command pool + buffers ----------------------------------------
        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(host_queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .expect("Failed to create command pool");

        let frame_count = swapchain_images.len();
        let command_buffers = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(frame_count as u32),
            )
            .expect("Failed to allocate command buffers");

        // -- Sync objects --------------------------------------------------
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(device, frame_count);

        // -- Vertex / index buffers ----------------------------------------
        let (vertex_buffers, vertex_buffer_memory, vertex_buffer_ptrs) = Self::create_host_buffers(
            device,
            &mem_props,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            frame_count,
        );
        let (index_buffers, index_buffer_memory, index_buffer_ptrs) = Self::create_host_buffers(
            device,
            &mem_props,
            vk::BufferUsageFlags::INDEX_BUFFER,
            frame_count,
        );

        // -- Black placeholder image ---------------------------------------
        let (black_image, black_image_memory, black_image_view) =
            Self::create_black_image(device, host_queue, host_queue_family, &mem_props);

        // -- Engine viewport descriptor set --------------------------------
        let engine_viewport_descriptor_set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(egui_descriptor_pool)
                    .set_layouts(&[egui_descriptor_set_layout]),
            )
            .expect("Failed to allocate engine viewport descriptor set")[0];

        // Point it at the black image initially
        device.update_descriptor_sets(
            std::slice::from_ref(
                &vk::WriteDescriptorSet::default()
                    .dst_set(engine_viewport_descriptor_set)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_array_element(0)
                    .dst_binding(0)
                    .image_info(std::slice::from_ref(
                        &vk::DescriptorImageInfo::default()
                            .image_view(black_image_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .sampler(egui_sampler),
                    )),
            ),
            &[],
        );

        let engine_viewport_texture_id = egui::TextureId::User(ENGINE_VIEWPORT_USER_ID);

        Self {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            host_queue,
            host_queue_family,
            queue_mutex,
            surface_loader,
            swapchain_loader,

            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_extent,
            swapchain_format,
            present_mode,
            fallback_extent,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,

            render_pass,
            framebuffers,

            command_pool,
            command_buffers,

            egui_pipeline_layout,
            egui_pipeline,
            egui_descriptor_pool,
            egui_descriptor_set_layout,
            egui_sampler,

            engine_viewport_texture_id,
            engine_viewport_descriptor_set,
            black_image,
            black_image_view,
            black_image_memory,

            managed_textures: HashMap::new(),

            vertex_buffers,
            vertex_buffer_memory,
            vertex_buffer_ptrs,
            index_buffers,
            index_buffer_memory,
            index_buffer_ptrs,

            mem_props,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Engine viewport
    // ─────────────────────────────────────────────────────────────────────

    /// Update the engine viewport descriptor to point at the given image view.
    pub(crate) unsafe fn set_engine_viewport(&mut self, image_view: vk::ImageView) {
        self.device.update_descriptor_sets(
            std::slice::from_ref(
                &vk::WriteDescriptorSet::default()
                    .dst_set(self.engine_viewport_descriptor_set)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_array_element(0)
                    .dst_binding(0)
                    .image_info(std::slice::from_ref(
                        &vk::DescriptorImageInfo::default()
                            .image_view(image_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .sampler(self.egui_sampler),
                    )),
            ),
            &[],
        );
    }

    /// Reset the engine viewport to the black placeholder.
    pub(crate) unsafe fn set_engine_viewport_black(&mut self) {
        self.set_engine_viewport(self.black_image_view);
    }

    /// The egui TextureId reserved for the engine viewport.
    pub(crate) fn engine_viewport_texture_id(&self) -> egui::TextureId {
        self.engine_viewport_texture_id
    }

    // ─────────────────────────────────────────────────────────────────────
    // Frame rendering
    // ─────────────────────────────────────────────────────────────────────

    /// Render one frame: update textures, record draw commands, submit, present.
    ///
    /// `compositor_timeline` is signaled with `signal_value` as part of the
    /// queue submit. Returns `Ok(FrameOutcome::_)` when the submit happened
    /// (signal value committed) and `Err(ERROR_OUT_OF_DATE_KHR)` when the
    /// submit was skipped because the swapchain is stale (signal NOT
    /// committed — caller must not advance the pool's compositor counter).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn render_frame(
        &mut self,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        scale_factor: f32,
        screen_size: [u32; 2],
        clear_color: [f32; 4],
        compositor_timeline: vk::Semaphore,
        signal_value: u64,
    ) -> Result<FrameOutcome, vk::Result> {
        // Wait for the previous work at this frame index to finish.
        self.device
            .wait_for_fences(
                std::slice::from_ref(&self.in_flight_fences[self.current_frame]),
                true,
                u64::MAX,
            )
            .inspect_err(|&e| {
                log::error!("wait_for_fences failed: {:?}", e);
            })?;

        // Acquire next swapchain image.
        let (image_index, _suboptimal) = match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available_semaphores[self.current_frame],
            vk::Fence::null(),
        ) {
            Ok(pair) => pair,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // No submit this frame — compositor_timeline is NOT signaled.
                // Caller must not commit signal_value.
                return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
            }
            Err(e) => panic!("acquire_next_image failed: {:?}", e),
        };
        let image_index = image_index as usize;

        // Update egui managed textures.
        self.apply_texture_delta(textures_delta);

        // Reset and begin command buffer.
        let cmd = self.command_buffers[self.current_frame];
        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .expect("reset_command_buffer failed");
        self.device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin_command_buffer failed");

        // Begin render pass (LOAD_OP_CLEAR).
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear_color,
            },
        }];
        self.device.cmd_begin_render_pass(
            cmd,
            &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index])
                .clear_values(&clear_values)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: self.swapchain_extent,
                }),
            vk::SubpassContents::INLINE,
        );

        // Record egui draw commands.
        self.record_egui_commands(cmd, &clipped_primitives, scale_factor, screen_size);

        self.device.cmd_end_render_pass(cmd);
        self.device
            .end_command_buffer(cmd)
            .expect("end_command_buffer failed");

        // Reset fence and submit.
        self.device
            .reset_fences(std::slice::from_ref(
                &self.in_flight_fences[self.current_frame],
            ))
            .expect("reset_fences failed");

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        // Signal both the binary render_finished (for queue_present) and the
        // timeline compositor semaphore (for the engine to wait on next frame).
        // Binary semaphores ignore the corresponding timeline value.
        let signal_semaphores = [
            self.render_finished_semaphores[self.current_frame],
            compositor_timeline,
        ];
        let signal_values = [0u64, signal_value];
        let cmd_bufs = [cmd];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_values);
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline_info);
        {
            let _qlock = self
                .queue_mutex
                .as_ref()
                .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()));
            self.device
                .queue_submit(
                    self.host_queue,
                    std::slice::from_ref(&submit_info),
                    self.in_flight_fences[self.current_frame],
                )
                .expect("queue_submit failed");
        }

        // Present — must only wait on binary semaphore (not timeline).
        let present_wait = [self.render_finished_semaphores[self.current_frame]];
        let swapchains = [self.swapchain];
        let image_indices = [image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&present_wait)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let _qlock = self
            .queue_mutex
            .as_ref()
            .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()));
        let result = self
            .swapchain_loader
            .queue_present(self.host_queue, &present_info);
        drop(_qlock);

        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();

        match result {
            Ok(false) => Ok(FrameOutcome::Rendered),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                Ok(FrameOutcome::RenderedNeedsRebuild)
            }
            Err(e) => panic!("queue_present failed: {:?}", e),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Swapchain recreation
    // ─────────────────────────────────────────────────────────────────────

    pub(crate) unsafe fn recreate_swapchain(&mut self) {
        self.device
            .device_wait_idle()
            .expect("device_wait_idle failed");

        // Destroy old framebuffers and image views.
        for fb in self.framebuffers.drain(..) {
            self.device.destroy_framebuffer(fb, None);
        }
        for iv in self.swapchain_image_views.drain(..) {
            self.device.destroy_image_view(iv, None);
        }

        // Destroy old sync objects.
        for &fence in &self.in_flight_fences {
            self.device.destroy_fence(fence, None);
        }
        for &sem in &self.image_available_semaphores {
            self.device.destroy_semaphore(sem, None);
        }
        for &sem in &self.render_finished_semaphores {
            self.device.destroy_semaphore(sem, None);
        }

        // Destroy old command buffers.
        self.device
            .free_command_buffers(self.command_pool, &self.command_buffers);

        // Destroy old vertex/index buffers.
        for i in 0..self.vertex_buffers.len() {
            self.device.unmap_memory(self.vertex_buffer_memory[i]);
            self.device.destroy_buffer(self.vertex_buffers[i], None);
            self.device.free_memory(self.vertex_buffer_memory[i], None);
        }
        for i in 0..self.index_buffers.len() {
            self.device.unmap_memory(self.index_buffer_memory[i]);
            self.device.destroy_buffer(self.index_buffers[i], None);
            self.device.free_memory(self.index_buffer_memory[i], None);
        }

        // Destroy old swapchain.
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);

        // Create new swapchain.
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            Self::create_swapchain_inner(
                self.physical_device,
                self.surface,
                &self.surface_loader,
                &self.swapchain_loader,
                self.present_mode,
                self.fallback_extent,
            );
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;

        // Recreate image views.
        self.swapchain_image_views =
            Self::create_image_views(&self.device, &self.swapchain_images, self.swapchain_format);

        // Recreate framebuffers.
        self.framebuffers = Self::create_framebuffers(
            &self.device,
            &self.swapchain_image_views,
            self.render_pass,
            self.swapchain_extent,
        );

        // Recreate sync objects.
        let frame_count = self.swapchain_images.len();
        let (fences, img_sems, rnd_sems) = Self::create_sync_objects(&self.device, frame_count);
        self.in_flight_fences = fences;
        self.image_available_semaphores = img_sems;
        self.render_finished_semaphores = rnd_sems;
        self.current_frame = 0;

        // Recreate command buffers.
        self.command_buffers = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(frame_count as u32),
            )
            .expect("Failed to allocate command buffers");

        // Recreate vertex/index buffers.
        let (vb, vbm, vbp) = Self::create_host_buffers(
            &self.device,
            &self.mem_props,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            frame_count,
        );
        let (ib, ibm, ibp) = Self::create_host_buffers(
            &self.device,
            &self.mem_props,
            vk::BufferUsageFlags::INDEX_BUFFER,
            frame_count,
        );
        self.vertex_buffers = vb;
        self.vertex_buffer_memory = vbm;
        self.vertex_buffer_ptrs = vbp;
        self.index_buffers = ib;
        self.index_buffer_memory = ibm;
        self.index_buffer_ptrs = ibp;
    }

    /// Current swapchain extent.
    /// Update the fallback extent used when the driver lets the client pick
    /// the swapchain size. Call this on every `WindowEvent::Resized` so the
    /// next swapchain recreation tracks the real window dimensions.
    pub(crate) fn set_fallback_extent(&mut self, extent: vk::Extent2D) {
        self.fallback_extent = extent;
    }

    pub(crate) fn swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    // ─────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────

    pub(crate) unsafe fn destroy(&mut self) {
        self.device
            .device_wait_idle()
            .expect("device_wait_idle failed");

        // Managed textures
        for (_, tex) in self.managed_textures.drain() {
            self.device.destroy_image_view(tex.image_view, None);
            self.device.destroy_image(tex.image, None);
            self.device.free_memory(tex.memory, None);
            // descriptor set freed when pool is destroyed
        }

        // Black image
        self.device.destroy_image_view(self.black_image_view, None);
        self.device.destroy_image(self.black_image, None);
        self.device.free_memory(self.black_image_memory, None);

        // Vertex/index buffers
        for i in 0..self.vertex_buffers.len() {
            self.device.unmap_memory(self.vertex_buffer_memory[i]);
            self.device.destroy_buffer(self.vertex_buffers[i], None);
            self.device.free_memory(self.vertex_buffer_memory[i], None);
        }
        for i in 0..self.index_buffers.len() {
            self.device.unmap_memory(self.index_buffer_memory[i]);
            self.device.destroy_buffer(self.index_buffers[i], None);
            self.device.free_memory(self.index_buffer_memory[i], None);
        }

        // Framebuffers + image views
        for fb in &self.framebuffers {
            self.device.destroy_framebuffer(*fb, None);
        }
        for iv in &self.swapchain_image_views {
            self.device.destroy_image_view(*iv, None);
        }

        // Sync objects
        for &fence in &self.in_flight_fences {
            self.device.destroy_fence(fence, None);
        }
        for &sem in &self.image_available_semaphores {
            self.device.destroy_semaphore(sem, None);
        }
        for &sem in &self.render_finished_semaphores {
            self.device.destroy_semaphore(sem, None);
        }

        // Command pool (also frees command buffers)
        self.device.destroy_command_pool(self.command_pool, None);

        // Pipeline
        self.device.destroy_pipeline(self.egui_pipeline, None);
        self.device
            .destroy_pipeline_layout(self.egui_pipeline_layout, None);

        // Render pass
        self.device.destroy_render_pass(self.render_pass, None);

        // Sampler
        self.device.destroy_sampler(self.egui_sampler, None);

        // Descriptor pool + layout (pool destruction frees all sets)
        self.device
            .destroy_descriptor_pool(self.egui_descriptor_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.egui_descriptor_set_layout, None);

        // Swapchain (surface is owned by Host, not compositor)
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Private helpers
    // ═════════════════════════════════════════════════════════════════════

    // ── Swapchain ────────────────────────────────────────────────────────

    unsafe fn create_swapchain_inner(
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        present_mode: vk::PresentModeKHR,
        fallback_extent: vk::Extent2D,
    ) -> (vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D) {
        let caps = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .expect("get_physical_device_surface_capabilities failed");
        let formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("get_physical_device_surface_formats failed");
        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .expect("get_physical_device_surface_present_modes failed");

        // Prefer B8G8R8A8_UNORM + SRGB_NONLINEAR.
        let surface_format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0]);

        // Prefer requested present mode, fall back to FIFO.
        let chosen_present_mode = present_modes
            .iter()
            .find(|&&m| m == present_mode)
            .copied()
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // Extent from surface capabilities.
        //
        // If the WM reports `u32::MAX` it's the sentinel "client picks any
        // size". The previous fallback hard-coded 1280×720 which produced
        // grossly under-sampled rendering on any normal-sized window (the
        // compositor then upscales 1280×720 to e.g. 2528×1372 on present and
        // everything looks blurry). We now clamp the caller-supplied
        // `fallback_extent` — ideally the current `window.inner_size()` — to
        // the driver's min/max so the swapchain matches the actual window.
        let extent = if caps.current_extent.width == u32::MAX {
            let w = fallback_extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width);
            let h = fallback_extent
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height);
            vk::Extent2D {
                width: w.max(1),
                height: h.max(1),
            }
        } else {
            // Guard against zero-extent (e.g. minimized window) — Vulkan
            // requires width and height >= 1 for swapchain creation.
            vk::Extent2D {
                width: caps.current_extent.width.max(1),
                height: caps.current_extent.height.max(1),
            }
        };

        // Image count: min + 1, clamped to max.
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count != 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(chosen_present_mode)
            .image_array_layers(1)
            .clipped(true);

        let swapchain = swapchain_loader
            .create_swapchain(&create_info, None)
            .expect("create_swapchain failed");
        let images = swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("get_swapchain_images failed");

        (swapchain, images, surface_format.format, extent)
    }

    unsafe fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Vec<vk::ImageView> {
        images
            .iter()
            .map(|&image| {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format)
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
                    .expect("Failed to create swapchain image view")
            })
            .collect()
    }

    // ── Render pass ──────────────────────────────────────────────────────

    unsafe fn create_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
        device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(
                        &vk::AttachmentDescription::default()
                            .format(format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
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
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)]),
                None,
            )
            .expect("Failed to create render pass")
    }

    // ── Framebuffers ─────────────────────────────────────────────────────

    unsafe fn create_framebuffers(
        device: &ash::Device,
        image_views: &[vk::ImageView],
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        image_views
            .iter()
            .map(|&iv| {
                let attachments = [iv];
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&attachments)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer")
            })
            .collect()
    }

    // ── Pipeline ─────────────────────────────────────────────────────────

    unsafe fn create_pipeline_layout(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> vk::PipelineLayout {
        device
            .create_pipeline_layout(
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
            .expect("Failed to create pipeline layout")
    }

    unsafe fn create_pipeline(
        device: &ash::Device,
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

        let vert_code = include_bytes!("shaders/spv/vert.spv");
        let frag_code = include_bytes!("shaders/spv/frag.spv");

        let vert_module = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: vert_code.len(),
                    p_code: vert_code.as_ptr().cast::<u32>(),
                    ..Default::default()
                },
                None,
            )
            .expect("Failed to create vertex shader module");
        let frag_module = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: frag_code.len(),
                    p_code: frag_code.as_ptr().cast::<u32>(),
                    ..Default::default()
                },
                None,
            )
            .expect("Failed to create fragment shader module");

        let main_fn = CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&main_fn),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&main_fn),
        ];

        let binding = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(4 * std::mem::size_of::<f32>() as u32 + 4 * std::mem::size_of::<u8>() as u32)];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&attributes)
            .vertex_binding_descriptions(&binding);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);

        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(
                    &vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vertex_input)
                        .input_assembly_state(&input_assembly)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&rasterization)
                        .multisample_state(&multisample)
                        .depth_stencil_state(&depth_stencil)
                        .color_blend_state(&color_blend)
                        .dynamic_state(&dynamic_state)
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0),
                ),
                None,
            )
            .expect("Failed to create graphics pipeline")[0];

        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        pipeline
    }

    // ── Sync objects ─────────────────────────────────────────────────────

    unsafe fn create_sync_objects(
        device: &ash::Device,
        count: usize,
    ) -> (Vec<vk::Fence>, Vec<vk::Semaphore>, Vec<vk::Semaphore>) {
        let mut fences = Vec::with_capacity(count);
        let mut img_sems = Vec::with_capacity(count);
        let mut rnd_sems = Vec::with_capacity(count);
        for _ in 0..count {
            fences.push(
                device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .expect("Failed to create fence"),
            );
            img_sems.push(
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .expect("Failed to create semaphore"),
            );
            rnd_sems.push(
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .expect("Failed to create semaphore"),
            );
        }
        (fences, img_sems, rnd_sems)
    }

    // ── Host-visible buffers ─────────────────────────────────────────────

    unsafe fn create_host_buffers(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        count: usize,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut u8>) {
        let mut buffers = Vec::with_capacity(count);
        let mut memories = Vec::with_capacity(count);
        let mut ptrs = Vec::with_capacity(count);

        for _ in 0..count {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(BUFFER_SIZE)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("Failed to create buffer");

            let reqs = device.get_buffer_memory_requirements(buffer);
            let mem_type = find_memory_type(
                mem_props,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(mem_type),
                    None,
                )
                .expect("Failed to allocate buffer memory");
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind buffer memory");

            let ptr = device
                .map_memory(memory, 0, BUFFER_SIZE, vk::MemoryMapFlags::empty())
                .expect("Failed to map buffer memory")
                .cast::<u8>();

            buffers.push(buffer);
            memories.push(memory);
            ptrs.push(ptr);
        }

        (buffers, memories, ptrs)
    }

    // ── Black placeholder image ──────────────────────────────────────────

    unsafe fn create_black_image(
        device: &ash::Device,
        queue: vk::Queue,
        queue_family: u32,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        // Create 1x1 R8G8B8A8_UNORM image.
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .expect("Failed to create black image");

        let reqs = device.get_image_memory_requirements(image);
        let mem_type = find_memory_type(
            mem_props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(mem_type),
                None,
            )
            .expect("Failed to allocate black image memory");
        device
            .bind_image_memory(image, memory, 0)
            .expect("Failed to bind black image memory");

        let image_view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .expect("Failed to create black image view");

        // Upload black pixel via staging buffer.
        let pixel: [u8; 4] = [0, 0, 0, 255];
        let staging_buf = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(4)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
            .expect("Failed to create staging buffer");
        let staging_reqs = device.get_buffer_memory_requirements(staging_buf);
        let staging_mem_type = find_memory_type(
            mem_props,
            staging_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let staging_mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_reqs.size)
                    .memory_type_index(staging_mem_type),
                None,
            )
            .expect("Failed to allocate staging memory");
        device
            .bind_buffer_memory(staging_buf, staging_mem, 0)
            .expect("Failed to bind staging memory");
        let ptr = device
            .map_memory(staging_mem, 0, 4, vk::MemoryMapFlags::empty())
            .expect("Failed to map staging memory")
            .cast::<u8>();
        std::ptr::copy_nonoverlapping(pixel.as_ptr(), ptr, 4);
        device.unmap_memory(staging_mem);

        // Record and submit transition + copy commands.
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
                None,
            )
            .expect("Failed to create temp command pool");
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("Failed to allocate temp command buffer")[0];
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("Failed to create temp fence");

        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        // UNDEFINED -> TRANSFER_DST_OPTIMAL
        let barrier_to_dst = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_to_dst],
        );

        device.cmd_copy_buffer_to_image(
            cmd,
            staging_buf,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(1)
                .buffer_image_height(1)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })],
        );

        // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        let barrier_to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_to_read],
        );

        device.end_command_buffer(cmd).unwrap();

        // NB: this helper is called only from `Compositor::new`, before the
        // engine thread has been spawned — no concurrent queue access is
        // possible yet, so no locking needed here.
        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                fence,
            )
            .unwrap();
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();

        // Cleanup temp resources.
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_buffer(staging_buf, None);
        device.free_memory(staging_mem, None);

        (image, memory, image_view)
    }

    // ── Texture management ───────────────────────────────────────────────

    unsafe fn apply_texture_delta(&mut self, delta: egui::TexturesDelta) {
        for (id, image_delta) in delta.set {
            self.update_texture(id, image_delta);
        }
        for id in delta.free {
            self.free_texture(id);
        }
    }

    unsafe fn update_texture(
        &mut self,
        texture_id: egui::TextureId,
        delta: egui::epaint::ImageDelta,
    ) {
        // Extract pixel data.
        let data: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                image
                    .pixels
                    .iter()
                    .flat_map(egui::Color32::to_array)
                    .collect()
            }
        };

        let width = delta.image.width() as u32;
        let height = delta.image.height() as u32;

        // Create staging buffer.
        let staging_buf = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(data.len() as vk::DeviceSize)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
            .expect("Failed to create staging buffer");
        let staging_reqs = self.device.get_buffer_memory_requirements(staging_buf);
        let staging_mem_type = find_memory_type(
            &self.mem_props,
            staging_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let staging_mem = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_reqs.size)
                    .memory_type_index(staging_mem_type),
                None,
            )
            .expect("Failed to allocate staging memory");
        self.device
            .bind_buffer_memory(staging_buf, staging_mem, 0)
            .unwrap();

        let ptr = self
            .device
            .map_memory(
                staging_mem,
                0,
                data.len() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to map staging memory")
            .cast::<u8>();
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        self.device.unmap_memory(staging_mem);

        // Create texture image (new full texture).
        let texture_image = self
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .expect("Failed to create texture image");

        let img_reqs = self.device.get_image_memory_requirements(texture_image);
        let img_mem_type = find_memory_type(
            &self.mem_props,
            img_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let texture_memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(img_reqs.size)
                    .memory_type_index(img_mem_type),
                None,
            )
            .expect("Failed to allocate texture memory");
        self.device
            .bind_image_memory(texture_image, texture_memory, 0)
            .unwrap();

        let texture_image_view = self
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(texture_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .expect("Failed to create texture image view");

        // Record commands to upload the texture.
        let cmd_pool = self
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.host_queue_family),
                None,
            )
            .unwrap();
        let cmd = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        let fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();

        self.device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        // UNDEFINED -> TRANSFER_DST_OPTIMAL
        self.cmd_image_barrier(
            cmd,
            texture_image,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
        );

        self.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buf,
            texture_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(width)
                .buffer_image_height(height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })],
        );

        // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        self.cmd_image_barrier(
            cmd,
            texture_image,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );

        self.device.end_command_buffer(cmd).unwrap();
        {
            let _qlock = self
                .queue_mutex
                .as_ref()
                .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()));
            self.device
                .queue_submit(
                    self.host_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    fence,
                )
                .unwrap();
        }
        self.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();

        if let Some(pos) = delta.pos {
            // Partial update: blit from newly uploaded texture into existing texture.
            if let Some(existing) = self.managed_textures.get(&texture_id) {
                self.device
                    .reset_command_pool(cmd_pool, vk::CommandPoolResetFlags::empty())
                    .unwrap();
                self.device.reset_fences(&[fence]).unwrap();

                self.device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();

                // Existing -> TRANSFER_DST (preserve contents!)
                self.cmd_image_barrier(
                    cmd,
                    existing.image,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                );

                // New texture -> TRANSFER_SRC
                self.cmd_image_barrier(
                    cmd,
                    texture_image,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                );

                let top_left = vk::Offset3D {
                    x: pos[0] as i32,
                    y: pos[1] as i32,
                    z: 0,
                };
                let bottom_right = vk::Offset3D {
                    x: pos[0] as i32 + width as i32,
                    y: pos[1] as i32 + height as i32,
                    z: 1,
                };
                let region = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: width as i32,
                            y: height as i32,
                            z: 1,
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [top_left, bottom_right],
                };
                self.device.cmd_blit_image(
                    cmd,
                    texture_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    existing.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                    vk::Filter::NEAREST,
                );

                // Existing -> SHADER_READ_ONLY
                self.cmd_image_barrier(
                    cmd,
                    existing.image,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                );

                self.device.end_command_buffer(cmd).unwrap();
                {
                    let _qlock = self
                        .queue_mutex
                        .as_ref()
                        .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()));
                    self.device
                        .queue_submit(
                            self.host_queue,
                            &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                            fence,
                        )
                        .unwrap();
                }
                self.device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .unwrap();

                // Destroy the temp texture (not needed — we blitted into existing).
                self.device.destroy_image(texture_image, None);
                self.device.destroy_image_view(texture_image_view, None);
                self.device.free_memory(texture_memory, None);
            } else {
                // No existing texture for partial update — shouldn't happen.
                // Clean up the temp resources we already created.
                self.device.destroy_image(texture_image, None);
                self.device.destroy_image_view(texture_image_view, None);
                self.device.free_memory(texture_memory, None);
            }
        } else {
            // Full texture replacement.
            let descriptor_set = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.egui_descriptor_pool)
                        .set_layouts(&[self.egui_descriptor_set_layout]),
                )
                .expect("Failed to allocate descriptor set")[0];

            self.device.update_descriptor_sets(
                std::slice::from_ref(
                    &vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_array_element(0)
                        .dst_binding(0)
                        .image_info(std::slice::from_ref(
                            &vk::DescriptorImageInfo::default()
                                .image_view(texture_image_view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .sampler(self.egui_sampler),
                        )),
                ),
                &[],
            );

            // Destroy old texture if it existed.
            if let Some(old) = self.managed_textures.remove(&texture_id) {
                self.device.destroy_image_view(old.image_view, None);
                self.device.destroy_image(old.image, None);
                self.device.free_memory(old.memory, None);
                self.device
                    .free_descriptor_sets(self.egui_descriptor_pool, &[old.descriptor_set])
                    .expect("free_descriptor_sets failed");
            }

            self.managed_textures.insert(
                texture_id,
                ManagedTexture {
                    image: texture_image,
                    memory: texture_memory,
                    image_view: texture_image_view,
                    descriptor_set,
                },
            );
        }

        // Cleanup staging resources.
        self.device.destroy_fence(fence, None);
        self.device.destroy_command_pool(cmd_pool, None);
        self.device.destroy_buffer(staging_buf, None);
        self.device.free_memory(staging_mem, None);
    }

    unsafe fn free_texture(&mut self, id: egui::TextureId) {
        if let Some(tex) = self.managed_textures.remove(&id) {
            self.device.destroy_image_view(tex.image_view, None);
            self.device.destroy_image(tex.image, None);
            self.device.free_memory(tex.memory, None);
        }
    }

    // ── egui draw recording ──────────────────────────────────────────────

    unsafe fn record_egui_commands(
        &mut self,
        cmd: vk::CommandBuffer,
        clipped_primitives: &[egui::ClippedPrimitive],
        scale_factor: f32,
        screen_size: [u32; 2],
    ) {
        let width_points = screen_size[0] as f32 / scale_factor;
        let height_points = screen_size[1] as f32 / scale_factor;

        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.egui_pipeline);
        self.device.cmd_bind_vertex_buffers(
            cmd,
            0,
            &[self.vertex_buffers[self.current_frame]],
            &[0],
        );
        self.device.cmd_bind_index_buffer(
            cmd,
            self.index_buffers[self.current_frame],
            0,
            vk::IndexType::UINT32,
        );
        self.device.cmd_push_constants(
            cmd,
            self.egui_pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytes_of(&width_points),
        );
        self.device.cmd_push_constants(
            cmd,
            self.egui_pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            4,
            bytes_of(&height_points),
        );

        let mut vertex_buffer_ptr = self.vertex_buffer_ptrs[self.current_frame];
        let vertex_buffer_end = vertex_buffer_ptr.add(BUFFER_SIZE as usize);
        let mut index_buffer_ptr = self.index_buffer_ptrs[self.current_frame];
        let index_buffer_end = index_buffer_ptr.add(BUFFER_SIZE as usize);

        let mut vertex_base: i32 = 0;
        let mut index_base: u32 = 0;

        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_primitives
        {
            let mesh = match primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh,
                egui::epaint::Primitive::Callback(_) => {
                    log::warn!("egui paint callbacks are not supported");
                    continue;
                }
            };
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            // Resolve descriptor set for this mesh's texture.
            let descriptor_set = match mesh.texture_id {
                egui::TextureId::Managed(_) => {
                    if let Some(tex) = self.managed_textures.get(&mesh.texture_id) {
                        tex.descriptor_set
                    } else {
                        log::error!("Missing managed texture: {:?}", mesh.texture_id);
                        continue;
                    }
                }
                egui::TextureId::User(id) => {
                    if id == ENGINE_VIEWPORT_USER_ID {
                        // Engine viewport
                        self.engine_viewport_descriptor_set
                    } else {
                        log::error!("Unknown user texture id: {}", id);
                        continue;
                    }
                }
            };

            // Copy vertex data.
            let v_size = std::mem::size_of::<egui::epaint::Vertex>();
            let v_bytes = mesh.vertices.len() * v_size;
            let i_bytes = mesh.indices.len() * std::mem::size_of::<u32>();

            let next_vptr = vertex_buffer_ptr.add(v_bytes);
            let next_iptr = index_buffer_ptr.add(i_bytes);

            if next_vptr > vertex_buffer_end || next_iptr > index_buffer_end {
                log::error!("egui paint out of buffer memory — skipping remaining primitives");
                break;
            }

            std::ptr::copy_nonoverlapping(
                mesh.vertices.as_ptr().cast::<u8>(),
                vertex_buffer_ptr,
                v_bytes,
            );
            std::ptr::copy_nonoverlapping(
                mesh.indices.as_ptr().cast::<u8>(),
                index_buffer_ptr,
                i_bytes,
            );

            vertex_buffer_ptr = next_vptr;
            index_buffer_ptr = next_iptr;

            // Bind descriptor set.
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.egui_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            // Scissor rect.
            let min = egui::Pos2 {
                x: (clip_rect.min.x * scale_factor).clamp(0.0, screen_size[0] as f32),
                y: (clip_rect.min.y * scale_factor).clamp(0.0, screen_size[1] as f32),
            };
            let max = egui::Pos2 {
                x: (clip_rect.max.x * scale_factor).clamp(min.x, screen_size[0] as f32),
                y: (clip_rect.max.y * scale_factor).clamp(min.y, screen_size[1] as f32),
            };
            self.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: min.x.round() as i32,
                        y: min.y.round() as i32,
                    },
                    extent: vk::Extent2D {
                        width: (max.x.round() - min.x.round()) as u32,
                        height: (max.y.round() - min.y.round()) as u32,
                    },
                }],
            );

            // Viewport.
            self.device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: screen_size[0] as f32,
                    height: screen_size[1] as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );

            // Draw.
            self.device.cmd_draw_indexed(
                cmd,
                mesh.indices.len() as u32,
                1,
                index_base,
                vertex_base,
                0,
            );

            vertex_base += mesh.vertices.len() as i32;
            index_base += mesh.indices.len() as u32;
        }
    }

    // ── Image barrier helper ─────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    unsafe fn cmd_image_barrier(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        self.device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
