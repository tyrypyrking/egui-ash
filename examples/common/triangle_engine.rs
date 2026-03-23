//! A reusable `TriangleEngine` that renders a rotating colored triangle.
//!
//! Uses the existing SPIR-V shaders at `examples/common/shaders/spv/triangle.{vert,frag}.spv`.
//! Demonstrates the v2 `EngineRenderer` trait with a real graphics pipeline
//! (dynamic rendering, Vulkan 1.3).

#![allow(dead_code)]

use ash::vk;
use egui_ash::{CompletedFrame, EngineContext, EngineEvent, EngineRenderer, RenderTarget};

// ─────────────────────────────────────────────────────────────────────────────
// UiState / EngineState
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct TriangleUiState {
    pub rotate_y: f32,
    pub auto_rotate: bool,
    pub bg_color: [f32; 3],
}

impl Default for TriangleUiState {
    fn default() -> Self {
        Self {
            rotate_y: 0.0,
            auto_rotate: true,
            bg_color: [0.1, 0.1, 0.15],
        }
    }
}

#[derive(Clone, Default)]
pub struct TriangleEngineState {
    pub frame_count: u64,
    pub current_angle: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// TriangleEngine
// ─────────────────────────────────────────────────────────────────────────────

pub struct TriangleEngine {
    // Pre-init fields (set in new(), used in init())
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,

    // Post-init fields (set in init())
    device: Option<ash::Device>,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    // Pipeline objects
    vert_shader: vk::ShaderModule,
    frag_shader: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    // Vertex buffer
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,

    // UBO
    ubo_buffer: vk::Buffer,
    ubo_memory: vk::DeviceMemory,
    ubo_mapped: *mut u8,

    // Descriptor
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    // State
    frame_count: u64,
    current_angle: f32,
}

// Safety: All Vulkan handles are thread-safe when accessed from a single thread.
// The engine runs on a single dedicated thread.
unsafe impl Send for TriangleEngine {}

impl TriangleEngine {
    pub fn new(instance: ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        Self {
            instance,
            physical_device,
            device: None,
            queue: vk::Queue::null(),
            queue_family_index: u32::MAX,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            vert_shader: vk::ShaderModule::null(),
            frag_shader: vk::ShaderModule::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            vertex_buffer: vk::Buffer::null(),
            vertex_buffer_memory: vk::DeviceMemory::null(),
            ubo_buffer: vk::Buffer::null(),
            ubo_memory: vk::DeviceMemory::null(),
            ubo_mapped: std::ptr::null_mut(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            frame_count: 0,
            current_angle: 0.0,
        }
    }

    fn create_shader_module(device: &ash::Device, code: &[u8]) -> vk::ShaderModule {
        // Align the code to 4 bytes as required by SPIR-V
        let code_aligned = ash::util::read_spv(&mut std::io::Cursor::new(code))
            .expect("failed to read SPIR-V");
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code_aligned);
        unsafe {
            device
                .create_shader_module(&create_info, None)
                .expect("failed to create shader module")
        }
    }
}

impl EngineRenderer for TriangleEngine {
    type UiState = TriangleUiState;
    type EngineState = TriangleEngineState;

    fn init(&mut self, ctx: EngineContext) {
        self.queue = ctx.queue;
        self.queue_family_index = ctx.queue_family_index;

        let device = &ctx.device;
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        // 1. Command pool + command buffer
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(ctx.queue_family_index);
        self.command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .expect("failed to create command pool")
        };
        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);
        self.command_buffer = unsafe {
            device
                .allocate_command_buffers(&cmd_alloc_info)
                .expect("failed to allocate command buffer")[0]
        };

        // 2. Load shaders
        let vert_code = include_bytes!("shaders/spv/triangle.vert.spv");
        let frag_code = include_bytes!("shaders/spv/triangle.frag.spv");
        self.vert_shader = Self::create_shader_module(device, vert_code);
        self.frag_shader = Self::create_shader_module(device, frag_code);

        // 3. Descriptor set layout: 1 UBO at binding 0, VERTEX stage
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .expect("failed to create descriptor set layout")
        };

        // 4. Pipeline layout
        let set_layouts = [self.descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        self.pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("failed to create pipeline layout")
        };

        // 5. Graphics pipeline with dynamic rendering (Vulkan 1.3)
        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vert_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.frag_shader)
                .name(entry_point),
        ];

        // Vertex input: stride 24 (6 floats), binding 0
        let binding_descs = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 24, // 6 * sizeof(f32)
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attribute_descs = [
            // location 0 = vec3 inPos at offset 0
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            // location 1 = vec3 inColor at offset 12
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
        ];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descs)
            .vertex_attribute_descriptions(&attribute_descs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // Dynamic viewport + scissor
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachment);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Dynamic rendering: color attachment format
        let color_formats = [ctx.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .push_next(&mut rendering_info);

        self.pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("failed to create graphics pipeline")[0]
        };

        // 6. Vertex buffer (device-local, uploaded via staging)
        #[rustfmt::skip]
        let vertices: [f32; 18] = [
            // pos              color
             0.0,  0.5, 0.0,   1.0, 0.0, 0.0, // top, red
             0.5, -0.5, 0.0,   0.0, 1.0, 0.0, // right, green
            -0.5, -0.5, 0.0,   0.0, 0.0, 1.0, // left, blue
        ];
        let vertex_data_size = (vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;

        // Staging buffer (HOST_VISIBLE)
        let staging_buf_info = vk::BufferCreateInfo::default()
            .size(vertex_data_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe {
            device
                .create_buffer(&staging_buf_info, None)
                .expect("failed to create staging buffer")
        };
        let staging_reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let staging_mem_type = find_memory_type(
            &mem_props,
            staging_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let staging_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_reqs.size)
            .memory_type_index(staging_mem_type);
        let staging_memory = unsafe {
            device
                .allocate_memory(&staging_alloc_info, None)
                .expect("failed to allocate staging memory")
        };
        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
                .expect("failed to bind staging buffer memory");
            let ptr = device
                .map_memory(
                    staging_memory,
                    0,
                    vertex_data_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("failed to map staging memory") as *mut f32;
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), ptr, vertices.len());
            device.unmap_memory(staging_memory);
        }

        // Device-local vertex buffer
        let vb_info = vk::BufferCreateInfo::default()
            .size(vertex_data_size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        self.vertex_buffer = unsafe {
            device
                .create_buffer(&vb_info, None)
                .expect("failed to create vertex buffer")
        };
        let vb_reqs = unsafe { device.get_buffer_memory_requirements(self.vertex_buffer) };
        let vb_mem_type = find_memory_type(
            &mem_props,
            vb_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let vb_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(vb_reqs.size)
            .memory_type_index(vb_mem_type);
        self.vertex_buffer_memory = unsafe {
            device
                .allocate_memory(&vb_alloc_info, None)
                .expect("failed to allocate vertex buffer memory")
        };
        unsafe {
            device
                .bind_buffer_memory(self.vertex_buffer, self.vertex_buffer_memory, 0)
                .expect("failed to bind vertex buffer memory");
        }

        // Upload via command buffer
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .expect("failed to begin command buffer");

            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: vertex_data_size,
            };
            device.cmd_copy_buffer(
                self.command_buffer,
                staging_buffer,
                self.vertex_buffer,
                &[copy_region],
            );

            device
                .end_command_buffer(self.command_buffer)
                .expect("failed to end command buffer");

            let cmd_bufs = [self.command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let fence_info = vk::FenceCreateInfo::default();
            let fence = device
                .create_fence(&fence_info, None)
                .expect("failed to create fence");
            device
                .queue_submit(self.queue, &[submit_info], fence)
                .expect("failed to submit upload command buffer");
            device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("failed to wait for fence");
            device.destroy_fence(fence, None);

            // Cleanup staging
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }

        // 7. UBO buffer (HOST_VISIBLE | HOST_COHERENT), 3 mat4 = 192 bytes
        let ubo_size: vk::DeviceSize = 3 * 64; // 3 * sizeof(mat4)
        let ubo_buf_info = vk::BufferCreateInfo::default()
            .size(ubo_size)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        self.ubo_buffer = unsafe {
            device
                .create_buffer(&ubo_buf_info, None)
                .expect("failed to create UBO buffer")
        };
        let ubo_reqs = unsafe { device.get_buffer_memory_requirements(self.ubo_buffer) };
        let ubo_mem_type = find_memory_type(
            &mem_props,
            ubo_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let ubo_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(ubo_reqs.size)
            .memory_type_index(ubo_mem_type);
        self.ubo_memory = unsafe {
            device
                .allocate_memory(&ubo_alloc_info, None)
                .expect("failed to allocate UBO memory")
        };
        unsafe {
            device
                .bind_buffer_memory(self.ubo_buffer, self.ubo_memory, 0)
                .expect("failed to bind UBO buffer memory");
            self.ubo_mapped = device
                .map_memory(self.ubo_memory, 0, ubo_size, vk::MemoryMapFlags::empty())
                .expect("failed to map UBO memory") as *mut u8;
        }

        // 8. Descriptor pool and set
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        self.descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .expect("failed to create descriptor pool")
        };

        let set_layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        self.descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .expect("failed to allocate descriptor set")[0]
        };

        // 9. Update descriptor set with UBO buffer
        let buffer_info = [vk::DescriptorBufferInfo {
            buffer: self.ubo_buffer,
            offset: 0,
            range: ubo_size,
        }];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info)];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        self.device = Some(ctx.device);
    }

    fn render(
        &mut self,
        target: RenderTarget,
        ui_state: &Self::UiState,
        engine_state: &mut Self::EngineState,
    ) -> CompletedFrame {
        let device = self.device.as_ref().expect("engine not initialized");
        let cmd = self.command_buffer;

        // Update angle
        if ui_state.auto_rotate {
            self.current_angle += 1.0;
        }
        let total_angle = self.current_angle + ui_state.rotate_y;
        let angle_rad = total_angle.to_radians();

        // Compute MVP matrices
        let aspect = target.extent.width as f32 / target.extent.height.max(1) as f32;
        let model = mat4_rotation_y(angle_rad);
        let view = mat4_look_at([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = mat4_perspective(45.0_f32.to_radians(), aspect, 0.1, 100.0);

        // Write UBO (3 mat4s, column-major)
        unsafe {
            let ptr = self.ubo_mapped as *mut [f32; 16];
            std::ptr::write(ptr, model);
            std::ptr::write(ptr.add(1), view);
            std::ptr::write(ptr.add(2), proj);
        }

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            // Reset and begin command buffer
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd, &begin_info)
                .expect("failed to begin command buffer");

            // Acquire barrier or UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
            if let Some(acquire) = target.acquire_barrier() {
                let barriers = [*acquire];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            } else {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }

            // Begin dynamic rendering
            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [
                        ui_state.bg_color[0],
                        ui_state.bg_color[1],
                        ui_state.bg_color[2],
                        1.0,
                    ],
                },
            };
            let color_attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(target.image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value)];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: target.extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments);
            device.cmd_begin_rendering(cmd, &rendering_info);

            // Set viewport and scissor
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: target.extent.width as f32,
                height: target.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: target.extent,
            }];
            device.cmd_set_viewport(cmd, 0, &viewports);
            device.cmd_set_scissor(cmd, 0, &scissors);

            // Bind pipeline, descriptor set, vertex buffer, draw
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);
            device.cmd_draw(cmd, 3, 1, 0, 0);

            device.cmd_end_rendering(cmd);

            // Release barrier or COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            if let Some(release) = target.release_barrier() {
                let barriers = [*release];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            } else {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }

            device
                .end_command_buffer(cmd)
                .expect("failed to end command buffer");

            // Submit with timeline semaphore wait + signal (Vulkan 1.3 synchronization2)
            let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(target.timeline)
                .value(target.wait_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(target.timeline)
                .value(target.signal_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmd_buf_infos =
                [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_semaphore_infos)
                .signal_semaphore_infos(&signal_semaphore_infos)
                .command_buffer_infos(&cmd_buf_infos);
            device
                .queue_submit2(self.queue, &[submit_info], vk::Fence::null())
                .expect("failed to submit command buffer");
        }

        self.frame_count += 1;
        engine_state.frame_count = self.frame_count;
        engine_state.current_angle = total_angle;

        target.complete()
    }

    fn handle_event(&mut self, _event: EngineEvent) {
        // No input handling needed for this engine.
    }

    fn destroy(&mut self) {
        if let Some(device) = self.device.take() {
            unsafe {
                device.device_wait_idle().ok();
                self.cleanup_gpu_resources(&device);
            }
        }
    }
}

impl TriangleEngine {
    unsafe fn cleanup_gpu_resources(&mut self, device: &ash::Device) {
        if self.pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.pipeline, None);
            self.pipeline = vk::Pipeline::null();
        }
        if self.pipeline_layout != vk::PipelineLayout::null() {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.pipeline_layout = vk::PipelineLayout::null();
        }
        if self.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
            self.descriptor_set = vk::DescriptorSet::null();
        }
        if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.descriptor_set_layout = vk::DescriptorSetLayout::null();
        }
        if !self.ubo_mapped.is_null() {
            device.unmap_memory(self.ubo_memory);
            self.ubo_mapped = std::ptr::null_mut();
        }
        if self.ubo_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.ubo_buffer, None);
            self.ubo_buffer = vk::Buffer::null();
        }
        if self.ubo_memory != vk::DeviceMemory::null() {
            device.free_memory(self.ubo_memory, None);
            self.ubo_memory = vk::DeviceMemory::null();
        }
        if self.vertex_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.vertex_buffer, None);
            self.vertex_buffer = vk::Buffer::null();
        }
        if self.vertex_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(self.vertex_buffer_memory, None);
            self.vertex_buffer_memory = vk::DeviceMemory::null();
        }
        if self.vert_shader != vk::ShaderModule::null() {
            device.destroy_shader_module(self.vert_shader, None);
            self.vert_shader = vk::ShaderModule::null();
        }
        if self.frag_shader != vk::ShaderModule::null() {
            device.destroy_shader_module(self.frag_shader, None);
            self.frag_shader = vk::ShaderModule::null();
        }
        if self.command_pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.command_pool, None);
            self.command_pool = vk::CommandPool::null();
        }
    }
}

impl Drop for TriangleEngine {
    fn drop(&mut self) {
        // Safety net: destroy GPU resources even after a panic.
        if let Some(device) = self.device.take() {
            unsafe {
                device.device_wait_idle().ok();
                self.cleanup_gpu_resources(&device);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan memory helpers
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
    panic!(
        "failed to find suitable memory type (bits={:#x}, flags={:?})",
        type_bits, flags
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4x4 matrix math helpers (column-major, no external dependencies)
// ─────────────────────────────────────────────────────────────────────────────

/// Identity matrix.
fn mat4_identity() -> [f32; 16] {
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    m
}

/// Rotation around the Y axis. Column-major.
fn mat4_rotation_y(angle_rad: f32) -> [f32; 16] {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    #[rustfmt::skip]
    let m = [
         c,  0.0,  -s, 0.0,
        0.0, 1.0, 0.0, 0.0,
         s,  0.0,   c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    m
}

/// Look-at view matrix. Column-major.
fn mat4_look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = vec3_normalize([
        center[0] - eye[0],
        center[1] - eye[1],
        center[2] - eye[2],
    ]);
    let s = vec3_normalize(vec3_cross(f, up));
    let u = vec3_cross(s, f);

    #[rustfmt::skip]
    let m = [
         s[0],  u[0], -f[0], 0.0,
         s[1],  u[1], -f[1], 0.0,
         s[2],  u[2], -f[2], 0.0,
        -vec3_dot(s, eye), -vec3_dot(u, eye), vec3_dot(f, eye), 1.0,
    ];
    m
}

/// Perspective projection matrix. Column-major.
/// Uses Vulkan clip-space conventions (Y flipped, depth 0..1).
fn mat4_perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y_rad / 2.0).tan();
    let range_inv = 1.0 / (near - far);

    // Vulkan: Y is flipped compared to OpenGL, depth [0, 1]
    #[rustfmt::skip]
    let m = [
        f / aspect,  0.0,  0.0,                       0.0,
        0.0,        -f,    0.0,                       0.0,
        0.0,         0.0,  far * range_inv,          -1.0,
        0.0,         0.0,  near * far * range_inv,    0.0,
    ];
    m
}

/// Multiply two 4x4 matrices (column-major): result = a * b.
#[allow(dead_code)]
fn mat4_multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut m = [0.0_f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            m[col * 4 + row] = sum;
        }
    }
    m
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}
