use ash::{vk, Device};
use bytemuck::bytes_of;
use egui_winit::winit;
use std::fmt::Debug;
use std::{
    collections::HashMap,
    ffi::CString,
    fmt::Formatter,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
};

use crate::allocator::{Allocation, AllocationCreateInfo, Allocator, MemoryLocation};
use crate::utils;

struct ViewportRendererState<A: Allocator + 'static> {
    width: u32,
    height: u32,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    swapchain_image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_allocations: Vec<A::Allocation>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_allocations: Vec<A::Allocation>,
    scale_factor: f32,
    physical_width: u32,
    physical_height: u32,
}

#[derive(Clone)]
struct ViewportRenderer<A: Allocator + 'static> {
    device: Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    allocator: A,
    state: Arc<Mutex<Option<ViewportRendererState<A>>>>,
}
impl<A: Allocator + 'static> ViewportRenderer<A> {
    fn new(device: Device, descriptor_set_layout: vk::DescriptorSetLayout, allocator: A) -> Self {
        Self {
            device,
            descriptor_set_layout,
            allocator,
            state: Arc::new(Mutex::new(None)),
        }
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
                    .dependencies(std::slice::from_ref(
                        &vk::SubpassDependency::default()
                            .src_subpass(vk::SUBPASS_EXTERNAL)
                            .dst_subpass(0)
                            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                    )),
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
                p_code: bytes_code.as_ptr() as *const u32,
                ..Default::default()
            };
            unsafe { device.create_shader_module(&shader_module_create_info, None) }
                .expect("Failed to create vertex shader module.")
        };
        let fragment_shader_module = {
            let bytes_code = include_bytes!("shaders/spv/frag.spv");
            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr() as *const u32,
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
                            .image(swapchain_image.clone())
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
            .map(|&image_views| unsafe {
                let attachments = &[image_views];
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

    fn create_buffers(
        device: &Device,
        swapchain_count: usize,
        allocator: &A,
    ) -> (
        Vec<vk::Buffer>,
        Vec<A::Allocation>,
        Vec<vk::Buffer>,
        Vec<A::Allocation>,
    ) {
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
                    .expect("Failed to create vertex buffer.")
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
                    .expect("Failed to create index buffer.")
            }

            vertex_buffers.push(vertex_buffer);
            vertex_buffer_allocations.push(vertex_buffer_allocation);
            index_buffers.push(index_buffer);
            index_buffer_allocations.push(index_buffer_allocation);
        }

        (
            vertex_buffers,
            vertex_buffer_allocations,
            index_buffers,
            index_buffer_allocations,
        )
    }

    fn update_swapchain(
        &mut self,
        width: u32,
        height: u32,
        swapchain_images: Vec<vk::Image>,
        surface_format: vk::Format,
        scale_factor: f32,
        physical_size: winit::dpi::PhysicalSize<u32>,
        allocator: A,
    ) {
        // wait device idle
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle.");
        }

        // cleanup framebuffers and others
        let (render_pass, pipeline_layout, pipeline) = {
            let Ok(mut state) = self.state.lock() else {
                panic!("Failed to lock state.");
            };
            if let Some(mut state) = state.take() {
                unsafe {
                    for vertex_buffer in state.vertex_buffers.drain(..) {
                        self.device.destroy_buffer(vertex_buffer, None);
                    }
                    for vertex_buffer_allocation in state.vertex_buffer_allocations.drain(..) {
                        self.allocator
                            .free(vertex_buffer_allocation)
                            .expect("Failed to free vertex buffer allocation.");
                    }
                    for index_buffer in state.index_buffers.drain(..) {
                        self.device.destroy_buffer(index_buffer, None);
                    }
                    for index_buffer_allocation in state.index_buffer_allocations.drain(..) {
                        self.allocator
                            .free(index_buffer_allocation)
                            .expect("Failed to free index buffer allocation.");
                    }
                    for framebuffer in state.framebuffers.drain(..) {
                        self.device.destroy_framebuffer(framebuffer, None);
                    }
                    for image_view in state.swapchain_image_views.drain(..) {
                        self.device.destroy_image_view(image_view, None);
                    }
                }

                (state.render_pass, state.pipeline_layout, state.pipeline)
            } else {
                let render_pass = Self::create_render_pass(&self.device, surface_format);
                let pipeline_layout =
                    Self::create_pipeline_layout(&self.device, self.descriptor_set_layout);
                let pipeline = Self::create_pipeline(&self.device, render_pass, pipeline_layout);
                (render_pass, pipeline_layout, pipeline)
            }
        };

        // Create Framebuffers
        let (framebuffers, swapchain_image_views) = Self::create_framebuffers(
            &self.device,
            &swapchain_images,
            render_pass,
            surface_format,
            width,
            height,
        );

        let (vertex_buffers, vertex_buffer_allocations, index_buffers, index_buffer_allocations) =
            Self::create_buffers(&self.device, swapchain_images.len(), &allocator);

        // update self
        let mut state = self.state.lock().expect("Failed to lock state.");
        *state = Some(ViewportRendererState {
            width,
            height,
            render_pass,
            pipeline_layout,
            pipeline,
            swapchain_image_views,
            framebuffers,
            vertex_buffers,
            vertex_buffer_allocations,
            index_buffers,
            index_buffer_allocations,
            scale_factor,
            physical_width: physical_size.width,
            physical_height: physical_size.height,
        });
    }

    // size for vertex buffer which egui-ash uses
    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    // size for index buffer which egui-ash uses
    fn index_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    fn create_egui_cmd(
        &self,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        managed_textures: Arc<Mutex<ManagedTextures<A>>>,
        user_textures: Arc<Mutex<UserTextures>>,
        scale_factor: f32,
        physical_size: winit::dpi::PhysicalSize<u32>,
    ) -> EguiCommand {
        EguiCommand {
            swapchain_recreate_required: {
                let this = self.clone();
                let state = this.state.lock().unwrap();
                if let Some(state) = &*state {
                    state.scale_factor != scale_factor
                } else {
                    false
                }
            },
            swapchain_updater: Some(Box::new({
                let mut this = self.clone();
                move |swapchain_update_info| {
                    let SwapchainUpdateInfo {
                        width,
                        height,
                        swapchain_images,
                        surface_format,
                    } = swapchain_update_info;
                    this.update_swapchain(
                        width,
                        height,
                        swapchain_images,
                        surface_format,
                        scale_factor,
                        physical_size,
                        this.allocator.clone(),
                    );
                }
            })),
            recorder: Box::new({
                let this = self.clone();
                move |cmd, index: usize| {
                    let state = this.state.lock().expect("Failed to lock state mutex.");
                    let state = state.as_ref().expect("State is none.");
                    let mut managed_textures =
                        managed_textures.lock().expect("Failed to lock textures.");
                    let mut user_textures =
                        user_textures.lock().expect("Failed to lock user textures.");

                    // update textures
                    managed_textures.update_textures(textures_delta);
                    user_textures.update_textures();

                    // get buffer ptr
                    let mut vertex_buffer_ptr = state.vertex_buffer_allocations[index]
                        .mapped_ptr()
                        .unwrap()
                        .as_ptr() as *mut u8;
                    let vertex_buffer_ptr_end =
                        unsafe { vertex_buffer_ptr.add(Self::vertex_buffer_size() as usize) };
                    let mut index_buffer_ptr = state.index_buffer_allocations[index]
                        .mapped_ptr()
                        .unwrap()
                        .as_ptr() as *mut u8;
                    let index_buffer_ptr_end =
                        unsafe { index_buffer_ptr.add(Self::index_buffer_size() as usize) };

                    // begin render pass
                    unsafe {
                        this.device.cmd_begin_render_pass(
                            cmd,
                            &vk::RenderPassBeginInfo::default()
                                .render_pass(state.render_pass)
                                .framebuffer(state.framebuffers[index])
                                .clear_values(&[])
                                .render_area(
                                    vk::Rect2D::default().extent(
                                        vk::Extent2D::default()
                                            .width(state.width)
                                            .height(state.height),
                                    ),
                                ),
                            vk::SubpassContents::INLINE,
                        );
                    }

                    // bind resources
                    unsafe {
                        this.device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            state.pipeline,
                        );
                        this.device.cmd_bind_vertex_buffers(
                            cmd,
                            0,
                            &[state.vertex_buffers[index]],
                            &[0],
                        );
                        this.device.cmd_bind_index_buffer(
                            cmd,
                            state.index_buffers[index],
                            0,
                            vk::IndexType::UINT32,
                        );
                        let width_points = state.physical_width as f32 / state.scale_factor;
                        let height_points = state.physical_height as f32 / state.scale_factor;
                        this.device.cmd_push_constants(
                            cmd,
                            state.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytes_of(&width_points),
                        );
                        this.device.cmd_push_constants(
                            cmd,
                            state.pipeline_layout,
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
                                    if let Some(&descriptor_set) =
                                        user_textures.texture_desc_sets.get(&id)
                                    {
                                        this.device.cmd_bind_descriptor_sets(
                                            cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            state.pipeline_layout,
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
                                    this.device.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        state.pipeline_layout,
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

                        // map memory
                        unsafe {
                            vertex_buffer_ptr.copy_from(v_slice.as_ptr() as *const u8, v_copy_size)
                        };
                        unsafe {
                            index_buffer_ptr.copy_from(i_slice.as_ptr() as *const u8, i_copy_size)
                        };

                        vertex_buffer_ptr = vertex_buffer_ptr_next;
                        index_buffer_ptr = index_buffer_ptr_next;

                        // record draw commands
                        unsafe {
                            let min = clip_rect.min;
                            let min = egui::Pos2 {
                                x: min.x * state.scale_factor as f32,
                                y: min.y * state.scale_factor as f32,
                            };
                            let min = egui::Pos2 {
                                x: f32::clamp(min.x, 0.0, state.physical_width as f32),
                                y: f32::clamp(min.y, 0.0, state.physical_height as f32),
                            };
                            let max = clip_rect.max;
                            let max = egui::Pos2 {
                                x: max.x * state.scale_factor as f32,
                                y: max.y * state.scale_factor as f32,
                            };
                            let max = egui::Pos2 {
                                x: f32::clamp(max.x, min.x, state.physical_width as f32),
                                y: f32::clamp(max.y, min.y, state.physical_height as f32),
                            };
                            this.device.cmd_set_scissor(
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
                            this.device.cmd_set_viewport(
                                cmd,
                                0,
                                std::slice::from_ref(
                                    &vk::Viewport::default()
                                        .x(0.0)
                                        .y(0.0)
                                        .width(state.physical_width as f32)
                                        .height(state.physical_height as f32)
                                        .min_depth(0.0)
                                        .max_depth(1.0),
                                ),
                            );
                            this.device.cmd_draw_indexed(
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
                        this.device.cmd_end_render_pass(cmd);
                    }
                }
            }),
        }
    }

    fn destroy(&mut self) {
        let mut state = self.state.lock().expect("Failed to lock state mutex.");
        if let Some(mut state) = state.take() {
            // wait device idle
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("Failed to wait device idle")
            };

            // destroy state
            unsafe {
                for image_view in state.swapchain_image_views.drain(..) {
                    self.device.destroy_image_view(image_view, None);
                }
                self.device.destroy_render_pass(state.render_pass, None);
                for vertex_buffer in state.vertex_buffers.drain(..) {
                    self.device.destroy_buffer(vertex_buffer, None);
                }
                for vertex_buffer_allocation in state.vertex_buffer_allocations.drain(..) {
                    self.allocator
                        .free(vertex_buffer_allocation)
                        .expect("Failed to free vertex buffer allocation.");
                }
                for index_buffer in state.index_buffers.drain(..) {
                    self.device.destroy_buffer(index_buffer, None);
                }
                for index_buffer_allocation in state.index_buffer_allocations.drain(..) {
                    self.allocator
                        .free(index_buffer_allocation)
                        .expect("Failed to free index buffer allocation.");
                }
                for framebuffer in state.framebuffers.drain(..) {
                    self.device.destroy_framebuffer(framebuffer, None);
                }
                for image_view in state.swapchain_image_views.drain(..) {
                    self.device.destroy_image_view(image_view, None);
                }
                self.device.destroy_pipeline(state.pipeline, None);
                self.device
                    .destroy_pipeline_layout(state.pipeline_layout, None);
            }
        }
    }
}

struct ManagedTextures<A: Allocator + 'static> {
    device: Device,
    queue: vk::Queue,
    queue_family_index: u32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    allocator: A,

    texture_desc_sets: HashMap<egui::TextureId, vk::DescriptorSet>,
    texture_images: HashMap<egui::TextureId, vk::Image>,
    texture_allocations: HashMap<egui::TextureId, A::Allocation>,
    texture_image_views: HashMap<egui::TextureId, vk::ImageView>,
}
impl<A: Allocator + 'static> ManagedTextures<A> {
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
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }
        .expect("Failed to create sampler.")
    }

    fn new(
        device: Device,
        queue: vk::Queue,
        queue_family_index: u32,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        allocator: A,
    ) -> Arc<Mutex<Self>> {
        let sampler = Self::create_sampler(&device);

        Arc::new(Mutex::new(Self {
            device,
            queue,
            queue_family_index,
            descriptor_pool,
            descriptor_set_layout,
            sampler,
            allocator,
            texture_desc_sets: HashMap::new(),
            texture_images: HashMap::new(),
            texture_allocations: HashMap::new(),
            texture_image_views: HashMap::new(),
        }))
    }

    fn update_texture(&mut self, texture_id: egui::TextureId, delta: egui::epaint::ImageDelta) {
        // Extract pixel data from egui
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
                    .flat_map(|color| color.to_array())
                    .collect()
            }
            egui::ImageData::Font(image) => image
                .srgba_pixels(None)
                .flat_map(|color| color.to_array())
                .collect(),
        };
        let cmd_pool = {
            unsafe {
                self.device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(self.queue_family_index),
                        None,
                    )
                    .unwrap()
            }
        };
        let cmd = {
            unsafe {
                self.device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_buffer_count(1u32)
                            .command_pool(cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )
                    .unwrap()[0]
            }
        };
        let cmd_fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap()
        };

        let (staging_buffer, staging_allocation) = {
            let buffer_size = data.len() as vk::DeviceSize;
            let buffer_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            let texture_buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.unwrap();
            let requirements =
                unsafe { self.device.get_buffer_memory_requirements(texture_buffer) };
            let allocation = self
                .allocator
                .allocate(A::AllocationCreateInfo::new(
                    Some("egui-ash image staging buffer"),
                    requirements,
                    MemoryLocation::cpu_to_gpu(),
                    true,
                ))
                .unwrap();
            unsafe {
                self.device
                    .bind_buffer_memory(texture_buffer, allocation.memory(), allocation.offset())
                    .unwrap()
            };
            (texture_buffer, allocation)
        };
        let ptr = staging_allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
        unsafe {
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        let (texture_image, texture_allocation) = {
            let extent = vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            };
            let handle = unsafe {
                self.device.create_image(
                    &vk::ImageCreateInfo::default()
                        .array_layers(1)
                        .extent(extent)
                        .flags(vk::ImageCreateFlags::empty())
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .image_type(vk::ImageType::TYPE_2D)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .mip_levels(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::TRANSFER_SRC,
                        ),
                    None,
                )
            }
            .unwrap();
            let requirements = unsafe { self.device.get_image_memory_requirements(handle) };
            let allocation = self
                .allocator
                .allocate(A::AllocationCreateInfo::new(
                    Some("egui-ash image buffer"),
                    requirements,
                    MemoryLocation::gpu_only(),
                    false,
                ))
                .unwrap();
            unsafe {
                self.device
                    .bind_image_memory(handle, allocation.memory(), allocation.offset())
                    .unwrap()
            };
            (handle, allocation)
        };
        let texture_image_view = {
            unsafe {
                self.device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .components(vk::ComponentMapping::default())
                            .flags(vk::ImageViewCreateFlags::empty())
                            .format(vk::Format::R8G8B8A8_UNORM)
                            .image(texture_image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_array_layer: 0,
                                base_mip_level: 0,
                                layer_count: 1,
                                level_count: 1,
                            })
                            .view_type(vk::ImageViewType::TYPE_2D),
                        None,
                    )
                    .unwrap()
            }
        };

        // begin cmd
        unsafe {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        // Transition texture image for transfer dst
        utils::insert_image_memory_barrier(
            &self.device,
            &cmd,
            &texture_image,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::NONE_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                base_mip_level: 0,
                layer_count: 1,
                level_count: 1,
            },
        );
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                texture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(
                    &vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(delta.image.width() as u32)
                        .buffer_image_height(delta.image.height() as u32)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_array_layer: 0,
                            layer_count: 1,
                            mip_level: 0,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: delta.image.width() as u32,
                            height: delta.image.height() as u32,
                            depth: 1,
                        }),
                ),
            );
        }
        utils::insert_image_memory_barrier(
            &self.device,
            &cmd,
            &texture_image,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                base_mip_level: 0,
                layer_count: 1,
                level_count: 1,
            },
        );

        unsafe {
            self.device.end_command_buffer(cmd).unwrap();
        }
        let cmd_buffs = [cmd];
        unsafe {
            self.device
                .queue_submit(
                    self.queue,
                    std::slice::from_ref(&vk::SubmitInfo::default().command_buffers(&cmd_buffs)),
                    cmd_fence,
                )
                .unwrap();
            self.device
                .wait_for_fences(&[cmd_fence], true, u64::MAX)
                .unwrap();
        }

        // texture is now in GPU memory, now we need to decide whether we should register it as new or update existing

        if let Some(pos) = delta.pos {
            // Blit texture data to existing texture if delta pos exists (e.g. font changed)
            let existing_texture = self.texture_images.get(&texture_id);
            if let Some(existing_texture) = existing_texture {
                let extent = vk::Extent3D {
                    width: delta.image.width() as u32,
                    height: delta.image.height() as u32,
                    depth: 1,
                };
                unsafe {
                    self.device
                        .reset_command_pool(cmd_pool, vk::CommandPoolResetFlags::empty())
                        .unwrap();
                    self.device.reset_fences(&[cmd_fence]).unwrap();

                    // begin cmd buff
                    self.device
                        .begin_command_buffer(
                            cmd,
                            &vk::CommandBufferBeginInfo::default()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )
                        .unwrap();

                    // Transition existing image for transfer dst
                    utils::insert_image_memory_barrier(
                        &self.device,
                        &cmd,
                        &existing_texture,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::AccessFlags::SHADER_READ,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    );
                    // Transition new image for transfer src
                    utils::insert_image_memory_barrier(
                        &self.device,
                        &cmd,
                        &texture_image,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::AccessFlags::SHADER_READ,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    );
                    let top_left = vk::Offset3D {
                        x: pos[0] as i32,
                        y: pos[1] as i32,
                        z: 0,
                    };
                    let bottom_right = vk::Offset3D {
                        x: pos[0] as i32 + delta.image.width() as i32,
                        y: pos[1] as i32 + delta.image.height() as i32,
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
                                x: extent.width as i32,
                                y: extent.height as i32,
                                z: extent.depth as i32,
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
                        *existing_texture,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                        vk::Filter::NEAREST,
                    );

                    // Transition existing image for shader read
                    utils::insert_image_memory_barrier(
                        &self.device,
                        &cmd,
                        &existing_texture,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    );
                    self.device.end_command_buffer(cmd).unwrap();
                    let cmd_buffs = [cmd];
                    self.device
                        .queue_submit(
                            self.queue,
                            std::slice::from_ref(
                                &vk::SubmitInfo::default().command_buffers(&cmd_buffs),
                            ),
                            cmd_fence,
                        )
                        .unwrap();
                    self.device
                        .wait_for_fences(&[cmd_fence], true, u64::MAX)
                        .unwrap();

                    // destroy texture_image and view
                    self.device.destroy_image(texture_image, None);
                    self.device.destroy_image_view(texture_image_view, None);
                    self.allocator.free(texture_allocation).unwrap();
                }
            } else {
                return;
            }
        } else {
            // Otherwise save the newly created texture

            // update dsc set
            let dsc_set = {
                unsafe {
                    self.device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(self.descriptor_pool)
                                .set_layouts(&[self.descriptor_set_layout]),
                        )
                        .unwrap()[0]
                }
            };
            unsafe {
                self.device.update_descriptor_sets(
                    std::slice::from_ref(
                        &vk::WriteDescriptorSet::default()
                            .dst_set(dsc_set)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .dst_array_element(0_u32)
                            .dst_binding(0_u32)
                            .image_info(std::slice::from_ref(
                                &vk::DescriptorImageInfo::default()
                                    .image_view(texture_image_view)
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                    .sampler(self.sampler),
                            )),
                    ),
                    &[],
                );
            }
            // destroy old texture
            if let Some((_, image)) = self.texture_images.remove_entry(&texture_id) {
                unsafe {
                    self.device.destroy_image(image, None);
                }
            }
            if let Some((_, image_view)) = self.texture_image_views.remove_entry(&texture_id) {
                unsafe {
                    self.device.destroy_image_view(image_view, None);
                }
            }
            // register new texture
            self.texture_images.insert(texture_id, texture_image);
            self.texture_allocations
                .insert(texture_id, texture_allocation);
            self.texture_image_views
                .insert(texture_id, texture_image_view);
            self.texture_desc_sets.insert(texture_id, dsc_set);
        }
        // cleanup
        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.destroy_command_pool(cmd_pool, None);
            self.allocator.free(staging_allocation).unwrap();
            self.device.destroy_fence(cmd_fence, None);
        }
    }

    fn free_texture(&mut self, id: egui::TextureId) {
        self.texture_desc_sets.remove_entry(&id);
        if let Some((_, image)) = self.texture_images.remove_entry(&id) {
            unsafe {
                self.device.destroy_image(image, None);
            }
        }
        if let Some((_, image_view)) = self.texture_image_views.remove_entry(&id) {
            unsafe {
                self.device.destroy_image_view(image_view, None);
            }
        }
        if let Some((_, allocation)) = self.texture_allocations.remove_entry(&id) {
            self.allocator.free(allocation).unwrap();
        }
    }

    fn update_textures(&mut self, textures_delta: egui::TexturesDelta) {
        for (id, image_delta) in textures_delta.set {
            self.update_texture(id, image_delta);
        }
        for id in textures_delta.free {
            self.free_texture(id);
        }
    }

    fn destroy(&mut self, device: &Device, allocator: &A) {
        // wait device idle
        unsafe {
            device
                .device_wait_idle()
                .expect("Failed to wait device idle")
        };

        // destroy images
        unsafe {
            for (_, image) in self.texture_images.drain() {
                device.destroy_image(image, None);
            }
            for (_, image_view) in self.texture_image_views.drain() {
                device.destroy_image_view(image_view, None);
            }
            for (_, allocation) in self.texture_allocations.drain() {
                allocator.free(allocation).unwrap();
            }
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

pub(crate) type ImageRegistryReceiver = Receiver<RegistryCommand>;

#[derive(Clone)]
pub struct ImageRegistry {
    sender: Sender<RegistryCommand>,
    counter: Arc<AtomicU64>,
}
impl ImageRegistry {
    pub(crate) fn new() -> (Self, ImageRegistryReceiver) {
        let (sender, receiver) = mpsc::channel();
        (
            Self {
                sender,
                counter: Arc::new(AtomicU64::new(0)),
            },
            receiver,
        )
    }

    pub fn register_user_texture(
        &self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> egui::TextureId {
        let id = egui::TextureId::User(self.counter.fetch_add(1, Ordering::SeqCst));
        self.sender
            .send(RegistryCommand::RegisterUserTexture {
                image_view,
                sampler,
                id,
            })
            .expect("Failed to send register user texture command.");
        id
    }

    pub fn unregister_user_texture(&self, id: egui::TextureId) {
        let _ = self
            .sender
            .send(RegistryCommand::UnregisterUserTexture { id });
    }
}
impl Debug for ImageRegistry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageRegistry").finish()
    }
}

pub(crate) enum RegistryCommand {
    RegisterUserTexture {
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        id: egui::TextureId,
    },
    UnregisterUserTexture {
        id: egui::TextureId,
    },
}

struct UserTextures {
    device: Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    texture_desc_sets: HashMap<u64, vk::DescriptorSet>,
    receiver: ImageRegistryReceiver,
}
impl UserTextures {
    fn new(
        device: Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        receiver: ImageRegistryReceiver,
    ) -> Arc<Mutex<Self>> {
        let texture_desc_sets = HashMap::new();

        Arc::new(Mutex::new(Self {
            device,
            descriptor_pool,
            descriptor_set_layout,
            texture_desc_sets,
            receiver,
        }))
    }

    fn register_user_texture(&mut self, id: u64, image_view: vk::ImageView, sampler: vk::Sampler) {
        let dsc_set = {
            unsafe {
                self.texture_desc_sets.insert(
                    id,
                    self.device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(self.descriptor_pool)
                                .set_layouts(&[self.descriptor_set_layout]),
                        )
                        .expect("Failed to allocate descriptor set")[0],
                );
            }
            self.texture_desc_sets.get(&id).unwrap()
        };
        unsafe {
            self.device.update_descriptor_sets(
                std::slice::from_ref(
                    &vk::WriteDescriptorSet::default()
                        .dst_set(*dsc_set)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_array_element(0_u32)
                        .dst_binding(0_u32)
                        .image_info(std::slice::from_ref(
                            &vk::DescriptorImageInfo::default()
                                .image_view(image_view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .sampler(sampler),
                        )),
                ),
                &[],
            );
        }
    }

    fn unregister_user_texture(&mut self, id: u64) {
        if let Some(desc_set) = self.texture_desc_sets.remove(&id) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("Failed to wait device idle");
                self.device
                    .free_descriptor_sets(self.descriptor_pool, &[desc_set])
                    .expect("Failed to free descriptor set.");
            }
        }
    }

    fn update_textures(&mut self) {
        for command in self.receiver.try_iter().collect::<Vec<_>>() {
            match command {
                RegistryCommand::RegisterUserTexture {
                    image_view,
                    sampler,
                    id,
                } => match id {
                    egui::TextureId::Managed(_) => {
                        panic!("This texture id is not for user texture: {:?}", id)
                    }
                    egui::TextureId::User(id) => {
                        self.register_user_texture(id, image_view, sampler);
                    }
                },
                RegistryCommand::UnregisterUserTexture { id } => match id {
                    egui::TextureId::Managed(_) => {
                        panic!("This texture id is not for user texture: {:?}", id)
                    }
                    egui::TextureId::User(id) => {
                        self.unregister_user_texture(id);
                    }
                },
            }
        }
    }
}

pub(crate) struct Renderer<A: Allocator + 'static> {
    device: Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    allocator: A,
    viewport_renderers: HashMap<egui::ViewportId, ViewportRenderer<A>>,

    managed_textures: Arc<Mutex<ManagedTextures<A>>>,
    user_textures: Arc<Mutex<UserTextures>>,
}
impl<A: Allocator + 'static> Renderer<A> {
    fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
        unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(1024)
                    .pool_sizes(std::slice::from_ref(
                        &vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024),
                    )),
                None,
            )
        }
        .expect("Failed to create descriptor pool.")
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(
                    &vk::DescriptorSetLayoutBinding::default()
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .binding(0)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                )),
                None,
            )
        }
        .expect("Failed to create descriptor set layout.")
    }

    pub(crate) fn new(
        device: Device,
        queue: vk::Queue,
        queue_family_index: u32,
        allocator: A,
        receiver: Receiver<RegistryCommand>,
    ) -> Arc<Mutex<Self>> {
        let descriptor_pool = Self::create_descriptor_pool(&device);
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);
        Arc::new(Mutex::new(Self {
            device: device.clone(),
            descriptor_pool,
            descriptor_set_layout,
            allocator: allocator.clone(),
            viewport_renderers: HashMap::new(),
            managed_textures: ManagedTextures::new(
                device.clone(),
                queue,
                queue_family_index,
                descriptor_pool,
                descriptor_set_layout,
                allocator,
            ),
            user_textures: UserTextures::new(
                device,
                descriptor_pool,
                descriptor_set_layout,
                receiver,
            ),
        }))
    }

    pub(crate) fn create_egui_cmd(
        &mut self,
        viewport_id: egui::ViewportId,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        scale_factor: f32,
        physical_size: winit::dpi::PhysicalSize<u32>,
    ) -> EguiCommand {
        let viewport_renderer = self
            .viewport_renderers
            .entry(viewport_id)
            .or_insert_with(|| {
                ViewportRenderer::new(
                    self.device.clone(),
                    self.descriptor_set_layout,
                    self.allocator.clone(),
                )
            });
        viewport_renderer.create_egui_cmd(
            clipped_primitives,
            textures_delta,
            self.managed_textures.clone(),
            self.user_textures.clone(),
            scale_factor,
            physical_size,
        )
    }

    pub(crate) fn destroy_viewports(&mut self, active_viewport_ids: &egui::ViewportIdSet) {
        let remove_viewports = self
            .viewport_renderers
            .keys()
            .filter(|id| !active_viewport_ids.contains(id))
            .filter(|id| id != &&egui::ViewportId::ROOT)
            .map(|id| id.clone())
            .collect::<Vec<_>>();

        for id in remove_viewports {
            if let Some(mut viewport_renderer) = self.viewport_renderers.remove(&id) {
                viewport_renderer.destroy();
            }
        }
    }

    pub(crate) fn destroy_root(&mut self) {
        // wait device idle
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle")
        };

        self.managed_textures
            .lock()
            .unwrap()
            .destroy(&self.device, &self.allocator);
        for (_, mut viewport_renderer) in self.viewport_renderers.drain() {
            viewport_renderer.destroy();
        }
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

/// struct to pass to `EguiCommand::update_swapchain` method.
pub struct SwapchainUpdateInfo {
    pub width: u32,
    pub height: u32,
    pub swapchain_images: Vec<vk::Image>,
    pub surface_format: vk::Format,
}

/// command recorder to record egui draw commands.
///
/// if you recreate swapchain, you must call `update_swapchain` method.
/// You also must call `update_swapchain` method when first time to record commands.
pub struct EguiCommand {
    swapchain_updater: Option<Box<dyn FnOnce(SwapchainUpdateInfo) + Send>>,
    recorder: Box<dyn FnOnce(vk::CommandBuffer, usize) + Send>,
    swapchain_recreate_required: bool,
}
impl EguiCommand {
    /// You must call this method once when first time to record commands
    /// and when you recreate swapchain.
    pub fn update_swapchain(&mut self, info: SwapchainUpdateInfo) {
        (self.swapchain_updater.take().expect(
            "The swapchain has been updated more than once. Always update swapchain more than once.",
        ))(info);
    }

    /// record commands to command buffer.
    pub fn record(self, cmd: vk::CommandBuffer, swapchain_index: usize) {
        (self.recorder)(cmd, swapchain_index);
    }

    /// Returns whether swapchain recreation is required.
    pub fn swapchain_recreate_required(&self) -> bool {
        self.swapchain_recreate_required
    }
}
impl Default for EguiCommand {
    fn default() -> Self {
        Self {
            swapchain_updater: None,
            recorder: Box::new(|_, _| {}),
            swapchain_recreate_required: false,
        }
    }
}
