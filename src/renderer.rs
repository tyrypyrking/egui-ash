use ash::{vk, Device, Entry, Instance};
use egui_winit::winit;
use std::fmt::Debug;
use std::{
    collections::HashMap,
    fmt::Formatter,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
};

use crate::allocator::{Allocation, AllocationCreateInfo, Allocator, MemoryLocation};
use crate::viewport_context::ViewportContext;

// ─────────────────────────────────────────────────────────────────────────────
// ManagedTextures
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) struct ManagedTextures<A: Allocator + 'static> {
    device: Device,
    queue: vk::Queue,
    queue_family_index: u32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    allocator: A,

    pub(crate) texture_desc_sets: HashMap<egui::TextureId, vk::DescriptorSet>,
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

    pub(crate) fn new(
        device: Device,
        queue: vk::Queue,
        queue_family_index: u32,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        allocator: A,
    ) -> Self {
        let sampler = Self::create_sampler(&device);
        Self {
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
        }
    }

    fn update_texture(&mut self, texture_id: egui::TextureId, delta: egui::epaint::ImageDelta) {
        use crate::utils;

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
                    .flat_map(egui::Color32::to_array)
                    .collect()
            }
        };
        let cmd_pool = unsafe {
            self.device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.queue_family_index),
                    None,
                )
                .unwrap()
        };
        let cmd = unsafe {
            self.device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(1u32)
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .unwrap()[0]
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
                    .unwrap();
            };
            (texture_buffer, allocation)
        };
        let ptr = staging_allocation.mapped_ptr().unwrap().as_ptr().cast::<u8>();
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
                    .unwrap();
            };
            (handle, allocation)
        };
        let texture_image_view = unsafe {
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
            cmd,
            texture_image,
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
            cmd,
            texture_image,
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

        if let Some(pos) = delta.pos {
            // Blit texture data to existing texture if delta pos exists (e.g. font changed)
            let existing_texture = self.texture_images.get(&texture_id);
            if let Some(&existing_texture) = existing_texture {
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

                    self.device
                        .begin_command_buffer(
                            cmd,
                            &vk::CommandBufferBeginInfo::default()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )
                        .unwrap();

                    utils::insert_image_memory_barrier(
                        &self.device,
                        cmd,
                        existing_texture,
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
                    utils::insert_image_memory_barrier(
                        &self.device,
                        cmd,
                        texture_image,
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
                        existing_texture,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                        vk::Filter::NEAREST,
                    );

                    utils::insert_image_memory_barrier(
                        &self.device,
                        cmd,
                        existing_texture,
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

                    self.device.destroy_image(texture_image, None);
                    self.device.destroy_image_view(texture_image_view, None);
                    self.allocator.free(texture_allocation).unwrap();
                }
            } else {
                return;
            }
        } else {
            // Save the newly created texture
            let dsc_set = unsafe {
                self.device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.descriptor_pool)
                            .set_layouts(&[self.descriptor_set_layout]),
                    )
                    .unwrap()[0]
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
            self.texture_allocations.insert(texture_id, texture_allocation);
            self.texture_image_views.insert(texture_id, texture_image_view);
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

    pub(crate) fn update_textures(&mut self, textures_delta: egui::TexturesDelta) {
        for (id, image_delta) in textures_delta.set {
            self.update_texture(id, image_delta);
        }
        for id in textures_delta.free {
            self.free_texture(id);
        }
    }

    pub(crate) fn destroy(&mut self, device: &Device, allocator: &A) {
        unsafe {
            device
                .device_wait_idle()
                .expect("Failed to wait device idle");

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

// ─────────────────────────────────────────────────────────────────────────────
// UserTextures
// ─────────────────────────────────────────────────────────────────────────────

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

    #[must_use]
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

pub(crate) struct UserTextures {
    device: Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) texture_desc_sets: HashMap<u64, vk::DescriptorSet>,
    receiver: ImageRegistryReceiver,
}

impl UserTextures {
    pub(crate) fn new(
        device: Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        receiver: ImageRegistryReceiver,
    ) -> Self {
        Self {
            device,
            descriptor_pool,
            descriptor_set_layout,
            texture_desc_sets: HashMap::new(),
            receiver,
        }
    }

    fn register_user_texture(&mut self, id: u64, image_view: vk::ImageView, sampler: vk::Sampler) {
        let dsc_set = unsafe {
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

    pub(crate) fn update_textures(&mut self) {
        for command in self.receiver.try_iter().collect::<Vec<_>>() {
            match command {
                RegistryCommand::RegisterUserTexture {
                    image_view,
                    sampler,
                    id,
                } => match id {
                    egui::TextureId::Managed(_) => {
                        panic!("This texture id is not for user texture: {id:?}")
                    }
                    egui::TextureId::User(id) => {
                        self.register_user_texture(id, image_view, sampler);
                    }
                },
                RegistryCommand::UnregisterUserTexture { id } => match id {
                    egui::TextureId::Managed(_) => {
                        panic!("This texture id is not for user texture: {id:?}")
                    }
                    egui::TextureId::User(id) => {
                        self.unregister_user_texture(id);
                    }
                },
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ViewportOps trait + impl (for EguiCommand)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) trait ViewportOps {
    fn update_swapchain(&mut self, info: SwapchainUpdateInfo, scale_factor: f32, physical_size: winit::dpi::PhysicalSize<u32>);
    fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        index: usize,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: egui::TexturesDelta,
        scale: f32,
        size: winit::dpi::PhysicalSize<u32>,
    );
}

struct ViewportOpsImpl<A: Allocator + 'static> {
    context: *mut ViewportContext<A>,
    managed_textures: *mut ManagedTextures<A>,
    user_textures: *mut UserTextures,
}

// SAFETY: ViewportOpsImpl is only used within the single-threaded render loop.
// The raw pointers point into Integration which has a stable heap address.
unsafe impl<A: Allocator + 'static> Send for ViewportOpsImpl<A> {}

impl<A: Allocator + 'static> ViewportOps for ViewportOpsImpl<A> {
    fn update_swapchain(&mut self, info: SwapchainUpdateInfo, scale_factor: f32, physical_size: winit::dpi::PhysicalSize<u32>) {
        unsafe { (*self.context).update_pipeline(info, scale_factor, physical_size) }
    }

    fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        index: usize,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: egui::TexturesDelta,
        scale: f32,
        size: winit::dpi::PhysicalSize<u32>,
    ) {
        unsafe {
            (*self.context).record_commands(
                cmd,
                index,
                clipped_primitives,
                textures_delta,
                scale,
                size,
                &mut *self.managed_textures,
                &mut *self.user_textures,
            );
        }
    }
}

// No-op ViewportOps for EguiCommand::default()
struct NoopViewportOps;
impl ViewportOps for NoopViewportOps {
    fn update_swapchain(&mut self, _info: SwapchainUpdateInfo, _scale_factor: f32, _physical_size: winit::dpi::PhysicalSize<u32>) {}
    fn record(&mut self, _cmd: vk::CommandBuffer, _index: usize, _clipped_primitives: &[egui::ClippedPrimitive], _textures_delta: egui::TexturesDelta, _scale: f32, _size: winit::dpi::PhysicalSize<u32>) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// SwapchainUpdateInfo + EguiCommand (public API)
// ─────────────────────────────────────────────────────────────────────────────

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
    pub(crate) clipped_primitives: Vec<egui::ClippedPrimitive>,
    pub(crate) textures_delta: egui::TexturesDelta,
    pub(crate) scale_factor: f32,
    pub(crate) physical_size: winit::dpi::PhysicalSize<u32>,
    pub(crate) swapchain_recreate_required: bool,
    pub(crate) ops: Box<dyn ViewportOps>,
}

impl EguiCommand {
    /// Returns whether swapchain recreation is required.
    #[must_use]
    pub fn swapchain_recreate_required(&self) -> bool {
        self.swapchain_recreate_required
    }

    /// You must call this method once when first time to record commands
    /// and when you recreate swapchain.
    pub fn update_swapchain(&mut self, info: SwapchainUpdateInfo) {
        self.ops.update_swapchain(info, self.scale_factor, self.physical_size);
    }

    /// record commands to command buffer.
    pub fn record(mut self, cmd: vk::CommandBuffer, swapchain_index: usize) {
        self.ops.record(
            cmd,
            swapchain_index,
            &self.clipped_primitives,
            self.textures_delta,
            self.scale_factor,
            self.physical_size,
        );
    }
}

impl Default for EguiCommand {
    fn default() -> Self {
        Self {
            clipped_primitives: vec![],
            textures_delta: egui::TexturesDelta::default(),
            scale_factor: 1.0,
            physical_size: winit::dpi::PhysicalSize::new(0, 0),
            swapchain_recreate_required: false,
            ops: Box::new(NoopViewportOps),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) struct Renderer<A: Allocator + 'static> {
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    present_mode: vk::PresentModeKHR,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    allocator: A,
    viewport_contexts: HashMap<egui::ViewportId, ViewportContext<A>>,
    pub(crate) managed_textures: ManagedTextures<A>,
    pub(crate) user_textures: UserTextures,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        entry: Entry,
        instance: Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        surface_loader: ash::khr::surface::Instance,
        swapchain_loader: ash::khr::swapchain::Device,
        queue: vk::Queue,
        queue_family_index: u32,
        command_pool: vk::CommandPool,
        present_mode: vk::PresentModeKHR,
        allocator: A,
        receiver: Receiver<RegistryCommand>,
    ) -> Self {
        let descriptor_pool = Self::create_descriptor_pool(&device);
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);
        let managed_textures = ManagedTextures::new(
            device.clone(),
            queue,
            queue_family_index,
            descriptor_pool,
            descriptor_set_layout,
            allocator.clone(),
        );
        let user_textures = UserTextures::new(
            device.clone(),
            descriptor_pool,
            descriptor_set_layout,
            receiver,
        );
        Self {
            entry,
            instance,
            physical_device,
            device,
            surface_loader,
            swapchain_loader,
            queue,
            command_pool,
            present_mode,
            descriptor_pool,
            descriptor_set_layout,
            allocator,
            viewport_contexts: HashMap::new(),
            managed_textures,
            user_textures,
        }
    }

    pub(crate) fn mark_viewport_dirty(&mut self, viewport_id: egui::ViewportId) {
        if let Some(ctx) = self.viewport_contexts.get_mut(&viewport_id) {
            ctx.mark_dirty();
        }
    }

    pub(crate) fn ensure_viewport_context(
        &mut self,
        viewport_id: egui::ViewportId,
        window: &winit::window::Window,
    ) {
        if !self.viewport_contexts.contains_key(&viewport_id) {
            if let Some(ctx) = ViewportContext::create(
                &self.entry,
                &self.instance,
                self.physical_device,
                self.device.clone(),
                &self.surface_loader,
                &self.swapchain_loader,
                self.queue,
                self.command_pool,
                window,
                self.present_mode,
                self.descriptor_set_layout,
                self.allocator.clone(),
            ) {
                self.viewport_contexts.insert(viewport_id, ctx);
            }
        } else if let Some(ctx) = self.viewport_contexts.get_mut(&viewport_id) {
            ctx.recreate_if_dirty(window);
        }
    }

    pub(crate) fn present_viewport_auto(
        &mut self,
        viewport_id: egui::ViewportId,
        window: &winit::window::Window,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        scale: f32,
        size: winit::dpi::PhysicalSize<u32>,
    ) {
        self.ensure_viewport_context(viewport_id, window);
        if let Some(ctx) = self.viewport_contexts.get_mut(&viewport_id) {
            let _ = ctx.present_managed(
                clipped_primitives,
                textures_delta,
                scale,
                size,
                &mut self.managed_textures,
                &mut self.user_textures,
            );
        }
    }

    pub(crate) fn create_egui_cmd(
        &mut self,
        viewport_id: egui::ViewportId,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        scale_factor: f32,
        physical_size: winit::dpi::PhysicalSize<u32>,
    ) -> EguiCommand {
        // For the Handle path: lazily create a handle-path ViewportContext if not present
        if !self.viewport_contexts.contains_key(&viewport_id) {
            let ctx = ViewportContext::new_for_handle_path(
                self.entry.clone(),
                self.instance.clone(),
                self.physical_device,
                self.device.clone(),
                self.surface_loader.clone(),
                self.swapchain_loader.clone(),
                self.queue,
                self.command_pool,
                self.descriptor_set_layout,
                self.allocator.clone(),
            );
            self.viewport_contexts.insert(viewport_id, ctx);
        }

        let ctx = self.viewport_contexts.get_mut(&viewport_id).unwrap();
        let swapchain_recreate_required = ctx.swapchain_recreate_required(scale_factor);

        let ops = Box::new(ViewportOpsImpl {
            context: ctx as *mut ViewportContext<A>,
            managed_textures: &mut self.managed_textures as *mut ManagedTextures<A>,
            user_textures: &mut self.user_textures as *mut UserTextures,
        });

        EguiCommand {
            clipped_primitives,
            textures_delta,
            scale_factor,
            physical_size,
            swapchain_recreate_required,
            ops,
        }
    }

    pub(crate) fn destroy_viewports(&mut self, active_viewport_ids: &egui::ViewportIdSet) {
        let remove_ids: Vec<_> = self
            .viewport_contexts
            .keys()
            .filter(|id| !active_viewport_ids.contains(id) && **id != egui::ViewportId::ROOT)
            .copied()
            .collect();
        for id in remove_ids {
            if let Some(mut ctx) = self.viewport_contexts.remove(&id) {
                ctx.destroy();
            }
        }
    }

    pub(crate) fn destroy(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }
        self.managed_textures.destroy(&self.device, &self.allocator);
        for (_, mut ctx) in self.viewport_contexts.drain() {
            ctx.destroy();
        }
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
