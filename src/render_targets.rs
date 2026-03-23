use ash::vk;

pub(crate) struct RenderTargetPool {
    device: ash::Device,
    images: [vk::Image; 2],
    image_views: [vk::ImageView; 2],
    memory: [vk::DeviceMemory; 2],
    timeline: vk::Semaphore,
    timeline_value: u64,
    extent: vk::Extent2D,
    format: vk::Format,
    host_queue_family: u32,
    engine_queue_family: u32,
    cross_family: bool,
}

impl RenderTargetPool {
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
        format: vk::Format,
        host_queue_family: u32,
        engine_queue_family: u32,
    ) -> Self {
        let cross_family = host_queue_family != engine_queue_family;

        let create_image = |dev: &ash::Device| -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let image = dev.create_image(&image_info, None).unwrap();

            let mem_reqs = dev.get_image_memory_requirements(image);
            let mem_props = instance.get_physical_device_memory_properties(physical_device);
            let mem_type_index = find_memory_type(
                &mem_props,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index);
            let memory = dev.allocate_memory(&alloc_info, None).unwrap();
            dev.bind_image_memory(image, memory, 0).unwrap();

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = dev.create_image_view(&view_info, None).unwrap();

            (image, memory, view)
        };

        let (img0, mem0, view0) = create_image(device);
        let (img1, mem1, view1) = create_image(device);

        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let timeline = device.create_semaphore(&sem_info, None).unwrap();

        Self {
            device: device.clone(),
            images: [img0, img1],
            image_views: [view0, view1],
            memory: [mem0, mem1],
            timeline,
            timeline_value: 0,
            extent,
            format,
            host_queue_family,
            engine_queue_family,
            cross_family,
        }
    }

    pub(crate) fn make_target(&mut self, index: usize) -> crate::types::RenderTarget {
        let wait_value = self.timeline_value;
        self.timeline_value += 1;
        let signal_value = self.timeline_value;

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let acquire_barrier = if self.cross_family {
            Some(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(self.host_queue_family)
                    .dst_queue_family_index(self.engine_queue_family)
                    .image(self.images[index])
                    .subresource_range(subresource_range),
            )
        } else {
            None
        };

        let release_barrier = if self.cross_family {
            Some(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(self.engine_queue_family)
                    .dst_queue_family_index(self.host_queue_family)
                    .image(self.images[index])
                    .subresource_range(subresource_range),
            )
        } else {
            None
        };

        crate::types::RenderTarget {
            image: self.images[index],
            image_view: self.image_views[index],
            extent: self.extent,
            format: self.format,
            timeline: self.timeline,
            wait_value,
            signal_value,
            acquire_barrier,
            release_barrier,
        }
    }

    pub(crate) fn timeline(&self) -> vk::Semaphore {
        self.timeline
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }

    pub(crate) fn image(&self, index: usize) -> vk::Image {
        self.images[index]
    }

    pub(crate) fn image_view(&self, index: usize) -> vk::ImageView {
        self.image_views[index]
    }

    /// Advance the shared timeline counter and return the new value.
    /// Used by the host compositor submit to signal the timeline after compositing,
    /// so that the engine's next `RenderTarget::wait_value` is satisfied.
    pub(crate) fn next_signal_value(&mut self) -> u64 {
        self.timeline_value += 1;
        self.timeline_value
    }

    pub(crate) unsafe fn destroy(&mut self) {
        for &view in &self.image_views {
            self.device.destroy_image_view(view, None);
        }
        for &image in &self.images {
            self.device.destroy_image(image, None);
        }
        for &mem in &self.memory {
            self.device.free_memory(mem, None);
        }
        self.device.destroy_semaphore(self.timeline, None);
    }
}

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
