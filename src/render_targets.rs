use crate::compositor::find_memory_type;
use ash::vk;

pub(crate) struct RenderTargetPool {
    device: ash::Device,
    images: [vk::Image; 2],
    image_views: [vk::ImageView; 2],
    memory: [vk::DeviceMemory; 2],
    // Engine signals this timeline after rendering a frame. Never CPU-signaled.
    engine_timeline: vk::Semaphore,
    // Compositor signals this timeline after sampling a slot. Never CPU-signaled.
    compositor_timeline: vk::Semaphore,
    engine_counter: u64,
    compositor_counter: u64,
    extent: vk::Extent2D,
    format: vk::Format,
}

impl RenderTargetPool {
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Self {
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

        let make_timeline = |dev: &ash::Device| -> vk::Semaphore {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            dev.create_semaphore(&sem_info, None).unwrap()
        };
        let engine_timeline = make_timeline(device);
        let compositor_timeline = make_timeline(device);

        Self {
            device: device.clone(),
            images: [img0, img1],
            image_views: [view0, view1],
            memory: [mem0, mem1],
            engine_timeline,
            compositor_timeline,
            engine_counter: 0,
            compositor_counter: 0,
            extent,
            format,
        }
    }

    pub(crate) fn make_target(&mut self, index: usize) -> crate::types::RenderTarget {
        // Engine waits on the compositor's most recently committed signal.
        // This guarantees the compositor has finished sampling this slot
        // before the engine overwrites it. Initial value 0 makes the very
        // first target wait trivially-satisfied.
        let wait_value = self.compositor_counter;

        // Engine signals the next monotone value on its own timeline.
        self.engine_counter += 1;
        let signal_value = self.engine_counter;

        // Cross-queue-family ownership transfer is not yet supported
        // (see VulkanContext docs and docs/known-limitations.md).
        // `Host::new` asserts host_queue_family_index == engine_queue_family_index,
        // so the barriers are always None here.
        crate::types::RenderTarget {
            image: self.images[index],
            image_view: self.image_views[index],
            extent: self.extent,
            format: self.format,
            wait_semaphore: self.compositor_timeline,
            wait_value,
            signal_semaphore: self.engine_timeline,
            signal_value,
            acquire_barrier: None,
            release_barrier: None,
        }
    }

    pub(crate) fn engine_timeline(&self) -> vk::Semaphore {
        self.engine_timeline
    }

    pub(crate) fn compositor_timeline(&self) -> vk::Semaphore {
        self.compositor_timeline
    }

    pub(crate) fn image(&self, index: usize) -> vk::Image {
        self.images[index]
    }

    pub(crate) fn image_view(&self, index: usize) -> vk::ImageView {
        self.image_views[index]
    }

    /// Peek the value the compositor should signal on its next GPU submit.
    /// Does not advance state — call `commit_compositor_signal` once the
    /// submit has succeeded.
    pub(crate) fn next_compositor_signal_value(&self) -> u64 {
        self.compositor_counter + 1
    }

    /// Commit a compositor signal value once the GPU submit is in flight.
    /// Must be called exactly once per successful submit.
    pub(crate) fn commit_compositor_signal(&mut self, value: u64) {
        debug_assert_eq!(value, self.compositor_counter + 1);
        self.compositor_counter = value;
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
        self.device.destroy_semaphore(self.engine_timeline, None);
        self.device
            .destroy_semaphore(self.compositor_timeline, None);
    }
}
