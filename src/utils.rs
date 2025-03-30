#[cfg(feature = "persistence")]
use egui_winit::winit::event_loop::EventLoopWindowTarget;

#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_image_memory_barrier(
    device: &ash::Device,
    cmd: &ash::vk::CommandBuffer,
    image: &ash::vk::Image,
    src_q_family_index: u32,
    dst_q_family_index: u32,
    src_access_mask: ash::vk::AccessFlags,
    dst_access_mask: ash::vk::AccessFlags,
    old_image_layout: ash::vk::ImageLayout,
    new_image_layout: ash::vk::ImageLayout,
    src_stage_mask: ash::vk::PipelineStageFlags,
    dst_stage_mask: ash::vk::PipelineStageFlags,
    subresource_range: ash::vk::ImageSubresourceRange,
) {
    let image_memory_barrier = ash::vk::ImageMemoryBarrier::builder()
        .src_queue_family_index(src_q_family_index)
        .dst_queue_family_index(dst_q_family_index)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_image_layout)
        .new_layout(new_image_layout)
        .image(*image)
        .subresource_range(subresource_range)
        .build();
    unsafe {
        device.cmd_pipeline_barrier(
            *cmd,
            src_stage_mask,
            dst_stage_mask,
            ash::vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[image_memory_barrier],
        );
    }
}

#[cfg(feature = "persistence")]
pub(crate) fn largest_monitor_point_size<E>(
    egui_zoom_factor: f32,
    event_loop: &EventLoopWindowTarget<E>,
) -> egui::Vec2 {
    let mut max_size = egui::Vec2::ZERO;

    let available_monitors = { event_loop.available_monitors() };

    for monitor in available_monitors {
        let size = monitor
            .size()
            .to_logical::<f32>(egui_zoom_factor as f64 * monitor.scale_factor());
        let size = egui::vec2(size.width, size.height);
        max_size = max_size.max(size);
    }

    if max_size == egui::Vec2::ZERO {
        egui::Vec2::splat(16000.0)
    } else {
        max_size
    }
}
