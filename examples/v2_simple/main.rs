//! Minimal v2 egui-ash example: a `ColorEngine` that clears render targets to
//! a solid color controlled by an egui sidebar.
//!
//! Demonstrates:
//! - `EngineRenderer` trait implementation
//! - egui sidebar with color picker + engine status
//! - Engine viewport displayed as an egui User texture
//! - Crash test button + restart capability

use ash::vk;
use std::ffi::CStr;
use std::ffi::CString;
use std::process::ExitCode;

use egui_ash::{
    EngineEvent, EngineHandle, EngineRenderer, EngineStatus, RunOption, VulkanContext,
};

// ─────────────────────────────────────────────────────────────────────────────
// UiState / EngineState — shared between UI and engine via ArcSwap
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct UiState {
    clear_color: [f32; 3],
    should_crash: bool,
}

#[derive(Clone, Default)]
struct EngineState {
    frame_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorEngine — trivial EngineRenderer that clears to a solid color
// ─────────────────────────────────────────────────────────────────────────────

struct ColorEngine {
    device: Option<ash::Device>,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    frame_count: u64,
}

impl ColorEngine {
    fn new() -> Self {
        Self {
            device: None,
            queue: vk::Queue::null(),
            queue_family_index: u32::MAX,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            frame_count: 0,
        }
    }
}

impl EngineRenderer for ColorEngine {
    type UiState = UiState;
    type EngineState = EngineState;

    fn init(&mut self, ctx: egui_ash::EngineContext) {
        self.queue = ctx.queue;
        self.queue_family_index = ctx.queue_family_index;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(ctx.queue_family_index);

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            let pool = ctx
                .device
                .create_command_pool(&pool_info, None)
                .expect("failed to create command pool");
            let cmd = ctx
                .device
                .allocate_command_buffers(&cmd_alloc_info.command_pool(pool))
                .expect("failed to allocate command buffer")[0];
            self.command_pool = pool;
            self.command_buffer = cmd;
        }

        self.device = Some(ctx.device);
    }

    fn render(
        &mut self,
        target: egui_ash::RenderTarget,
        ui_state: &Self::UiState,
        engine_state: &mut Self::EngineState,
    ) -> egui_ash::CompletedFrame {
        if ui_state.should_crash {
            panic!("Crash requested by UI");
        }

        let device = self.device.as_ref().expect("engine not initialized");
        let cmd = self.command_buffer;

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

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            // If cross-queue-family: insert acquire barrier
            // Otherwise: transition from UNDEFINED to TRANSFER_DST_OPTIMAL
            if let Some(acquire) = target.acquire_barrier() {
                // Cross-family acquire. The barrier transitions to
                // COLOR_ATTACHMENT_OPTIMAL; we still need to go to
                // TRANSFER_DST_OPTIMAL for vkCmdClearColorImage.
                let barriers = [*acquire];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);

                // COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_DST_OPTIMAL
                let to_transfer = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [to_transfer];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            } else {
                // Same-family: simple transition UNDEFINED -> TRANSFER_DST_OPTIMAL
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }

            // Clear the image
            let clear_color = vk::ClearColorValue {
                float32: [
                    ui_state.clear_color[0],
                    ui_state.clear_color[1],
                    ui_state.clear_color[2],
                    1.0,
                ],
            };
            let ranges = [subresource_range];
            device.cmd_clear_color_image(
                cmd,
                target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &ranges,
            );

            // Transition to SHADER_READ_ONLY_OPTIMAL (for compositor sampling)
            if let Some(release) = target.release_barrier() {
                // First: TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL
                // (the release barrier expects COLOR_ATTACHMENT_OPTIMAL as old_layout)
                let to_color_attach = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [to_color_attach];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);

                // Cross-family release barrier
                let barriers = [*release];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            } else {
                // Same-family: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
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

            // Submit with timeline semaphore wait + signal
            let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(target.timeline)
                .value(target.wait_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(target.timeline)
                .value(target.signal_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmd_buf_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
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

        target.complete()
    }

    fn handle_event(&mut self, _event: EngineEvent) {
        // No input handling needed for this simple example.
    }

    fn destroy(&mut self) {
        if let Some(device) = self.device.take() {
            unsafe {
                device.device_wait_idle().ok();
                if self.command_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.command_pool, None);
                    self.command_pool = vk::CommandPool::null();
                }
            }
        }
    }
}

impl Drop for ColorEngine {
    fn drop(&mut self) {
        // Safety net: destroy GPU resources even after a panic.
        if let Some(device) = self.device.take() {
            unsafe {
                device.device_wait_idle().ok();
                if self.command_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.command_pool, None);
                    self.command_pool = vk::CommandPool::null();
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan setup
// ─────────────────────────────────────────────────────────────────────────────

const ENABLE_VALIDATION: bool = cfg!(debug_assertions);

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = CStr::from_ptr((*data).p_message);
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => eprintln!("[VK ERROR] {:?}", msg),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => eprintln!("[VK WARN]  {:?}", msg),
        _ => {}
    }
    vk::FALSE
}

/// Pick surface instance extensions for the current platform at compile time.
fn platform_surface_extensions() -> Vec<*const i8> {
    let mut exts = vec![ash::khr::surface::NAME.as_ptr()];

    #[cfg(target_os = "linux")]
    {
        // Enable both X11 and Wayland so the example works everywhere.
        exts.push(ash::khr::xlib_surface::NAME.as_ptr());
        exts.push(ash::khr::wayland_surface::NAME.as_ptr());
        exts.push(ash::khr::xcb_surface::NAME.as_ptr());
    }
    #[cfg(target_os = "windows")]
    {
        exts.push(ash::khr::win32_surface::NAME.as_ptr());
    }
    #[cfg(target_os = "macos")]
    {
        exts.push(ash::ext::metal_surface::NAME.as_ptr());
    }

    exts
}

/// Resource holder for Vulkan objects owned by main. Destroyed after `run()`.
struct VulkanResources {
    instance: ash::Instance,
    device: ash::Device,
    debug_utils_loader: ash::ext::debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for VulkanResources {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);
            if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn create_vulkan_context() -> (VulkanContext, VulkanResources) {
    let entry = ash::Entry::linked();

    // ── Instance ──
    let app_name = CString::new("v2_simple").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_3);

    let mut instance_extensions = platform_surface_extensions();
    if ENABLE_VALIDATION {
        instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
    let layer_names: Vec<*const i8> = if ENABLE_VALIDATION {
        vec![validation_layer.as_ptr()]
    } else {
        vec![]
    };

    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(debug_callback));

    let mut instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&layer_names);
    if ENABLE_VALIDATION {
        instance_info = instance_info.push_next(&mut debug_create_info);
    }

    let instance = unsafe {
        entry
            .create_instance(&instance_info, None)
            .expect("failed to create Vulkan instance")
    };

    let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
    let debug_messenger = if ENABLE_VALIDATION {
        unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_create_info, None)
                .expect("failed to create debug messenger")
        }
    } else {
        vk::DebugUtilsMessengerEXT::null()
    };

    // ── Physical device + queue families ──
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
    };

    let mut chosen: Option<(vk::PhysicalDevice, u32, u32, u32)> = None; // (pd, family, host_idx, engine_idx)

    for &pd in &physical_devices {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(pd) };

        // Strategy 1: Find a graphics family with 2+ queues
        for (i, qf) in queue_families.iter().enumerate() {
            if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) && qf.queue_count >= 2 {
                chosen = Some((pd, i as u32, 0, 1));
                break;
            }
        }
        if chosen.is_some() {
            break;
        }

        // Fallback: Single queue (same queue for host and engine)
        for (i, qf) in queue_families.iter().enumerate() {
            if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                chosen = Some((pd, i as u32, 0, 0));
                break;
            }
        }
        if chosen.is_some() {
            break;
        }
    }

    let (physical_device, queue_family, host_queue_idx, engine_queue_idx) =
        chosen.expect("no suitable physical device found");

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
    eprintln!(
        "Selected GPU: {:?} (family={}, host_q={}, engine_q={})",
        device_name, queue_family, host_queue_idx, engine_queue_idx
    );

    // ── Device ──
    let queue_count = if host_queue_idx == engine_queue_idx {
        1
    } else {
        2
    };
    let queue_priorities: Vec<f32> = vec![1.0; queue_count as usize];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_priorities);
    let queue_create_infos = [queue_create_info];

    let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

    // Enable Vulkan 1.2 features (timeline semaphores) + 1.3 features (synchronization2)
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true);
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(true);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .push_next(&mut vulkan_12_features)
        .push_next(&mut vulkan_13_features);

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("failed to create device")
    };

    let host_queue = unsafe { device.get_device_queue(queue_family, host_queue_idx) };
    let engine_queue = unsafe { device.get_device_queue(queue_family, engine_queue_idx) };

    let vulkan_context = VulkanContext {
        entry,
        instance: instance.clone(),
        physical_device,
        device: device.clone(),
        host_queue,
        host_queue_family_index: queue_family,
        engine_queue,
        engine_queue_family_index: queue_family,
    };

    let resources = VulkanResources {
        instance,
        device,
        debug_utils_loader,
        debug_messenger,
    };

    (vulkan_context, resources)
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let (vulkan, _resources) = create_vulkan_context();

    let engine = ColorEngine::new();

    let options = RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash v2 simple")
                .with_inner_size(egui::vec2(900.0, 600.0)),
        ),
        follow_system_theme: false,
        default_theme: egui_ash::winit::window::Theme::Dark,
        ..Default::default()
    };

    let mut clear_color = [0.2_f32, 0.3, 0.8];

    egui_ash::run(
        "v2_simple",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &EngineStatus,
              ui_state: &mut UiState,
              engine_state: &EngineState,
              handle: &EngineHandle<ColorEngine>| {
            egui::SidePanel::left("controls")
                .default_width(250.0)
                .show(ctx, |ui| {
                    ui.heading("Color Engine");
                    ui.separator();

                    ui.label("Clear Color");
                    ui.color_edit_button_rgb(&mut clear_color);
                    ui.add(
                        egui::Slider::new(&mut clear_color[0], 0.0..=1.0)
                            .text("R"),
                    );
                    ui.add(
                        egui::Slider::new(&mut clear_color[1], 0.0..=1.0)
                            .text("G"),
                    );
                    ui.add(
                        egui::Slider::new(&mut clear_color[2], 0.0..=1.0)
                            .text("B"),
                    );

                    ui.separator();
                    ui.heading("Engine Status");

                    let health_text = match &status.health {
                        egui_ash::EngineHealth::Starting => "Starting...".to_string(),
                        egui_ash::EngineHealth::Running => "Running".to_string(),
                        egui_ash::EngineHealth::Stopped => "Stopped".to_string(),
                        egui_ash::EngineHealth::Crashed { message } => {
                            format!("CRASHED: {}", message)
                        }
                    };
                    ui.label(format!("Health: {}", health_text));
                    ui.label(format!("Frames delivered: {}", status.frames_delivered));

                    if let Some(ft) = status.last_frame_time {
                        ui.label(format!("Last frame: {:.2}ms", ft.as_secs_f64() * 1000.0));
                    }

                    ui.label(format!(
                        "Engine frame count: {}",
                        engine_state.frame_count
                    ));

                    ui.separator();

                    // Crash test button
                    if status.health.is_alive()
                        && ui.button("Crash Engine").clicked()
                    {
                        ui_state.should_crash = true;
                    }

                    // Restart button (only if crashed/stopped)
                    let can_restart = status.health.is_crashed()
                        || matches!(status.health, egui_ash::EngineHealth::Stopped);
                    if can_restart && ui.button("Restart Engine").clicked() {
                        let _ = handle.restart(ColorEngine::new());
                    }
                });

            egui::CentralPanel::default().show(ctx, |ui| {
                let available = ui.available_size();
                ui.image(egui::load::SizedTexture::new(
                    status.viewport_texture_id,
                    available,
                ));
            });

            // Publish current color to engine
            ui_state.clear_color = clear_color;
        },
    )
}
