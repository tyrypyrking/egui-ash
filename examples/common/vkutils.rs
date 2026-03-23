//! Shared Vulkan setup helpers for v2 egui-ash examples.
//!
//! Creates a `VulkanContext` with:
//! - Vulkan 1.3 API version
//! - Timeline semaphores (1.2 feature)
//! - Synchronization2 (1.3 feature)
//! - VK_KHR_swapchain device extension
//! - A graphics queue family with 2+ queues (separate host/engine)
//! - Optional validation layers (debug builds only)

use ash::vk;
use egui_ash::VulkanContext;
use std::ffi::{CStr, CString};

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

/// RAII holder for Vulkan objects owned by the example's main function.
/// Destroyed after `run()` returns.
pub struct VulkanResources {
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

/// Create a `VulkanContext` and its backing resources for an example.
///
/// `app_name` is used in the Vulkan application info.
pub fn create_vulkan_context(app_name: &str) -> (VulkanContext, VulkanResources) {
    let entry = ash::Entry::linked();

    // -- Instance --
    let app_name_c = CString::new(app_name).unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name_c)
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

    // -- Physical device + queue families --
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
    };

    let mut chosen: Option<(vk::PhysicalDevice, u32, u32, u32)> = None;

    for &pd in &physical_devices {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(pd) };

        // Find a graphics family with 2+ queues
        for (i, qf) in queue_families.iter().enumerate() {
            if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) && qf.queue_count >= 2 {
                chosen = Some((pd, i as u32, 0, 1));
                break;
            }
        }
        if chosen.is_some() {
            break;
        }
    }

    let (physical_device, queue_family, host_queue_idx, engine_queue_idx) = chosen
        .expect("no GPU found with a graphics queue family supporting 2+ queues");

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
    eprintln!(
        "Selected GPU: {:?} (family={}, host_q={}, engine_q={})",
        device_name, queue_family, host_queue_idx, engine_queue_idx
    );

    // -- Device --
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
    let mut vulkan_12_features =
        vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(true)
        .dynamic_rendering(true);

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
