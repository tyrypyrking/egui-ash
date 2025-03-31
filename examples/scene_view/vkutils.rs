use ash::{ext::debug_utils, vk, Device, Entry, Instance};
use egui_ash::{
    raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _},
    winit,
};
use std::{collections::HashSet, ffi::CString};

const ENABLE_VALIDATION_LAYERS: bool = true;
const VALIDATION: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[VERBOSE]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[WARNING]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[ERROR]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[INFO]",
        _ => panic!("[UNKNOWN]"),
    };
    let types = match message_types {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
        _ => panic!("[UNKNOWN]"),
    };
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    println!("[DEBUG]{}{}{:?}", severity, types, message);

    vk::FALSE
}

pub fn create_entry() -> Entry {
    ash::Entry::linked()
}

pub fn create_instance(
    required_instance_extensions: &[CString],
    entry: &Entry,
) -> (Instance, debug_utils::Instance, vk::DebugUtilsMessengerEXT) {
    let mut debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let app_name = std::ffi::CString::new("egui-ash example simple").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(1, 0, 0, 0))
        .api_version(vk::API_VERSION_1_0);
    let mut extension_names = vec![debug_utils::NAME.as_ptr()];
    for ext in required_instance_extensions {
        let name = ext.as_ptr();
        extension_names.push(name);
    }
    let raw_layer_names = VALIDATION
        .iter()
        .map(|l| std::ffi::CString::new(*l).unwrap())
        .collect::<Vec<_>>();
    let layer_names = raw_layer_names
        .iter()
        .map(|l| l.as_ptr())
        .collect::<Vec<*const i8>>();
    let instance_create_info = vk::InstanceCreateInfo::default()
        .push_next(&mut debug_utils_messenger_create_info)
        .application_info(&app_info)
        .enabled_extension_names(&extension_names);
    let instance_create_info = if ENABLE_VALIDATION_LAYERS {
        instance_create_info.enabled_layer_names(&layer_names)
    } else {
        instance_create_info
    };
    let instance = unsafe {
        entry
            .create_instance(&instance_create_info, None)
            .expect("Failed to create instance")
    };

    // setup debug utils
    let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
    let debug_messenger = if ENABLE_VALIDATION_LAYERS {
        unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
                .expect("Failed to create debug utils messenger")
        }
    } else {
        vk::DebugUtilsMessengerEXT::null()
    };

    (instance, debug_utils_loader, debug_messenger)
}

pub fn create_surface_loader(entry: &Entry, instance: &Instance) -> ash::khr::surface::Instance {
    ash::khr::surface::Instance::new(&entry, &instance)
}

pub fn create_swapchain_loader(
    instance: &Instance,
    device: &Device,
) -> ash::khr::swapchain::Device {
    ash::khr::swapchain::Device::new(&instance, &device)
}

pub fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &winit::window::Window,
) -> vk::SurfaceKHR {
    unsafe {
        ash_window::create_surface(
            entry,
            instance,
            window
                .display_handle()
                .expect("Unable to get a display handle")
                .as_raw(),
            window
                .window_handle()
                .expect("Unable to get a window handle")
                .as_raw(),
            None,
        )
        .expect("Failed to create surface")
    }
}

pub fn create_physical_device(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    required_device_extensions: &[CString],
) -> (vk::PhysicalDevice, vk::PhysicalDeviceMemoryProperties, u32) {
    let mut queue_family_index: Option<usize> = None;

    let (physical_device, physical_device_memory_properties) = {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        let physical_device = physical_devices.into_iter().find(|physical_device| {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
            for (i, queue_family) in queue_families.iter().enumerate() {
                let mut graphics_queue = false;
                let mut present_queue = false;
                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    graphics_queue = true;
                }
                let present_support = unsafe {
                    surface_loader
                        .get_physical_device_surface_support(*physical_device, i as u32, surface)
                        .unwrap()
                };
                if present_support {
                    present_queue = true;
                }
                if graphics_queue && present_queue {
                    queue_family_index = Some(i);
                    break;
                }
            }
            let is_queue_family_supported = queue_family_index.is_some();

            // check device extensions
            let device_extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(*physical_device)
                    .unwrap()
            };
            let mut device_extensions_name = vec![];
            for device_extension in device_extensions {
                let name = unsafe {
                    std::ffi::CStr::from_ptr(device_extension.extension_name.as_ptr()).to_owned()
                };
                device_extensions_name.push(name);
            }
            let mut required_extensions = HashSet::new();
            for extension in required_device_extensions.iter() {
                required_extensions.insert(extension.to_owned());
            }
            for extension_name in device_extensions_name {
                required_extensions.remove(&extension_name);
            }
            let is_device_extension_supported = required_extensions.is_empty();

            // check swapchain support
            let surface_formats = unsafe {
                surface_loader
                    .get_physical_device_surface_formats(*physical_device, surface)
                    .unwrap()
            };
            let surface_present_modes = unsafe {
                surface_loader
                    .get_physical_device_surface_present_modes(*physical_device, surface)
                    .unwrap()
            };
            let is_swapchain_supported =
                !surface_formats.is_empty() && !surface_present_modes.is_empty();

            is_queue_family_supported && is_swapchain_supported && is_device_extension_supported
        });
        let physical_device = physical_device.expect("Failed to get physical device");
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        (physical_device, physical_device_memory_properties)
    };

    (
        physical_device,
        physical_device_memory_properties,
        queue_family_index.unwrap() as u32,
    )
}

pub fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    required_device_extensions: &[CString],
) -> (Device, vk::Queue) {
    let queue_priorities = [1.0_f32];
    let mut queue_create_infos = vec![];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);
    queue_create_infos.push(queue_create_info);

    let physical_device_features = vk::PhysicalDeviceFeatures::default();

    let enable_extension_names = required_device_extensions
        .iter()
        .map(|s| s.as_ptr())
        .collect::<Vec<_>>();

    // device create info
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_features(&physical_device_features)
        .enabled_extension_names(&enable_extension_names);

    // create device
    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create device")
    };

    // get device queue
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    (device, queue)
}

pub fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
    let command_pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);
    unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .expect("Failed to create command pool")
    }
}

pub fn insert_image_memory_barrier(
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
    let image_memory_barrier = ash::vk::ImageMemoryBarrier::default()
        .src_queue_family_index(src_q_family_index)
        .dst_queue_family_index(dst_q_family_index)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_image_layout)
        .new_layout(new_image_layout)
        .image(*image)
        .subresource_range(subresource_range);
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
