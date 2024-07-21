use ash::{ext::debug_utils, vk, Device, Entry, Instance};
use egui_ash::{
    raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _},
    winit, App, AppCreator, AshRenderState, CreationContext, RunOption, Theme,
};
use gpu_allocator::vulkan::*;
use std::{
    collections::HashSet,
    ffi::CString,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

struct MyApp {
    entry: Entry,
    instance: Instance,
    device: Device,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,

    // native ash user texture image
    image_registry: egui_ash::ImageRegistry,
    texture_id: egui::TextureId,
    image: vk::Image,
    image_view: vk::ImageView,
    image_allocation: Option<Allocation>,
    sampler: vk::Sampler,
}
impl App for MyApp {
    fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(&ctx, |ui| {
            ui.heading("Hello");
            ui.label("Hello native ash image texture!");
            ui.separator();
            ui.hyperlink("https://github.com/emilk/egui");

            egui::Window::new("My Window")
                .id(egui::Id::new("my_window"))
                .resizable(true)
                .scroll([true, true])
                .show(&ctx, |ui| {
                    ui.label("user texture");
                    ui.image(egui::load::SizedTexture::new(
                        self.texture_id,
                        egui::Vec2::new(256.0, 256.0),
                    ));
                });
        });
    }
}
impl Drop for MyApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.image_registry.unregister_user_texture(self.texture_id);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            {
                let mut allocator = self.allocator.lock().unwrap();
                allocator
                    .free(self.image_allocation.take().unwrap())
                    .unwrap();
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.surface_loader.destroy_surface(self.surface, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct MyAppCreator;
impl MyAppCreator {
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

    fn create_entry() -> Entry {
        ash::Entry::linked()
    }

    fn create_instance(
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
            .pfn_user_callback(Some(Self::vulkan_debug_utils_callback));

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
        let raw_layer_names = Self::VALIDATION
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
        let instance_create_info = if Self::ENABLE_VALIDATION_LAYERS {
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
        let debug_messenger = if Self::ENABLE_VALIDATION_LAYERS {
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

    fn create_surface_loader(entry: &Entry, instance: &Instance) -> ash::khr::surface::Instance {
        ash::khr::surface::Instance::new(&entry, &instance)
    }

    fn create_swapchain_loader(
        instance: &Instance,
        device: &Device,
    ) -> ash::khr::swapchain::Device {
        ash::khr::swapchain::Device::new(&instance, &device)
    }

    fn create_surface(
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
                    .expect("Failed to get a display handle")
                    .as_raw(),
                window
                    .window_handle()
                    .expect("Failed to get a window handle")
                    .as_raw(),
                None,
            )
            .expect("Failed to create surface")
        }
    }

    fn create_physical_device(
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
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(*physical_device)
                };
                for (i, queue_family) in queue_families.iter().enumerate() {
                    let mut graphics_queue = false;
                    let mut present_queue = false;
                    if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_queue = true;
                    }
                    let present_support = unsafe {
                        surface_loader
                            .get_physical_device_surface_support(
                                *physical_device,
                                i as u32,
                                surface,
                            )
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
                        std::ffi::CStr::from_ptr(device_extension.extension_name.as_ptr())
                            .to_owned()
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

    fn create_device(
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

    fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create command pool")
        }
    }

    fn load_and_create_image(
        device: &Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        allocator: &Arc<Mutex<Allocator>>,
    ) -> (vk::Image, vk::ImageView, Allocation, vk::Sampler) {
        let mut allocator = allocator.lock().unwrap();

        let image =
            image::open("./examples/native_image/Mandrill.bmp").expect("Failed to open image file");
        let (image_width, image_height) = (image.width(), image.height());
        let image_data = image.to_rgba8().into_raw();

        let image_size = (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as u64;

        // Create Staging buffer
        let staging_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(image_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .expect("Failed to create staging buffer")
        };
        let staging_buffer_requirements =
            unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let staging_buffer_allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "User Texture Staging Buffer",
                requirements: staging_buffer_requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate staging buffer memory");
        unsafe {
            device
                .bind_buffer_memory(
                    staging_buffer,
                    staging_buffer_allocation.memory(),
                    staging_buffer_allocation.offset(),
                )
                .expect("Failed to bind staging buffer memory")
        }
        // Map staging buffer
        unsafe {
            let mapped_memory = staging_buffer_allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            mapped_memory.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());
        }

        let format = vk::Format::R8G8B8A8_UNORM;

        // Create Image
        let image = unsafe {
            device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width: image_width,
                            height: image_height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
                    None,
                )
                .expect("Failed to create image")
        };
        let image_requirements = unsafe { device.get_image_memory_requirements(image) };
        let image_allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "User Texture Image",
                requirements: image_requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate image memory");
        unsafe {
            device
                .bind_image_memory(image, image_allocation.memory(), image_allocation.offset())
                .expect("Failed to bind image memory")
        };

        // Copy data from buffer to image
        unsafe {
            let command = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("Failed to allocate command buffer")[0];

            // Begin command
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin command buffer");

            // Change image layout to transfer dst optimal
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )],
            );

            // Copy data from buffer to image
            device.cmd_copy_buffer_to_image(
                command,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: image_width,
                        height: image_height,
                        depth: 1,
                    })
                    .buffer_offset(0)
                    .buffer_image_height(0)
                    .buffer_row_length(0)
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })],
            );

            // Change image layout to shader read only optimal
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )],
            );

            // End command
            device
                .end_command_buffer(command)
                .expect("Failed to end command buffer");

            // Submit command
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command])],
                    vk::Fence::null(),
                )
                .expect("Failed to submit command buffer");
            device
                .queue_wait_idle(queue)
                .expect("Failed to wait queue idle");
        }

        // Delete staging buffer
        allocator
            .free(staging_buffer_allocation)
            .expect("Failed to free staging buffer");
        unsafe {
            device.destroy_buffer(staging_buffer, None);
        }

        // Create Image View
        let image_view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .image(image)
                        .format(format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .expect("Failed to create image view")
        };

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }
        .expect("Failed to create sampler");

        (image, image_view, image_allocation, sampler)
    }
}
impl AppCreator<Arc<Mutex<Allocator>>> for MyAppCreator {
    type App = MyApp;

    fn create(&self, cc: CreationContext) -> (Self::App, AshRenderState<Arc<Mutex<Allocator>>>) {
        // create vk objects
        let entry = Self::create_entry();
        let (instance, debug_utils_loader, debug_messenger) =
            Self::create_instance(&cc.required_instance_extensions, &entry);
        let surface_loader = Self::create_surface_loader(&entry, &instance);
        let surface = Self::create_surface(&entry, &instance, cc.main_window);
        let (physical_device, _physical_device_memory_properties, queue_family_index) =
            Self::create_physical_device(
                &instance,
                &surface_loader,
                surface,
                &cc.required_device_extensions,
            );
        let (device, queue) = Self::create_device(
            &instance,
            physical_device,
            queue_family_index,
            &cc.required_device_extensions,
        );
        let swapchain_loader = Self::create_swapchain_loader(&instance, &device);
        let command_pool = Self::create_command_pool(&device, queue_family_index);

        // create allocator
        let allocator = {
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .expect("Failed to create allocator")
        };
        let allocator = Arc::new(Mutex::new(allocator));

        // setup context
        cc.context.set_visuals(egui::style::Visuals::dark());

        let (image, image_view, image_allocation, sampler) =
            Self::load_and_create_image(&device, queue, command_pool, &allocator);
        let texture_id = cc.image_registry.register_user_texture(image_view, sampler);

        let app = MyApp {
            entry,
            instance,
            device,
            debug_utils_loader: debug_utils_loader,
            debug_messenger,
            physical_device,
            surface_loader,
            swapchain_loader,
            surface,
            queue,
            command_pool,
            allocator: ManuallyDrop::new(allocator.clone()),

            image_registry: cc.image_registry,
            texture_id,
            image,
            image_view,
            image_allocation: Some(image_allocation),
            sampler,
        };
        let ash_render_state = AshRenderState {
            entry: app.entry.clone(),
            instance: app.instance.clone(),
            physical_device: app.physical_device,
            device: app.device.clone(),
            surface_loader: app.surface_loader.clone(),
            swapchain_loader: app.swapchain_loader.clone(),
            queue: app.queue,
            queue_family_index,
            command_pool: app.command_pool,
            allocator: allocator.clone(),
        };

        (app, ash_render_state)
    }
}

fn main() -> std::process::ExitCode {
    egui_ash::run(
        "egui-ash-native-image",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(egui::ViewportBuilder::default().with_title("egui-ash")),
            follow_system_theme: false,
            default_theme: Theme::Dark,
            ..Default::default()
        },
    )
}
