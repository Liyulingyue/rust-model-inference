#![cfg(feature = "vulkan")]
//! Compile one SPIR-V file into a compute pipeline and report the result.
//!
//! The GB10 driver (580.173.02) traps inside `libnvidia-gpucomp.so` while
//! compiling `q8_matmul_grouped.spv`, which takes the whole process down. A
//! probe that only touches `vkCreateComputePipelines` lets a variant be tested
//! in a second instead of rebuilding the crate for every shader edit.
//!
//! Usage: `cargo run --profile release-fast --features vulkan --example
//! spv_compile_probe -- <path.spv> [<path.spv> ...]`
use ash::vk;
use std::ffi::CStr;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: spv_compile_probe <file.spv> [...]");
        std::process::exit(2);
    }

    unsafe {
        let entry = ash::Entry::load().expect("vulkan loader");
        let app = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: std::ptr::null(),
            application_version: 0,
            p_engine_name: CStr::from_bytes_with_nul(b"spv-probe\0").unwrap().as_ptr(),
            engine_version: 0,
            api_version: vk::API_VERSION_1_3,
        };
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo {
                    s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::InstanceCreateFlags::empty(),
                    p_application_info: &app,
                    enabled_layer_count: 0,
                    pp_enabled_layer_names: std::ptr::null(),
                    enabled_extension_count: 0,
                    pp_enabled_extension_names: std::ptr::null(),
                },
                None,
            )
            .expect("instance");

        let devices = instance.enumerate_physical_devices().expect("devices");
        let (physical, queue_family) = devices
            .into_iter()
            .find_map(|device| {
                let props = instance.get_physical_device_properties(device);
                if props.device_type == vk::PhysicalDeviceType::CPU {
                    return None;
                }
                let families = instance.get_physical_device_queue_family_properties(device);
                families
                    .iter()
                    .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|index| (device, index as u32))
            })
            .expect("a non-CPU compute device");
        let props = instance.get_physical_device_properties(physical);
        let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
        eprintln!("device: {name}");

        let priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: priority.as_ptr(),
        }];
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo {
                    s_type: vk::StructureType::DEVICE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::DeviceCreateFlags::empty(),
                    queue_create_info_count: queue_info.len() as u32,
                    p_queue_create_infos: queue_info.as_ptr(),
                    enabled_layer_count: 0,
                    pp_enabled_layer_names: std::ptr::null(),
                    enabled_extension_count: 0,
                    pp_enabled_extension_names: std::ptr::null(),
                    p_enabled_features: std::ptr::null(),
                },
                None,
            )
            .expect("device");

        // q8_matmul_grouped binds three read-only storage buffers plus the
        // arena, and passes a 108-byte push constant block.
        let bindings: Vec<_> = (0..4u32)
            .map(|binding| vk::DescriptorSetLayoutBinding {
                binding,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
            })
            .collect();
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                    binding_count: bindings.len() as u32,
                    p_bindings: bindings.as_ptr(),
                },
                None,
            )
            .expect("descriptor set layout");
        let layouts = [set_layout];
        let push_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 128,
        }];
        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::PipelineLayoutCreateFlags::empty(),
                    set_layout_count: layouts.len() as u32,
                    p_set_layouts: layouts.as_ptr(),
                    push_constant_range_count: push_range.len() as u32,
                    p_push_constant_ranges: push_range.as_ptr(),
                },
                None,
            )
            .expect("pipeline layout");

        for path in &paths {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let mut cursor = std::io::Cursor::new(bytes.as_slice());
            let words = ash::util::read_spv(&mut cursor).expect("spirv words");
            let module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: vk::ShaderModuleCreateFlags::empty(),
                        code_size: bytes.len(),
                        p_code: words.as_ptr(),
                    },
                    None,
                )
                .expect("shader module");
            eprintln!("compiling {path}");
            let stage = vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                stage: vk::ShaderStageFlags::COMPUTE,
                module,
                p_name: CStr::from_bytes_with_nul(b"main\0").unwrap().as_ptr(),
                p_specialization_info: std::ptr::null(),
            };
            let info = vk::ComputePipelineCreateInfo {
                s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::PipelineCreateFlags::empty(),
                stage,
                layout: pipeline_layout,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: 0,
            };
            match device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) {
                Ok(pipelines) => {
                    println!("OK   {path}");
                    device.destroy_pipeline(pipelines[0], None);
                }
                Err((_, error)) => println!("FAIL {path}: {error:?}"),
            }
            device.destroy_shader_module(module, None);
        }
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
