use ash::vk;
use std::ffi::CStr;
use std::sync::Mutex;

#[cfg(feature = "vulkan")]
const SHADER: &[u8] = include_bytes!("../shaders/bin/q8_matmul.spv");

pub struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    _physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    mutex: Mutex<()>,
}

struct BufferInfo {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
}

impl VulkanContext {
    pub fn new() -> Result<Self, VulkanError> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| VulkanError::InitFailed(e.to_string()))?;
            let instance = Self::create_instance(&entry)?;
            let (physical_device, queue_family) = Self::find_physical_device(&instance)?;
            let device = Self::create_device(&instance, physical_device, queue_family)?;
            let queue = device.get_device_queue(queue_family, 0);
            let (pipeline_layout, descriptor_set_layout, descriptor_pool, descriptor_set) =
                Self::create_compute_pipeline(&device)?;
            let pipeline = Self::create_pipeline(&device, pipeline_layout)?;
            let command_pool = Self::create_command_pool(&device, queue_family)?;

            Ok(Self {
                entry,
                instance,
                device,
                _physical_device: physical_device,
                queue,
                queue_family,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set,
                command_pool,
                mutex: Mutex::new(()),
            })
        }
    }

    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, VulkanError> {
        let engine_name = CStr::from_bytes_with_nul(b"rust-model-inference\0").unwrap();
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: std::ptr::null(),
            application_version: 0,
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_3,
        };

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            p_application_info: &app_info,
            flags: Default::default(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };

        unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| {
                    eprintln!("[GPU] vkCreateInstance failed: {}", e);
                    VulkanError::InitFailed(e.to_string())
                })
        }
    }

    fn find_physical_device(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, u32), VulkanError> {
        unsafe {
            let devices = match instance.enumerate_physical_devices() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[GPU] enumerate_physical_devices failed: {}", e);
                    return Err(VulkanError::InitFailed(e.to_string()));
                }
            };

            for device in devices {
                let props = instance.get_physical_device_properties(device);
                let name = CStr::from_ptr(props.device_name.as_ptr());
                eprintln!("Found device: {:?}", name);

                let queue_families = instance.get_physical_device_queue_family_properties(device);

                for (i, family) in queue_families.iter().enumerate() {
                    if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                        eprintln!("  Using queue family {} for compute", i);
                        return Ok((device, i as u32));
                    }
                }
            }

            Err(VulkanError::NoComputeDevice)
        }
    }

    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> Result<ash::Device, VulkanError> {
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
        };

        let features = vk::PhysicalDeviceFeatures {
            shader_int64: 1,
            ..Default::default()
        };

        let device_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            p_enabled_features: &features,
            ..Default::default()
        };

        unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))
        }
    }

    fn create_compute_pipeline(
        device: &ash::Device,
    ) -> Result<
        (
            vk::PipelineLayout,
            vk::DescriptorSetLayout,
            vk::DescriptorPool,
            vk::DescriptorSet,
        ),
        VulkanError,
    > {
        let binding_descs = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            binding_count: 4,
            p_bindings: binding_descs.as_ptr(),
        };

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?
        };

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 16, // 4 * u32 = 16 bytes
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        set_layout_count: 1,
                        p_set_layouts: &descriptor_set_layout,
                        push_constant_range_count: 1,
                        p_push_constant_ranges: &push_constant_range,
                    },
                    None,
                )
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?
        };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4,
        }];

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        max_sets: 1,
                        pool_size_count: 1,
                        p_pool_sizes: pool_sizes.as_ptr(),
                    },
                    None,
                )
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?
        };

        let descriptor_set = unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &descriptor_set_layout,
            };
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?[0]
        };

        Ok((
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
        ))
    }

    fn create_pipeline(
        device: &ash::Device,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, VulkanError> {
        eprintln!("[GPU] Creating shader module, size={}", SHADER.len());
        let shader_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        code_size: SHADER.len(),
                        p_code: SHADER.as_ptr() as *const u32,
                    },
                    None,
                )
                .map_err(|e| VulkanError::ShaderCompileFailed(e.to_string()))?
        };
        eprintln!("[GPU] Shader module created successfully");

        let stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: CStr::from_bytes_with_nul(b"main\0").unwrap().as_ptr(),
            p_specialization_info: std::ptr::null(),
        };

        let pipeline = unsafe {
            let create_info = vk::ComputePipelineCreateInfo {
                s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: Default::default(),
                stage,
                layout: pipeline_layout,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: 0,
            };
            match device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None) {
                Ok(pipelines) => pipelines[0],
                Err((_, _)) => return Err(VulkanError::ShaderCompileFailed("Pipeline creation failed".to_string())),
            }
        };

        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(pipeline)
    }

    fn create_command_pool(
        device: &ash::Device,
        queue_family: u32,
    ) -> Result<vk::CommandPool, VulkanError> {
        unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: vk::CommandPoolCreateFlags::TRANSIENT,
                        queue_family_index: queue_family,
                    },
                    None,
                )
                .map_err(|e| VulkanError::InitFailed(e.to_string()))
        }
    }

    pub unsafe fn matmul_q8_0(
        &self,
        weight: &[u8],
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<(), VulkanError> {
        let _lock = self.mutex.lock().unwrap();
        let blocks_per_row = n_in / 32;
        let weight_row_stride = blocks_per_row * 34;
        eprintln!("[GPU] matmul: n_in={}, n_out={}, blocks_per_row={}, weight_row_stride={}, weight.len={}, input_q8.len={}, input_scales.len={}, output.len={}",
                  n_in, n_out, blocks_per_row, weight_row_stride, weight.len(), input_q8.len(), input_scales.len(), output.len());

        let weight_buffer = self.create_buffer(weight, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        eprintln!("[GPU] weight buffer created: size={}", weight_buffer.size);
        let input_buffer = self.create_buffer(input_q8, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let scale_buffer = self.create_buffer_f32(input_scales, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let output_buffer = self.create_buffer_f32_mut(output, vk::BufferUsageFlags::STORAGE_BUFFER)?;

        self.update_descriptor_sets(
            weight_buffer.buffer,
            input_buffer.buffer,
            scale_buffer.buffer,
            output_buffer.buffer,
        )?;

        let command_buffer = self.begin_command_buffer()?;

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            let push_constants: [u32; 4] = [
                n_in as u32,
                n_out as u32,
                blocks_per_row as u32,
                weight_row_stride as u32,
            ];

            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&push_constants),
            );

            self.device.cmd_dispatch(command_buffer, n_out as u32, 1, 1);
        }

        self.end_and_submit_command_buffer(command_buffer)?;

        eprintln!("[GPU] Command completed, reading output...");
        self.read_buffer_f32(output_buffer, output)?;
        eprintln!("[GPU] Output read complete");

        unsafe {
            self.destroy_buffer(weight_buffer);
            self.destroy_buffer(input_buffer);
            self.destroy_buffer(scale_buffer);
        }
        eprintln!("[GPU] Buffers destroyed");

        Ok(())
    }

    unsafe fn create_buffer(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferInfo, VulkanError> {
        let buffer_size = data.len() as u64;

        let buffer = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo {
                    s_type: vk::StructureType::BUFFER_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    size: buffer_size,
                    usage,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    queue_family_index_count: 0,
                    p_queue_family_indices: std::ptr::null(),
                },
                None,
            )
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        let mem_reqs = self.device.get_buffer_memory_requirements(buffer);
        let mem_type = self
            .find_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or(VulkanError::OutOfMemory)?;

        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                    p_next: std::ptr::null(),
                    allocation_size: mem_reqs.size,
                    memory_type_index: mem_type,
                },
                None,
            )
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        self.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        let mapped = self
            .device
            .map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
        self.device.unmap_memory(memory);

        Ok(BufferInfo {
            buffer,
            memory,
            size: data.len(),
        })
    }

    unsafe fn create_buffer_f32(
        &self,
        data: &[f32],
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferInfo, VulkanError> {
        let bytes = bytemuck::cast_slice(data);
        let mut info = self.create_buffer(bytes, usage)?;
        info.size = data.len() * 4;
        Ok(info)
    }

    unsafe fn create_buffer_f32_mut(
        &self,
        data: &mut [f32],
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferInfo, VulkanError> {
        let bytes = bytemuck::cast_slice_mut(data);
        let mut info = self.create_buffer(bytes, usage)?;
        info.size = data.len() * 4;
        Ok(info)
    }

    unsafe fn read_buffer_f32(
        &self,
        buffer_info: BufferInfo,
        output: &mut [f32],
    ) -> Result<(), VulkanError> {
        let buffer_size = buffer_info.size as u64;

        let mapped = self
            .device
            .map_memory(
                buffer_info.memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        std::ptr::copy_nonoverlapping(
            mapped as *const u8,
            output.as_mut_ptr() as *mut u8,
            buffer_size as usize,
        );

        self.device.unmap_memory(buffer_info.memory);
        self.device.destroy_buffer(buffer_info.buffer, None);
        self.device.free_memory(buffer_info.memory, None);

        Ok(())
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        unsafe {
            let mem_props = self
                .instance
                .get_physical_device_memory_properties(self._physical_device);
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
                {
                    return Some(i);
                }
            }
            None
        }
    }

    unsafe fn destroy_buffer(&self, buffer_info: BufferInfo) {
        self.device.destroy_buffer(buffer_info.buffer, None);
        self.device.free_memory(buffer_info.memory, None);
    }

    unsafe fn update_descriptor_sets(
        &self,
        weight_buffer: vk::Buffer,
        input_buffer: vk::Buffer,
        scale_buffer: vk::Buffer,
        output_buffer: vk::Buffer,
    ) -> Result<(), VulkanError> {
        let weight_info = vk::DescriptorBufferInfo {
            buffer: weight_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let input_info = vk::DescriptorBufferInfo {
            buffer: input_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let scale_info = vk::DescriptorBufferInfo {
            buffer: scale_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let output_info = vk::DescriptorBufferInfo {
            buffer: output_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };

        let desc_writes = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: self.descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &weight_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: self.descriptor_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &input_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: self.descriptor_set,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &scale_info,
                p_texel_buffer_view: std::ptr::null(),
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: self.descriptor_set,
                dst_binding: 3,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &output_info,
                p_texel_buffer_view: std::ptr::null(),
            },
        ];

        self.device.update_descriptor_sets(&desc_writes, &[]);

        Ok(())
    }

    unsafe fn begin_command_buffer(&self) -> Result<vk::CommandBuffer, VulkanError> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        };

        let command_buffer = self
            .device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?[0];

        self.device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        Ok(command_buffer)
    }

    unsafe fn end_and_submit_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), VulkanError> {
        self.device
            .end_command_buffer(command_buffer)
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };

        self.device
            .queue_submit(self.queue, &[submit_info], vk::Fence::null())
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        eprintln!("[GPU] Submitted command buffer, waiting for queue idle...");

        self.device
            .queue_wait_idle(self.queue)
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        eprintln!("[GPU] Queue idle complete");

        self.device
            .free_command_buffers(self.command_pool, std::slice::from_ref(&command_buffer));

        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Default for VulkanContext {
    fn default() -> Self {
        Self::new().expect("failed to create Vulkan context")
    }
}

#[derive(Debug)]
pub enum VulkanError {
    InitFailed(String),
    NoComputeDevice,
    ShaderCompileFailed(String),
    OutOfMemory,
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::InitFailed(s) => write!(f, "Vulkan init failed: {}", s),
            VulkanError::NoComputeDevice => write!(f, "No compute device found"),
            VulkanError::ShaderCompileFailed(s) => write!(f, "Shader compile failed: {}", s),
            VulkanError::OutOfMemory => write!(f, "Out of GPU memory"),
        }
    }
}

impl std::error::Error for VulkanError {}