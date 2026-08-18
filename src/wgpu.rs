#[cfg(feature = "wgpu")]
use wgpu::{Device, Queue, ComputePipeline, Buffer};

#[cfg(feature = "wgpu")]
pub struct WgpuContext {
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "wgpu")]
impl WgpuContext {
    pub async fn new() -> Result<Self, WgpuError> {
        Self::new_impl().await
    }
    
    pub fn new_blocking() -> Result<Self, WgpuError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| WgpuError::new(&format!("Failed to create runtime: {}", e)))?;
        rt.block_on(Self::new_impl())
    }
    
    async fn new_impl() -> Result<Self, WgpuError> {
        eprintln!("[WGPU] Creating instance with DX12 backend...");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        eprintln!("[WGPU] Instance created, requesting adapter...");
        
        let adapter = match instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await {
            Ok(a) => a,
            Err(e) => return Err(WgpuError::new(&format!("No GPU adapter found: {}", e))),
        };
        eprintln!("[WGPU] Adapter found, getting device...");

        let adapter_info = adapter.get_info();
        log::info!(
            "WGPU Selected GPU: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let supports_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        log::info!("WGPU SHADER_F16 support: {}", supports_f16);

        let required_features = if supports_f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WGPU Device"),
                required_features,
                memory_hints: Default::default(),
                required_limits: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            },
        ).await.map_err(|e| WgpuError::new(&e.to_string()))?;
        eprintln!("[WGPU] Device created, compiling shader...");

        let shader_source = include_str!("../shaders/wgsl/matvec_q8_0.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec_q8_0"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec_q8_0_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matvec_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_q8_0_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    fn create_u8_buffer(&self, data: &[u8], usage: wgpu::BufferUsages) -> Buffer {
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_buffer"),
            size: data.len() as u64,
            usage,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut().copy_from_slice(data);
        buffer.unmap();
        buffer
    }

    fn create_f32_buffer(&self, data: &[f32], usage: wgpu::BufferUsages) -> Buffer {
        let bytes = bytemuck::cast_slice(data);
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_f32_buffer"),
            size: bytes.len() as u64,
            usage,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
        buffer.unmap();
        buffer
    }

    pub unsafe fn matmul_q8_0(
        &self,
        weight: &[u8],
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<(), WgpuError> {
        let blocks_per_row = n_in / 32;
        let weight_row_stride = blocks_per_row * 34;

        let weight_buffer = self.create_u8_buffer(weight, wgpu::BufferUsages::STORAGE);
        let input_buffer = self.create_u8_buffer(input_q8, wgpu::BufferUsages::STORAGE);
        let scale_buffer = self.create_f32_buffer(input_scales, wgpu::BufferUsages::STORAGE);

        let output_bytes = bytemuck::cast_slice_mut(output);
        let output_buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_output_buffer"),
            size: output_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        output_buffer.slice(..).get_mapped_range_mut().copy_from_slice(output_bytes);
        output_buffer.unmap();

        let dims_data: [u32; 4] = [
            n_in as u32,
            n_out as u32,
            blocks_per_row as u32,
            weight_row_stride as u32,
        ];
        let dims_buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dims_buffer"),
            size: (dims_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        dims_buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&dims_data));
        dims_buffer.unmap();

        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_q8_0_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scale_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matvec_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_out as u32, 1, 1);
        }

        self.queue().submit(Some(encoder.finish()));
        self.device().poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        
        let staging_buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: output_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let mut copy_encoder = self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });
        copy_encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_bytes.len() as u64);
        self.queue().submit(Some(copy_encoder.finish()));
        self.device().poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        
        let (tx, rx) = std::sync::mpsc::channel();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device().poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        
        match rx.recv() {
            Ok(Ok(())) => {
                let staging_slice = staging_buffer.slice(..);
                let staging_mapped = staging_slice.get_mapped_range();
                output_bytes.copy_from_slice(&staging_mapped);
            }
            Ok(Err(e)) => return Err(WgpuError::new(&format!("Map async failed: {:?}", e))),
            Err(e) => return Err(WgpuError::new(&format!("Recv failed: {:?}", e))),
        }

        Ok(())
    }
}

#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub struct WgpuError {
    message: String,
}

#[cfg(feature = "wgpu")]
impl WgpuError {
    pub fn new(msg: &str) -> Self {
        Self {
            message: msg.to_string(),
        }
    }
}

#[cfg(feature = "wgpu")]
impl std::fmt::Display for WgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuError: {}", self.message)
    }
}

#[cfg(feature = "wgpu")]
impl std::error::Error for WgpuError {}
