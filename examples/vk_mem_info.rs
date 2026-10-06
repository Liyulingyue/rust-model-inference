//! Dump the Vulkan memory heaps.
//!
//! `alloc_persistently_mapped` asks only for HOST_VISIBLE|HOST_COHERENT, so on
//! a device that exposes a separate DEVICE_LOCAL heap the weights are not
//! where the fastest reads come from, and no timing explains why on its own.
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;

fn main() {
    enable_gpu();
    let Some(context) = get_vulkan_context() else {
        eprintln!("no vulkan context");
        return;
    };
    println!("device: {}", context.device_name());
    println!("memory types:");
    print!("{}", context.memory_type_report());
}
