//! Float conversions + GPU/CPU feature detection.

use std::sync::atomic::Ordering;

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::AtomicBool;

#[cfg(target_arch = "x86_64")]
static GPU_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(not(target_arch = "x86_64"))]
static GPU_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn enable_gpu() {
    GPU_ENABLED.store(true, Ordering::Relaxed);
}

pub fn gpu_requested() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}

/// True when a GPU backend is active and healthy, i.e. matmul outputs are
/// produced by a fenced GPU dispatch owned by pool thread 0. Trunks use this
/// to route *element-wise post-matmul work* (silu, gating) to thread 0 over
/// the whole buffer instead of per-thread row slices — on the GPU path there
/// are no per-thread row owners, so per-thread post-op slices would read the
/// buffer before the dispatch completes.
#[cfg(feature = "vulkan")]
pub fn gpu_matmul_active() -> bool {
    !crate::core::thread_pool::gpu_matmul_disabled()
        && !crate::vulkan::gpu_broken()
        && get_vulkan_context().is_some()
}

#[cfg(not(feature = "vulkan"))]
pub fn gpu_matmul_active() -> bool {
    false
}

#[cfg(feature = "vulkan")]
pub fn gpu_broken() -> bool {
    crate::vulkan::gpu_broken()
}

#[cfg(feature = "vulkan")]
pub fn mark_gpu_broken(reason: &str) {
    crate::vulkan::mark_gpu_broken(reason);
}

#[cfg(feature = "vulkan")]
use std::sync::OnceLock;

#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanContext;

#[cfg(feature = "wgpu")]
use std::sync::OnceLock;

#[cfg(feature = "wgpu")]
use crate::wgpu::WgpuContext;

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::asm;

#[cfg(target_arch = "x86_64")]
static HAS_AVX2_FMA: AtomicBool = AtomicBool::new(false);
#[cfg(target_arch = "x86_64")]
static HAS_F16C: AtomicBool = AtomicBool::new(false);
#[cfg(target_arch = "x86_64")]
static INIT_DONE: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "vulkan")]
static VULKAN_CONTEXT: OnceLock<Result<VulkanContext, String>> = OnceLock::new();

#[cfg(feature = "vulkan")]
use std::sync::Mutex;

#[cfg(feature = "vulkan")]
pub fn get_vulkan_context() -> Option<&'static VulkanContext> {
    if !GPU_ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    // Warmup runs INSIDE the init closure: the driver JITs the compute
    // pipeline on first dispatch (seconds on Meteor Lake), and any thread
    // reaching a dispatch before that completes wedges the watchdog. The
    // OnceLock serializes context creation + warmup across all callers.
    let result =
        VULKAN_CONTEXT.get_or_init(|| match VulkanContext::new().map_err(|e| e.to_string()) {
            Ok(ctx) => {
                eprintln!("[GPU] Warming up Vulkan pipeline (driver JIT)...");
                let t0 = std::time::Instant::now();
                match unsafe { ctx.warmup() } {
                    Ok(()) => {
                        eprintln!("[GPU] Warmup done in {:.1}s", t0.elapsed().as_secs_f64());
                        Ok(ctx)
                    }
                    Err(e) => {
                        eprintln!("[GPU] Warmup failed: {e}. Falling back to CPU.");
                        crate::vulkan::mark_gpu_broken(&e.to_string());
                        Err(format!("warmup failed: {e}"))
                    }
                }
            }
            Err(e) => Err(e),
        });
    match result {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("[GPU] Vulkan init failed: {}. Falling back to CPU.", e);
            None
        }
    }
}

#[cfg(feature = "wgpu")]
static WGPU_CONTEXT: OnceLock<Result<WgpuContext, String>> = OnceLock::new();

#[cfg(feature = "wgpu")]
static WGPU_INIT_THREAD: std::sync::OnceLock<std::sync::Mutex<Option<WgpuContext>>> =
    std::sync::OnceLock::new();

#[cfg(feature = "wgpu")]
pub fn get_wgpu_context() -> Option<&'static WgpuContext> {
    if !GPU_ENABLED.load(Ordering::Relaxed) {
        return None;
    }

    let cell = WGPU_INIT_THREAD.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cell.lock().ok()?;

    if guard.is_none() {
        eprintln!("[GPU] Creating wgpu context via blocking thread...");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                WgpuContext::new_blocking()
            }));
            match result {
                Ok(Ok(ctx)) => {
                    let _ = tx.send(Ok(ctx));
                }
                Ok(Err(e)) => {
                    let _ = tx.send(Err(e.to_string()));
                }
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    let _ = tx.send(Err(msg));
                }
            }
        });

        eprintln!("[GPU] Waiting for wgpu init...");
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(Ok(ctx)) => {
                eprintln!("[GPU] Wgpu context created successfully!");
                *guard = Some(ctx);
            }
            Ok(Err(e)) => {
                eprintln!("[GPU] WGPU init failed: {}. Falling back to CPU.", e);
            }
            Err(e) => {
                eprintln!("[GPU] WGPU init timeout: {:?}. Falling back to CPU.", e);
            }
        }
    }

    guard
        .as_ref()
        .map(|ctx| unsafe { std::mem::transmute(ctx) })
}

#[cfg(target_arch = "x86_64")]
fn init_cpu_features() {
    if INIT_DONE.load(Ordering::Relaxed) {
        return;
    }
    let avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    let f16c = is_x86_feature_detected!("f16c");
    HAS_AVX2_FMA.store(avx2_fma, Ordering::Relaxed);
    HAS_F16C.store(f16c, Ordering::Relaxed);
    INIT_DONE.store(true, Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_avx2_fma() -> bool {
    if !INIT_DONE.load(Ordering::Relaxed) {
        init_cpu_features();
    }
    HAS_AVX2_FMA.load(Ordering::Relaxed)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub const fn has_avx2_fma() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn has_f16c() -> bool {
    if !INIT_DONE.load(Ordering::Relaxed) {
        init_cpu_features();
    }
    HAS_F16C.load(Ordering::Relaxed)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub const fn has_f16c() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub const fn has_neon() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub const fn has_neon() -> bool {
    false
}

#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    {
        unsafe {
            use std::arch::x86_64::*;
            let v = _mm_set1_epi16(bits as i16);
            _mm_cvtss_f32(_mm_cvtph_ps(v))
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
    {
        half::f16::from_bits(bits).to_f32()
    }
}

#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    {
        unsafe {
            use std::arch::x86_64::*;
            let fv = _mm_set_ss(v);
            let hv = _mm_cvtps_ph(fv, 0);
            _mm_extract_epi16(hv, 0) as u16
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
    {
        half::f16::from_f32(v).to_bits()
    }
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    crate::core::tensor::bf16_to_f32(bits)
}

#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

pub fn f32_slice_to_f16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() {
            unsafe {
                f32_slice_to_f16_avx2(src, dst);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                f32_slice_to_f16_neon(src, dst);
            }
            return;
        }
    }
    for i in 0..src.len() {
        dst[i] = f32_to_f16(src[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn f32_slice_to_f16_neon(src: &[f32], dst: &mut [u16]) {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= src.len() {
        let halves = vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(src.as_ptr().add(i))));
        vst1_u16(dst.as_mut_ptr().add(i), halves);
        i += 4;
    }
    while i < src.len() {
        dst[i] = f32_to_f16(src[i]);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn f32_slice_to_f16_avx2(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        let h = _mm256_cvtps_ph(v, 0);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        i += 8;
    }
    while i < n {
        dst[i] = f32_to_f16(src[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_round_trip() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-2.0), 0xc000);
    }
}
