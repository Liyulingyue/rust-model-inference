use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "vulkan")]
thread_local! {
    static GPU_MATMUL_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "vulkan")]
pub(crate) fn gpu_matmul_disabled() -> bool {
    GPU_MATMUL_DISABLED.get()
}

#[cfg(feature = "vulkan")]
pub(crate) struct GpuMatmulScope {
    previous: bool,
}

#[cfg(feature = "vulkan")]
impl GpuMatmulScope {
    fn set(disabled: bool) -> Self {
        Self {
            previous: GPU_MATMUL_DISABLED.replace(disabled),
        }
    }
}

#[cfg(feature = "vulkan")]
impl Drop for GpuMatmulScope {
    fn drop(&mut self) {
        GPU_MATMUL_DISABLED.set(self.previous);
    }
}

// ============================================================================
// TODO(architecture): Two-pool concurrency model — see below for unification
// direction. Currently the codebase uses two distinct thread scheduling
// systems:
//
//   1. ComputePool (this file): hand-rolled spin-loop + atomics thread pool.
//      Used by LLM prefill/decode and any code path that needs predictable
//      low-latency scheduling. One ComputePool per inference task (Arc-shared
//      within a request), explicit row-partitioned work via (ith, nth).
//
//   2. rayon global pool: third-party library with work-stealing. Used by
//      audio conv chunks (src/models/qwen3/asr/model.rs), vision patch encoding
//      (src/models/vision.rs), and qwen35 SSM (src/models/qwen35.rs).
//
// The two pools are kept in lock-step via `app::init_rayon_global_pool(n)` at
// startup (see src/main.rs and src/app/cli.rs), which calls
// `rayon::ThreadPoolBuilder::num_threads(n).build_global()` to size rayon's
// global pool to match the resolved `--threads N`.
//
// ## Current rationale (why two pools, not one)
//   - LLM is the hot path: per-task lifecycle, byte-exact output, no
//     work-stealing reordering. ComputePool gives this for free.
//   - Other parallel work is stateless batch processing: rayon is simpler
//     and avoids boilerplate.
//   - They never run concurrently today (audio conv finishes before LLM
//     prefill starts), so oversubscription is not an issue.
//
// ## Future unification — preferred direction: migrate audio/vision/qwen35
//    TO ComputePool, NOT LLM to rayon
//   - LLM is the riskiest surface to change (byte-exact requirement,
//     every model goes through it). Migrating LLM would require parity
//     testing across all model variants.
//   - ComputePool's `compute_with_chunks(n, f)` is a drop-in replacement
//     for `(0..n).into_par_iter().for_each(...)` — minimal API churn.
//   - Single-pool architecture would remove the `init_rayon_global_pool`
//     workaround and the risk of two-pool thread-count drift.
//
//   Affected files when migrating:
//     src/models/qwen3/asr/model.rs   — encode_convolution (8 Mel chunks)
//     src/models/qwen3/asr/model.rs   — audio 18 transformer layers (if MT'd)
//     src/models/vision.rs         — patch parallel (image encoder)
//     src/models/qwen35.rs         — SSM chunks (2 sites)
//
//   When this work happens, the rayon dependency can be dropped from
//   Cargo.toml.
// ============================================================================

pub struct ComputePool {
    n_threads: usize,
    inner: Arc<Inner>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

struct Inner {
    call_fn: AtomicUsize,
    call_data: AtomicUsize,
    n_complete: AtomicI32,
    epoch: AtomicU32,
    shutdown: AtomicBool,
    chunk_counter: AtomicI32,
    chunk_barrier: std::sync::atomic::AtomicUsize,
    chunk_n_chunks: std::sync::atomic::AtomicI32,
    #[cfg(feature = "vulkan")]
    gpu_matmul_disabled: AtomicBool,
    #[cfg(all(test, feature = "vulkan"))]
    gpu_disabled_workers: AtomicUsize,
}

type CallFn = unsafe fn(usize, usize, usize);

impl ComputePool {
    pub fn new(n_threads: usize) -> Self {
        if n_threads <= 1 {
            return ComputePool {
                n_threads: 1,
                inner: Arc::new(Inner {
                    call_fn: AtomicUsize::new(0),
                    call_data: AtomicUsize::new(0),
                    n_complete: AtomicI32::new(1),
                    epoch: AtomicU32::new(0),
                    shutdown: AtomicBool::new(true),
                    chunk_counter: AtomicI32::new(0),
                    chunk_barrier: AtomicUsize::new(0),
                    chunk_n_chunks: AtomicI32::new(0),
                    #[cfg(feature = "vulkan")]
                    gpu_matmul_disabled: AtomicBool::new(false),
                    #[cfg(all(test, feature = "vulkan"))]
                    gpu_disabled_workers: AtomicUsize::new(0),
                }),
                threads: Vec::new(),
            };
        }

        let inner = Arc::new(Inner {
            call_fn: AtomicUsize::new(0),
            call_data: AtomicUsize::new(0),
            n_complete: AtomicI32::new(n_threads as i32),
            epoch: AtomicU32::new(0),
            shutdown: AtomicBool::new(false),
            chunk_counter: AtomicI32::new(0),
            chunk_barrier: AtomicUsize::new(0),
            chunk_n_chunks: AtomicI32::new(0),
            #[cfg(feature = "vulkan")]
            gpu_matmul_disabled: AtomicBool::new(false),
            #[cfg(all(test, feature = "vulkan"))]
            gpu_disabled_workers: AtomicUsize::new(0),
        });

        let mut threads = Vec::with_capacity(n_threads - 1);
        let start_barrier = Arc::new(std::sync::Barrier::new(n_threads));
        for tid in 1..n_threads {
            let inner = inner.clone();
            let nt = n_threads;
            let barrier = start_barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                worker_loop(tid, nt, &inner);
            }));
        }

        start_barrier.wait();

        ComputePool {
            n_threads,
            inner,
            threads,
        }
    }

    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn disable_gpu_matmul_for_scope() -> GpuMatmulScope {
        GpuMatmulScope::set(true)
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn clear_gpu_disabled_workers_for_test(&self) {
        self.inner.gpu_disabled_workers.store(0, Ordering::Relaxed);
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn gpu_disabled_workers_for_test(&self) -> usize {
        self.inner.gpu_disabled_workers.load(Ordering::Relaxed)
    }

    pub fn compute<F: Fn(usize, usize)>(&self, f: F) {
        if self.n_threads <= 1 {
            #[cfg(all(test, feature = "vulkan"))]
            if gpu_matmul_disabled() {
                self.inner
                    .gpu_disabled_workers
                    .fetch_or(1, Ordering::Relaxed);
            }
            f(0, 1);
            return;
        }

        let boxed = Box::new(f);
        let data_ptr = Box::into_raw(boxed) as usize;

        unsafe fn call_closure<F: Fn(usize, usize)>(ith: usize, nth: usize, data: usize) {
            let f = &*(data as *const F);
            f(ith, nth);
        }

        let call_fn = call_closure::<F> as *const () as usize;

        // Wait for all workers to be idle from previous compute
        while self.inner.n_complete.load(Ordering::Acquire) < self.n_threads as i32 {
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(Ordering::SeqCst);

        // Reset completion counter before publishing new work
        self.inner.n_complete.store(0, Ordering::Relaxed);
        self.inner.call_fn.store(call_fn, Ordering::Relaxed);
        self.inner.call_data.store(data_ptr, Ordering::Relaxed);
        #[cfg(feature = "vulkan")]
        self.inner
            .gpu_matmul_disabled
            .store(gpu_matmul_disabled(), Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);

        // Publish: epoch increment signals workers to start
        self.inner.epoch.fetch_add(1, Ordering::SeqCst);

        // Main thread does its share
        #[cfg(all(test, feature = "vulkan"))]
        if gpu_matmul_disabled() {
            self.inner
                .gpu_disabled_workers
                .fetch_or(1, Ordering::Relaxed);
        }
        unsafe {
            call_closure::<F>(0, self.n_threads, data_ptr);
        }
        self.inner.n_complete.fetch_add(1, Ordering::SeqCst);

        // Wait for all threads to finish
        while self.inner.n_complete.load(Ordering::Acquire) < self.n_threads as i32 {
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(Ordering::SeqCst);

        unsafe {
            drop(Box::from_raw(data_ptr as *mut F));
        }
    }

    pub fn compute_with_chunks<F: Fn(usize, i32) + Send + Sync + 'static>(
        &self,
        n_chunks: i32,
        f: F,
    ) {
        let f = Arc::new(f);
        if self.n_threads <= 1 {
            let mut chunk_id = 0;
            while chunk_id < n_chunks {
                f(0, chunk_id);
                chunk_id += 1;
            }
            return;
        }

        while self.inner.n_complete.load(Ordering::Acquire) < self.n_threads as i32 {
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(Ordering::SeqCst);

        self.inner.n_complete.store(0, Ordering::Relaxed);

        let barrier = Arc::new(std::sync::Barrier::new(self.n_threads));
        let mut handles = Vec::with_capacity(self.n_threads - 1);
        for tid in 1..self.n_threads {
            let barrier = barrier.clone();
            let inner = self.inner.clone();
            let f = f.clone();
            let handle = std::thread::spawn(move || {
                barrier.wait();
                let mut chunk_id = inner.chunk_counter.fetch_add(1, Ordering::Relaxed);
                loop {
                    if chunk_id >= n_chunks {
                        break;
                    }
                    f(tid, chunk_id);
                    chunk_id = inner.chunk_counter.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(handle);
        }

        self.inner.call_fn.store(0, Ordering::Relaxed);
        self.inner.call_data.store(0, Ordering::Relaxed);
        self.inner.chunk_counter.store(0, Ordering::Relaxed);
        self.inner
            .chunk_barrier
            .store(Arc::as_ptr(&barrier) as usize, Ordering::Relaxed);
        self.inner.chunk_n_chunks.store(n_chunks, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);

        self.inner.epoch.fetch_add(1, Ordering::SeqCst);

        while self.inner.n_complete.load(Ordering::Acquire) < (self.n_threads - 1) as i32 {
            std::hint::spin_loop();
        }

        barrier.wait();

        let mut chunk_id = self.inner.chunk_counter.fetch_add(1, Ordering::Relaxed);
        loop {
            if chunk_id >= n_chunks {
                break;
            }
            f(0, chunk_id);
            chunk_id = self.inner.chunk_counter.fetch_add(1, Ordering::Relaxed);
        }

        self.inner.n_complete.fetch_add(1, Ordering::SeqCst);

        while self.inner.n_complete.load(Ordering::Acquire) < self.n_threads as i32 {
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(Ordering::SeqCst);

        self.inner.call_fn.store(0, Ordering::Relaxed);
        self.inner.call_data.store(0, Ordering::Relaxed);
        self.inner.chunk_barrier.store(0, Ordering::Relaxed);
        self.inner.chunk_n_chunks.store(0, Ordering::Relaxed);

        for h in handles {
            let _ = h.join();
        }
    }

    pub fn next_chunk(&self) -> i32 {
        0
    }
}

impl Drop for ComputePool {
    fn drop(&mut self) {
        while self.inner.n_complete.load(Ordering::Acquire) < self.n_threads as i32 {
            std::hint::spin_loop();
        }
        self.inner.n_complete.store(0, Ordering::Relaxed);
        self.inner.shutdown.store(true, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);
        self.inner.epoch.fetch_add(1, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn worker_loop(tid: usize, n_threads: usize, inner: &Inner) {
    let mut my_epoch: u32 = 0;
    loop {
        while inner.epoch.load(Ordering::Acquire) == my_epoch {
            if inner.shutdown.load(Ordering::Acquire) {
                return;
            }
            // While the GPU matmul path is active the CPU workers have no
            // hot-path role (one fenced GPU dispatch covers all rows), so the
            // idle spin here only competes with the driver's submission
            // threads — observed to hang ANV/Meteor Lake. Despin when GPU is
            // on; the pure-CPU path keeps the tight spin.
            std::hint::spin_loop();
        }
        my_epoch = inner.epoch.load(Ordering::Acquire);
        if inner.shutdown.load(Ordering::Acquire) {
            return;
        }

        let call_fn = inner.call_fn.load(Ordering::Acquire);
        let call_data = inner.call_data.load(Ordering::Acquire);
        if call_fn != 0 {
            #[cfg(feature = "vulkan")]
            let gpu_matmul_disabled = inner.gpu_matmul_disabled.load(Ordering::Acquire);
            #[cfg(all(test, feature = "vulkan"))]
            if gpu_matmul_disabled {
                if let Some(worker_bit) = 1usize.checked_shl(tid as u32) {
                    inner
                        .gpu_disabled_workers
                        .fetch_or(worker_bit, Ordering::Relaxed);
                }
            }
            #[cfg(feature = "vulkan")]
            let _gpu_matmul_scope = GpuMatmulScope::set(gpu_matmul_disabled);
            let f: CallFn = unsafe { std::mem::transmute(call_fn) };
            unsafe {
                f(tid, n_threads, call_data);
            }
        }

        inner.n_complete.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn single_thread_pool_drops_without_spinning() {
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let pool = ComputePool::new(1);
            pool.compute(|ith, nth| assert_eq!((ith, nth), (0, 1)));
            drop(pool);
            tx.send(()).unwrap();
        });

        rx.recv_timeout(Duration::from_secs(1))
            .expect("single-thread ComputePool::drop timed out");
        handle.join().unwrap();
    }

    #[test]
    fn chunks_begin_at_zero_and_execute_exactly_once() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let worker_seen = Arc::clone(&seen);
        ComputePool::new(2).compute_with_chunks(7, move |_thread, chunk| {
            worker_seen.lock().unwrap().push(chunk);
        });

        let mut seen = seen.lock().unwrap().clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..7).collect::<Vec<_>>());
    }
}
