use crate::core::memory::{BlockAllocator, MemoryArena};
use crate::core::traits::ModelConfig;

pub fn run_self_test() {
    println!("=== RustModelInference MVP Self-Test ===\n");
    let config = ModelConfig::qwen2_0_6b();
    println!(
        "[Config] Qwen2-0.6B: n_embd={}, n_layer={}, n_head={}, n_ff={}",
        config.n_embd, config.n_layer, config.n_head, config.n_ff
    );

    let mut alloc = BlockAllocator::new(64);
    let b0 = alloc.alloc().unwrap();
    let b1 = alloc.alloc().unwrap();
    alloc.free(b1);
    let b3 = alloc.alloc().unwrap();
    println!(
        "BlockAllocator: alloc {},{}, free {}, re-alloc {} [OK]",
        b0, b1, b1, b3
    );

    let mut arena = MemoryArena::new(1024, 1024);
    let ptr = arena.scratch_slice().as_ptr() as usize;
    arena.scratch_slice()[0] = 42.0;
    assert_eq!(arena.scratch_slice().as_ptr() as usize, ptr);
    println!("MemoryArena: ptr stable [OK]");

    println!("\nUsage: cargo run -- --model <path.gguf> --prompt \"hello\"");
    println!("       cargo run -- --model <path.gguf>  (interactive mode)");
    println!("       cargo run -- --model <llm.gguf> --mmproj <mmproj.gguf> --image <image.png> --prompt \"describe\"");
    println!(
        "       cargo run --release --bin rust-model-inference -- --model models/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0.gguf --mmproj models/qwen3-asr-0.6b/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf --audio sample.wav --language English --max-tokens 256 --threads 8"
    );
}
