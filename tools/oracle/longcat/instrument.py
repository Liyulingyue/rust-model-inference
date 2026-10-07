"""Only used against a disposable, fixed-revision reference checkout."""
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
p = root / 'ggml/src/ggml-cpu/CMakeLists.txt'
s = p.read_text()
marker = '    if (GGML_SYSTEM_ARCH STREQUAL "ARM")'
if 'set(GGML_SYSTEM_ARCH "RMI_SCALAR")' not in s:
    assert s.count(marker) == 1
    s = s.replace(marker, '    set(GGML_SYSTEM_ARCH "RMI_SCALAR")\n' + marker)
p.write_text(s)
p = root / 'CMakeLists.txt'
s = p.read_text()
if 'add_executable(longcat-oracle' not in s:
    s += '\nadd_executable(longcat-oracle longcat-oracle.cpp)\ntarget_link_libraries(longcat-oracle PRIVATE stable-diffusion)\n'
p.write_text(s)
p = root / 'src/model/diffusion/flux.hpp'
s = p.read_text()
# Capture copies are scheduled into the graph so in-place later ops cannot overwrite them.
replacements = {
    '            sd::ggml_graph_cut::mark_graph_cut(img, "flux.prelude", "img");':
        '            ctx->capture_tensor("longcat.prelude.img", img);\n            ctx->capture_tensor("longcat.prelude.txt", txt);\n            ctx->capture_tensor("longcat.prelude.vec", vec);\n',
    '                sd::ggml_graph_cut::mark_graph_cut(img, "flux.double_blocks." + std::to_string(i), "img");':
        '                ctx->capture_tensor("longcat.double." + std::to_string(i) + ".img", img);\n                ctx->capture_tensor("longcat.double." + std::to_string(i) + ".txt", txt);\n',
    '                sd::ggml_graph_cut::mark_graph_cut(txt_img, "flux.single_blocks." + std::to_string(i), "txt_img");':
        '                ctx->capture_tensor("longcat.single." + std::to_string(i), txt_img);\n',
}
for marker, added in replacements.items():
    if added.strip() not in s:
        assert s.count(marker) == 1
        s = s.replace(marker, added + marker)
p.write_text(s)

# Select ggml's existing F32 GELU branch instead of its default FP16 lookup approximation.
p = root / 'ggml/src/ggml-cpu/vec.h'
s = p.read_text()
s = s.replace('#define GGML_GELU_FP16\n', '// RMI scalar parity: F32 GELU, no FP16 lookup table.\n')
s = s.replace('#define GGML_GELU_QUICK_FP16\n', '// RMI scalar parity: F32 quick GELU.\n')
p.write_text(s)
