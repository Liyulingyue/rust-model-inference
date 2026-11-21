#!/usr/bin/env python3
"""Generate the IQ2/IQ3/IQ1 matmul shaders (GLSL compute) from the grid
tables in src/ops/quant/iq_tables_data.rs. Run from the repo root:
    python3 shaders/gen_iq_shaders.py
then compile each emitted shader and refresh shaders/manifest.sha256.

Decode/accumulation semantics mirror vec_dot_iq*_q8k_avx2 exactly:
8 i32 lane accumulators (lane k = values 4k..4k+3 via madd pairing),
per-lane f32 accumulation via acc_lane[k] = fma(d, float(lane[k]), acc_lane[k]),
final reduction = hsum_float_8's add tree, and the IQ1 delta terms in a
separate f32 accumulator matching the CPU's 0.125-weighted epilogue.
"""
import re

TABLES_FILE = "src/ops/quant/iq_tables_data.rs"

PRELUDE = """#version 450

layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Arena { uint data[]; } arena;
layout(std430, binding = 1) readonly buffer W0 { uint data[]; } w0;
layout(std430, binding = 2) readonly buffer W1 { uint data[]; } w1;
layout(std430, binding = 3) readonly buffer W2 { uint data[]; } w2;
layout(push_constant) uniform PC { uint q8_word; uint scales_word; uint n_in; uint blocks_per_row; uint output0_word; uint rows0; uint output0_stride; uint output1_word; uint rows1; uint output1_stride; uint output2_word; uint rows2; uint output2_stride; uint group_count; uint input_sum_word; uint input_stride; uint q8_stride; uint scales_stride; uint input_sum_stride; uint q8k_stride; uint q8k_scales_stride; uint token_rows; } pc;

float f16_decode(uint bits) { uint sign = (bits >> 15u) & 1u; int exponent = int((bits >> 10u) & 0x1fu); uint mantissa = bits & 0x3ffu; float value = exponent == 0 ? float(mantissa) * exp2(-24.0) : (exponent == 31 ? 0.0 : (1.0 + float(mantissa) / 1024.0) * exp2(float(exponent - 15))); return sign == 0u ? value : -value; }
uint weight_word(uint slot, uint index) { return slot == 0u ? w0.data[index] : (slot == 1u ? w1.data[index] : w2.data[index]); }
uint wbyte(uint slot, uint abs_byte) { return (weight_word(slot, abs_byte >> 2u) >> ((abs_byte & 3u) * 8u)) & 0xffu; }
uint wshort(uint slot, uint abs_byte) { return wbyte(slot, abs_byte) | (wbyte(slot, abs_byte + 1u) << 8u); }
uint wword4(uint slot, uint abs_byte) { return wbyte(slot, abs_byte) | (wbyte(slot, abs_byte + 1u) << 8u) | (wbyte(slot, abs_byte + 2u) << 16u) | (wbyte(slot, abs_byte + 3u) << 24u); }
int q8(uint index) { uint value = (arena.data[pc.q8_word + (gl_WorkGroupID.z / pc.group_count) * pc.q8_stride + (index >> 2u)] >> ((index & 3u) * 8u)) & 0xffu; return value >= 128u ? int(value) - 256 : int(value); }
int i8v(uint b) { return b >= 128u ? int(b) - 256 : int(b); }

shared int input_q8[256];
"""

MAIN_OPEN = """void main() {
    uint token_row = gl_WorkGroupID.z / pc.group_count; uint slot = gl_WorkGroupID.z % pc.group_count;
    uint row = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * 64u + gl_LocalInvocationID.x;
    uint rows = slot == 0u ? pc.rows0 : (slot == 1u ? pc.rows1 : pc.rows2);
    bool valid_row = slot < pc.group_count && row < rows;
    precise float acc_lane[8] = float[8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    {extra_acc}
    for (uint block = 0u; block < pc.blocks_per_row; ++block) {
        for (uint local = gl_LocalInvocationID.x; local < 256u; local += 64u) {
            input_q8[local] = q8(block * 256u + local);
        }
        barrier();
        if (valid_row) {
            uint rb = row * pc.blocks_per_row * {bb}u + block * {bb}u;
"""

MAIN_CLOSE = """        }
        barrier();
    }
    if (!valid_row) return;
    uint output_word = slot == 0u ? pc.output0_word : (slot == 1u ? pc.output1_word : pc.output2_word);
    uint out_stride = slot == 0u ? pc.output0_stride : (slot == 1u ? pc.output1_stride : pc.output2_stride);
    arena.data[output_word + token_row * out_stride + row] = floatBitsToUint({final});
}
"""

HSUM_TREE = """    precise float t0 = acc_lane[4] + acc_lane[0];
    precise float t1 = acc_lane[5] + acc_lane[1];
    precise float t2 = acc_lane[6] + acc_lane[2];
    precise float t3 = acc_lane[7] + acc_lane[3];
    precise float u0 = t0 + t2;
    precise float u1 = t1 + t3;"""

LANE_ACC = """            for (uint k = 0u; k < 8u; ++k) {
                acc_lane[k] = fma(d, float(lane[k]), acc_lane[k]);
            }"""


def load_tables():
    src = open(TABLES_FILE).read()
    def u64t(name):
        m = re.search(r"pub const %s: \[u64; (\d+)\] = \[(.*?)\];" % name, src, re.S)
        return [int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]+)", m.group(2))]
    def u32t(name):
        m = re.search(r"pub const %s: \[u32; (\d+)\] = \[(.*?)\];" % name, src, re.S)
        return [int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]+)", m.group(2))]
    ks = re.search(r"pub const KSIGNS_IQ2XS: \[u8; 128\] = \[(.*?)\];", src, re.S)
    ksigns = [int(v) for v in re.findall(r"\b(\d+)\b", ks.group(1))]
    return dict(
        IQ2XXS_GRID=u64t("IQ2XXS_GRID"), IQ2XS_GRID=u64t("IQ2XS_GRID"),
        IQ2S_GRID=u64t("IQ2S_GRID"), IQ3XXS_GRID=u32t("IQ3XXS_GRID"),
        IQ3S_GRID=u32t("IQ3S_GRID"), IQ1S_GRID=u64t("IQ1S_GRID"), KSIGNS=ksigns)


def u64_tables(name, vals):
    n = len(vals)
    lo = [f"0x{v & 0xffffffff:08x}u" for v in vals]
    hi = [f"0x{v >> 32:08x}u" for v in vals]
    return (f"const uint {name}_LO[{n}] = uint[{n}]({', '.join(lo)});\n"
            f"const uint {name}_HI[{n}] = uint[{n}]({', '.join(hi)});\n")

def u32_table(name, vals):
    n = len(vals)
    return f"const uint {name}[{n}] = uint[{n}]({', '.join(f'0x{v:08x}u' for v in vals)});\n"

def ksigns_table(vals):
    words = [f"0x{vals[i]|(vals[i+1]<<8)|(vals[i+2]<<16)|(vals[i+3]<<24):08x}u" for i in range(0, 128, 4)]
    return ("const uint KSIGNS_WORDS[32] = uint[32](" + ", ".join(words) + ");\n"
            "uint ksigns(uint index) { uint w = KSIGNS_WORDS[index >> 2u]; return (w >> ((index & 3u) * 8u)) & 0xffu; }\n")


def emit(kind, tables):
    spec = BODIES[kind]
    parts = [PRELUDE]
    for t in spec["tables"]:
        if t == "KSIGNS":
            parts.append(ksigns_table(tables["KSIGNS"]))
        elif t in ("IQ2XXS_GRID", "IQ2XS_GRID", "IQ2S_GRID", "IQ1S_GRID"):
            parts.append(u64_tables(t, tables[t]))
        else:
            parts.append(u32_table(t, tables[t]))
    parts.append(MAIN_OPEN.replace("{bb}", str(spec["bb"])).replace("{extra_acc}", spec.get("extra_acc", "")))
    parts.append(spec["body"] + "\n")
    parts.append(MAIN_CLOSE.replace("{final}", spec["final"]))
    return "\n".join(parts)


def _final(factor="1.0"):
    # hsum_float_8's add tree as a pure expression:
    # t0=x[4]+x[0], t1=x[5]+x[1], t2=x[6]+x[2], t3=x[7]+x[3]
    # u0=t0+t2, u1=t1+t3, result=u0+u1
    a = "acc_lane"
    expr = f"(({a}[4] + {a}[0]) + ({a}[6] + {a}[2])) + (({a}[5] + {a}[1]) + ({a}[7] + {a}[3]))"
    if factor == "1.0":
        return expr
    return factor + " * (" + expr + ")"


BODIES = {}

def _def(name, bb, tables, body, final, extra_acc=""):
    BODIES[name] = dict(bb=bb, tables=tables, body=body, final=_final(final), extra_acc=extra_acc)

# iq2_xxs
_def("iq2_xxs", 66, ["IQ2XXS_GRID", "KSIGNS"], """\
            precise float q8_scale = uintBitsToFloat(arena.data[pc.scales_word + token_row * pc.scales_stride + block]);
            precise float d = f16_decode(wshort(slot, rb) & 0xffffu) * q8_scale;
            int lane[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0);
            for (uint g = 0u; g < 8u; ++g) {
                uint a0 = wword4(slot, rb + 2u + 8u * g);
                uint a1 = wword4(slot, rb + 6u + 8u * g);
                uint ls = 2u * ((a1 >> 28u) & 0xfu) + 1u;
                for (uint k = 0u; k < 8u; ++k) {
                    uint l = k / 2u;
                    uint gidx = (a0 >> (8u * l)) & 0xffu;
                    uint sb = ksigns((a1 >> (7u * l)) & 0x7fu);
                    int dot = 0;
                    for (uint m = 0u; m < 4u; ++m) {
                        uint j = 4u * (k % 2u) + m;
                        uint b = j < 4u ? (IQ2XXS_GRID_LO[gidx] >> (8u * j)) & 0xffu
                                        : (IQ2XXS_GRID_HI[gidx] >> (8u * (j - 4u))) & 0xffu;
                        int sgn = ((sb >> j) & 1u) == 1u ? -1 : 1;
                        dot += i8v(b) * sgn * input_q8[32u * g + 4u * k + m];
                    }
                    lane[k] += dot * int(ls);
                }
            }
""" + LANE_ACC, "0.125f")

# iq2_xs
_def("iq2_xs", 74, ["IQ2XS_GRID", "KSIGNS"], """\
            precise float q8_scale = uintBitsToFloat(arena.data[pc.scales_word + token_row * pc.scales_stride + block]);
            precise float d = f16_decode(wshort(slot, rb) & 0xffffu) * q8_scale;
            int lane[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0);
            for (uint g = 0u; g < 8u; ++g) {
                uint sc_byte = wbyte(slot, rb + 66u + g);
                uint ls1 = 2u * (sc_byte & 0xfu) + 1u;
                uint ls2 = 2u * (sc_byte >> 4u) + 1u;
                for (uint k = 0u; k < 8u; ++k) {
                    uint l = k / 2u;
                    uint q = wshort(slot, rb + 2u + 8u * g + 2u * l);
                    uint gidx = q & 511u;
                    uint sb = ksigns((q >> 9u) & 0x7fu);
                    int dot = 0;
                    for (uint m = 0u; m < 4u; ++m) {
                        uint j = 4u * (k % 2u) + m;
                        uint b = j < 4u ? (IQ2XS_GRID_LO[gidx] >> (8u * j)) & 0xffu
                                        : (IQ2XS_GRID_HI[gidx] >> (8u * (j - 4u))) & 0xffu;
                        int sgn = ((sb >> j) & 1u) == 1u ? -1 : 1;
                        dot += i8v(b) * sgn * input_q8[32u * g + 4u * k + m];
                    }
                    lane[k] += dot * int((k < 4u) ? ls1 : ls2);
                }
            }
""" + LANE_ACC, "0.125f")

# iq2_s
_def("iq2_s", 82, ["IQ2S_GRID"], """\
            precise float q8_scale = uintBitsToFloat(arena.data[pc.scales_word + token_row * pc.scales_stride + block]);
            precise float d = f16_decode(wshort(slot, rb) & 0xffffu) * q8_scale;
            int lane[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0);
            for (uint g = 0u; g < 8u; ++g) {
                uint sc_byte = wbyte(slot, rb + 74u + g);
                uint ls1 = 1u + 2u * (sc_byte & 0xfu);
                uint ls2 = 1u + 2u * (sc_byte >> 4u);
                uint qh_b = wbyte(slot, rb + 66u + g);
                uint sign_bits = wword4(slot, rb + 34u + 4u * g);
                for (uint k = 0u; k < 8u; ++k) {
                    uint l = k / 2u;
                    uint gidx = wbyte(slot, rb + 2u + 4u * g + l) | ((qh_b << (8u - 2u * l)) & 0x300u);
                    int dot = 0;
                    for (uint m = 0u; m < 4u; ++m) {
                        uint j = 4u * (k % 2u) + m;
                        uint b = j < 4u ? (IQ2S_GRID_LO[gidx] >> (8u * j)) & 0xffu
                                        : (IQ2S_GRID_HI[gidx] >> (8u * (j - 4u))) & 0xffu;
                        int sgn = ((sign_bits >> (8u * l + j)) & 1u) == 1u ? -1 : 1;
                        dot += i8v(b) * sgn * input_q8[32u * g + 4u * k + m];
                    }
                    lane[k] += dot * int((k < 4u) ? ls1 : ls2);
                }
            }
""" + LANE_ACC, "0.125f")

# iq3_xxs
_def("iq3_xxs", 98, ["IQ3XXS_GRID", "KSIGNS"], """\
            precise float q8_scale = uintBitsToFloat(arena.data[pc.scales_word + token_row * pc.scales_stride + block]);
            precise float d = f16_decode(wshort(slot, rb) & 0xffffu) * q8_scale;
            int lane[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0);
            for (uint g = 0u; g < 8u; ++g) {
                uint aux = wword4(slot, rb + 66u + 4u * g);
                uint ls = 2u * ((aux >> 28u) & 0xfu) + 1u;
                for (uint k = 0u; k < 8u; ++k) {
                    // Lane k covers values 4k..4k+3, using grid entry qs[8g+k].
                    // Sign bits: k%2==0 → bits 0-3 of sb, k%2==1 → bits 4-7.
                    uint grid_entry = IQ3XXS_GRID[wbyte(slot, rb + 2u + 8u * g + k)];
                    uint l = k / 2u;
                    uint sb = ksigns((aux >> (7u * l)) & 0x7fu);
                    uint sign_base = 4u * (k % 2u);
                    int dot = 0;
                    for (uint j = 0u; j < 4u; ++j) {
                        int v = i8v((grid_entry >> (8u * j)) & 0xffu);
                        int sgn = ((sb >> (sign_base + j)) & 1u) == 1u ? -1 : 1;
                        dot += v * sgn * input_q8[32u * g + 4u * k + j];
                    }
                    lane[k] += dot * int(ls);
                }
            }
""" + LANE_ACC, "0.25f")

# iq3_s
_def("iq3_s", 110, ["IQ3S_GRID"], """\
            precise float q8_scale = uintBitsToFloat(arena.data[pc.scales_word + token_row * pc.scales_stride + block]);
            precise float d = f16_decode(wshort(slot, rb) & 0xffffu) * q8_scale;
            int lane[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0);
            for (uint g = 0u; g < 8u; ++g) {
                uint qh_b = wbyte(slot, rb + 66u + g);
                uint sc_byte = wbyte(slot, rb + 106u + g / 2u);
                uint ls = (g % 2u == 0u) ? 2u * (sc_byte & 0xfu) + 1u : 2u * (sc_byte >> 4u) + 1u;
                for (uint k = 0u; k < 8u; ++k) {
                    uint l = k / 2u;
                    uint grid_entry = IQ3S_GRID[wbyte(slot, rb + 2u + 8u * g + k) | ((qh_b << (8u - k)) & 0x100u)];
                    uint sb = wbyte(slot, rb + 74u + 4u * g + l);
                    uint sign_base = 4u * (k % 2u);
                    int dot = 0;
                    for (uint j = 0u; j < 4u; ++j) {
                        int v = i8v((grid_entry >> (8u * j)) & 0xffu);
                        int sgn = ((sb >> (sign_base + j)) & 1u) == 1u ? -1 : 1;
                        dot += v * sgn * input_q8[32u * g + 4u * k + j];
                    }
                    lane[k] += dot * int(ls);
                }
            }
""" + LANE_ACC, "1.0f")

# iq1_s
_def("iq1_s", 50, ["IQ1S_GRID"], """\
            precise float q8_scale = uintBitsToFloat(arena.data[pc.scales_word + token_row * pc.scales_stride + block]);
            precise float d = f16_decode(wshort(slot, rb) & 0xffffu) * q8_scale;
            int lane[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0);
            int total_delta = 0;
            for (uint ib = 0u; ib < 8u; ++ib) {
                uint qh_val = wshort(slot, rb + 34u + 2u * ib);
                uint ls = 2u * ((qh_val >> 12u) & 7u) + 1u;
                int delta = (qh_val & 0x8000u) == 0x8000u ? -1 : 1;
                int dot = 0;
                int bsum_pair = 0;
                for (uint l = 0u; l < 4u; ++l) {
                    uint gidx = wbyte(slot, rb + 2u + 4u * ib + l) | (((qh_val >> (3u * l)) & 7u) << 8u);
                    int dot2 = 0;
                    for (uint j = 0u; j < 8u; ++j) {
                        uint b = j < 4u ? (IQ1S_GRID_LO[gidx] >> (8u * j)) & 0xffu
                                        : (IQ1S_GRID_HI[gidx] >> (8u * (j - 4u))) & 0xffu;
                        int qv = input_q8[32u * ib + 8u * l + j];
                        dot += i8v(b) * qv;
                        bsum_pair += qv;
                    }
                }
                lane[ib] += dot * int(ls);
                total_delta += int(ls) * delta * bsum_pair;
            }
""" + LANE_ACC + """
            acc_delta += d * float(total_delta);""", "1.0f",
    extra_acc="    precise float acc_delta = 0.0;")

# iq1_m
_def("iq1_m", 56, ["IQ1S_GRID"], """\
            uint sc0 = wshort(slot, rb + 48u);
            uint sc1 = wshort(slot, rb + 50u);
            uint sc2 = wshort(slot, rb + 52u);
            uint sc3 = wshort(slot, rb + 54u);
            uint scale_u16 = (sc0 >> 12u) | ((sc1 >> 8u) & 0xf0u) | ((sc2 >> 4u) & 0xf00u) | (sc3 & 0xf000u);
            precise float q8_scale = uintBitsToFloat(arena.data[pc.scales_word + token_row * pc.scales_stride + block]);
            precise float d = f16_decode(scale_u16 & 0xffffu) * q8_scale;
            int lane[8] = int[8](0, 0, 0, 0, 0, 0, 0, 0);
            int total_delta = 0;
            for (uint ib = 0u; ib < 8u; ++ib) {
                uint sc_pair = wshort(slot, rb + 48u + 2u * (ib / 2u));
                uint shift = 6u * (ib % 2u);
                uint ls1 = 2u * ((sc_pair >> shift) & 7u) + 1u;
                uint ls2 = 2u * ((sc_pair >> (shift + 3u)) & 7u) + 1u;
                uint qh0 = wbyte(slot, rb + 32u + 2u * ib);
                uint qh1 = wbyte(slot, rb + 32u + 2u * ib + 1u);
                int dv0 = (qh0 & 0x08u) == 0x08u ? -1 : 1;
                int dv1 = (qh0 & 0x80u) == 0x80u ? -1 : 1;
                int dv2 = (qh1 & 0x08u) == 0x08u ? -1 : 1;
                int dv3 = (qh1 & 0x80u) == 0x80u ? -1 : 1;
                int gdot[4]; int qsum[4];
                for (uint l = 0u; l < 4u; ++l) {
                    uint qhb = (l < 2u) ? qh0 : qh1;
                    uint sh = (l % 2u == 0u) ? 8u : 4u;
                    uint gidx = wbyte(slot, rb + 4u * ib + l) | ((wbyte(slot, rb + 32u + 2u * ib + l / 2u) << sh) & 0x700u);
                    int gd = 0; int qs8 = 0;
                    for (uint j = 0u; j < 8u; ++j) {
                        uint b = j < 4u ? (IQ1S_GRID_LO[gidx] >> (8u * j)) & 0xffu
                                        : (IQ1S_GRID_HI[gidx] >> (8u * (j - 4u))) & 0xffu;
                        int qv = input_q8[32u * ib + 8u * l + j];
                        gd += i8v(b) * qv;
                        qs8 += qv;
                    }
                    gdot[l] = gd; qsum[l] = qs8;
                }
                int delta_r0 = int(1u - 2u * ((qh0 >> 3u) & 1u));
                int delta_r1 = int(1u - 2u * ((qh0 >> 7u) & 1u));
                int delta_r2 = int(1u - 2u * ((qh1 >> 3u) & 1u));
                int delta_r3 = int(1u - 2u * ((qh1 >> 7u) & 1u));
                lane[2u * (ib % 2u)] += (gdot[0] + gdot[1]) * int(ls1)
                    + (qsum[0] * delta_r0 + qsum[1] * delta_r1) * int(ls1);
                lane[2u * (ib % 2u) + 1u] += (gdot[2] + gdot[3]) * int(ls2)
                    + (qsum[2] * delta_r2 + qsum[3] * delta_r3) * int(ls2);
                total_delta += (qsum[0] * delta_r0 + qsum[1] * delta_r1) * int(ls1)
                             + (qsum[2] * delta_r2 + qsum[3] * delta_r3) * int(ls2);
            }
""" + LANE_ACC + """
            acc_delta = fma(d, float(total_delta), acc_delta);""", "1.0f",
    extra_acc="    precise float acc_delta = 0.0;")


def main():
    tables = load_tables()
    for kind in BODIES:
        path = f"shaders/glsl/{kind}_matmul.comp"
        open(path, "w").write(emit(kind, tables))
        print(f"wrote {path}")

main()
