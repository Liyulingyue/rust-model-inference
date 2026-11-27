// Copyright (c) 2017-2025, Arm Limited.
// SPDX-License-Identifier: MIT
// Table/polynomial approximations from ARM-software/optimized-routines math/{expf,logf}.
// Reference: f67e7daa5457646fad9f75580263dba654c988aa (MIT option).
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to
// do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef RMI_HOST_LIBM_GLSL
#define RMI_HOST_LIBM_GLSL
#include "exp_accurate.glsl"

// The Linux x86 host reference uses these libm approximations. Half-ULP
// differences from correctly rounded exp can change the next Q8 activation.
layout(constant_id = 0) const bool host_glibc_libm = false;

vec2 libm_add(vec2 a, vec2 b) {
    precise float s = a.x + b.x;
    precise float v = s - a.x;
    precise float e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
    precise float h = s + e;
    return vec2(h, e - (h - s));
}

vec2 libm_mul(vec2 a, vec2 b) {
    precise float p = a.x * b.x;
    float ah = uintBitsToFloat(floatBitsToUint(a.x) & 0xfffff000u);
    float bh = uintBitsToFloat(floatBitsToUint(b.x) & 0xfffff000u);
    precise float al = a.x - ah;
    precise float bl = b.x - bh;
    precise float e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    e += a.x * b.y + a.y * b.x;
    precise float h = p + e;
    return vec2(h, e - (h - p));
}

const vec2 libm_exp_table[32] = vec2[32](
    vec2(1.0, 0.0),
    vec2(1.0218971967697144, -4.811559861650494e-08),
    vec2(1.0442737340927124, 4.8334701574503924e-08),
    vec2(1.0671404600143433, -5.9337519786595294e-08),
    vec2(1.0905077457427979, -1.3077539939843064e-08),
    vec2(1.114386796951294, -5.435540018083884e-08),
    vec2(1.1387885808944702, 5.386222312608879e-08),
    vec2(1.1637248992919922, -4.051441493402308e-08),
    vec2(1.1892070770263672, 3.797635272917432e-08),
    vec2(1.215247392654419, -3.267394887984665e-08),
    vec2(1.2418577671051025, 4.496838101886169e-08),
    vec2(1.2690509557724, 1.419333317542737e-09),
    vec2(1.2968395948410034, -4.018999533172973e-08),
    vec2(1.3252366781234741, -3.496373324196611e-08),
    vec2(1.3542555570602417, -1.0123349269974824e-08),
    vec2(1.3839099407196045, -5.875577357983275e-08),
    vec2(1.4142135381698608, 2.4203234971764687e-08),
    vec2(1.4451807737350464, 3.324199937537742e-08),
    vec2(1.4768261909484863, -4.500898853621038e-08),
    vec2(1.5091644525527954, -2.4959373234878512e-08),
    vec2(1.5422108173370361, 8.070904833346049e-09),
    vec2(1.5759809017181396, -5.661025426206834e-08),
    vec2(1.610490322113037, 9.836217174097328e-09),
    vec2(1.6457555294036865, -5.124972091152813e-08),
    vec2(1.6817928552627563, -2.4755326677450284e-08),
    vec2(1.7186193466186523, -4.8496175963919086e-08),
    vec2(1.7562521696090698, -9.235770370707996e-09),
    vec2(1.7947090864181519, -1.141504490931311e-08),
    vec2(1.8340080976486206, -1.1239277952768134e-08),
    vec2(1.8741676807403564, -4.663005626071026e-08),
    vec2(1.9152065515518188, 9.845328108326612e-09),
    vec2(1.9571441411972046, -1.7021804410433106e-08)
);
float libm_exp(float x) {
    if (isnan(x) || isinf(x) || abs(x) > 80.0) return exp_accurate(x);
    vec2 z = libm_mul(vec2(x, 0.0), vec2(46.16624069213867, 6.163081707200035e-07));
    float k = roundEven(z.x + z.y);
    vec2 r = libm_add(z, vec2(-k, 0.0));
    vec2 z0 = libm_add(libm_mul(vec2(1.6938358839979628e-06, 4.109405754592656e-14), r),
        vec2(0.00023459810472559184, -6.8305018759151626e-12));
    vec2 y = libm_add(libm_mul(vec2(0.021660849452018738, -5.5405603716085494e-11), r), vec2(1.0, 0.0));
    y = libm_add(libm_mul(z0, libm_mul(r, r)), y);
    int ki = int(k);
    int n = ki >> 5;
    y = libm_mul(y, libm_exp_table[ki & 31]);
    precise float value = y.x + y.y;
    return value * uintBitsToFloat(uint(n + 127) << 23u);
}

const vec2 libm_log_invc[16] = vec2[16](
    vec2(1.39890718460083, -2.2454301529251097e-08),
    vec2(1.3403141498565674, 3.9807233065403125e-08),
    vec2(1.2864322662353516, -5.611123654603034e-08),
    vec2(1.2367150783538818, -5.692689342140511e-08),
    vec2(1.1906976699829102, 4.668826392162373e-08),
    vec2(1.147982120513916, -1.8458273132182512e-08),
    vec2(1.1082251071929932, 3.7634222849192156e-08),
    vec2(1.0711297988891602, -5.758342069839273e-08),
    vec2(1.0364372730255127, 5.95177063189567e-09),
    vec2(1.0, 0.0),
    vec2(0.9492859840393066, -4.465400937903041e-09),
    vec2(0.8951049447059631, -1.8450627692345734e-09),
    vec2(0.8476821780204773, -1.5985367696202957e-08),
    vec2(0.8050314784049988, 6.76420119916088e-09),
    vec2(0.7664670944213867, 6.46292397377124e-09),
    vec2(0.7314286231994629, -1.9883135138343278e-08)
);

const vec2 libm_log_c[16] = vec2[16](
    vec2(-0.33569133281707764, -5.117452017699975e-10),
    vec2(-0.2929040491580963, -7.2193109268425815e-09),
    vec2(-0.2518726587295532, 6.358163440189912e-10),
    vec2(-0.21245868504047394, -3.0306985809858134e-09),
    vec2(-0.1745394468307495, -5.006707048949011e-09),
    vec2(-0.13800570368766785, -3.544307958236459e-09),
    vec2(-0.10275976359844208, -3.3870093396615175e-09),
    vec2(-0.06871392577886581, 1.3086605132883733e-09),
    vec2(-0.035789139568805695, 8.289828823393464e-10),
    vec2(0.0, 0.0),
    vec2(0.052045177668333054, -2.390380948735782e-10),
    vec2(0.11081431061029434, 2.3775850177543134e-09),
    vec2(0.1652495265007019, -4.131187836264871e-09),
    vec2(0.2168738842010498, 6.115949968688028e-09),
    vec2(0.26596349477767944, 8.034460208250493e-09),
    vec2(0.31275567412376404, -7.716407957047977e-09)
);

float libm_log(float x) {
    if (x <= 0.0 || isinf(x) || isnan(x) || x < uintBitsToFloat(0x00800000u)) return log(x);
    uint ix = floatBitsToUint(x);
    uint tmp = ix - 0x3f330000u;
    uint i = (tmp >> 19u) & 15u;
    int k = int(tmp) >> 23;
    float z = uintBitsToFloat(ix - (tmp & 0xff800000u));
    vec2 r = libm_add(libm_mul(vec2(z, 0.0), libm_log_invc[i]), vec2(-1.0, 0.0));
    vec2 y0 = libm_add(libm_log_c[i], libm_mul(vec2(float(k), 0.0), vec2(0.6931471824645996, -1.9046542121259336e-09)));
    vec2 r2 = libm_mul(r, r);
    vec2 y = libm_add(libm_mul(vec2(0.33345675468444824, 1.1059618110209612e-08), r), vec2(-0.4999997615814209, 1.3001210774632455e-08));
    y = libm_add(libm_mul(vec2(-0.25089341402053833, -8.121832983931654e-09), r2), y);
    y = libm_add(libm_mul(y, r2), libm_add(y0, r));
    return y.x + y.y;
}

float exp_reference(float x) {
    return host_glibc_libm ? libm_exp(x) : exp_accurate(x);
}

float log_reference(float x) {
    return host_glibc_libm ? libm_log(x) : log(x);
}
#endif
