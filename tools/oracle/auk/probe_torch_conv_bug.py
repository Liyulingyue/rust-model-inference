"""Minimal reproduction of the PyTorch failure in the official AuK VAE.

The official alias-free resampler calls

    F.conv1d(x, self.filter.expand(C, -1, -1), stride=2, groups=C)

The weight is an *expanded* tensor, so its strides are (0, K, 1). PyTorch's
grouped convolution then hits an internal `at::parallel_for` chunking bug and
libgomp aborts with "illegal immediate parameter (range error)" once the
output is large enough.

This script shows:
  * the failure depends only on the output size, not on AuK;
  * making the weight contiguous avoids it (the upstream fix would be
    `self.filter.expand(C, -1, -1).contiguous()`);
  * the Rust AuK VAE port implements the same math directly and has no such
    length limit.
"""
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

K = 12
C = 24
STRIDE = 2
FILTER = torch.randn(1, 1, K)
FILTER = FILTER / FILTER.sum()


def try_conv(length: int, contiguous: bool) -> str:
    x = torch.randn(1, C, length)
    weight = FILTER.expand(C, -1, -1)
    if contiguous:
        weight = weight.contiguous()
    try:
        out = F.conv1d(x, weight, stride=STRIDE, groups=C)
        return f"OK out={tuple(out.shape)} numel={out.numel()}"
    except RuntimeError as error:
        return f"FAILED -> {error}"


print(f"weight strides (expanded) = {FILTER.expand(C, -1, -1).stride()}")
print()
for frames in (50, 75, 100, 150, 200, 400):
    length = frames * 480 + 11
    print(f"latent frames {frames:>4} (x len {length:>6}): "
          f"expanded={try_conv(length, contiguous=False)}")
    print(f"{'':<22}contiguous={try_conv(length, contiguous=True)}")
