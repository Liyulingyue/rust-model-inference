#!/usr/bin/env python3
"""Small deterministic media requests; run on the model host with FFmpeg on PATH."""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((64, 64))
    image = np.stack(((xx*3+yy)%256, (yy*3)%256, (xx+yy)%256), -1).astype(np.uint8)
    image[16:48, 16:48] = [240, 20, 20]
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    inline = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    frames = np.stack([np.roll(image, i*3, axis=1) for i in range(4)])
    clip = output / "moving-box.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "64x64", "-r", "2", "-i", "pipe:0", "-c:v", "ffv1", str(clip)],
        input=frames.tobytes(), check=True)
    video = "data:video/x-matroska;base64," + base64.b64encode(clip.read_bytes()).decode()
    requests = {
        "image": dict(state="A test image.", images=[inline], questions={"q":dict(type="noul", instructions="Is a red box visible?")}),
        "box": dict(state="A test image.", assets={"@picture":inline}, questions={"q":dict(type="box", instructions="The red box in @picture")}),
        "video": dict(state="A short clip.", videos=[video], questions={"q":dict(type="noul", instructions="Is a red box moving?")}),
        "video-box": dict(state="A short clip.", videos=[video], questions={"q":dict(type="box", instructions="The red box", at=0.25)}),
        "track": dict(state="A short clip.", videos=[video], questions={"q":dict(type="track", instructions="The red box", rate=1)}),
    }
    for name, request in requests.items():
        (output / f"{name}.json").write_text(json.dumps(request))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
