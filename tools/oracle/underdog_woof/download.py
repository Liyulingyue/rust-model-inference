"""Download pinned conversion/oracle inputs directly onto the target machine."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

from tools.converter.underdog_woof.convert_underdog_woof import CONTRACT, sha256


def download(directory):
    source = CONTRACT["source"]
    for name, expected in source["hashes"].items():
        path = Path(directory) / name
        if path.is_file() and sha256(path) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://modelscope.cn/models/{source['repository']}/resolve/{source['revision']}/{name}"
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Underdog-Woof source hash mismatch: {name}")
            if name == "woof_int8.npz" and os.stat(temporary).st_size != source["weight_size"]:
                raise ValueError("Underdog-Woof weight size mismatch")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    download(parser.parse_args().directory)
