"""Download only the pinned PiVo conversion/oracle assets from ModelScope."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def matches(path, row):
    return (path.is_file() and path.stat().st_size == row['Size']
            and hashlib.sha256(path.read_bytes()).hexdigest() == row['Sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name('sources.json').read_text())
    for row in pins['files']:
        path = args.directory / row['Path']
        if not matches(path, row):
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://modelscope.cn/models/{pins['repo']}/resolve/{row['Revision']}/{row['Path']}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if len(data) != row['Size'] or hashlib.sha256(data).hexdigest() != row['Sha256']:
                raise ValueError(f'PiVo download size/hash mismatch: {path}')
            path.write_bytes(data)
        assert matches(path, row), path
        print(row['Path'])


if __name__ == '__main__':
    main()
