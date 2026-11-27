"""Stream the pinned ModelScope release directly to the host running this script."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = '287c5fdc8c83e43c4d207d21bc51024a52fdff38'
REPO = 'https://modelscope.cn/models/microsoft/Mage-VL-Cowork'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    root = p.parse_args().destination
    root.mkdir(parents=True,exist_ok=True)
    with urlopen(f'https://modelscope.cn/api/v1/models/microsoft/Mage-VL-Cowork/repo/files?Revision={REVISION}&Recursive=true',timeout=120) as response:
        manifest = json.load(response)
    (root/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
    for entry in manifest['Data']['Files']:
        name = entry['Path']
        if entry['Type']!='blob' or '/' in name or name in {'.','..','.gitattributes'}:
            continue
        path = root/name
        if path.exists():
            with path.open('rb') as source:
                digest = hashlib.file_digest(source,'sha256').hexdigest()
            if path.stat().st_size==entry['Size'] and digest==entry['Sha256']:
                continue
            raise ValueError(f'{path} exists with different contents')
        temporary = path.with_name(path.name+'.part')
        digest = hashlib.sha256()
        size = 0
        try:
            with urlopen(f'{REPO}/resolve/{REVISION}/{name}',timeout=120) as source,temporary.open('wb') as destination:
                while data:=source.read(8*1024*1024):
                    destination.write(data)
                    digest.update(data)
                    size += len(data)
            if size!=entry['Size'] or digest.hexdigest()!=entry['Sha256']:
                raise ValueError(f'{name}: size/SHA256 mismatch')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(name,size,digest.hexdigest(),flush=True)


if __name__=='__main__':
    main()
