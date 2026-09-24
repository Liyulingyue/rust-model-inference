import struct, sys

for path in ['models/Fun-ASR-Nano-GGUF/paraformer-q8.gguf',
             'models/Fun-ASR-Nano-GGUF/sensevoice-small-q8.gguf']:
    print(f"\n=== {path} ===")
    with open(path, 'rb') as f:
        magic = f.read(4)
        version, n_kv, n_tensors = struct.unpack('<III', f.read(12))
        print(f'GGUF v{version}, {n_kv} KV, {n_tensors} tensors')
        for _ in range(n_kv):
            klen = struct.unpack('<I', f.read(4))[0]
            key = f.read(klen).decode('utf-8', errors='replace')
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 8:  # string
                vlen = struct.unpack('<I', f.read(4))[0]
                val = f.read(vlen).decode('utf-8', errors='replace')
                print(f'  {key} = {val}')
            elif vtype == 4:  # uint32
                val = struct.unpack('<I', f.read(4))[0]
                print(f'  {key} = {val}')
            elif vtype == 5:  # int32
                val = struct.unpack('<i', f.read(4))[0]
                print(f'  {key} = {val}')
            elif vtype == 10:  # uint64
                val = struct.unpack('<Q', f.read(8))[0]
                print(f'  {key} = {val}')
            elif vtype == 6:  # float32
                val = struct.unpack('<f', f.read(4))[0]
                print(f'  {key} = {val:.6}')
            elif vtype == 7:  # bool
                val = struct.unpack('<?', f.read(1))[0]
                print(f'  {key} = {val}')
            elif vtype == 2:  # array
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<I', f.read(4))[0]
                print(f'  {key} = array(type={atype}, len={alen})')
                if atype == 8:
                    for _ in range(min(alen, 4)):
                        vlen = struct.unpack('<I', f.read(4))[0]
                        v = f.read(vlen).decode('utf-8', errors='replace')
                        print(f'    [{v}]')
                    for _ in range(max(0, alen - 4)):
                        vlen = struct.unpack('<I', f.read(4))[0]
                        f.read(vlen)
                elif atype == 4:
                    f.read(alen * 4)
                elif atype == 5:
                    f.read(alen * 4)
                elif atype == 6:
                    f.read(alen * 4)
                elif atype == 7:
                    f.read(alen)
                elif atype == 10:
                    f.read(alen * 8)
                else:
                    print(f'    UNKNOWN array type {atype}')
                    break
            else:
                print(f'  {key} = (type {vtype})')
                break
        # Print first 10 tensor names
        print('  First 10 tensors:')
        for i in range(min(n_tensors, 10)):
            tlen = struct.unpack('<I', f.read(4))[0]
            tname = f.read(tlen).decode('utf-8', errors='replace')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack('<Q', f.read(8))[0])
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            print(f'    {tname} dims={dims} type={ttype}')
