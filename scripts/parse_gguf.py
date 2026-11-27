import struct, sys

with open('models/Bonsai-8B-q1.gguf', 'rb') as f:
    magic = f.read(4)
    version, n_kv, n_tensors = struct.unpack('<III', f.read(12))
    print(f'GGUF v{version}, {n_kv} KV, {n_tensors} tensors')
    for _ in range(min(n_kv, 30)):
        klen = struct.unpack('<I', f.read(4))[0]
        key = f.read(klen).decode('utf-8', errors='replace')
        vtype = struct.unpack('<I', f.read(4))[0]
        if vtype == 8:
            vlen = struct.unpack('<I', f.read(4))[0]
            val = f.read(vlen).decode('utf-8', errors='replace')
            print(f'{key} = {val}')
        elif vtype == 4:
            val = struct.unpack('<I', f.read(4))[0]
            print(f'{key} = {val}')
        elif vtype == 5:
            val = struct.unpack('<i', f.read(4))[0]
            print(f'{key} = {val}')
        elif vtype == 10:
            val = struct.unpack('<Q', f.read(8))[0]
            print(f'{key} = {val}')
        elif vtype == 6:
            val = struct.unpack('<f', f.read(4))[0]
            print(f'{key} = {val:.6}')
        elif vtype == 7:
            val = struct.unpack('<?', f.read(1))[0]
            print(f'{key} = {val}')
        elif vtype == 2:
            atype = struct.unpack('<I', f.read(4))[0]
            alen = struct.unpack('<I', f.read(4))[0]
            print(f'{key} = array(type={atype}, len={alen})')
            if atype == 8:
                for _ in range(alen):
                    vlen = struct.unpack('<I', f.read(4))[0]
                    f.read(vlen)
            elif atype == 4:
                f.read(alen * 4)
            elif atype == 5:
                f.read(alen * 4)
            elif vtype == 6:
                f.read(alen * 4)
            elif atype == 7:
                f.read(alen)
            elif atype == 10:
                f.read(alen * 8)
            else:
                break
        else:
            print(f'{key} = (type {vtype})')
            break
