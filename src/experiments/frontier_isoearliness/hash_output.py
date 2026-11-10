"""Write and verify a SHA-256 manifest for the complete output directory."""
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=root/'SHA256SUMS'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
entries=[]
for path in sorted(root.rglob('*')):
    if path.is_symlink():raise ValueError('Symlink not supported: '+str(path))
    if path.is_file() and path!=manifest:
        rel=path.relative_to(root).as_posix()
        if '\n' in rel or '\r' in rel:raise ValueError('Invalid manifest path')
        entries.append((digest(path),rel))
with manifest.open('x') as f:
    for h,rel in entries:f.write(h+'  '+rel+'\n')
for h,rel in entries:
    if digest(root/rel)!=h:raise ValueError('Output changed during hashing: '+rel)
print('OUTPUT_HASHES_VERIFIED',len(entries),'MANIFEST_SHA256',digest(manifest),flush=True)
