import json,hashlib
from settings import *
assert (OUT/'ANALYSIS_COMPLETE').exists()
a=json.loads((OUT/'source_audit.json').read_text());assert not any(a[k] for k in ['changed','removed','added'])
r=json.loads((OUT/'artifact_audit.json').read_text());assert r['model_artifacts']==260
files=[]
for p in sorted(OUT.rglob('*')):
 if p.is_symlink():raise RuntimeError('Unexpected symlink '+str(p))
 if p.is_file() and p.name not in ('SHA256SUMS','MANIFEST_READY'):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  files.append(h.hexdigest()+'  '+p.relative_to(OUT).as_posix())
with (OUT/'SHA256SUMS').open('x') as f:f.write('\n'.join(files)+'\n')
with (OUT/'MANIFEST_READY').open('x') as f:f.write('complete\n')
print('MANIFEST_READY',len(files),'files',flush=True)
