"""Create the final SHA256 manifest only after training, reporting and source audit succeed."""
import json,hashlib
from pathlib import Path
from settings import *
assert (OUT/'ANALYSIS_COMPLETE').exists()
status=json.loads((OUT/'orchestration_status.json').read_text());assert not status['failed']
audit=json.loads((OUT/'source_audit.json').read_text());assert not any(audit[k] for k in ('changed','removed','added'))
result=json.loads((OUT/'artifact_audit.json').read_text());assert result['networks']==168
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
