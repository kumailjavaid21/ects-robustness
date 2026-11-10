"""Finalize only after complete no-retraining replay; hash every output before hand-off."""
import os,sys,time,json,subprocess,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY='/home/ubuntu/host_share/kumail_mad/.venv/bin/python'
def log(*a):print(*a,flush=True)
def main():
    status=[HERE/'logs/worker_A_status.json',HERE/'logs/worker_B_status.json']
    while not all(p.exists() for p in status):time.sleep(30)
    errors=[]
    for p in status:errors+=json.loads(p.read_text()).get('errors',[])
    if errors:
        (HERE/'logs/finalize_error.json').write_text(json.dumps(errors,indent=2));log('FINALIZE_STOPPED worker errors');return 1
    if not (HERE/'posthoc_stats.csv').exists():
        (HERE/'logs/finalize_error.json').write_text(json.dumps({'error':'Fixed-point analysis missing'},indent=2));log('FINALIZE_STOPPED fixed-point analysis missing');return 1
    log('WORKERS_COMPLETE; running audit')
    for script in ('audit.py','report.py'):
        v=subprocess.run([PY,'-B',str(HERE/script)],cwd=str(HERE),check=False)
        if v.returncode:
            (HERE/'logs/finalize_error.json').write_text(json.dumps({'script':script,'exit_code':v.returncode},indent=2));log('FINALIZE_STOPPED',script,v.returncode);return v.returncode
    (HERE/'finalize_status.json').write_text(json.dumps({'status':'complete','retraining':False,'manifest':'SHA256SUMS'},indent=2))
    log('Analysis complete; writing SHA-256 manifest as final operation.')
    sys.stdout.flush();sys.stderr.flush();target=HERE/'SHA256SUMS'
    if target.exists():raise RuntimeError('Manifest already exists')
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and not p.is_symlink() and p!=target)
    with target.open('x') as out:
        for p in files:
            h=hashlib.sha256()
            with p.open('rb') as inp:
                for b in iter(lambda:inp.read(1024*1024),b''):h.update(b)
            out.write(h.hexdigest()+'  '+str(p.relative_to(HERE)).replace(os.sep,'/')+'\n')
    (HERE/'MANIFEST_READY').write_text('complete\n')
    return 0
if __name__=='__main__':sys.exit(main())
