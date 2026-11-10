"""Wait for the two light workers, then audit and publish a hash manifest."""
import os,sys,time,json,subprocess,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY='/home/ubuntu/host_share/kumail_mad/.venv/bin/python'
def log(*a):print(*a,flush=True)
def main():
    status=[HERE/'logs/worker_A_status.json',HERE/'logs/worker_B_status.json']
    while not all(p.exists() for p in status):
        time.sleep(30)
    problems=[]
    for p in status:
        v=json.loads(p.read_text());problems+=v.get('errors',[])
    if problems:
        (HERE/'logs/finalize_error.json').write_text(json.dumps(problems,indent=2))
        log('FINALIZE_STOPPED: worker errors; inspect finalize_error.json')
        return 1
    log('WORKERS_COMPLETE; running audit')
    for script in ['audit.py','scale_report.py']:
        r=subprocess.run([PY,'-B',str(HERE/script)],cwd=str(HERE),check=False)
        if r.returncode:
            (HERE/'logs/finalize_error.json').write_text(json.dumps({'script':script,'exit_code':r.returncode},indent=2))
            log('FINALIZE_STOPPED:',script,'exit',r.returncode)
            return r.returncode
    (HERE/'finalize_status.json').write_text(json.dumps({'status':'complete','workers':2,'manifest':'SHA256SUMS'},indent=2))
    log('Analysis complete; writing SHA-256 manifest as final operation.')
    sys.stdout.flush();sys.stderr.flush()
    target=HERE/'SHA256SUMS'
    if target.exists():raise RuntimeError('Manifest already exists; refusing to replace it')
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and not p.is_symlink() and p!=target)
    with target.open('x') as out:
        for p in files:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            out.write(h.hexdigest()+'  '+str(p.relative_to(HERE)).replace(os.sep,'/')+'\n')
    (HERE/'MANIFEST_READY').write_text('complete\n')
    return 0
if __name__=='__main__':sys.exit(main())
