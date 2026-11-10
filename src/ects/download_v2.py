"""Fetch and verify the ECTS v2 UCR splits, retaining all v1 data."""
import concurrent.futures,json
from .download import run,DATA
NAMES=('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop','GunPoint','Coffee')
def main():
    result=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for entry in pool.map(run,NAMES):
            print('DATASET_READY',entry['name'],flush=True);result.append(entry)
    manifest=DATA/'manifest_v2.json'
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        original={f['path']:f['sha256'] for d in prior['datasets'] for f in d['files']}
        current={f['path']:f['sha256'] for d in result for f in d['files']}
        if original!=current:raise ValueError('Existing manifest does not match verified data')
        print('MANIFEST_VERIFIED',flush=True)
    else:
        with manifest.open('x') as f:
            json.dump({'source':'ChengsenWang/UCR-2018 at hf-mirror.com','datasets':result},f,indent=2)
if __name__=='__main__':main()
