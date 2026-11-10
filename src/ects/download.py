"""Acquire the requested UCR splits from a verified public Hugging Face mirror."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/ucr'
NAMES=('GunPoint','ECG200','Coffee','ItalyPowerDemand','TwoLeadECG','Wafer')
API='https://hf-mirror.com/api/datasets/ChengsenWang/UCR-2018/tree/main/'
BASE='https://hf-mirror.com/datasets/ChengsenWang/UCR-2018/resolve/main/'
def run(name):
    response=requests.get(API+name,timeout=30);response.raise_for_status()
    byname={x['path'].split('/')[-1]:x for x in response.json()}
    destination=DATA/name;destination.mkdir(parents=True,exist_ok=True);metadata=[]
    for split in ('train','test'):
        key=split+'.parquet';entry=byname[key];url=BASE+entry['path'];expected=entry['lfs']['oid']
        target=destination/key
        if target.exists():
            actual=hashlib.sha256(target.read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('Existing file differs: '+str(target))
        else:
            for attempt in range(4):
                try:
                    r=requests.get(url,timeout=(15,90));r.raise_for_status();blob=r.content
                    actual=hashlib.sha256(blob).hexdigest()
                    if actual!=expected:raise ValueError(f'Checksum mismatch for {url}: {actual} vs {expected}')
                    with target.open('xb') as f:f.write(blob)
                    break
                except (requests.RequestException,ValueError) as exc:
                    print('DOWNLOAD_RETRY',name,split,attempt+1,repr(exc),flush=True)
                    if attempt==3:raise
                    time.sleep(2*(attempt+1))
        metadata.append({'split':split,'path':str(target),'url':url,'sha256':expected,'bytes':entry['size']})
    return {'name':name,'files':metadata}
def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=[]
        for result in pool.map(run,NAMES):
            results.append(result);print('DATASET_DOWNLOADED',result['name'],flush=True)
    path=DATA/'manifest.json'
    with path.open('x') as f:json.dump({'source':'Hugging Face mirror of ChengsenWang/UCR-2018','datasets':results},f,indent=2)
if __name__=='__main__':main()
