"""Resume bounded ranges of the official wheel; verify PyPI SHA-256 before use."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/'ocp-package-index.json').read_text())
wheel=next(f for f in meta['urls'] if 'cp313-cp313-win_amd64' in f['filename'])
parts=ROOT/'.downloads';parts.mkdir(exist_ok=True)
size=wheel['size'];chunk=2*1024*1024


def fetch(index):
    start=index*chunk;end=min(size,start+chunk)-1
    path=parts/f'ocp-{index:02}.part';expected=end-start+1
    for attempt in range(10):
        got=path.stat().st_size if path.exists() else 0
        if got==expected:return path
        assert got<expected
        try:
            req=urllib.request.Request(wheel['url'],headers={'Range':f'bytes={start+got}-{end}'})
            with urllib.request.urlopen(req,timeout=30) as response, path.open('ab') as out:
                assert response.status==206
                while data:=response.read(16384):out.write(data)
        except Exception as exc:
            print(f'part {index}: retry {attempt+1}, {type(exc).__name__}',flush=True)
    raise RuntimeError(f'Incomplete part {index}')


with ThreadPoolExecutor(max_workers=12) as pool:
    paths=list(pool.map(fetch,range((size+chunk-1)//chunk)))
dest=ROOT/wheel['filename']
with dest.open('wb') as out:
    for path in paths:out.write(path.read_bytes())
digest=hashlib.sha256(dest.read_bytes()).hexdigest()
assert dest.stat().st_size==size and digest==wheel['digests']['sha256']
print(json.dumps({'file':str(dest),'sha256':digest,'verified':True}),flush=True)
