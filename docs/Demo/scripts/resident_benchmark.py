"""Compare the saved one-shot API with resident calls on the same surface/grid.

Measured end-to-end calls; this is a workflow amortization comparison, not a
matched kernel throughput benchmark. Summary calls deliberately return less data.
"""
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from class_a import gpu_batch
from engineering_fixtures import polynomial_patch
from resident_surface import SurfaceWorker

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/resident-surfaces/benchmark'


def main():
    OUT.mkdir(parents=True,exist_ok=True);patch=polynomial_patch(5,bump=2);cold=[]
    for i in range(3):
        start=time.perf_counter();legacy,_=gpu_batch([patch],OUT/f'legacy-{i}',samples=17);cold.append((time.perf_counter()-start)*1000)
    start=time.perf_counter()
    with SurfaceWorker(OUT/'resident') as worker:
        startup_ms=(time.perf_counter()-start)*1000;ref=worker.upload(patch);full=worker.analyze(ref,full=True,samples=17)
        maximum=float(np.max(np.abs(full[:,:45]-legacy[0]['grid'].reshape(-1,48)[:,:45])))
        if maximum>1e-8:raise RuntimeError('Legacy/resident differential outputs disagree')
        full_times=[];summary_times=[];before=worker.stats()
        for _ in range(7):
            start=time.perf_counter();worker.analyze(ref,full=True,samples=17);full_times.append((time.perf_counter()-start)*1000)
        for _ in range(7):
            start=time.perf_counter();worker.analyze(ref,samples=17);summary_times.append((time.perf_counter()-start)*1000)
        after=worker.stats()
        report={'pass':True,'comparison':'same degree-5 polynomial patch, 17x17 grid, common fields 0..44; per-call wall time',
            'legacy_cold_process_ms':cold,'legacy_median_ms':statistics.median(cold),'resident_startup_ms':startup_ms,'resident_startup':worker.startup,
            'resident_full_grid_warm_ms':full_times,'resident_full_grid_median_ms':statistics.median(full_times),
            'resident_summary_warm_ms':summary_times,'resident_summary_median_ms':statistics.median(summary_times),
            'warm_full_grid_call_ratio':statistics.median(cold)/statistics.median(full_times),'common_field_max_absolute_difference':maximum,
            'warm_upload_bytes':after['upload_bytes']-before['upload_bytes'],'warm_geometry_readbacks':after['geometry_readbacks']-before['geometry_readbacks'],
            'summary_payload_bytes':64,'grid_payload_bytes':17*17*48*8,'resident_stats':after,
            'kernel_speedup_claimed':False,'summary_returns_less_data':True,
            'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/resident_benchmark.py','scripts/resident_surface.py','scripts/nurbs_surface.py','cuda/resident_surface.cu','src/bin/surface_worker.rs','scripts/class_a.py']}}
        if report['warm_upload_bytes'] or report['warm_geometry_readbacks']:raise RuntimeError('Unexpected warm transfer')
        (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
