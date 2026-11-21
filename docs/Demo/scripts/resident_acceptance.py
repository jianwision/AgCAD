"""Run the resident-surface suite and retain per-case/source evidence."""
import hashlib
import io
import json
from pathlib import Path
import unittest
from engineering_acceptance import EvidenceResult
from resident_demo import SOURCES

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'scripts'),pattern='test_resident_surfaces.py')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=EvidenceResult).run(suite)
    directory=ROOT/'output/resident-surfaces';directory.mkdir(exist_ok=True)
    (directory/'tests.log').write_text(log.getvalue(),encoding='utf-8')
    report={'pass':result.wasSuccessful(),'tests_run':result.testsRun,'cases':result.cases,
            'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [*SOURCES,'scripts/test_resident_surfaces.py','scripts/resident_acceptance.py']}}
    (directory/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(log.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
