"""Run engineering acceptance with a machine-readable per-case receipt."""
import hashlib
import io
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class EvidenceResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.cases=[]
    def addSuccess(self,test):super().addSuccess(test);self.cases.append({'test':test.id(),'pass':True})
    def addFailure(self,test,error):super().addFailure(test,error);self.cases.append({'test':test.id(),'pass':False})
    def addError(self,test,error):super().addError(test,error);self.cases.append({'test':test.id(),'pass':False})


if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'scripts'),pattern='test_engineering_support.py')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=EvidenceResult).run(suite)
    directory=ROOT/'output/engineering';directory.mkdir(exist_ok=True)
    (directory/'tests.log').write_text(log.getvalue(),encoding='utf-8')
    report={'pass':result.wasSuccessful(),'tests_run':result.testsRun,'cases':result.cases,
            'test_source_sha256':hashlib.sha256((ROOT/'scripts/test_engineering_support.py').read_bytes()).hexdigest()}
    paths=['scripts/class_a.py','scripts/design_intent.py','scripts/tolerance_topology.py','scripts/engineering_runtime.py',
           'scripts/model_session.py','scripts/model_adaptor.py','scripts/session_protocol.py','cuda/class_a.cu','src/bin/class_a_suite.rs']
    report['sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    (directory/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(log.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
