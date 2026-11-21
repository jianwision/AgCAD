"""Verify saved session evidence and optionally package sources plus CAD results.

Artifact verification does not rerun the acceptance tests or invoke an LLM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
V6_SHA256='c1307c5c53e32269554c1109c85fd6eb448663ce8dd87316657c4a7a6f5d4fac'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def verify():
    evidence={}
    for mode in ('demo','cuda'):
        directory=ROOT/'output/sessions'/mode
        report=load(directory/'validation.json');worker=load(directory/'worker-report.json')
        assert report['pass'] and report['goal_met'] and report['cad']['pass']
        assert all(report['checks'].values())
        assert report['goal_criterion']['required_for_delivery']
        assert report['cad']['faces']==4 and report['cad']['solids']==0
        assert report['cad']['probe_count']==324 and report['cad']['probe_max_mm']<1e-8
        assert digest(directory/'session_hood.step')==report['cad']['sha256']
        assert digest(ROOT/'output/diagnostics/model/probe_policy.f64')==report['weights_sha256']
        for source,expected in report['sources'].items():assert digest(ROOT/source)==expected,source
        assert report['worker']==worker
        assert worker['context_creations']==worker['template_uploads']==1
        assert worker['geometry_downloads']==1 and worker['budget_reserved']<=worker['budget_limit']
        assert report['device_model_inferences']==16 and not report['external_model_service_invoked']
        events=load(directory/'ipc-events.json');model_events=load(directory/'model-events.json')
        rollouts=[e for e in events if e['operation']==2]
        assert len(rollouts)==1 and rollouts[0]['response_bytes']==32+8*(64+26*16)
        assert model_events['device_model_inferences']==16
        if mode=='demo':
            assert report['default_demo_is_recorded_proposals_not_live_llm']
            assert len(report['switches'])==1 and report['cpu_gpu_model_proposal_max_mm']<1e-9
        evidence[mode]={'validation_sha256':digest(directory/'validation.json'),
                        'step_sha256':report['cad']['sha256'],'goal_error_mm':report['probe_max_error_mm']}
    links=0
    for name in ('MODEL_SESSIONS.md','SELF_IMPROVEMENT_REVIEW_V2.md','README.md'):
        source=(ROOT/name).read_text(encoding='utf-8')
        assert source.count('```')%2==0,name
        for link in re.findall(r'\]\(([^)]+)\)',source):
            if re.match(r'^[a-zA-Z]+:',link) or link.startswith('#'):continue
            assert (ROOT/link.split('#',1)[0]).exists(),(name,link)
            links+=1
    old=ROOT/'output/delivery/rust-cuda-modeling-demo-v6.zip'
    preserved=digest(old)==V6_SHA256 if old.exists() else None
    assert preserved is not False
    receipt={'pass':True,'artifact_verification_only':True,'sessions':evidence,
             'markdown_links_checked':links,'existing_v6_unchanged':preserved,
             'external_llm_invoked':False,'general_primitive_dag_integrated':False}
    path=ROOT/'output/sessions/delivery-verification.json'
    path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def package(verification):
    files=set()
    patterns=['*.md','primitive_*.json','Cargo.toml','Cargo.lock','run.ps1','cuda/*.cuh','cuda/*.cu',
              'src/**/*.rs','scripts/*.py','configs/*.json','trainer/Cargo.toml','trainer/Cargo.lock',
              'trainer/src/**/*.rs','models/*']
    for pattern in patterns:files.update(p for p in ROOT.glob(pattern) if p.is_file())
    suffixes={'.json','.html','.step','.stl','.jpg','.png','.cu','.ptx','.bin','.npz','.log','.f64'}
    for name in ('primitives','car','car-variant','constraints','harness','diagnostics','planning'):
        files.update(p for p in (ROOT/'output'/name).rglob('*') if p.is_file() and p.suffix in suffixes)
    for name in ('demo','cuda'):
        files.update(p for p in (ROOT/'output/sessions'/name).iterdir() if p.is_file() and p.suffix in suffixes)
    files.add(ROOT/'output/sessions/delivery-verification.json')
    files=sorted(p for p in files if p.name not in {'task_plan.md','progress.md','findings.md','solidworks_validation.json'})
    dest=ROOT/'output/delivery';dest.mkdir(exist_ok=True)
    manifest={'release':'model-sessions-v1','scope':'Persistent model-driven fixed-hood sessions; explicit CPU STEP backend',
              'verification':verification,'files':[{'path':p.relative_to(ROOT).as_posix(),
                  'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]}
    manifest_path=dest/'manifest-model-sessions-v1.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path=dest/'rust-cuda-modeling-sessions-v1.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in files:archive.write(file,file.relative_to(ROOT).as_posix())
        archive.write(manifest_path,manifest_path.name)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for entry in manifest['files']:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256']
    receipt={'file':str(path),'bytes':path.stat().st_size,'files':len(files)+1,'sha256':digest(path),
             'zip_crc_passed':True,'all_manifest_hashes_passed':True}
    (dest/'receipt-model-sessions-v1.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',action='store_true')
    args=parser.parse_args();result=verify()
    if args.package:result['package']=package(result)
    print(json.dumps(result,indent=2))
