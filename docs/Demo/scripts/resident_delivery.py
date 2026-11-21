"""Verify saved resident evidence and package source/results without rerunning tests."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
import engineering_delivery as previous
from engineering_delivery import digest,load,require,verify_sources
import tolerance_topology as topo
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone

ROOT=Path(__file__).resolve().parents[1]
ENGINEERING_ZIP_SHA256='6fc04a6f56ed64e089b428f7dae948a5bb7297c02925bb37a58ab9abc4b16135'


def verify():
    directory=ROOT/'output/resident-surfaces';acceptance=load(directory/'acceptance.json')
    require(acceptance['pass'] and acceptance['tests_run']==18,'Resident acceptance incomplete')
    require(len(acceptance['cases'])==18 and all(c['pass'] for c in acceptance['cases']),'Resident case failed')
    verify_sources(acceptance)
    demo=load(directory/'demo/validation.json');benchmark=load(directory/'benchmark/validation.json')
    for report in (demo,benchmark):require(report['pass'],'Demo/benchmark failed');verify_sources(report)
    require(digest(directory/'demo/joint_g3_panel.step')==demo['cad']['sha256'],'Changed delivered STEP')
    require(demo['cad']['pass'] and all(c['pass'] for c in demo['cad']['user_contract_checks']),'Final contract check failed')
    require(demo['resident_stats']['context_creations']==1,'Demo did not reuse context')
    require(demo['resident_stats']['graph_builds']==1 and demo['resident_stats']['graph_launches']==256,'Missing graph execution')
    require(demo['optimization']['intermediate_geometry_readbacks']==0,'Unexpected intermediate geometry transfer')
    require(demo['optimization']['after']['objective']<demo['optimization']['before']['objective'],'Objective did not improve')
    require(demo['independent_dense_quality']['curvature_gradient_reduction_fraction']>0,'Denser independent quality failed')
    require(not demo['class_a_certified'] and not demo['external_llm_invoked'] and not demo['new_cumodel_training_performed'],'Incorrect provenance/scope')
    require(benchmark['warm_upload_bytes']==benchmark['warm_geometry_readbacks']==0,'Unexpected warm transfers')
    require(benchmark['summary_payload_bytes']==64 and benchmark['common_field_max_absolute_difference']<1e-8,'Benchmark equivalence/accounting failed')
    require(not benchmark['kernel_speedup_claimed'],'Unsupported kernel speed claim')
    native=load(directory/'native-nurbs-validation.json');require(native['pass'] and native['jet_max_mm']<2e-8,'Native NURBS D3 comparison failed')
    preview=load(directory/'demo/quality-preview.json')
    for key,path in (('source_grid_sha256',directory/'demo/quality-grid.npz'),
                     ('script_sha256',ROOT/'scripts/resident_preview.py'),
                     ('image_sha256',directory/'demo/quality-preview.png')):
        require(digest(path)==preview[key],'Stale preview evidence: '+key)
    require(preview['shared_color_scales'] and preview['physical_3d_aspect'],'Misleading preview scales')
    network=load(directory/'network-validation.json');require(network['pass'],'Joint network test failed')
    require(network['stats_before']['geometry_readbacks']==network['stats_after']['geometry_readbacks'],'Network downloaded geometry during search')
    require(network['stats_after']['download_bytes']-network['stats_before']['download_bytes']==144,'Network final metric readback mismatch')
    for name in ('acceptance-worker','demo/gpu','benchmark/resident'):
        require((directory/name/'worker-source.cu').read_bytes()==(ROOT/'cuda/resident_surface.cu').read_bytes(),'Stale CUDA evidence: '+name)
    worker=load(directory/'demo/gpu/worker-report.json');require(worker['optimization_intermediate_readbacks']==0,'Intermediate readbacks reported')
    steps={}
    for name,faces in (('demo/joint_g3_panel.step',3),('joint-skin.step',3),('rational-skin.step',1)):
        path=directory/name;reader=STEPControl_Reader()
        require(reader.ReadFile(str(path))==IFSelect_RetDone and reader.TransferRoots()>0,'Cannot reimport '+name)
        info=topo.inspect_shape(reader.OneShape())
        require(info['valid'] and not info['nonmanifold_edges'] and info['faces']==faces and info['solids']==0,'Invalid delivered topology: '+name)
        require(info['maximum_native_tolerance_mm']<=topo.TolerancePolicy().maximum_native_mm,'Native tolerance exceeded')
        steps[name]={'sha256':digest(path),'inspection':info}
    previous_result=previous.verify()
    require(digest(ROOT/'output/delivery/rust-cuda-engineering-support-v1.zip')==ENGINEERING_ZIP_SHA256,'Previous engineering ZIP changed')
    source=(ROOT/'RESIDENT_SURFACES.md').read_text(encoding='utf-8');require(source.count('```')%2==0,'Unclosed documentation fence')
    links=0
    for link in re.findall(r'\]\(([^)]+)\)',source):
        if re.match(r'^[a-zA-Z]+:',link) or link.startswith('#'):continue
        require((ROOT/link.split('#',1)[0]).exists(),'Missing resident documentation link: '+link);links+=1
    receipt={'pass':True,'artifact_verification_only':True,'resident_tests':18,'engineering_regression_tests':20,
        'acceptance_sha256':digest(directory/'acceptance.json'),'demo_sha256':digest(directory/'demo/validation.json'),
        'benchmark_sha256':digest(directory/'benchmark/validation.json'),'native_d3_max_mm':native['jet_max_mm'],
        'preview_sha256':preview['image_sha256'],'preview_source_hashes_verified':True,
        'reopened_steps':steps,'previous_artifacts_verified':previous_result['pass'],'previous_engineering_zip_unchanged':True,
        'markdown_links_checked':links+previous_result['markdown_links_checked'],'external_llm_invoked':False,
        'solidworks_import_performed':False,'global_class_a_certified':False,'arbitrary_rational_network_optimization':False}
    (directory/'delivery-verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def package(verification):
    files=set()
    patterns=['*.md','primitive_*.json','Cargo.toml','Cargo.lock','run.ps1','cuda/*.cuh','cuda/*.cu','src/**/*.rs',
        'scripts/*.py','configs/*.json','trainer/Cargo.toml','trainer/Cargo.lock','trainer/src/**/*.rs','models/*']
    for pattern in patterns:files.update(p for p in ROOT.glob(pattern) if p.is_file())
    suffixes={'.json','.html','.step','.stl','.jpg','.png','.cu','.ptx','.bin','.npz','.log','.f64','.txt'}
    for name in ('primitives','car','car-variant','constraints','harness','diagnostics','planning','engineering','resident-surfaces'):
        files.update(p for p in (ROOT/'output'/name).rglob('*') if p.is_file() and p.suffix in suffixes)
    for name in ('demo','cuda'):files.update(p for p in (ROOT/'output/sessions'/name).iterdir() if p.is_file() and p.suffix in suffixes)
    files.add(ROOT/'output/sessions/delivery-verification.json')
    files=sorted(p for p in files if p.name not in {'task_plan.md','progress.md','findings.md','solidworks_validation.json'})
    dest=ROOT/'output/delivery';dest.mkdir(exist_ok=True)
    manifest={'release':'resident-surfaces-v1','scope':'Resident positive-weight NURBS evaluation/refinement; coupled polynomial Bezier boundary-jet optimization; explicit CPU CAD export',
        'verification':verification,'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]}
    manifest_path=dest/'manifest-resident-surfaces-v1.json';manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path=dest/'rust-cuda-resident-surfaces-v1.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in files:archive.write(file,file.relative_to(ROOT).as_posix())
        archive.write(manifest_path,manifest_path.name)
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None,'ZIP CRC failure')
        for entry in manifest['files']:require(hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256'],'ZIP digest mismatch: '+entry['path'])
    receipt={'file':str(path),'bytes':path.stat().st_size,'files':len(files)+1,'sha256':digest(path),'zip_crc_passed':True,'all_manifest_hashes_passed':True}
    (dest/'receipt-resident-surfaces-v1.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',action='store_true');args=parser.parse_args();result=verify()
    if args.package:result['package']=package(result)
    print(json.dumps(result,indent=2))
