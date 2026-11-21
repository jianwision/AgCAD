"""Verify saved engineering evidence and package the reproducible source delivery.

This checks current source hashes and reopens STEP files. It does not rerun the
acceptance suite, contact a model service, or perform a SolidWorks import.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import session_delivery
import tolerance_topology as topo
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader


ROOT = Path(__file__).resolve().parents[1]
SESSION_ZIP_SHA256 = 'e1fa066bb83ad58f527d795231b3aa70553eaa65375c69fc1fb8d343b1b5a0f3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sources(report):
    for source, expected in report['sources'].items():
        require(digest(ROOT / source) == expected, f'Stale source evidence: {source}')


def verify():
    directory = ROOT / 'output/engineering'
    acceptance = load(directory / 'acceptance.json')
    require(acceptance['pass'] and acceptance['tests_run'] == 20, 'Engineering acceptance incomplete')
    require(len(acceptance['cases']) == 20 and all(c['pass'] for c in acceptance['cases']), 'Acceptance case failed')
    require(digest(ROOT / 'scripts/test_engineering_support.py') == acceptance['test_source_sha256'], 'Stale test evidence')
    verify_sources(acceptance)
    report = load(directory / 'demo/validation.json')
    verify_sources(report)
    require(report['pass'] and report['cad']['pass'], 'Engineering demo failed')
    require(report['design_intent']['hard_constraints_pass'], 'Design intent failed')
    require(report['design_intent']['finite_difference_calls_included'], 'Incomplete solver accounting')
    require(report['design_intent']['function_evaluations'] <= report['design_intent']['evaluation_budget'], 'Solver exceeded budget')
    require(all(c['pass'] for c in report['cad']['user_contract_checks']), 'User delivery rule failed')
    require(report['model_decisions'] == 3 and not report['external_llm_invoked'], 'Incorrect model provenance')
    require(not report['class_a_certified'] and not report['general_134_primitive_integration'], 'Unsupported scope claim')
    require(digest(directory / 'demo/class_a_panel.step') == report['cad']['sha256'], 'STEP digest changed')
    native = load(directory / 'class-a/native-validation.json')
    require(native['pass'] and native['jet_max_mm'] < 1e-8, 'Native derivative comparison failed')
    for gpu in (native['gpu'], report['gpu_display_batch']):
        require(gpu['context_creations'] == gpu['final_readbacks'] == 1, 'Unexpected GPU batch accounting')
        require(gpu['intermediate_readbacks'] == 0, 'Unexpected intermediate GPU readback')
    require((directory / 'class-a/source.cu').read_bytes() == (ROOT / 'cuda/class_a.cu').read_bytes(), 'Stale saved CUDA source')

    # Read native shapes again rather than treating file existence as delivery.
    steps = {}
    expected = {
        'trimmed-skin.step': (1, 0),
        'split-solid.step': (12, 2),
        'intent-bound-skin.step': (1, 0),
        'sewn-solid.step': (6, 1),
        'section-curves.step': (0, 0),
        'demo/class_a_panel.step': (2, 0),
    }
    for name, (faces, solids) in expected.items():
        path = directory / name
        reader = STEPControl_Reader()
        require(reader.ReadFile(str(path)) == IFSelect_RetDone, f'Cannot read {name}')
        require(reader.TransferRoots() > 0, f'Cannot transfer {name}')
        shape = reader.OneShape()
        info = topo.inspect_shape(shape)
        require(info['valid'] and not info['nonmanifold_edges'], f'Invalid topology: {name}')
        require(info['faces'] == faces and info['solids'] == solids, f'Unexpected topology: {name}')
        require(info['maximum_native_tolerance_mm'] <= topo.TolerancePolicy().maximum_native_mm, f'Native tolerance exceeded: {name}')
        steps[name] = {'sha256': digest(path), 'inspection': info}

    sessions = session_delivery.verify()
    session_zip = ROOT / 'output/delivery/rust-cuda-modeling-sessions-v1.zip'
    require(digest(session_zip) == SESSION_ZIP_SHA256, 'Previous session ZIP changed')
    require(sessions['existing_v6_unchanged'] is True, 'Previous v6 ZIP missing or changed')
    links = 0
    for name in ('ENGINEERING_SUPPORT.md', 'README.md', 'MODEL_SESSIONS.md', 'SELF_IMPROVEMENT_REVIEW_V2.md'):
        source = (ROOT / name).read_text(encoding='utf-8')
        require(source.count('```') % 2 == 0, f'Unclosed Markdown fence: {name}')
        for link in re.findall(r'\]\(([^)]+)\)', source):
            if re.match(r'^[a-zA-Z]+:', link) or link.startswith('#'):
                continue
            require((ROOT / link.split('#', 1)[0]).exists(), f'Missing link: {name}: {link}')
            links += 1
    receipt = {
        'pass': True,
        'artifact_verification_only': True,
        'engineering_tests': 20,
        'acceptance_sha256': digest(directory / 'acceptance.json'),
        'demo_validation_sha256': digest(directory / 'demo/validation.json'),
        'native_validation_sha256': digest(directory / 'class-a/native-validation.json'),
        'native_d3_max_mm': native['jet_max_mm'],
        'reopened_steps': steps,
        'sessions': sessions,
        'markdown_links_checked': links,
        'previous_session_zip_unchanged': True,
        'external_llm_invoked': False,
        'solidworks_import_performed': False,
        'class_a_certified': False,
        'general_resident_gpu_cad_dag': False,
        'topology_and_nonlinear_solver_backend': 'CPU OCCT / SciPy',
    }
    (directory / 'delivery-verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def package(verification):
    files = set()
    patterns = ['*.md', 'primitive_*.json', 'Cargo.toml', 'Cargo.lock', 'run.ps1',
                'cuda/*.cuh', 'cuda/*.cu', 'src/**/*.rs', 'scripts/*.py', 'configs/*.json',
                'trainer/Cargo.toml', 'trainer/Cargo.lock', 'trainer/src/**/*.rs', 'models/*']
    for pattern in patterns:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    suffixes = {'.json', '.html', '.step', '.stl', '.jpg', '.png', '.cu', '.ptx', '.bin', '.npz', '.log', '.f64'}
    for name in ('primitives', 'car', 'car-variant', 'constraints', 'harness', 'diagnostics', 'planning', 'engineering'):
        files.update(p for p in (ROOT / 'output' / name).rglob('*') if p.is_file() and p.suffix in suffixes)
    for name in ('demo', 'cuda'):
        files.update(p for p in (ROOT / 'output/sessions' / name).iterdir() if p.is_file() and p.suffix in suffixes)
    files.add(ROOT / 'output/sessions/delivery-verification.json')
    files = sorted(p for p in files if p.name not in {'task_plan.md', 'progress.md', 'findings.md', 'solidworks_validation.json'})
    dest = ROOT / 'output/delivery'
    dest.mkdir(exist_ok=True)
    manifest = {
        'release': 'engineering-support-v1',
        'scope': 'Bounded Class-A tools, unit-aware design intent, tolerance-controlled CPU B-rep; shared ModelAdaptor tools',
        'verification': verification,
        'excluded': ['toolchains', 'Python dependencies', 'build caches', 'previous ZIP archives', 'planning notes'],
        'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files],
    }
    manifest_path = dest / 'manifest-engineering-support-v1.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    path = dest / 'rust-cuda-engineering-support-v1.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files:
            archive.write(file, file.relative_to(ROOT).as_posix())
        archive.write(manifest_path, manifest_path.name)
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'ZIP CRC failure')
        for entry in manifest['files']:
            require(hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256'], f"ZIP digest mismatch: {entry['path']}")
    receipt = {'file': str(path), 'bytes': path.stat().st_size, 'files': len(files) + 1,
               'sha256': digest(path), 'zip_crc_passed': True, 'all_manifest_hashes_passed': True}
    (dest / 'receipt-engineering-support-v1.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', action='store_true')
    args = parser.parse_args()
    result = verify()
    if args.package:
        result['package'] = package(result)
    print(json.dumps(result, indent=2))
