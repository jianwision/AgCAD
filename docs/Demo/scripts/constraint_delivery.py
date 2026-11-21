"""Generate an offline viewer and package the validated constraint extension."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
from constraint_demo import OUT, ROOT, read_results, patch_nets, patch_value


def generate():
    report = json.loads((OUT/'validation.json').read_text())
    assert report['pass']
    _, results = read_results(OUT)
    arrays = []
    for result in results:
        vertices = []
        for p in patch_nets(result['x']).reshape(4, 4, 4, 3):
            for i in range(20):
                for j in range(16):
                    for a, b in [(i, j), (i+1, j), (i+1, j+1), (i, j), (i+1, j+1), (i, j+1)]:
                        point = patch_value(p, a/20, b/16)-np.array([0, 0, 210])
                        vertices.extend(point)
        arrays.append(base64.b64encode(np.array(vertices, dtype='<f4').tobytes()).decode())
    source = (ROOT/'output/primitives/advanced_viewer.html').read_text(encoding='utf-8')
    source = re.sub(r'<title>.*?</title>', '<title>GPU 多曲面约束求解</title>', source)
    start, end = source.index('<aside>'), source.index('</aside>')+len('</aside>')
    aside = '''<aside><div><div class="eyebrow">RUST × CUDA</div><h1>多曲面联合设计</h1><p>四片曲面，保持边界与 C2 接缝</p></div>
<label for="candidate">外形方案</label><select id="candidate" class="button">'''
    aside += ''.join(f'<option value="{i}" {"selected" if i == 5 else ""}>方案 {i+1} · 目标冠高 {220+10*i} mm</option>' for i in range(8))
    aside += '''</select><dl><div><dt>联合约束</dt><dd>400 条硬约束</dd></div><div><dt>设计目标</dt><dd>484 条软目标</dd></div></dl>
<p id="status">8 个候选均通过约束检查</p><button class="button" id="reset">重置视角</button>
<a class="button" href="constraint_hood.step" download>下载方案 6 · STEP</a>
<p class="small">拖动旋转，滚轮缩放。切换显示已求解方案；目标冠高是拟合目标，并非实际测得高度。STEP 为四片缝合曲面。</p>
<span id="count" hidden></span></aside>'''
    source = source[:start]+aside+source[end:]
    source = re.sub(r'const radius=[^,]+, triangleCount=\d+;', 'const radius=1500, triangleCount=2560;', source)
    source = source.replace('zoom=4.3', 'zoom=2.8')
    source = source.replace('gl_FragColor=vec4(sqrt(color),1.);',
                            'float seam=min(abs(world.x),abs(world.y));color=mix(vec3(.07,.57,.62),color,smoothstep(.0004,.002,seam));gl_FragColor=vec4(sqrt(color),1.);')
    start = source.index("const encoded='")
    end = source.index("const canvas=", start)
    data_script = 'const candidates='+json.dumps(arrays)+';\n'
    data_script += "let vertices=new Float32Array(Uint8Array.from(atob(candidates[5]),x=>x.charCodeAt(0)).buffer);\n"
    source = source[:start]+data_script+source[end:]
    marker = 'new ResizeObserver(draw).observe(canvas);draw();'
    source = source.replace(marker, marker+"\ndocument.querySelector('#candidate').onchange=e=>{vertices=new Float32Array(Uint8Array.from(atob(candidates[Number(e.target.value)]),x=>x.charCodeAt(0)).buffer);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);document.querySelector('#status').textContent='方案 '+(Number(e.target.value)+1)+' · 约束检查通过';draw();};")
    (OUT/'viewer.html').write_text(source, encoding='utf-8')
    # Keep the existing gallery discoverable without changing its 134-item count.
    index = ROOT/'output/primitives/index.html'
    page = index.read_text(encoding='utf-8')
    if '../constraints/viewer.html' not in page:
        page = page.replace('</nav>', '<a href="../constraints/viewer.html">多曲面约束设计</a></nav>') if '</nav>' in page else page.replace('</aside>', '<a href="../constraints/viewer.html">多曲面约束设计</a></aside>', 1)
        # Existing workbench uses a header rather than a navigation element.
        if '../constraints/viewer.html' not in page:
            page = page.replace('</header>', '<a href="../constraints/viewer.html">多曲面约束设计</a></header>', 1)
        index.write_text(page, encoding='utf-8')
    print('Generated offline constraint viewer')


def package():
    report = json.loads((OUT/'validation.json').read_text())
    baseline = json.loads((ROOT/'output/primitives/validation.json').read_text())
    assert report['pass'] and baseline['all_passed'] and baseline['operator_count'] == 134
    files = []
    for pattern in ['*.md', 'primitive_*.json', 'Cargo.toml', 'Cargo.lock', 'run.ps1', 'cuda/*.cuh', 'cuda/*.cu', 'src/**/*.rs', 'scripts/*.py', 'trainer/Cargo.toml', 'trainer/Cargo.lock', 'trainer/src/**/*.rs', 'models/*']:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file())
    for name in ('primitives', 'car', 'car-variant', 'constraints'):
        files.extend(p for p in (ROOT/'output'/name).rglob('*') if p.is_file() and p.suffix in {'.json', '.html', '.step', '.stl', '.jpg', '.png', '.cu', '.ptx', '.bin', '.log'})
    files = sorted(p for p in set(files) if p.name not in {'task_plan.md', 'findings.md', 'progress.md', 'solidworks_validation.json'})
    dest = ROOT/'output/delivery'
    dest.mkdir(exist_ok=True)
    manifest = {'version': 5, 'baseline_operator_count': 134, 'constraint_candidate_count': 8,
                'constraint_positive_gpu_cases': 15, 'constraint_negative_cases': 14,
                'step_roundtrip_count': 17,
                'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                           'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    manifest_path = dest/'manifest-v5.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    path = dest/'rust-cuda-modeling-demo-v5.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(manifest_path, manifest_path.name)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for entry in manifest['files']:
            assert hashlib.sha256(z.read(entry['path'])).hexdigest() == entry['sha256']
    receipt = {'file': str(path), 'bytes': path.stat().st_size, 'files': len(files)+1,
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'zip_crc_passed': True, 'manifest_sha256_passed': True}
    (dest/'receipt-v5.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    generate()
    if '--package' in sys.argv:
        package()
