"""Generate the offline capability browser and a hash-verified source/artifact ZIP."""
from pathlib import Path
import json,hashlib,zipfile,sys,html,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/primitives';DEST=ROOT/'output/delivery'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def generate():
    report=load('validation.json');composition=load('composition_validation.json');gpu=load('gpu_report.json')
    assert report['all_passed'] and report['operator_count']==report['passed']==134 and composition['pass']
    ops=report['operators'];steps=report['step_roundtrips']+[composition['step_roundtrip']];assert len(steps)==16 and all(s['valid_after'] for s in steps)
    family={}
    for op in ops:family.setdefault(op['family'],[]).append(op)
    md=['# 可组合建模能力 v4','',f'实际验收：**{len(ops)}/{len(ops)}**，**{len(steps)} 份 STEP** 导出并由 OCCT 重读。',
        'Rust/NVRTC/CUDA 几何 + 显式 CPU CAD/网格拓扑后端。当前神经策略仍为 769 参数、四指令槽；扩展库尚未全部训练进模型。','',
        '## 本轮扩展','',
        '在 85 项基础上补齐 44 项缺少的能力；再加入吸附、网格合并、沿曲线、沿网格、沿曲面移动 5 项。',
        'GPU 曲面可组合为缝合、加厚、打孔后的有效实体，实例见 `output/primitives/composed_panel.step`。','',
        '## 执行与复现','',
        '```powershell','.\\run.ps1 suite','.\\run.ps1 test','.\\run.ps1 lint','python scripts/build_delivery.py','python scripts/build_delivery.py --package','```','',
        f'最终 GPU 批次下载 {gpu["download_bytes"]:,} 字节，1 次最终读回、0 次中间读回。',
        'CAD 内核和部分网格拓扑操作在 CPU 上执行，不包含在这个 GPU 批次计时内。','',
        '## 原语目录','',
        '每行对应实际执行过的验收项。入口、后端、范围、独立误差和负例详见 `primitive_catalog.json` 与 `output/primitives/validation*.json`。','']
    for group,items in family.items():
        md.extend([f'### {group}（{len(items)}）','','| 原语 | 入口 | 后端 / 范围 |','|---|---|---|'])
        for op in items:md.append('| '+op['name']+' | `'+op.get('entry',op.get('source',''))+'` | '+(op.get('backend','')+'；'+op.get('limits','')).replace('|','/').replace('\n',' ')+' |')
        md.append('')
    md.extend(['## STEP 交付','','| 文件 | 实体数 | 面数 | 字节 |','|---|---:|---:|---:|'])
    for s in steps:md.append(f'| {Path(s["file"]).name} | {s["counts"]["solids"]} | {s["counts"]["faces"]} | {s["bytes"]} |')
    md.extend(['','另保留 `output/car/car.step` 与 `output/car-variant/car.step` 的神经网络车身示例。',
        '解析/B-spline 曲面按原生 CAD 几何交付；多边形编辑结果导出为分面 B-rep，不能将其当作自动恢复的光顺 NURBS。',
        'STEP 已由 OCCT 校验，未声称在本机 SolidWorks/CATIA 中完成实机导入验证。','',
        '## 关键边界','',
        '- G1/G2 自动匹配限兼容 bicubic Bezier 边界；双片平顺保持 C2 与兼容外边界，不代表全车 Class-A。',
        '- 闭环 sweep 修正离散帧扭转，几何接缝验收 G0；任意自交、拐角奇异和全局连续性需进一步处理。',
        '- 沿曲线按数值弧长移动；沿网格使用跨边平行输运的直线走步，顶点歧义拒绝、开边界截停；沿曲面按 UV 轨迹贴附，不是世界坐标测地线。',
        '- merge 明确区分索引合并/焊点、曲面缝合和实体 Boolean union；索引 merge 不自动删除内部面。',
        '- 自交修复是显式删面策略，非流形修复是分裂顶点扇；均报告新边界，不冒充任意闭实体重构。',
        '- 小特征修复支持同域/共面碎片和原生小边修复，任意曲面薄片无法保证自动消除。',
        '- 对象 id 稳定；几何子选择属于结果 revision，重建后须重选，不承诺通用持久拓扑命名。'])
    (ROOT/'CAPABILITIES_V4.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    links=''.join(f'<a href="{Path(s["file"]).name}" download>{html.escape(Path(s["file"]).stem)} <small>{s["counts"]["solids"]} solids · {s["bytes"]:,} B</small></a>' for s in steps)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU CAD · 原语工作台</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f4;color:#1c302d;font:15px/1.5 system-ui}header{padding:25px 36px;background:#173c35;color:#fff}h1{margin:0 0 8px;font-size:27px}header p{margin:0;color:#c8dbd6}.layout{display:grid;grid-template-columns:280px 1fr;gap:24px;padding:24px;max-width:1600px;margin:auto}aside,section{background:white;border:1px solid #dce5e0;border-radius:12px;padding:20px}aside a{display:block;text-decoration:none;color:#225f50;padding:8px 0;border-bottom:1px solid #edf0ee;font-size:13px}small{display:block;color:#758b85;font-size:11px}h2{font-size:17px;margin:0 0 15px}.stats{display:flex;gap:12px;margin:0 0 16px}.stats span{background:#e7f2ec;padding:8px 14px;border-radius:8px;font-weight:600}iframe{width:100%;height:470px;border:0;border-radius:8px;background:#101e22}select,input{border:1px solid #b8ccc3;border-radius:6px;padding:9px;font:inherit;background:white}select{max-width:100%;margin-bottom:12px}input{width:100%;margin:18px 0 12px}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;border-bottom:1px solid #e0e9e4;padding:10px;vertical-align:top}td:nth-child(1){font-weight:600;color:#255c4d;white-space:nowrap}td:last-child{min-width:250px}th{color:#70877e}.scroll{overflow:auto;max-height:650px}.note{font-size:12px;color:#70877e;margin:12px 0 0}@media(max-width:850px){.layout{grid-template-columns:1fr}aside{order:2}header{padding:20px}.stats{flex-wrap:wrap}}</style>
<header><h1>GPU CAD · 可组合原语</h1><p>Rust + CUDA 几何 / 原生 CAD 拓扑 / STEP 交付</p></header><main class="layout"><aside><h2>已验证 STEP</h2>__LINKS__<a href="../car/car.step" download>神经网络车身示例<small>5 个原生实体</small></a><p class="note">每份文件均通过 OCCT 导出重读。多边形结果为分面 B-rep；当前模型仍只学习四个指令槽。</p></aside><section><div class="stats"><span>134 / 134 已通过</span><span>16 份 STEP</span><span>1 次 GPU 读回</span></div>
<select aria-label="选择预览" id="preview"><option value="composed_panel.html">组合：双片 C2 曲面 → 加厚 → 打孔</option><option value="motion_viewer.html">吸附 / 合并 / 沿曲线、网格、曲面移动</option><option value="advanced_viewer.html">高级扫掠 / 闭环 / G1 G2 / 平顺</option><option value="operations_viewer.html">Bend / Twist / Taper / Sweep / Loft / 细分</option><option value="topology_viewer.html">Revolve / 删面 / 填洞 / 平滑</option><option value="solid_viewer.html">十类基本实体</option><option value="curve_viewer.html">扩展曲线</option><option value="surface_viewer.html">原始曲面基线</option><option value="viewer.html">原始多边形基线</option></select><iframe id="frame" title="建模结果预览" src="composed_panel.html"></iframe><p class="note">拖动旋转，滚轮缩放。GPU 控制网与 CPU 拓扑后端在目录中逐项标明。</p>
<input id="search" aria-label="搜索建模原语" placeholder="搜索 snap / merge / move / loft / G2 / fillet …"><div class="scroll"><table><thead><tr><th>原语</th><th>后端 / 类别</th><th>输入范围和限制</th></tr></thead><tbody id="rows"></tbody></table></div><p class="note" id="matches"></p></section></main>
<script>const ops=__OPS__;const esc=s=>String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const render=()=>{let q=document.querySelector('#search').value.toLowerCase(),items=ops.filter(x=>JSON.stringify(x).toLowerCase().includes(q));document.querySelector('#rows').innerHTML=items.map(x=>'<tr><td>'+esc(x.name)+'</td><td>'+esc(x.backend)+'<small>'+esc(x.family)+'</small></td><td>'+esc(x.limits)+'</td></tr>').join('');document.querySelector('#matches').textContent=items.length+' / '+ops.length+' 项'};document.querySelector('#search').addEventListener('input',render);document.querySelector('#preview').addEventListener('change',e=>document.querySelector('#frame').src=e.target.value);render();</script></html>'''
    page=page.replace('__LINKS__',links).replace('__OPS__',json.dumps(ops,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('<style>','<style>section{min-width:0}',1)
    page=page.replace('src="composed_panel.html"','src="composed_panel.html?embed=1"').replace('src=e.target.value','src=e.target.value+\'?embed=1\'')
    (OUT/'index.html').write_text(page,encoding='utf-8')
    # An actual line preview of the GPU movement traces, embedded for offline use.
    motion=load('motion.json');paths=[]
    for offset,width,n,shift in [(16,7,33,(0,0,0)),(448,8,33,(0,3,0)),(736,9,17,(0,5,0))]:
        for i in range(n-1):
            for j in [i,i+1]:paths.extend(motion[offset+width*j+k]+shift[k] for k in range(3))
    template=(OUT/'curve_viewer.html').read_text(encoding='utf-8');start=template.index('const vertices=');end=template.index(';',start)
    center=[(min(paths[k::3])+max(paths[k::3]))/2 for k in range(3)];paths=[v-center[k%3] for k,v in enumerate(paths)]
    template=template[:start]+'const vertices=new Float32Array('+json.dumps(paths)+')'+template[end:]
    template=re.sub(r'const radius=[^,]+, triangleCount=\d+;',f'const radius=1.5, triangleCount={len(paths)//6};',template).replace('三角形','轨迹线段')
    template=re.sub(r'<h1>.*?</h1>','<h1>贴附移动轨迹</h1>',template)
    template=template.replace('扩展曲线原语','贴附移动轨迹').replace('</aside>','<p class="small">依次为：按弧长沿螺旋线移动；跨倾斜网格移动；NURBS UV 贴附。轨迹点来自实际 GPU 输出。</p><a href="placement_operations.step" download>下载放置实例 STEP</a></aside>')
    (OUT/'motion_viewer.html').write_text(template,encoding='utf-8')
    embedded='<style>.embedded main{display:block;min-height:100vh}.embedded aside{display:none}.embedded section{height:100vh;min-height:100vh}</style><script>if(new URLSearchParams(location.search).has("embed"))document.documentElement.classList.add("embedded");</script>'
    for p in OUT.glob('*.html'):
        if p.name=='index.html':continue
        source=p.read_text(encoding='utf-8')
        if '.embedded main' not in source:p.write_text(source.replace('<main>',embedded+'<main>',1),encoding='utf-8')
    verification={'version':4,'operator_count':134,'all_passed':True,'step_roundtrip_count':16,'gpu':gpu,'composition':composition,'scope':'Bounded GPU geometry with explicit CPU CAD/topology backends; four-slot neural policy unchanged.'}
    (OUT/'delivery_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print('Generated capability matrix and offline browser: 134 operators, 16 STEP files')
def package():
    report=load('delivery_verification.json');assert report['all_passed'] and report['operator_count']==134
    DEST.mkdir(exist_ok=True);files=[]
    for pattern in ['*.md','primitive_*.json','Cargo.toml','Cargo.lock','run.ps1','cuda/*.cuh','cuda/*.cu','src/**/*.rs','scripts/*.py','trainer/Cargo.toml','trainer/Cargo.lock','trainer/src/**/*.rs','models/*']:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file())
    for directory in ['primitives','car','car-variant']:
        files.extend(p for p in (ROOT/'output'/directory).glob('*') if p.is_file() and p.suffix in {'.json','.html','.step','.stl','.jpg','.png','.cu','.ptx'})
    files.extend(p for p in (ROOT/'output').glob('*') if p.is_file() and p.name in {'unit-tests.log','lint.log','suite-final.log','report.json','viewer.html','preview.png','model.stl','model-source.cu'})
    files=sorted(p for p in set(files) if p.name not in {'task_plan.md','progress.md','findings.md','solidworks_validation.json'});manifest={'version':4,'operator_count':134,'step_roundtrip_count':16,'files':[{ 'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    manifest_path=DEST/'manifest-v4.json';manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path=DEST/'rust-cuda-modeling-demo-v4.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(manifest_path,'manifest-v4.json')
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for entry in manifest['files']:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    receipt={'file':str(path),'bytes':path.stat().st_size,'files':len(files)+1,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'zip_crc_passed':True,'all_manifest_hashes_passed':True}
    (DEST/'receipt-v4.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    generate()
    if '--package' in sys.argv:package()
