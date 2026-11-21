"""Offline GPU harness trace viewer and verified source/result package."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from harness_demo import ROOT, OUT, read_results, REASONS


def generate():
    report = json.loads((OUT/'validation.json').read_text())
    assert report['pass']
    _, results, _ = read_results()
    program = json.loads((OUT/'program.json').read_text())
    candidates = [{'poles':r['x'].tolist(), 'state':r['state'].tolist(), 'trace':r['trace'].tolist(),
                   'initial':program['requests'][i]['initial']} for i,r in enumerate(results)]
    engine = (ROOT/'output/primitives/advanced_viewer.html').read_text(encoding='utf-8')
    engine = engine[engine.index('const canvas='):engine.index('}catch(error)')]
    engine = engine.replace('zoom=4.3', 'zoom=2.8').replace('preview.png','preview.jpg')
    engine = engine.replace('gl_FragColor=vec4(sqrt(color),1.);',
                            'float seam=min(abs(world.x),abs(world.y));color=mix(vec3(.07,.57,.62),color,smoothstep(.0004,.002,seam));gl_FragColor=vec4(sqrt(color),1.);')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU 建模执行轨迹</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1320;color:#e6eff8;font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{display:grid;grid-template-columns:300px minmax(0,1fr);grid-template-rows:minmax(350px,57vh) auto;min-height:100vh}aside{grid-row:1/3;padding:30px 24px;border-right:1px solid #273448;display:flex;flex-direction:column;gap:18px}.eyebrow{letter-spacing:3px;color:#73d8da;font-size:10px}h1{font-size:28px;line-height:1.35;margin:10px 0}h2{font-size:16px;margin:0 0 8px}p{color:#9eb1c6;margin:0}.button{display:block;width:100%;border:1px solid #36546a;border-radius:7px;background:#14283a;color:#e2f5ff;padding:9px 12px;font:inherit;text-align:center;text-decoration:none;cursor:pointer}.button:hover{background:#204057}.small{font-size:12px}label{color:#b7d1df;font-size:12px}dl{margin:0;display:grid;grid-template-columns:1fr 1fr;gap:12px;border-top:1px solid #273448;border-bottom:1px solid #273448;padding:14px 0}dt{font-size:11px;color:#8fa9bc}dd{margin:3px 0 0;font-size:18px}#scene{position:relative;min-width:0;background:radial-gradient(ellipse at 50% 50%,#1e334b,#0b1320 75%)}#view{width:100%;height:100%;position:absolute;touch-action:none}.overlay{position:absolute;left:24px;top:20px;color:#b8d5e6;font-size:12px;pointer-events:none}#error{position:absolute;bottom:30px;left:24px;color:#ffbba8}.hint{position:absolute;bottom:14px;width:100%;text-align:center;color:#87a9bc;font-size:11px;pointer-events:none}#history{padding:18px 24px 24px;border-top:1px solid #273448;min-width:0}.heading{display:flex;justify-content:space-between;gap:20px;align-items:baseline}.badge{font-size:11px;color:#79d3b5;background:#13362f;padding:3px 9px;border-radius:12px}#chart{width:100%;height:85px;display:block}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{font-weight:400;color:#88a4b9;text-align:left}th,td{padding:6px 9px;border-bottom:1px solid #1d3042}td:first-child,th:first-child{padding-left:0}tr[data-round]{cursor:pointer}tr[data-round]:hover,tr.active{background:#142f40}.ok{color:#73d7b6}.bad{color:#e8b782}.idle{color:#8198ac}.table-scroll{overflow:auto}#detail{margin-top:10px;font-size:11px;min-height:19px}a{color:#8fd8eb}#status{font-size:12px;min-height:38px}
@media(max-width:750px){main{display:flex;flex-direction:column}aside{padding:20px;border:0;gap:12px}h1{font-size:24px}dl{grid-template-columns:repeat(4,1fr)}dd{font-size:15px}#scene{height:52vh;min-height:320px}#history{padding:16px}aside .small{display:none}}
</style><main><aside><div><div class="eyebrow">RUST × CUDA</div><h1>曲面建模<br>执行轨迹</h1><p>32 个候选 · 同一设计目标<br>试验、检查、提交与回退</p></div>
<label for="candidate">查看候选的最终曲面</label><select id="candidate" class="button"></select>
<dl><div><dt>目标冠高</dt><dd>270 mm</dd></div><div><dt>最终参数</dt><dd id="parameter"></dd></div><div><dt>已提交</dt><dd id="commits"></dd></div><div><dt>已拒绝</dt><dd id="rejects"></dd></div></dl>
<p id="status"></p><button id="reset" class="button">重置视角</button><a class="button" href="harness_hood.step" download>下载选出方案 01 · STEP</a><a class="button" href="trace.json" download>下载完整轨迹</a>
<p class="small">冠高是截面拟合目标。曲面保持边界与 C2 接缝；STEP 为四面曲面壳。当前显示候选的最终控制网，表格展示其操作历史。</p><p class="small"><a href="validation.json">独立检查报告</a> · <a href="../primitives/index.html">原语库</a></p></aside>
<section id="scene"><canvas id="view"></canvas><div class="overlay" id="model-label"></div><div id="error"></div><div class="hint">拖动旋转 · 滚轮缩放 · 青色线为曲面接缝</div></section>
<article id="history"><div class="heading"><h2>目标误差与事务轨迹</h2><span class="badge">完整 GPU 重放 · 逐位一致</span></div><svg id="chart" viewBox="0 0 700 85" role="img" aria-label="不可变目标分数随轮次的变化"></svg><div class="table-scroll"><table><thead><tr><th>轮次</th><th>操作</th><th>结果</th><th>修订</th><th>目标分数</th><th>预算</th></tr></thead><tbody id="events"></tbody></table></div><p id="detail">选择一轮查看检查指标。分数越低越好，只有通过检查且改善目标时才提交。</p></article></main><script>'use strict';try{
'''
    page += 'const candidates='+json.dumps(candidates, separators=(',',':'))+';\n'
    page += 'const reasons='+json.dumps(REASONS)+';\n'
    page += '''const labels=['提交','复用','停止','已停用','预算耗尽','修订过期','模板需重编译','目标版本过期','非法操作','参数越界','参数非有限','硬约束失败','局部翻折','数值检查失败','未改善','初始状态无效'];
const ops=['AUTO','SET','EDIT','NOOP','STOP'];
const formatOp=r=>r[3]===3?'—':(ops[r[1]]||'未知 #'+r[1])+(r[1]===0?' → '+ops[r[2]]:'');
function mesh(poles){const points=[];const basis=t=>[(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3];
for(let patch=0;patch<4;patch++)for(let i=0;i<20;i++)for(let j=0;j<16;j++)for(const [a,b] of [[i,j],[i+1,j],[i+1,j+1],[i,j],[i+1,j+1],[i,j+1]]){const u=basis(a/20),v=basis(b/16);for(let axis=0;axis<3;axis++){let x=0;for(let l=0;l<4;l++)for(let m=0;m<4;m++)x+=u[l]*v[m]*poles[patch*48+l*12+m*3+axis];points.push(x-(axis===2?210:0));}}return new Float32Array(points);}
const radius=1500;let vertices=mesh(candidates[0].poles);
'''
    page += engine
    page += '''const select=document.querySelector('#candidate');candidates.forEach((c,i)=>{const option=document.createElement('option');option.value=i;option.textContent=String(i+1).padStart(2,'0')+' · 初始目标 '+c.initial.toFixed(1)+' mm';select.append(option);});
function showCandidate(id){const c=candidates[id],s=c.state;vertices=mesh(c.poles);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);draw();
document.querySelector('#parameter').textContent=s[4].toFixed(1)+' mm';document.querySelector('#commits').textContent=s[10];document.querySelector('#rejects').textContent=s[11];document.querySelector('#model-label').textContent='候选 '+String(id+1).padStart(2,'0')+' / 32 · 四片 C2 曲面 · 最终状态';
document.querySelector('#status').textContent=(s[0]===2?'预算已用完，保留最后有效曲面。':'已停止，保留最后有效曲面。')+'剩余预算 '+s[2]+'，复用 '+s[12]+' 次。';
const scores=[c.trace[0][11],...c.trace.map(r=>r[13])],low=Math.min(...scores),high=Math.max(...scores),span=Math.max(high-low,1),xy=scores.map((v,i)=>[20+i*82,65-50*(v-low)/span]);
document.querySelector('#chart').innerHTML='<path d="M'+xy.map(p=>p.join(',')).join('L')+'" fill="none" stroke="#6bd0ce" stroke-width="2"/>'+xy.map(([x,y],i)=>'<circle cx="'+x+'" cy="'+y+'" r="3" fill="#6bd0ce"/><text x="'+x+'" y="82" fill="#7895aa" text-anchor="middle" font-size="9">'+(i===0?'初始':i-1)+'</text>').join('')+'<text x="696" y="12" text-anchor="end" fill="#a3bece" font-size="10">'+scores[0].toFixed(2)+' → '+scores.at(-1).toFixed(2)+'</text>';
document.querySelector('#events').innerHTML=c.trace.map((r,i)=>'<tr data-round="'+i+'"><td>'+i+'</td><td>'+formatOp(r)+'</td><td class="'+(r[3]<3?'ok':r[3]===3?'idle':'bad')+'" title="'+reasons[r[3]]+'">'+labels[r[3]]+'</td><td>'+r[4]+' → '+r[5]+'</td><td>'+r[13].toFixed(2)+'</td><td>'+r[6]+' → '+r[7]+'</td></tr>').join('');
document.querySelector('#detail').textContent='选择一轮查看检查指标。分数越低越好，只有通过检查且改善目标时才提交。';
document.querySelectorAll('tr[data-round]').forEach(row=>row.onclick=()=>{document.querySelectorAll('tr.active').forEach(x=>x.classList.remove('active'));row.classList.add('active');const r=c.trace[Number(row.dataset.round)];document.querySelector('#detail').textContent='轮 '+r[0]+' · '+reasons[r[3]]+' · 试验分数 '+r[12].toPrecision(7)+' · 硬约束比值 '+r[14].toExponential(2)+' · 局部 Jacobian '+r[15].toExponential(2)+' · 相对梯度 '+r[16].toExponential(2)+' · '+(r[18]?'本轮已求值':'本轮复用或跳过');});}
select.onchange=()=>showCandidate(Number(select.value));showCandidate(0);
}catch(error){document.querySelector('#error').textContent=error.message;console.error(error);}</script></html>'''
    (OUT/'viewer.html').write_text(page, encoding='utf-8')
    index = ROOT/'output/primitives/index.html'
    source = index.read_text(encoding='utf-8')
    if '../harness/viewer.html' not in source:
        source = source.replace('</header>', '<a href="../harness/viewer.html">GPU 执行轨迹</a></header>',1)
        index.write_text(source, encoding='utf-8')
    print('Generated offline harness trace viewer')


def package():
    report = json.loads((OUT/'validation.json').read_text())
    assert report['pass'] and report['device_replay_bit_exact']
    assert json.loads((ROOT/'output/primitives/validation.json').read_text())['all_passed']
    files = []
    for pattern in ['*.md','primitive_*.json','Cargo.toml','Cargo.lock','run.ps1','cuda/*.cuh','cuda/*.cu','src/**/*.rs','scripts/*.py','trainer/Cargo.toml','trainer/Cargo.lock','trainer/src/**/*.rs','models/*']:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file())
    for name in ('primitives','car','car-variant','constraints','harness'):
        files.extend(p for p in (ROOT/'output'/name).rglob('*') if p.is_file() and p.suffix in {'.json','.html','.step','.stl','.jpg','.png','.cu','.ptx','.bin','.npz','.log'})
    files = sorted(p for p in set(files) if p.name not in {'task_plan.md','findings.md','progress.md','solidworks_validation.json'})
    manifest = {'version':6,'baseline_operator_count':134,'constraint_positive_gpu_cases':15,
                'harness_main_candidates':32,'harness_rounds':8,'harness_verifier_fault_injections':3,
                'harness_host_input_rejections':10,'step_roundtrip_count_primitive_constraint_harness':18,
                'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
                          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    dest = ROOT/'output/delivery'
    dest.mkdir(exist_ok=True)
    manifest_path = dest/'manifest-v6.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path = dest/'rust-cuda-modeling-demo-v6.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:
            archive.write(p,p.relative_to(ROOT).as_posix())
        archive.write(manifest_path,manifest_path.name)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for entry in manifest['files']:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256']
    receipt = {'file':str(path),'bytes':path.stat().st_size,'files':len(files)+1,
               'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'zip_crc_passed':True,'manifest_sha256_passed':True}
    (dest/'receipt-v6.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    generate()
    if '--package' in sys.argv:
        package()
