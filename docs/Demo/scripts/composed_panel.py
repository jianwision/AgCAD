"""GPU C2 panel -> native sew/thicken -> two drilled holes -> STEP.

Run after run.ps1 suite. This example consumes the GPU output unchanged;
the B-rep features below are explicitly CPU OCCT operations.
"""
import json,math
from acceptance import OUT,build_surface,properties,count,roundtrip,xyz,check
from cad_ops import sew_surfaces,valid
from cad_advanced import untrim_surface,thicken
from cad_topology_operations import boolean_cut,boolean_common,boolean_union
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2,gp_Dir,gp_Pnt
from OCP.TopAbs import TopAbs_SOLID
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer

def main():
    gpu=json.loads((OUT/'advanced_surfaces.json').read_text());left,right=[build_surface(gpu['surfaces'][i]) for i in [7,8]]
    shell=sew_surfaces([untrim_surface(left),untrim_surface(right)])
    panel=thicken(shell,.05);before=properties(panel)[1];assert before>0 and count(panel,TopAbs_SOLID)==1
    tools=[BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,.5,-.5),gp_Dir(0,0,1)),.08,2.).Shape() for x in [.5,1.5]]
    cutter=boolean_union(*tools);removed=boolean_common(panel,cutter);drilled=boolean_cut(panel,cutter);after=properties(drilled)[1];loss=properties(removed)[1]
    assert 0<after<before and loss>0;check(before,after+loss,1e-8);assert count(drilled,TopAbs_SOLID)==1
    result=roundtrip('composed_panel.step',drilled,1)
    BRepMesh_IncrementalMesh(drilled,.003,False,.15,False).Perform();writer=StlAPI_Writer();writer.ASCIIMode=False
    assert writer.Write(drilled,str(OUT/'composed_panel.stl'))
    # Preview actual final CAD tessellation; this is not reported as GPU mesh generation.
    from pathlib import Path
    import struct,re
    data=(OUT/'composed_panel.stl').read_bytes();n=struct.unpack_from('<I',data,80)[0];points=[]
    for i in range(n):points.extend(struct.unpack_from('<9f',data,84+50*i+12))
    center=[(min(points[k::3])+max(points[k::3]))/2 for k in range(3)];radius=max(max(points[k::3])-min(points[k::3]) for k in range(3))*.65
    points=[v-center[k%3] for k,v in enumerate(points)]
    template=(OUT/'advanced_viewer.html').read_text(encoding='utf-8');start=template.index('const vertices=');end=template.index(';',start)
    template=template[:start]+'const vertices=new Float32Array('+json.dumps(points)+')'+template[end:]
    template=re.sub(r'const radius=[^,]+, triangleCount=\d+;',f'const radius={radius}, triangleCount={n};',template)
    template=template.replace('在 GPU 上生成的三维网格','GPU 曲面经 CAD 特征处理后的网格')
    template=template.replace('高级曲面操作','曲面组合示例').replace('advanced_surfaces.stl','composed_panel.stl').replace('导向 / 缩放 / 变截面 / 闭环扫掠，G1/G2 匹配及联合平顺','GPU 双片曲面 → C2 平顺 → 缝合加厚 → 两孔切除').replace('</aside>','<a href="composed_panel.step" download>下载 STEP</a></aside>')
    (OUT/'composed_panel.html').write_text(template,encoding='utf-8')
    report={'pass':True,'pipeline':['GPU C2 fairing of two bicubic patches','CPU sew','CPU .05 mm thickening','CPU Boolean cut two .08 mm radius holes','STEP reimport'],
        'native_solids':1,'volume_before_mm3':before,'volume_after_mm3':after,'removed_volume_mm3':loss,'volume_conservation_residual_mm3':abs(before-after-loss),'step_roundtrip':result,'preview_triangles':n}
    (OUT/'composition_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
