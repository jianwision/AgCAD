"""Build exact CAD geometry from GPU poles. This CPU stage builds topology,
not a fit to STL. Wheel solids are analytic tori, with GPU-predicted dimensions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.python-deps'))
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCP.Geom import Geom_BSplineSurface
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array2OfReal, TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeTorus
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID, TopAbs_FACE
from OCP.BRepLib import BRepLib
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static


def knots(full):
    counted = Counter(full)
    k, m = TColStd_Array1OfReal(1, len(counted)), TColStd_Array1OfInteger(1, len(counted))
    for i, (value, count) in enumerate(counted.items(), 1):
        k.SetValue(i, value); m.SetValue(i, count)
    return k, m


def count(shape, kind):
    ex = TopExp_Explorer(shape, kind); n = 0
    while ex.More():
        n += 1; ex.Next()
    return n


def volume(shape):
    props = GProp_GProps(); BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'output/car')
    args = parser.parse_args(); out = args.out.resolve()
    data = json.loads((out/'car_surface.json').read_text())
    nu, nv = data['size']; cp = data['poles_xyzw']
    poles = TColgp_Array2OfPnt(1, nu, 1, nv)
    weights = TColStd_Array2OfReal(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            x, y, z, w = cp[i*nv+j]
            poles.SetValue(i+1, j+1, gp_Pnt(x,y,z)); weights.SetValue(i+1,j+1,w)
    uk, um = knots(data['u_knots']); vk, vm = knots(data['v_knots'])
    surface = Geom_BSplineSurface(poles, weights, uk, vk, um, vm, 3, 3, False, False)
    # Independent CAD evaluator compared with 1,200 points sampled on CUDA.
    sampled = data['profiles_xyz']; max_error = 0.
    for i in range(25):
        for j in range(48):
            point = surface.Value(i/24., 3+14*j/48.)
            q = sampled[3*(48*i+j):3*(48*i+j)+3]
            max_error = max(max_error, point.Distance(gp_Pnt(*q)))
    assert max_error < 1e-7, max_error
    skin = BRepBuilderAPI_MakeFace(surface, 1e-7).Face()
    sewing = BRepBuilderAPI_Sewing(1e-6); sewing.Add(skin)
    for u in [0., 1.]:
        edge = BRepBuilderAPI_MakeEdge(surface.UIso(u)).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        cap = BRepBuilderAPI_MakeFace(wire, True).Face()
        sewing.Add(cap)
    sewing.Perform(); sewn = sewing.SewedShape()
    shells = TopExp_Explorer(sewn, TopAbs_SHELL)
    assert shells.More(), 'Sewing did not produce a shell'
    body = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shells.Current())).Solid()
    assert BRepLib.OrientClosedSolid_s(body), 'Unable to orient closed solid'
    assert BRepCheck_Analyzer(body).IsValid(), 'Invalid body B-rep'
    commands = data['commands']; length, width = commands[1:3]; tyre_scale = commands[17]
    builder = BRep_Builder(); compound = TopoDS_Compound(); builder.MakeCompound(compound); builder.Add(compound, body)
    if commands[16] == 3:
        for x in [-1450*length, 1450*length]:
            for y in [-900*width, 900*width]:
                axis = gp_Ax2(gp_Pnt(x,y,350*tyre_scale), gp_Dir(0,1,0))
                tyre = BRepPrimAPI_MakeTorus(axis, 280*tyre_scale, 70*tyre_scale).Shape()
                assert BRepCheck_Analyzer(tyre).IsValid()
                builder.Add(compound, tyre)
    assert BRepCheck_Analyzer(compound).IsValid()
    Interface_Static.SetCVal_s('write.step.schema', 'AP214IS')
    Interface_Static.SetCVal_s('write.step.unit', 'MM')
    writer = STEPControl_Writer()
    assert writer.Transfer(compound, STEPControl_AsIs) == IFSelect_RetDone
    path = out/'car.step'
    assert writer.Write(str(path)) == IFSelect_RetDone
    # Roundtrip in the CAD kernel, not just an existence check.
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    assert reader.TransferRoots() > 0
    reopened = reader.OneShape()
    assert BRepCheck_Analyzer(reopened).IsValid()
    solids = count(compound, TopAbs_SOLID)
    assert count(reopened, TopAbs_SOLID) == solids
    va, vb = volume(compound), volume(reopened)
    assert va > 0 and abs(va-vb)/va < 1e-8
    report = {'kernel':'Open CASCADE 7.9.3', 'gpu_cad_sample_points':1200,
              'max_gpu_cad_error_mm':max_error, 'source':'exact GPU NURBS poles, no mesh fitting',
              'step':str(path), 'bytes':path.stat().st_size, 'solid_count':solids,
              'faces':count(compound,TopAbs_FACE), 'valid_before':True, 'valid_after':True,
              'volume_before_mm3':va, 'volume_after_mm3':vb,
              'limitations':'Concept car with closed wheel covers; independent tyre solids may intersect body. No native feature history or Class-A claim.'}
    (out/'step_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
