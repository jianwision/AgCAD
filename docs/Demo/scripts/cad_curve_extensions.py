"""Convert typed GPU curve records to native CAD curves.

Helix and Archimedean spiral are interpolated approximations; the acceptance
runner measures error at points between all supplied samples before delivery.
"""
from cad_ops import ROOT
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec
from OCP.Geom import Geom_BSplineCurve, Geom_Parabola, Geom_Hyperbola, Geom_Circle, Geom_TrimmedCurve, Geom_OffsetCurve
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GeomConvert import GeomConvert_ApproxCurve
from OCP.GeomAbs import GeomAbs_C2
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_HArray1OfReal
from export_step import knots


def curve_axis(s):
    return gp_Ax2(gp_Pnt(*s[8:11]),gp_Dir(*s[17:20]),gp_Dir(*s[11:14]))


def curve_for_step(curve,kind):
    # OCCT's free-curve STEP writer omits this native offset-curve type.
    # Convert explicitly, then measure its error in the independent validator.
    if kind!=8:return curve
    approximation=GeomConvert_ApproxCurve(curve,1e-9,GeomAbs_C2,128,8)
    if not approximation.IsDone() or approximation.MaxError()>1e-9:
        raise ValueError('Offset-curve STEP approximation missed its tolerance')
    return approximation.Curve()


def curve_from_record(records,samples,index):
    s=records[index];kind=int(s[0]);first,last=s[3:5]
    if kind in [0,6]:
        n,p=int(s[1]),int(s[2]);points=TColgp_Array1OfPnt(1,n);weights=TColStd_Array1OfReal(1,n)
        for i in range(n):
            points.SetValue(i+1,gp_Pnt(*s[24+4*i:27+4*i]));weights.SetValue(i+1,s[27+4*i])
        k,m=knots(s[152:152+n+p+1]);return Geom_BSplineCurve(points,weights,k,m,p,False)
    if kind==1:base=Geom_Parabola(curve_axis(s),s[5])
    if kind==2:base=Geom_Hyperbola(curve_axis(s),s[5],s[6])
    if kind==5:base=Geom_Circle(curve_axis(s),s[5])
    if kind in [1,2,5]:return Geom_TrimmedCurve(base,first,last)
    if kind in [3,4]:
        jets=samples[index];n=len(jets);points=TColgp_HArray1OfPnt(1,n);parameters=TColStd_HArray1OfReal(1,n)
        for i,q in enumerate(jets):
            points.SetValue(i+1,gp_Pnt(*q[:3]));parameters.SetValue(i+1,first+(last-first)*i/(n-1))
        fit=GeomAPI_Interpolate(points,parameters,False,1e-10)
        fit.Load(gp_Vec(*jets[0,3:6]),gp_Vec(*jets[-1,3:6]),False);fit.Perform()
        if not fit.IsDone():raise ValueError('Analytic curve spline approximation failed')
        return fit.Curve()
    if kind==7:return curve_from_record(records,samples,int(s[21])).Reversed()
    if kind==8:return Geom_OffsetCurve(curve_from_record(records,samples,int(s[21])),-s[22],gp_Dir(*s[17:20]))
    raise ValueError('Unsupported curve record')
