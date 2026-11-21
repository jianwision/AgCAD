// Reusable curve and differential-geometry operators. Caller validates counts,
// nondecreasing knots and positive weights before device evaluation.
struct CurveJet { P3 p,d,dd; };
__device__ CurveJet nurbs_curve(const double* poles,int count,int degree,const double* knots,double t){
    double n[40]={0},d[40]={0},dd[40]={0};basis(count,degree,knots,t,n,d,dd);
    P3 a=p3(0,0,0),b=a,c=a;double w=0,dw=0,ddw=0;
    for(int i=0;i<count;++i){P3 p=p3(poles[4*i],poles[4*i+1],poles[4*i+2]);double q=poles[4*i+3];a=plus(a,times(p,n[i]*q));b=plus(b,times(p,d[i]*q));c=plus(c,times(p,dd[i]*q));w+=n[i]*q;dw+=d[i]*q;ddw+=dd[i]*q;}
    CurveJet s;s.p=times(a,1/w);s.d=times(minus(b,times(s.p,dw)),1/w);s.dd=times(minus(minus(c,times(s.d,2*dw)),times(s.p,ddw)),1/w);return s;
}
__device__ P3 line_point(P3 a,P3 b,double t){return mix_point(a,b,t);}
__device__ P3 polyline_point(const P3* p,int count,double t){double a=fmin(fmax(t,0.),1.)*(count-1);int i=min(int(a),count-2);return mix_point(p[i],p[i+1],a-i);}
__device__ P3 ellipse_point(P3 origin,P3 x,P3 y,double a,double b,double angle){return plus(origin,plus(times(x,a*cos(angle)),times(y,b*sin(angle))));}
__device__ P3 catmull_rom(P3 p0,P3 p1,P3 p2,P3 p3_,double t){P3 cp[4]={p1,plus(p1,times(minus(p2,p0),1./6)),minus(p2,times(minus(p3_,p1),1./6)),p2};return bezier(cp,3,t);}
__device__ P3 rotate_axis(P3 p,P3 center,P3 axis,double angle){axis=unit(axis);P3 v=minus(p,center);return plus(center,plus(plus(times(v,cos(angle)),times(vector_cross(axis,v),sin(angle))),times(axis,scalar(axis,v)*(1-cos(angle)))));}
__device__ P3 scale_about(P3 p,P3 center,P3 scale){P3 v=minus(p,center);return plus(center,p3(v.x*scale.x,v.y*scale.y,v.z*scale.z));}
__device__ P3 mirror_plane(P3 p,P3 origin,P3 normal){normal=unit(normal);return minus(p,times(normal,2*scalar(minus(p,origin),normal)));}
__device__ P3 curvature(Jet s){
    double E=scalar(s.du,s.du),F=scalar(s.du,s.dv),G=scalar(s.dv,s.dv),det=E*G-F*F;
    if(det<=1e-24)return p3(0,0,0);P3 n=surface_normal(s);
    double e=scalar(n,s.duu),f=scalar(n,s.duv),g=scalar(n,s.dvv);
    return p3((e*g-f*f)/det,(E*g-2*F*f+G*e)/(2*det),sqrt(det));
}
__device__ double normal_curvature(Jet s,P3 direction){
    double E=scalar(s.du,s.du),F=scalar(s.du,s.dv),G=scalar(s.dv,s.dv),det=E*G-F*F;
    double x=scalar(s.du,direction),y=scalar(s.dv,direction),a=(G*x-F*y)/det,b=(E*y-F*x)/det;
    return scalar(surface_normal(s),plus(plus(times(s.duu,a*a),times(s.duv,2*a*b)),times(s.dvv,b*b)))/scalar(direction,direction);
}
// Gap mm, tangent-plane angle deg, max normal-curvature mismatch (1/mm).
// Three tangent directions detect the full symmetric second fundamental form.
__device__ P3 continuity_metrics(Jet a,Jet b){
    P3 na=surface_normal(a),nb=surface_normal(b),t=unit(a.dv),r=vector_cross(na,t);double sign=scalar(na,nb)<0?-1:1;
    P3 directions[3]={t,r,unit(plus(t,r))};double jump=0;
    for(int i=0;i<3;++i)jump=fmax(jump,fabs(normal_curvature(a,directions[i])-sign*normal_curvature(b,directions[i])));
    P3 d=minus(a.p,b.p);return p3(sqrt(scalar(d,d)),acos(fmin(1.,fabs(scalar(na,nb))))*57.29577951308232,jump);
}

__device__ double trimmed_parameter(double start,double end,double t){return start+(end-start)*t;}
