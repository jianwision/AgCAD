__device__ bool snap_grid(P3 p,P3 origin,double step,P3& result){
    if(!finite_point(p)||!finite_point(origin)||!isfinite(step)||step<=0)return false;
    result=p3(origin.x+step*floor((p.x-origin.x)/step+.5),origin.y+step*floor((p.y-origin.y)/step+.5),origin.z+step*floor((p.z-origin.z)/step+.5));return true;
}
__device__ bool snap_vertices(P3 p,const P3* points,int count,double radius,P3& result,int& index){
    if(count<1||count>4096||radius<0||!finite_point(p))return false;double best=radius*radius;index=-1;
    for(int i=0;i<count;++i){if(!finite_point(points[i]))return false;P3 q=minus(points[i],p);double d=scalar(q,q);if(d<best||(index<0&&d==best)){best=d;index=i;result=points[i];}}
    return index>=0;
}
__device__ bool move_curve_arc(const double* curve,double start,double distance,bool clamp,P3& result,P3& tangent,double& parameter){
    if(!isfinite(start+distance)||start<curve[3]||start>curve[4])return false;
    double length=curve_arc_length(curve,curve[3],curve[4],64),target=curve_arc_length(curve,curve[3],start,64)+distance;
    if(length<1e-12||(!clamp&&(target<0||target>length)))return false;target=fmax(0.,fmin(length,target));
    parameter=curve_distance_parameter(curve,target/length);CurveJet jet=analytic_curve_jet(curve,parameter);result=jet.p;tangent=unit(jet.d);return scalar(jet.d,jet.d)>1e-20;
}
__device__ P3 barycentric(P3 p,P3 a,P3 b,P3 c){P3 v0=minus(b,a),v1=minus(c,a),v2=minus(p,a);double x=scalar(v0,v0),y=scalar(v0,v1),z=scalar(v1,v1),d=scalar(v2,v0),e=scalar(v2,v1),den=x*z-y*y;double v=(z*d-y*e)/den,w=(x*e-y*d)/den;return p3(1-v-w,v,w);}
__device__ double component(P3 p,int j){return j==0?p.x:j==1?p.y:p.z;}
// Intrinsic straightest walk: parallel transport direction through each shared
// edge by the adjacent face-normal rotation. Vertex hits reject as ambiguous;
// boundary hits clamp and report status=2. 128 vertices, 256 triangles max.
__device__ bool move_mesh_path(const P3* p,int nv,const int* triangles,int nt,int start_face,P3 start,P3 direction,double distance,P3& result,P3& tangent,int& face,int& status){
    if(nv<3||nv>128||nt<1||nt>256||start_face<0||start_face>=nt||distance<0||!isfinite(distance)||!finite_point(start)||!finite_point(direction))return false;
    for(int k=0;k<3*nt;++k)if(triangles[k]<0||triangles[k]>=nv)return false;
    face=start_face;result=start;status=0;P3 oldnormal=p3(0,0,0);double remaining=distance;
    for(int step=0;step<256;++step){const int* tri=triangles+3*face;P3 a=p[tri[0]],b=p[tri[1]],c=p[tri[2]],normal=vector_cross(minus(b,a),minus(c,a));
        if(scalar(normal,normal)<1e-20)return false;normal=unit(normal);
        if(step){P3 axis=vector_cross(oldnormal,normal);double sine=sqrt(scalar(axis,axis));if(sine>1e-12)direction=rotate_axis(direction,p3(0,0,0),axis,atan2(sine,scalar(oldnormal,normal)));else if(scalar(oldnormal,normal)<0)return false;}
        else if(fabs(scalar(minus(result,a),normal))>1e-9)return false;
        direction=minus(direction,times(normal,scalar(direction,normal)));if(scalar(direction,direction)<1e-20)return false;direction=unit(direction);tangent=direction;
        P3 bary=barycentric(result,a,b,c),rate=minus(barycentric(plus(result,direction),a,b,c),bary);if(fmin(bary.x,fmin(bary.y,bary.z))<-1e-8)return false;
        double limit=1e100;int opposite=-1;bool ambiguous=false;
        for(int j=0;j<3;++j)if(component(rate,j)<-1e-12){double hit=-component(bary,j)/component(rate,j);if(hit>=-1e-9){hit=fmax(0.,hit);if(fabs(hit-limit)<1e-9)ambiguous=true;else if(hit<limit){limit=hit;opposite=j;ambiguous=false;}}}
        if(remaining<=limit+1e-12){result=plus(result,times(direction,remaining));status=1;return true;}
        if(opposite<0||ambiguous)return false;result=plus(result,times(direction,limit));remaining-=limit;
        int ea=tri[(opposite+1)%3],eb=tri[(opposite+2)%3],next=-1;
        for(int f=0;f<nt;++f)if(f!=face){const int* other=triangles+3*f;bool has_a=false,has_b=false;for(int j=0;j<3;++j){has_a|=other[j]==ea;has_b|=other[j]==eb;}if(has_a&&has_b){if(next>=0)return false;next=f;}}
        if(next<0){status=2;return true;}oldnormal=normal;face=next;
    }return false;
}
__device__ bool move_surface_uv(const double* s,double u,double v,double du,double dv,P3& result,P3& tangent,P3& normal){
    if(!isfinite(u+v+du+dv)||u<s[4]||u>s[5]||v<s[6]||v>s[7]||u+du<s[4]||u+du>s[5]||v+dv<s[6]||v+dv>s[7])return false;
    Jet jet=nurbs_surface(s+96,int(s[0]),int(s[1]),int(s[2]),int(s[3]),s+16,s+56,u+du,v+dv);P3 n=vector_cross(jet.du,jet.dv),t=plus(times(jet.du,du),times(jet.dv,dv));
    if(scalar(n,n)<1e-20||scalar(t,t)<1e-20)return false;result=jet.p;tangent=unit(t);normal=unit(n);return true;
}
const int MOTION_WORDS=1024;
__device__ void store_motion(double* a,int k,P3 p){a[k]=p.x;a[k+1]=p.y;a[k+2]=p.z;}
extern "C" __global__ void motion_suite(double* arena,unsigned offset){
    if(blockIdx.x||threadIdx.x)return;double* a=arena+offset;P3 result,tangent,normal;int index;
    snap_grid(p3(.24,-.26,1.12),p3(0,0,0),.1,result);store_motion(a,0,result);
    P3 candidates[3]={p3(1,0,0),p3(0,1,0),p3(0,0,1)};snap_vertices(p3(.9,.1,0),candidates,3,.3,result,index);store_motion(a,3,result);a[6]=index;
    double curve[EC_WORDS]={0};make_conic_curve(curve,3,p3(0,0,0),p3(1,0,0),p3(0,1,0),1.2,0,.4,0,6.283185307179586);double parameter=0;
    for(int i=0;i<33;++i){if(!move_curve_arc(curve,.4,2.*i/32,false,result,tangent,parameter)){a[1000]=1;return;}store_motion(a,16+7*i,result);store_motion(a,19+7*i,tangent);a[22+7*i]=parameter;}
    for(int i=0;i<EC_WORDS;++i)a[256+i]=curve[i];
    P3 points[6]={p3(0,0,0),p3(1,0,0),p3(1,1,0),p3(0,1,0),p3(2,0,1),p3(2,1,1)};int ids[12]={0,1,2,0,2,3,1,4,5,1,5,2};
    for(int i=0;i<33;++i){int face,status;if(!move_mesh_path(points,6,ids,4,1,p3(.2,.3,0),p3(1,0,0),1.5*i/32,result,tangent,face,status)){a[1000]=2;return;}store_motion(a,448+8*i,result);store_motion(a,451+8*i,tangent);a[454+8*i]=face;a[455+8*i]=status;}
    int face,status;move_mesh_path(points,6,ids,4,1,p3(.2,.3,0),p3(1,0,0),4,result,tangent,face,status);store_motion(a,720,result);a[723]=status;
    double s[SURF_WORDS]={0};P3 cp[4]={p3(0,0,0),p3(0,1,0),p3(1,0,0),p3(1,1,.2)};make_bezier_patch(s,cp,2,2);
    for(int i=0;i<17;++i){if(!move_surface_uv(s,.1,.2,.6*i/16.,.3*i/16.+1e-10,result,tangent,normal)){a[1000]=3;return;}store_motion(a,736+9*i,result);store_motion(a,739+9*i,tangent);store_motion(a,742+9*i,normal);}
    a[992]=!snap_grid(p3(0,0,0),p3(0,0,0),0,result);a[993]=!snap_vertices(p3(10,0,0),candidates,3,.1,result,index);a[994]=!move_curve_arc(curve,.4,100,false,result,tangent,parameter);a[995]=!move_surface_uv(s,.1,.2,2,0,result,tangent,normal);
}
