__device__ Jet surface_jet(const double* arena,int id,double u,double v){const double* s=surf_record(arena,id);return nurbs_surface(s+96,int(s[0]),int(s[1]),int(s[2]),int(s[3]),s+16,s+56,u,v);}
__device__ P3 surface_point(const double* arena,int id,double u,double v){const double* s=surf_record(arena,id);if(int(s[8])==1)return offset_point(surface_jet(arena,int(s[9]),u,v),s[10]);return surface_jet(arena,id,u,v).p;}
__device__ P3 rail(int side,int j){return p3(side,j/3.,.1*(j==1||j==2)+(side?.05:0));}
__device__ P3 rail_d(int side){return p3(1,0,side?-.5:.7);}
__device__ P3 rail_dd(int side){return p3(0,0,side?.2:.3);}
extern "C" __global__ void build_surfaces(double* packet,unsigned offset){double* arena=packet+offset;
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=SURF_COUNT)return;double* s=surf_record(arena,id);
    if(id==0)make_plane(s,p3(0,0,0),p3(1,0,0),p3(0,1,0));
    if(id==1){P3 cp[16];for(int i=0;i<4;++i)for(int j=0;j<4;++j)cp[4*i+j]=p3(i/3.,j/3.,.4*sin(i*1.0471975511965976)*sin(j*1.0471975511965976));if(!make_bezier_patch(s,cp,4,4))s[11]=1;}
    if(id==2){double cp[24],ku[4]={0,0,1,1},kv[6]={0,0,0,1,1,1};for(int i=0;i<2;++i)for(int j=0;j<3;++j){int k=4*(3*i+j);cp[k]=i;cp[k+1]=j==2?0:1;cp[k+2]=j==0?0:1;cp[k+3]=j==1?sqrt(.5):1;}if(!make_nurbs_patch(s,2,3,1,2,ku,kv,cp))s[11]=1;}
    if(id==3){P3 a[4],b[4];double k[8]={0,0,0,0,1,1,1,1};for(int j=0;j<4;++j){a[j]=p3(0,j/3.,.1*(j==1||j==2));b[j]=p3(1,j/3.,.2+.1*j);}if(!make_ruled(s,a,b,4,3,k))s[11]=1;}
    if(id==4){double cp[16]={0,0,0,1,.3,.2,.5,1,.7,-.2,.5,1,1,0,0,1},k[8]={0,0,0,0,1,1,1,1};if(!make_extruded(s,cp,4,3,k,p3(0,1,.2)))s[11]=1;}
    if(id==5){double circle[9][2]={{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1},{1,0}},cp[36],k[12]={0,0,0,.25,.25,.5,.5,.75,.75,1,1,1};for(int j=0;j<9;++j){cp[4*j]=1+.25*circle[j][0];cp[4*j+1]=.25*circle[j][1];cp[4*j+2]=0;cp[4*j+3]=j%2?sqrt(.5):1;}if(!make_revolved(s,cp,9,2,k,0,6.283185307179586))s[11]=1;}
    if(id==6){double a[16]={0,0,0,1,.3,0,.2,1,.7,0,.4,1,1,0,.1,1},b[16]={0,0,0,1,0,.3,.2,1,0,.7,.2,1,0,1,0,1},k[8]={0,0,0,0,1,1,1,1};if(!make_translation_sweep(s,a,4,3,k,b,4,3,k))s[11]=1;}
    if(id==7){P3 sections[16];double k[8]={0,0,0,0,1,1,1,1};for(int i=0;i<4;++i)for(int j=0;j<4;++j)sections[4*i+j]=p3(i/3.,j/3.,.1*i*i+.15*sin(j*1.0471975511965976)*(1+.2*i));if(!make_loft(s,sections,4,4,3,k))s[11]=1;}
    if(id==8){P3 c0[4]={p3(0,0,0),p3(1./3,0,.2),p3(2./3,0,.2),p3(1,0,0)},c1[4]={p3(0,1,0),p3(1./3,1,.4),p3(2./3,1,.4),p3(1,1,0)},d0[4]={p3(0,0,0),p3(0,1./3,.1),p3(0,2./3,.1),p3(0,1,0)},d1[4]={p3(1,0,0),p3(1,1./3,.3),p3(1,2./3,.3),p3(1,1,0)};
        if(!make_coons(s,c0,c1,d0,d1))s[11]=1;}
    if(id==9){if(!make_offset(s,1,.07))s[11]=1;}
    if(id==10||id==11){P3 a[4],b[4],da[4],db[4],dda[4],ddb[4];double k[8]={0,0,0,0,1,1,1,1};for(int j=0;j<4;++j){a[j]=rail(0,j);b[j]=rail(1,j);da[j]=rail_d(0);db[j]=rail_d(1);dda[j]=rail_dd(0);ddb[j]=rail_dd(1);}if(!make_hermite_blend(s,id==10?1:2,a,b,da,db,dda,ddb,4,3,k))s[11]=1;}
    if(id==12||id==13){setup_surface(s,3,4,2,3);int side=id-12;for(int j=0;j<4;++j){P3 a=rail(side,j),d=rail_d(side),dd=rail_dd(side);if(side){set_pole(s,0,j,a);set_pole(s,1,j,plus(a,times(d,.5)));set_pole(s,2,j,plus(plus(a,d),times(dd,.5)));}else{set_pole(s,2,j,a);set_pole(s,1,j,minus(a,times(d,.5)));set_pole(s,0,j,plus(minus(a,d),times(dd,.5)));}}}
}
__device__ P3 curve_fixture(int id,double t){
    P3 p[4]={p3(0,0,0),p3(.3,.7,.1),p3(.7,-.2,.4),p3(1,1,.2)};
    if(id==0)return line_point(p[0],p[3],t);
    if(id==1)return polyline_point(p,4,t);
    if(id==2)return ellipse_point(p3(0,0,0),p3(1,0,0),p3(0,1,0),1,1,6.283185307179586*t);
    if(id==3)return ellipse_point(p3(0,0,.2),p3(1,0,0),p3(0,1,0),2,.5,6.283185307179586*t);
    if(id==4)return bezier(p,3,t);
    if(id==5){double cp[12]={1,0,0,1,1,1,0,0.7071067811865475244,0,1,0,1},k[6]={0,0,0,1,1,1};return nurbs_curve(cp,3,2,k,t).p;}
    if(id==6)return catmull_rom(p[0],p[1],p[2],p[3],t);
    return bezier(p,3,trimmed_parameter(.2,.8,t));
}
extern "C" __global__ void sample_curves(double* packet,unsigned offset){double* out=packet+offset;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=8*33)return;P3 p=curve_fixture(i/33,(i%33)/32.);out[3*i]=p.x;out[3*i+1]=p.y;out[3*i+2]=p.z;}
extern "C" __global__ void sample_surfaces(double* packet,unsigned surface_offset,unsigned output_offset){const double* arena=packet+surface_offset;double* out=packet+output_offset;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=SURF_COUNT*SURF_SAMPLES)return;int k=i%SURF_SAMPLES;P3 p=surface_point(arena,i/SURF_SAMPLES,(k/9)/8.,(k%9)/8.);out[3*i]=p.x;out[3*i+1]=p.y;out[3*i+2]=p.z;}
extern "C" __global__ void sample_metrics(double* packet,unsigned surface_offset,unsigned output_offset){const double* arena=packet+surface_offset;double* out=packet+output_offset;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=33)return;double v=i/32.;P3 a=continuity_metrics(surface_jet(arena,12,1,v),surface_jet(arena,10,0,v)),b=continuity_metrics(surface_jet(arena,12,1,v),surface_jet(arena,11,0,v)),c=continuity_metrics(surface_jet(arena,11,1,v),surface_jet(arena,13,0,v));P3 q[3]={a,b,c};for(int j=0;j<3;++j){out[9*i+3*j]=q[j].x;out[9*i+3*j+1]=q[j].y;out[9*i+3*j+2]=q[j].z;}
    if(i==0){P3 p=p3(1,2,3),center=p3(.1,.2,.3),axis=p3(1,2,-1);P3 x[5]={plus(p,p3(2,-1,.5)),rotate_axis(p,center,axis,.7),scale_about(p,center,p3(2,.5,-1)),mirror_plane(p,center,axis),curvature(surface_jet(arena,2,.4,.5))};for(int j=0;j<5;++j){out[297+3*j]=x[j].x;out[297+3*j+1]=x[j].y;out[297+3*j+2]=x[j].z;}Jet a=surface_jet(arena,12,1,.5),b=surface_jet(arena,11,0,.5);b.p.z+=.1;P3 bad=continuity_metrics(a,b);out[312]=bad.x;out[313]=bad.y;out[314]=bad.z;}
}

extern "C" __global__ void check_surface_builders(double* packet,unsigned surface_offset,unsigned output_offset){
    if(blockIdx.x||threadIdx.x)return;double* arena=packet+surface_offset;double* out=packet+output_offset;
    double temp[SURF_WORDS],cp[16]={0,0,0,1,0,1,0,1,1,0,0,1,1,1,0,1},k[4]={0,0,1,1};
    int invalid_degree=int(arena[2])+1;
    out[0]=make_nurbs_patch(temp,2,2,invalid_degree,1,k,k,cp)?0:1;
    cp[3]=-arena[99];out[1]=make_nurbs_patch(temp,2,2,1,1,k,k,cp)?0:1;cp[3]=1;
    double bad[4]={0,0,1,.5};out[2]=make_nurbs_patch(temp,2,2,1,1,bad,k,cp)?0:1;
    P3 a[4]={p3(0,0,0),p3(.3,0,0),p3(.7,0,0),p3(1,0,0)},b[4]={p3(0,1,0),p3(.3,1,0),p3(.7,1,0),p3(1,1,0)};
    out[3]=make_coons(temp,a,b,a,b)?0:1;
    out[4]=make_loft(temp,a,12,4,3,k)?0:1;
}
