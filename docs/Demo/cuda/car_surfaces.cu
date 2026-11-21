// Composable styling recipe: periodic cubic NURBS skin, mirrored section poles,
// surfaces of revolution for four tyres. Export the very same control net to CAD.
const int NU=14,NV=17;
__device__ void car_knots(double* u,double* v){
    for(int i=0;i<18;++i)u[i]=i<4?0:(i>=14?1:(i-3)/11.0);
    for(int j=0;j<21;++j)v[j]=j;
}
extern "C" __global__ void make_control_net(double* packet){
    const double rows[14][3]={{680,760,560},{850,800,640},{930,880,690},{950,1020,720},{935,1210,740},{910,1330,750},{900,1350,740},{910,1340,730},{920,1240,720},{940,1050,700},{910,850,670},{830,760,620},{650,680,550},{400,570,480}};
    unsigned id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=NU*NV||packet[32]!=0||packet[0]!=1||packet[8]!=2)return;
    int i=id/NV,j=id%NV;double x=-2300+4600.0*i/(NU-1),w=rows[i][0],roof=rows[i][1],belt=rows[i][2],floor=160;
    x*=packet[1];w*=packet[2];floor*=packet[3];roof*=packet[3];belt*=packet[3];if(i>6&&i<12)roof-=packet[4]*25*(i-6);double h=roof-belt;P3 half[8]={p3(x,0,roof),p3(x,.52*w,roof-12),p3(x,.70*w,belt+.63*h),p3(x,.92*w,belt+.10*h),p3(x,w,belt-35),p3(x,.98*w,floor+120),p3(x,.70*w,floor),p3(x,0,floor)};
    int k=j%14;P3 cp=k<=7?half[k]:mirror_y(half[14-k]);
    packet[64+4*id]=cp.x;packet[64+4*id+1]=cp.y;packet[64+4*id+2]=cp.z;packet[64+4*id+3]=1;
}
__device__ Jet car(const double* cp,double u,double v){double ku[18],kv[21];car_knots(ku,kv);return nurbs_surface(cp,NU,NV,3,3,ku,kv,u,3+14*v);}
__device__ void put(double* out,unsigned offset,P3 p){out[offset]=p.x;out[offset+1]=p.y;out[offset+2]=p.z;}
__device__ void triangle(double* out,unsigned index,P3 a,P3 b,P3 c){put(out,9*index,a);put(out,9*index+3,b);put(out,9*index+6,c);}
__device__ P3 tyre(const double* packet,unsigned wheel,double u,double v){
    double a=6.283185307179586*u,b=6.283185307179586*v;
    P3 ring=revolve_y(p3(280+70*cos(b),-70*sin(b),0),a);
    return plus(times(ring,packet[17]),p3((wheel<2?-1450:1450)*packet[1],(wheel%2?-900:900)*packet[2],350*packet[17]));
}
extern "C" __global__ void evaluate_car(double* packet,unsigned steps,unsigned around,unsigned triangle_offset){
    unsigned id=blockIdx.x*blockDim.x+threadIdx.x,body=steps*around,caps=2*around,wheel_cells=64*24;
    if(packet[32]!=0||packet[24]!=4)return;double* out=packet+triangle_offset;
    if(id<body){unsigned i=id/around,j=id%around;double u=double(i)/steps,uu=double(i+1)/steps,v=double(j)/around,vv=double((j+1)%around)/around;
        P3 a=car(packet+64,u,v).p,b=car(packet+64,uu,v).p,c=car(packet+64,uu,vv).p,d=car(packet+64,u,vv).p;triangle(out,2*id,a,b,c);triangle(out,2*id+1,a,c,d);
    }else if(id<body+caps){unsigned local=id-body,side=local/around,j=local%around;double u=side?1:0;P3 a=car(packet+64,u,double(j)/around).p,b=car(packet+64,u,double((j+1)%around)/around).p;
        P3 center=p3((side?2300:-2300)*packet[1],0,(side?350:450)*packet[3]);triangle(out,2*body+local,center,side?b:a,side?a:b);
    }else if(id<body+caps+4*wheel_cells&&packet[16]==3){unsigned local=id-body-caps,w=local/wheel_cells,k=local%wheel_cells,i=k/24,j=k%24;double u=double(i)/64,uu=double((i+1)%64)/64,v=double(j)/24,vv=double((j+1)%24)/24;
        P3 a=tyre(packet,w,u,v),b=tyre(packet,w,uu,v),c=tyre(packet,w,uu,vv),d=tyre(packet,w,u,vv);unsigned t=2*body+caps+2*local;triangle(out,t,a,b,c);triangle(out,t+1,a,c,d);
    }
}
extern "C" __global__ void cad_profiles(double* packet,unsigned offset){unsigned id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=25*48||packet[32]!=0)return;unsigned i=id/48,j=id%48;put(packet,offset+3*id,car(packet+64,double(i)/24,double(j)/48).p);}
extern "C" __global__ void primitive_checks(double* packet,unsigned offset){
    if(blockIdx.x||threadIdx.x||packet[32]!=0)return;P3 cp[4]={p3(0,0,0),p3(1,2,0),p3(2,2,0),p3(3,0,0)};
    put(packet,offset,bezier(cp,3,.5));put(packet,offset+3,mirror_y(p3(1,2,3)));
    put(packet,offset+6,revolve_y(p3(2,1,0),1.5707963267948966));
    put(packet,offset+9,ruled(p3(0,0,0),p3(2,4,6),.25));
    put(packet,offset+12,sweep_frame(p3(1,2,3),p3(1,0,0),p3(0,1,0),2,3));
    put(packet,offset+15,coons(p3(.3,0,0),p3(.3,1,0),p3(0,.7,0),p3(1,.7,0),p3(0,0,0),p3(1,0,0),p3(0,1,0),p3(1,1,0),.3,.7));
    double poles[64],knots[8]={0,0,0,0,1,1,1,1};for(int i=0;i<4;++i)for(int j=0;j<4;++j){int k=4*(i*4+j);poles[k]=i/3.0;poles[k+1]=j/3.0;poles[k+2]=0;poles[k+3]=1;}
    Jet s=nurbs_surface(poles,4,4,3,3,knots,knots,.25,.75);put(packet,offset+18,s.p);put(packet,offset+21,s.du);put(packet,offset+24,s.dv);put(packet,offset+27,s.duu);
    put(packet,offset+30,offset_point(s,2));put(packet,offset+33,blend_c2(p3(0,0,0),p3(1,0,0),p3(0,0,0),p3(1,0,0),p3(1,0,0),p3(0,0,0),.4));
    P3 seam[3]={p3(0,0,0),p3(0,0,0),p3(0,0,0)};
    for(int i=0;i<=32;++i){Jet a=car(packet+64,i/32.0,0),b=car(packet+64,i/32.0,1);P3 diff[3]={minus(a.p,b.p),minus(a.dv,b.dv),minus(a.dvv,b.dvv)};for(int k=0;k<3;++k){seam[k].x=fmax(seam[k].x,fabs(diff[k].x));seam[k].y=fmax(seam[k].y,fabs(diff[k].y));seam[k].z=fmax(seam[k].z,fabs(diff[k].z));}}
    for(int k=0;k<3;++k)put(packet,offset+36+3*k,seam[k]);
    double arc[24],q=sqrt(.5),ku[4]={0,0,1,1},kv[6]={0,0,0,1,1,1};
    for(int i=0;i<2;++i)for(int j=0;j<3;++j){int k=4*(3*i+j);arc[k]=i;arc[k+1]=j==2?0:1;arc[k+2]=j==0?0:1;arc[k+3]=j==1?q:1;}
    put(packet,offset+48,nurbs_surface(arc,2,3,1,2,ku,kv,.4,.5).p);
    double matrix[12]={1,0,0,4,0,2,0,5,0,0,3,6};put(packet,offset+51,affine(p3(1,2,3),matrix));
    double poly[8]={0,0,1,0,1,1,0,1};put(packet,offset+45,p3(trim_polygon(.5,.5,poly,4)?1:0,trim_polygon(2,.5,poly,4)?1:0,0));
}

