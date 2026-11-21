const int OM_COUNT=7,OM_SURF_COUNT=2,OM_MESH_OFFSET=2*SURF_WORDS;
const int OM_SAMPLE_OFFSET=OM_MESH_OFFSET+OM_COUNT*MESH_WORDS;
const int OM_AUX_OFFSET=OM_SAMPLE_OFFSET+2*81*3,OM_AUX_WORDS=256;
extern "C" __global__ void build_operation_meshes(double* packet,unsigned offset){
    double* a=packet+offset;int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=OM_COUNT)return;GMesh m={a+OM_MESH_OFFSET+id*MESH_WORDS};bool ok=true;
    if(id<3){P3 axis=id==0?p3(1,0,0):p3(0,0,1),x=id==0?p3(0,1,0):p3(1,0,0),y=id==0?p3(0,0,1):p3(0,1,0);ok=make_prism_stations(m,64,3,.4,.3,axis,x,y);
        for(int i=0;i<vertices(m);++i){P3 p=vertex(m,i);set_vertex(m,i,id==0?bend_point(p,2.6):id==1?twist_point(p,.9):taper_point(p,.18));}}
    if(id==3){P3 path[17],t[17],profile[8];double angle=4.39822971502571;
        for(int i=0;i<17;++i){double u=i/16.;path[i]=p3(1.2*cos(angle*u),1.2*sin(angle*u),1.5*u);t[i]=p3(-1.2*angle*sin(angle*u),1.2*angle*cos(angle*u),1.5);}
        for(int j=0;j<8;++j){double q=6.283185307179586*j/8;profile[j]=p3(.22*cos(q),.22*sin(q),0);}
        ok=make_path_sweep(a,m,path,t,17,profile,8,p3(0,0,1));if(!ok)a[11]=1;}
    if(id==4){P3 p[8];for(int i=0;i<8;++i)p[i]=p3(i&1?1:-1,i&2?1:-1,i&4?1:-1);int faces[24]={0,2,3,1,4,5,7,6,0,4,6,2,1,3,7,5,0,1,5,4,2,6,7,3};ok=catmull_clark(m,p,8,faces,6);}
    if(id==5){P3 p[4]={p3(1,1,1),p3(1,-1,-1),p3(-1,1,-1),p3(-1,-1,1)};int ids[12]={0,1,2,0,3,1,0,2,3,1,3,2};ok=loop_subdivide(m,p,4,ids,4);}
    if(id==6){P3 p[25];for(int i=0;i<5;++i)for(int j=0;j<5;++j)p[5*i+j]=p3(i/4.,j/4.,.2*i*j/16.);ok=resample_grid(m,p,5,5,16,12);}
    if(!ok)m.a[2]=201;else mesh_normals(m);
}
extern "C" __global__ void build_operation_loft(double* packet,unsigned offset){
    if(blockIdx.x||threadIdx.x)return;double* a=packet+offset;P3 sections[16];double cx[4]={0,.3,1.4,2},z[4]={0,.4,-.1,.5},knots[8]={0,0,0,0,1,1,1,1};
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)sections[4*i+j]=p3(cx[i],j/3.-.5,z[i]+.18*sin(j*1.0471975511965976)*(1+.2*i));
    if(!make_chordal_loft(a+SURF_WORDS,sections,4,4,3,knots))a[SURF_WORDS+11]=1;
}
extern "C" __global__ void sample_operation_surfaces(double* packet,unsigned offset){
    double* a=packet+offset;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2*81)return;const double* s=a+(i/81)*SURF_WORDS;if(s[11])return;int k=i%81;P3 p=nurbs_surface(s+96,int(s[0]),int(s[1]),int(s[2]),int(s[3]),s+16,s+56,(k/9)/8.,(k%9)/8.).p;ec_set(a+OM_SAMPLE_OFFSET,3*i,p);
}
extern "C" __global__ void operation_resample_checks(double* packet,unsigned offset){
    if(blockIdx.x||threadIdx.x)return;double* out=packet+offset+OM_AUX_OFFSET;
    P3 p[5]={p3(0,0,0),p3(1,0,0),p3(1,2,0),p3(1,2,1),p3(2,2,1)},sample[65];
    if(!resample_polyline(sample,65,p,5)){out[7]=1;return;}for(int i=0;i<65;++i)ec_set(out,16+3*i,sample[i]);
    GMesh dummy={nullptr};P3 path[2]={p3(0,0,0),p3(1,0,0)},t[2]={p3(1,0,0),p3(-1,0,0)},profile[4]={p3(1,0,0),p3(0,1,0),p3(-1,0,0),p3(0,-1,0)};
    out[0]=!make_path_sweep(nullptr,dummy,path,t,2,profile,4,p3(0,0,1));
    P3 sections[8];for(int i=0;i<8;++i)sections[i]=p3(i%4,0,0);double k[8]={0,0,0,0,1,1,1,1};out[1]=!make_chordal_loft(nullptr,sections,2,4,3,k);
    int quad_ids[4]={0,1,2,3},tri_ids[3]={0,1,2};out[2]=!catmull_clark(dummy,profile,4,quad_ids,1);out[3]=!loop_subdivide(dummy,profile,4,tri_ids,1);
    P3 same[2]={p[0],p[0]};out[4]=!resample_polyline(sample,65,same,2);out[5]=!resample_grid(dummy,profile,2,2,0,4);out[6]=!make_chordal_loft(nullptr,sections,12,4,3,k);
}
