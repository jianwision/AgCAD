__device__ bool delete_triangles(GMesh m,const int* remove,int count){
    if(count!=triangles(m))return false;for(int i=0;i<count;++i)if(remove[i]!=0&&remove[i]!=1)return false;
    int kept=0;for(int i=0;i<count;++i)if(!remove[i]){for(int j=0;j<3;++j)m.a[8+6*MV+3*kept+j]=index_at(m,i,j);++kept;}m.a[1]=kept;return true;
}
// Oriented boundary loop must oppose the remaining mesh's halfedges.
__device__ bool fill_planar_hole(GMesh m,const int* loop,int count){
    if(count<3||count>64)return false;P3 points[64];
    for(int i=0;i<count;++i){int a=loop[i],b=loop[(i+1)%count];if(a<0||a>=vertices(m)||b<0||b>=vertices(m))return false;for(int j=0;j<i;++j)if(loop[j]==a)return false;points[i]=vertex(m,a);int edge_count=0,reverse=0;
        for(int f=0;f<triangles(m);++f)for(int k=0;k<3;++k){int x=index_at(m,f,k),y=index_at(m,f,(k+1)%3);if((x==a&&y==b)||(x==b&&y==a)){++edge_count;if(x==b&&y==a)++reverse;}}if(edge_count!=1||reverse!=1)return false;}
    if(!triangulate_polygon(m,points,count))return false;weld_vertices(m,1e-12);return m.a[2]==0;
}
__device__ bool revolve_profile(GMesh m,const P3* profile,int count,int segments){
    if(count<3||count>32||segments<3||count*segments>MV||2*count*segments>MT)return false;
    double area=0;for(int j=0;j<count;++j){if(!finite_point(profile[j])||profile[j].x<=0||fabs(profile[j].y)>1e-12)return false;P3 p=profile[(j+1)%count];area+=profile[j].x*p.z-profile[j].z*p.x;}if(area<=1e-12)return false;
    for(int i=0;i<segments;++i){double t=6.283185307179586*i/segments;for(int j=0;j<count;++j)add_vertex(m,p3(profile[j].x*cos(t),profile[j].x*sin(t),profile[j].z));}
    for(int i=0;i<segments;++i)for(int j=0;j<count;++j)quad(m,i*count+j,((i+1)%segments)*count+j,((i+1)%segments)*count+(j+1)%count,i*count+(j+1)%count);return true;
}
__device__ bool smooth_mesh(GMesh m,int iterations,double lambda,double mu){
    int nv=vertices(m),nt=triangles(m);if(nv<3||nv>512||nt<1||nt>1024||iterations<1||iterations>64||!isfinite(lambda)||!isfinite(mu)||lambda<=0||lambda>=1||mu>0||mu<=-1)return false;
    int ea[1536],eb[1536],ne=0;for(int f=0;f<nt;++f)for(int j=0;j<3;++j){int a=index_at(m,f,j),b=index_at(m,f,(j+1)%3);if(a<0||a>=nv||b<0||b>=nv||a==b)return false;if(a>b){int q=a;a=b;b=q;}bool found=false;for(int e=0;e<ne;++e)if(ea[e]==a&&eb[e]==b){found=true;break;}if(!found){if(ne==1536)return false;ea[ne]=a;eb[ne++]=b;}}
    P3 old[512],next[512];for(int it=0;it<iterations;++it)for(int pass=0;pass<(mu?2:1);++pass){double rate=pass?mu:lambda;for(int i=0;i<nv;++i)old[i]=vertex(m,i);
        for(int i=0;i<nv;++i){P3 sum=p3(0,0,0);int n=0;for(int e=0;e<ne;++e)if(ea[e]==i||eb[e]==i){sum=plus(sum,old[ea[e]==i?eb[e]:ea[e]]);++n;}if(!n)return false;next[i]=plus(old[i],times(minus(times(sum,1./n),old[i]),rate));}
        for(int i=0;i<nv;++i)set_vertex(m,i,next[i]);}return true;
}
__device__ bool laplacian_smooth(GMesh m,int iterations,double lambda){return smooth_mesh(m,iterations,lambda,0);}
__device__ bool taubin_smooth(GMesh m,int iterations,double lambda,double mu){return mu<0&&smooth_mesh(m,iterations,lambda,mu);}

const int TM_COUNT=5,TM_AUX_OFFSET=TM_COUNT*MESH_WORDS,TM_AUX_WORDS=128;
extern "C" __global__ void build_topology_meshes(double* packet,unsigned offset){
    double* a=packet+offset;int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=TM_COUNT)return;GMesh m={a+id*MESH_WORDS};bool ok=true;
    if(id<2){ok=make_box(m,p3(-1,-1,-1),p3(1,1,1));int remove[12]={0,0,1,1,0,0,0,0,0,0,0,0};ok=ok&&delete_triangles(m,remove,12);if(id==1){int loop[4]={4,5,7,6};ok=ok&&fill_planar_hole(m,loop,4);}}
    if(id==2){P3 profile[4]={p3(.6,0,-.5),p3(1.2,0,-.5),p3(1.2,0,.5),p3(.6,0,.5)};ok=revolve_profile(m,profile,4,48);}
    if(id>=3){ok=make_uv_sphere(m,p3(0,0,0),1,16,8);for(int i=0;i<vertices(m);++i){P3 p=vertex(m,i);double phi=acos(fmin(1.,fmax(-1.,p.z))),theta=atan2(p.y,p.x);set_vertex(m,i,times(p,1+.12*sin(3*theta)*sin(2*phi)));}ok=ok&&(id==3?laplacian_smooth(m,6,.3):taubin_smooth(m,6,.3,-.31));}
    if(!ok)m.a[2]=202;else mesh_normals(m);
}
extern "C" __global__ void topology_descriptors_checks(double* packet,unsigned offset){
    if(blockIdx.x||threadIdx.x)return;double* out=packet+offset+TM_AUX_OFFSET;
    double boxes[12]={0,0,0,2,2,2,1,.5,.5,3,1.5,1.5};for(int i=0;i<12;++i)out[16+i]=boxes[i];
    P3 profile[4]={p3(.6,0,-.5),p3(1.2,0,-.5),p3(1.2,0,.5),p3(.6,0,.5)};for(int i=0;i<4;++i)ec_set(out,32+3*i,profile[i]);
    GMesh dummy={nullptr};out[0]=!revolve_profile(dummy,profile,2,48);out[1]=!fill_planar_hole(dummy,nullptr,2);
    out[2]=6;out[3]=.3;out[4]=-.31;
}
