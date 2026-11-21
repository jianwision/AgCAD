// Bounded indexed mesh operations. Millimetres; counts/indices are exact
// integers stored in a packed double arena for a single final transfer.
const int MV=4096,MT=4096,MESH_WORDS=8+6*MV+3*MT,MESH_COUNT=16;
struct GMesh{double* a;};
__device__ int vertices(GMesh m){return int(m.a[0]);}
__device__ int triangles(GMesh m){return int(m.a[1]);}
__device__ P3 vertex(GMesh m,int i){return p3(m.a[8+3*i],m.a[9+3*i],m.a[10+3*i]);}
__device__ void set_vertex(GMesh m,int i,P3 p){m.a[8+3*i]=p.x;m.a[9+3*i]=p.y;m.a[10+3*i]=p.z;}
__device__ int add_vertex(GMesh m,P3 p){int n=vertices(m);if(n>=MV){m.a[2]=1;return 0;}set_vertex(m,n,p);m.a[0]=n+1;return n;}
__device__ int index_at(GMesh m,int t,int k){return int(m.a[8+6*MV+3*t+k]);}
__device__ void add_triangle(GMesh m,int a,int b,int c){int t=triangles(m);if(t>=MT){m.a[2]=1;return;}int k=8+6*MV+3*t;m.a[k]=a;m.a[k+1]=b;m.a[k+2]=c;m.a[1]=t+1;}
__device__ void quad(GMesh m,int a,int b,int c,int d){add_triangle(m,a,b,c);add_triangle(m,a,c,d);}
__device__ double orient(P3 a,P3 b,P3 c,P3 n){return scalar(vector_cross(minus(b,a),minus(c,a)),n);}
// Ear clipping in the polygon's plane. Supports a simple concave polygon, no
// holes, <=64 vertices. Rejects nonplanarity, duplicate vertices and self-crossing.
__device__ bool triangulate_polygon(GMesh m,const P3* p,int n){
    if(n<3||n>64){m.a[2]=100;return false;}P3 normal=p3(0,0,0);
    for(int i=0;i<n;++i)normal=plus(normal,vector_cross(p[i],p[(i+1)%n]));
    if(scalar(normal,normal)<1e-20){m.a[2]=102;return false;}normal=unit(normal);
    for(int i=0;i<n;++i){if(fabs(scalar(minus(p[i],p[0]),normal))>1e-8){m.a[2]=101;return false;}
        for(int j=i+1;j<n;++j){P3 d=minus(p[i],p[j]);if(scalar(d,d)<1e-20){m.a[2]=102;return false;}}
        for(int j=i+2;j<n;++j){if((j+1)%n==i)continue;P3 a=p[i],b=p[(i+1)%n],c=p[j],d=p[(j+1)%n];if(orient(a,b,c,normal)*orient(a,b,d,normal)<-1e-20&&orient(c,d,a,normal)*orient(c,d,b,normal)<-1e-20){m.a[2]=102;return false;}}}
    int ids[64],base=vertices(m);for(int i=0;i<n;++i){ids[i]=i;add_vertex(m,p[i]);}int left=n;
    for(int pass=0;left>3&&pass<n*n;++pass){bool found=false;for(int k=0;k<left;++k){int ia=ids[(k+left-1)%left],ib=ids[k],ic=ids[(k+1)%left];if(orient(p[ia],p[ib],p[ic],normal)<=1e-12)continue;bool occupied=false;
            for(int l=0;l<left;++l){int q=ids[l];if(q==ia||q==ib||q==ic)continue;if(orient(p[ia],p[ib],p[q],normal)>=-1e-12&&orient(p[ib],p[ic],p[q],normal)>=-1e-12&&orient(p[ic],p[ia],p[q],normal)>=-1e-12){occupied=true;break;}}
            if(occupied)continue;add_triangle(m,base+ia,base+ib,base+ic);for(int l=k;l<left-1;++l)ids[l]=ids[l+1];--left;found=true;break;}
        if(!found){m.a[2]=103;return false;}}
    if(left!=3){m.a[2]=103;return false;}add_triangle(m,base+ids[0],base+ids[1],base+ids[2]);return true;
}
__device__ bool extrude_polygon(GMesh m,const P3* p,int n,P3 direction){
    P3 normal=p3(0,0,0);for(int i=0;i<n;++i)normal=plus(normal,vector_cross(p[i],p[(i+1)%n]));
    if(scalar(normal,direction)<=1e-12){m.a[2]=104;return false;}if(!triangulate_polygon(m,p,n))return false;int count=triangles(m);
    for(int i=0;i<n;++i)add_vertex(m,plus(p[i],direction));for(int i=0;i<count;++i){int a=index_at(m,i,0),b=index_at(m,i,1),c=index_at(m,i,2),k=8+6*MV+3*i;m.a[k]=c;m.a[k+2]=a;add_triangle(m,a+n,b+n,c+n);}
    for(int i=0;i<n;++i)quad(m,i,(i+1)%n,(i+1)%n+n,i+n);return true;
}
__device__ void weld_vertices(GMesh m,double tolerance){
    if(!isfinite(tolerance)||tolerance<=0){m.a[2]=106;return;}int map[MV],n=vertices(m),kept=0;for(int i=0;i<n;++i){P3 p=vertex(m,i);int id=-1;for(int j=0;j<kept;++j){P3 d=minus(p,vertex(m,j));if(scalar(d,d)<=tolerance*tolerance){id=j;break;}}if(id<0){id=kept++;set_vertex(m,id,p);}map[i]=id;}
    for(int i=0;i<triangles(m);++i)for(int k=0;k<3;++k)m.a[8+6*MV+3*i+k]=map[index_at(m,i,k)];m.a[0]=kept;
}
__device__ void subdivide_triangles(GMesh src,GMesh dst){
    if(6*triangles(src)>MV||4*triangles(src)>MT){dst.a[2]=1;return;}for(int t=0;t<triangles(src);++t){P3 a=vertex(src,index_at(src,t,0)),b=vertex(src,index_at(src,t,1)),c=vertex(src,index_at(src,t,2));int k=vertices(dst);add_vertex(dst,a);add_vertex(dst,b);add_vertex(dst,c);add_vertex(dst,mix_point(a,b,.5));add_vertex(dst,mix_point(b,c,.5));add_vertex(dst,mix_point(c,a,.5));add_triangle(dst,k,k+3,k+5);add_triangle(dst,k+3,k+1,k+4);add_triangle(dst,k+5,k+4,k+2);add_triangle(dst,k+3,k+4,k+5);}weld_vertices(dst,1e-12);
}
__device__ void mesh_normals(GMesh m){
    int base=8+3*MV;for(int i=0;i<3*vertices(m);++i)m.a[base+i]=0;
    for(int t=0;t<triangles(m);++t){int ids[3]={index_at(m,t,0),index_at(m,t,1),index_at(m,t,2)};P3 a=vertex(m,ids[0]),b=vertex(m,ids[1]),c=vertex(m,ids[2]),n=vector_cross(minus(b,a),minus(c,a));for(int k=0;k<3;++k){int i=base+3*ids[k];m.a[i]+=n.x;m.a[i+1]+=n.y;m.a[i+2]+=n.z;}}
    for(int v=0;v<vertices(m);++v){int i=base+3*v;P3 n=unit(p3(m.a[i],m.a[i+1],m.a[i+2]));m.a[i]=n.x;m.a[i+1]=n.y;m.a[i+2]=n.z;}
}
// Vertex truncation for closed convex tri-valent polyhedra, <=64 vertices,
// <=96 undirected edges, convex faces <=16 corners. Reject other valences.
__device__ bool bevel_vertices(GMesh m,const P3* p,int nv,const int* faces,const int* offsets,int nf,double t){
    if(nv>64||t<=0||t>=.5){m.a[2]=105;return false;}int ea[192],eb[192],ev[192],ne=0;P3 center=p3(0,0,0);for(int i=0;i<nv;++i)center=plus(center,times(p[i],1./nv));
    for(int f=0;f<nf;++f){int a=offsets[f],b=offsets[f+1];if(b-a<3||b-a>16){m.a[2]=105;return false;}for(int i=a;i<b;++i){int v=faces[i],w=faces[i+1==b?a:i+1];for(int sign=0;sign<2;++sign){int from=sign?w:v,to=sign?v:w,id=-1;for(int k=0;k<ne;++k)if(ea[k]==from&&eb[k]==to)id=k;if(id<0){if(ne==192){m.a[2]=105;return false;}ea[ne]=from;eb[ne]=to;ev[ne]=add_vertex(m,mix_point(p[from],p[to],t));++ne;}}}}
    for(int v=0;v<nv;++v){int ids[3],n=0;for(int k=0;k<ne;++k)if(ea[k]==v){if(n==3){m.a[2]=105;return false;}ids[n++]=ev[k];}if(n!=3){m.a[2]=105;return false;}P3 a=vertex(m,ids[0]),b=vertex(m,ids[1]),c=vertex(m,ids[2]);if(orient(a,b,c,minus(p[v],center))<0){int q=ids[1];ids[1]=ids[2];ids[2]=q;}add_triangle(m,ids[0],ids[1],ids[2]);}
    for(int f=0;f<nf;++f){int start=offsets[f],end=offsets[f+1],poly[32],n=0;for(int i=start;i<end;++i){int v=faces[i],prev=faces[i==start?end-1:i-1],next=faces[i+1==end?start:i+1];int to[2]={prev,next};for(int j=0;j<2;++j)for(int k=0;k<ne;++k)if(ea[k]==v&&eb[k]==to[j])poly[n++]=ev[k];}for(int i=1;i<n-1;++i)add_triangle(m,poly[0],poly[i],poly[i+1]);}return true;
}

// Parameterized constructors. Counts, radii and capacities are validated before writes.
__device__ bool make_box(GMesh m,P3 minimum,P3 maximum){
    if(minimum.x>=maximum.x||minimum.y>=maximum.y||minimum.z>=maximum.z){m.a[2]=110;return false;}
    for(int i=0;i<8;++i)add_vertex(m,p3(i&1?maximum.x:minimum.x,i&2?maximum.y:minimum.y,i&4?maximum.z:minimum.z));
    int f[24]={0,2,3,1,4,5,7,6,0,4,6,2,1,3,7,5,0,1,5,4,2,6,7,3};for(int i=0;i<6;++i)quad(m,f[4*i],f[4*i+1],f[4*i+2],f[4*i+3]);return true;
}
__device__ bool make_uv_sphere(GMesh m,P3 center,double radius,int segments,int rings){
    if(radius<=0||segments<3||rings<2||2+(rings-1)*segments>MV||2*segments*(rings-1)>MT){m.a[2]=110;return false;}
    const double pi=3.14159265358979323846;int n=segments,r=rings;add_vertex(m,plus(center,p3(0,0,radius)));
    for(int i=1;i<r;++i)for(int j=0;j<n;++j)add_vertex(m,plus(center,times(p3(sin(pi*i/r)*cos(2*pi*j/n),sin(pi*i/r)*sin(2*pi*j/n),cos(pi*i/r)),radius)));
    int bottom=add_vertex(m,plus(center,p3(0,0,-radius)));for(int j=0;j<n;++j){add_triangle(m,0,1+j,1+(j+1)%n);add_triangle(m,bottom,1+(r-2)*n+(j+1)%n,1+(r-2)*n+j);}
    for(int i=0;i<r-2;++i)for(int j=0;j<n;++j)quad(m,1+i*n+j,1+(i+1)*n+j,1+(i+1)*n+(j+1)%n,1+i*n+(j+1)%n);return true;
}
__device__ bool make_frustum(GMesh m,P3 origin,double bottom_radius,double top_radius,double height,int n){
    if(bottom_radius<=0||top_radius<0||height<=0||n<3||2*n+2>MV||4*n>MT){m.a[2]=110;return false;}const double pi=3.14159265358979323846;
    for(int j=0;j<n;++j)add_vertex(m,plus(origin,p3(bottom_radius*cos(2*pi*j/n),bottom_radius*sin(2*pi*j/n),0)));
    if(top_radius>0){for(int j=0;j<n;++j)add_vertex(m,plus(origin,p3(top_radius*cos(2*pi*j/n),top_radius*sin(2*pi*j/n),height)));int a=add_vertex(m,origin),b=add_vertex(m,plus(origin,p3(0,0,height)));for(int j=0;j<n;++j){int k=(j+1)%n;quad(m,j,k,n+k,n+j);add_triangle(m,a,k,j);add_triangle(m,b,n+j,n+k);}}
    else{int top=add_vertex(m,plus(origin,p3(0,0,height))),bottom=add_vertex(m,origin);for(int j=0;j<n;++j){int k=(j+1)%n;add_triangle(m,j,k,top);add_triangle(m,bottom,k,j);}}return true;
}
__device__ bool make_torus(GMesh m,P3 center,double major,double minor,int n,int r){
    if(major<=minor||minor<=0||n<3||r<3||n*r>MV||2*n*r>MT){m.a[2]=110;return false;}const double pi=3.14159265358979323846;
    for(int i=0;i<n;++i)for(int j=0;j<r;++j){double u=2*pi*i/n,v=2*pi*j/r;add_vertex(m,plus(center,p3((major+minor*cos(v))*cos(u),(major+minor*cos(v))*sin(u),minor*sin(v))));}
    for(int i=0;i<n;++i)for(int j=0;j<r;++j)quad(m,i*r+j,((i+1)%n)*r+j,((i+1)%n)*r+(j+1)%r,i*r+(j+1)%r);return true;
}
__device__ bool make_grid(GMesh m,P3 origin,P3 u,P3 v,int us,int vs,double warp){
    P3 n=vector_cross(u,v);if(us<1||vs<1||(us+1)*(vs+1)>MV||2*us*vs>MT||scalar(n,n)<1e-20){m.a[2]=110;return false;}n=unit(n);const double pi=3.14159265358979323846;
    for(int i=0;i<=us;++i)for(int j=0;j<=vs;++j)add_vertex(m,plus(origin,plus(plus(times(u,double(i)/us),times(v,double(j)/vs)),times(n,warp*sin(pi*i/us)*sin(pi*j/vs)))));
    for(int i=0;i<us;++i)for(int j=0;j<vs;++j)quad(m,(vs+1)*i+j,(vs+1)*(i+1)+j,(vs+1)*(i+1)+j+1,(vs+1)*i+j+1);return true;
}
__device__ bool inset_polygon(GMesh m,const P3* p,int n,double fraction){
    if(n<3||n>64||fraction<=0||fraction>=1){m.a[2]=100;return false;}P3 center=p3(0,0,0),normal=p3(0,0,0);
    for(int i=0;i<n;++i){center=plus(center,times(p[i],1./n));normal=plus(normal,vector_cross(p[i],p[(i+1)%n]));}
    if(scalar(normal,normal)<1e-20){m.a[2]=102;return false;}normal=unit(normal);
    for(int i=0;i<n;++i)if(fabs(scalar(minus(p[i],p[0]),normal))>1e-8||orient(p[i],p[(i+1)%n],p[(i+2)%n],normal)<=1e-12){m.a[2]=107;return false;}
    for(int i=0;i<n;++i)add_vertex(m,p[i]);for(int i=0;i<n;++i)add_vertex(m,mix_point(p[i],center,fraction));int c=add_vertex(m,center);
    for(int i=0;i<n;++i){int j=(i+1)%n;quad(m,i,j,n+j,n+i);add_triangle(m,c,n+i,n+j);}return true;
}
__device__ void mesh_affine(GMesh m,const double* matrix){
    for(int i=0;i<vertices(m);++i)set_vertex(m,i,affine(vertex(m,i),matrix));
    double det=matrix[0]*(matrix[5]*matrix[10]-matrix[6]*matrix[9])-matrix[1]*(matrix[4]*matrix[10]-matrix[6]*matrix[8])+matrix[2]*(matrix[4]*matrix[9]-matrix[5]*matrix[8]);
    if(fabs(det)<1e-20){m.a[2]=108;return;}if(det<0)for(int t=0;t<triangles(m);++t){int k=8+6*MV+3*t;double q=m.a[k+1];m.a[k+1]=m.a[k+2];m.a[k+2]=q;}
}
extern "C" __global__ void mesh_base(double* packet,unsigned offset){double* arena=packet+offset;
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>8)return;GMesh m={arena+id*MESH_WORDS};const double pi=3.14159265358979323846;
    if(id==0)make_box(m,p3(-1,-1,-1),p3(1,1,1));
    if(id==1)make_uv_sphere(m,p3(0,0,0),1,24,12);
    if(id==2||id==3)make_frustum(m,p3(0,0,0),1,id==2?1:0,2,24);
    if(id==4)make_torus(m,p3(0,0,0),1,.3,24,16);
    if(id==5)make_grid(m,p3(0,0,0),p3(1,0,0),p3(0,1,0),8,8,.15);
    if(id==6){P3 p[6]={p3(0,0,0),p3(2,0,0),p3(2,1,0),p3(1,1,0),p3(1,2,0),p3(0,2,0)};triangulate_polygon(m,p,6);}
    if(id==7){P3 p[6]={p3(0,0,0),p3(2,0,0),p3(2,1,0),p3(1,1,0),p3(1,2,0),p3(0,2,0)};extrude_polygon(m,p,6,p3(.2,.1,2));}
    if(id==8){P3 p[5];for(int i=0;i<5;++i)p[i]=p3(cos(2*pi*i/5),sin(2*pi*i/5),0);inset_polygon(m,p,5,.25);}
}
extern "C" __global__ void mesh_derived(double* packet,unsigned offset){double* arena=packet+offset;
    int id=9+blockIdx.x*blockDim.x+threadIdx.x;if(id>=MESH_COUNT)return;GMesh m={arena+id*MESH_WORDS},box={arena},torus={arena+4*MESH_WORDS};
    if(id==9)subdivide_triangles(box,m);
    if(id==10){bool seen[8]={false};for(int t=0;t<triangles(box);++t){int k=vertices(m);for(int j=0;j<3;++j){int index=index_at(box,t,j);double noise=seen[index]?1e-12:0;add_vertex(m,plus(vertex(box,index),p3(noise,-noise,0)));seen[index]=true;}add_triangle(m,k,k+1,k+2);}weld_vertices(m,1e-10);}
    if(id==11){for(int i=0;i<vertices(torus);++i)add_vertex(m,vertex(torus,i));for(int t=0;t<triangles(torus);++t)add_triangle(m,index_at(torus,t,0),index_at(torus,t,1),index_at(torus,t,2));}
    if(id==12){P3 p[8];for(int i=0;i<8;++i)p[i]=vertex(box,i);int faces[24]={0,2,3,1,4,5,7,6,0,4,6,2,1,3,7,5,0,1,5,4,2,6,7,3},offsets[7]={0,4,8,12,16,20,24};bevel_vertices(m,p,8,faces,offsets,6,.2);}
    if(id==13){P3 p[4]={p3(0,0,0),p3(1,1,0),p3(0,1,0),p3(1,0,0)};triangulate_polygon(m,p,4);}
    if(id==14){P3 p[3]={p3(0,0,0),p3(1,0,0),p3(0,1,0)};extrude_polygon(m,p,3,p3(1,0,0));}
    if(id==15){P3 p[5]={p3(-1,-1,0),p3(1,-1,0),p3(1,1,0),p3(-1,1,0),p3(0,0,1)};int faces[16]={0,3,2,1,0,1,4,1,2,4,2,3,4,3,0,4},offsets[6]={0,4,7,10,13,16};bevel_vertices(m,p,5,faces,offsets,5,.2);}
}
extern "C" __global__ void mesh_finalize(double* packet,unsigned offset){double* arena=packet+offset;int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=MESH_COUNT)return;GMesh m={arena+id*MESH_WORDS};if(m.a[2])return;mesh_normals(m);}
