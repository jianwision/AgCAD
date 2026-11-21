// Batched control-net constructors. Every record has a bounded NURBS object;
// offset records reference another object without fitting it to a control net.
const int SURF_COUNT=14,SURF_WORDS=4192,SURF_SAMPLES=81;
__device__ double* surf_record(double* arena,int id){return arena+id*SURF_WORDS;}
__device__ const double* surf_record(const double* arena,int id){return arena+id*SURF_WORDS;}
__device__ void setup_surface(double* s,int nu,int nv,int pu,int pv){
    s[0]=nu;s[1]=nv;s[2]=pu;s[3]=pv;s[4]=0;s[5]=1;s[6]=0;s[7]=1;s[8]=0;s[11]=0;
    for(int i=0;i<nu+pu+1;++i)s[16+i]=i<=pu?0:1;
    for(int j=0;j<nv+pv+1;++j)s[56+j]=j<=pv?0:1;
}
__device__ void set_pole(double* s,int i,int j,P3 p,double w=1){int k=96+4*(i*int(s[1])+j);s[k]=p.x;s[k+1]=p.y;s[k+2]=p.z;s[k+3]=w;}
