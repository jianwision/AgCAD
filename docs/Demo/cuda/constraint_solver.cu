// Fixed-topology affine least squares, hard equalities eliminated by CPU SVD.
// One block per candidate for reductions; rows/variables are separate threads.
extern "C" {
__device__ int tid(){return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ long istride(int n,int m,int k,int r){return (long)m*k+m+(long)n*k+n+(long)r*n+r+k;}
__device__ long wstride(int n,int m,int k,int r){return 7L*k+2*m+8+n+r;}
__device__ double reduce_sum(double value){
    __shared__ double s[256];
    int t=threadIdx.x;s[t]=value;__syncthreads();
    for(int d=128;d;d/=2){if(t<d)s[t]+=s[t+d];__syncthreads();}
    double result=s[0];__syncthreads();return result;
}
__global__ void initialize(const double* in,double* out,int n,int m,int k,int r){
    int c=blockIdx.y,j=tid();if(j>=k)return;
    const double* b=in+16+c*istride(n,m,k,r);const double* d=b+(long)m*k;
    const double* diagonal=d+m+(long)n*k+n+(long)r*n+r;
    double* w=out+c*wstride(n,m,k,r);
    double rhs=0;for(int i=0;i<m;i++)rhs+=b[(long)i*k+j]*d[i];
    w[j]=0;w[k+j]=rhs;w[2*k+j]=rhs/diagonal[j];w[3*k+j]=rhs/diagonal[j];
}
__global__ void initialize_reduce(double* out,int n,int m,int k,int r,double tol){
    int c=blockIdx.x,t=threadIdx.x;double* w=out+c*wstride(n,m,k,r);
    double rho=0,norm=0;for(int j=t;j<k;j+=256){rho+=w[k+j]*w[2*k+j];norm+=w[k+j]*w[k+j];}
    rho=reduce_sum(rho);norm=reduce_sum(norm);
    if(t==0){double* s=w+7*k+2*m;s[0]=norm<=tol*tol?1:0;s[1]=0;s[2]=rho;s[3]=rho;s[6]=norm;}
}
__global__ void matvec(const double* in,double* out,int n,int m,int k,int r,int final){
    int c=blockIdx.y,i=tid();if(i>=m)return;
    const double* b=in+16+c*istride(n,m,k,r);double* w=out+c*wstride(n,m,k,r);
    if(!final && w[7*k+2*m]!=0)return;
    const double* v=w+(final?0:3*k);double sum=0;
    for(int j=0;j<k;j++)sum+=b[(long)i*k+j]*v[j];
    w[7*k+i]=sum;
}
__global__ void transpose(const double* in,double* out,int n,int m,int k,int r,int final){
    int c=blockIdx.y,j=tid();if(j>=k)return;
    const double* b=in+16+c*istride(n,m,k,r);const double* d=b+(long)m*k;
    double* w=out+c*wstride(n,m,k,r);if(!final && w[7*k+2*m]!=0)return;
    double sum=0;for(int i=0;i<m;i++)sum+=b[(long)i*k+j]*(w[7*k+i]-(final?d[i]:0));
    w[(final?5:4)*k+j]=sum;
}
__global__ void step_size(double* out,int n,int m,int k,int r){
    int c=blockIdx.x,t=threadIdx.x;double* w=out+c*wstride(n,m,k,r);double* s=w+7*k+2*m;
    if(s[0]!=0)return;
    double php=0;for(int j=t;j<k;j+=256)php+=w[3*k+j]*w[4*k+j];php=reduce_sum(php);
    if(t==0){if(!(php>0)||!isfinite(php)||!isfinite(s[2]))s[0]=2;else s[4]=s[2]/php;}
}
__global__ void update(const double* in,double* out,int n,int m,int k,int r){
    int c=blockIdx.y,j=tid();if(j>=k)return;
    const double* b=in+16+c*istride(n,m,k,r);const double* diagonal=b+(long)m*k+m+(long)n*k+n+(long)r*n+r;
    double* w=out+c*wstride(n,m,k,r);double* s=w+7*k+2*m;if(s[0]!=0)return;
    w[j]+=s[4]*w[3*k+j];w[k+j]-=s[4]*w[4*k+j];w[2*k+j]=w[k+j]/diagonal[j];
}
__global__ void convergence(double* out,int n,int m,int k,int r,double tol){
    int c=blockIdx.x,t=threadIdx.x;double* w=out+c*wstride(n,m,k,r);double* s=w+7*k+2*m;
    if(s[0]!=0)return;
    double rho=0,norm=0;for(int j=t;j<k;j+=256){rho+=w[k+j]*w[2*k+j];norm+=w[k+j]*w[k+j];}
    rho=reduce_sum(rho);norm=reduce_sum(norm);
    if(t==0){s[1]+=1;s[5]=rho/s[2];s[2]=rho;
        if(!isfinite(norm)||!isfinite(rho))s[0]=2;
        else if(norm<=tol*tol*fmax(s[6],1.))s[0]=1;}
}
__global__ void direction(double* out,int n,int m,int k,int r){
    int c=blockIdx.y,j=tid();if(j>=k)return;
    double* w=out+c*wstride(n,m,k,r);double* s=w+7*k+2*m;if(s[0]!=0)return;
    w[3*k+j]=w[2*k+j]+s[5]*w[3*k+j];
}
__global__ void reconstruct(const double* in,double* out,int n,int m,int k,int r){
    int c=blockIdx.y,i=tid();if(i>=n)return;
    const double* b=in+16+c*istride(n,m,k,r);const double* z=b+(long)m*k+m;
    const double* x0=z+(long)n*k;double* w=out+c*wstride(n,m,k,r);
    double value=x0[i];for(int j=0;j<k;j++)value+=z[(long)i*k+j]*w[j];
    w[7*k+2*m+8+i]=value;
}
__global__ void residuals(const double* in,double* out,int n,int m,int k,int r){
    int c=blockIdx.y,i=tid();if(i>=r)return;
    const double* b=in+16+c*istride(n,m,k,r);const double* a=b+(long)m*k+m+(long)n*k+n;
    const double* targets=a+(long)r*n;double* w=out+c*wstride(n,m,k,r);const double* x=w+7*k+2*m+8;
    double value=-targets[i];for(int j=0;j<n;j++)value+=a[(long)i*n+j]*x[j];
    w[7*k+2*m+8+n+i]=value;
}
__global__ void probes(const double* in,double* out,int n,int m,int k,int r){
    int c=blockIdx.y,i=tid();const double* b=in+16+c*istride(n,m,k,r);double* w=out+c*wstride(n,m,k,r);
    if(i<m){double value=0;for(int j=0;j<k;j++)value+=b[(long)i*k+j]*sin(.37*(j+1));w[7*k+m+i]=value;}
    if(i<k){double value=0;for(int j=0;j<m;j++)value+=b[(long)j*k+i]*cos(.23*(j+1));w[6*k+i]=value;}
}
__global__ void finalize(double* out,int n,int m,int k,int r,double tol){
    int c=blockIdx.x,t=threadIdx.x;double* w=out+c*wstride(n,m,k,r);double* s=w+7*k+2*m;
    double norm=0;for(int j=t;j<k;j+=256)norm+=w[5*k+j]*w[5*k+j];norm=reduce_sum(norm);
    if(t==0){s[7]=sqrt(norm/fmax(s[6],1.));
        if(s[0]!=2)s[0]=(isfinite(norm)&&s[7]<=10*tol)?1:3;}
}
}
