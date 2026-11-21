// Persistent positive-weight NURBS jets and constrained multi-patch search.
// Basis tables are compiled once on CPU; all repeated geometry stays on device.
#define SLOT_WORDS 4096
__device__ double dp(const double*a,const double*b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ void cross3(const double*a,const double*b,double*c){c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
__device__ int comb(int n,int k){return k==0||k==n?1:n;}
__device__ int jet_index(int a,int b){const int du[10]={0,1,0,2,1,0,3,2,1,0},dv[10]={0,0,1,0,1,2,0,1,2,3};for(int i=0;i<10;i++)if(du[i]==a&&dv[i]==b)return i;return 0;}
__device__ void evaluate(const double*cp,int nv,int p,int q,const double*bs,double*out){
    const int du[10]={0,1,0,2,1,0,3,2,1,0},dv[10]={0,0,1,0,1,2,0,1,2,3};
    double h[10][4]={0};for(int i=0;i<48;i++)out[i]=0;
    for(int i=0;i<=p;i++)for(int j=0;j<=q;j++){
        const double*c=cp+4*((int(bs[0])+i)*nv+int(bs[1])+j);
        for(int d=0;d<10;d++){
            double b=bs[2+8*du[d]+i]*bs[34+8*dv[d]+j]*c[3];
            for(int a=0;a<3;a++)h[d][a]+=b*c[a];h[d][3]+=b;
        }
    }
    for(int d=0;d<10;d++)for(int axis=0;axis<3;axis++){
        double v=h[d][axis];
        for(int a=0;a<=du[d];a++)for(int b=0;b<=dv[d];b++)if(a||b)
            v-=comb(du[d],a)*comb(dv[d],b)*h[jet_index(a,b)][3]*out[3*jet_index(du[d]-a,dv[d]-b)+axis];
        out[3*d+axis]=v/h[0][3];
    }
    double*a=out+3,*b=out+6,n[3];cross3(a,b,n);
    double E=dp(a,a),F=dp(a,b),G=dp(b,b),D=dp(n,n),jac=sqrt(D);out[33]=jac;out[47]=jac*bs[66];
    if(E<1e-24||G<1e-24||jac<1e-12*sqrt(E*G))return;
    for(int k=0;k<3;k++)out[30+k]=n[k]/jac;
    double*normal=out+30;double e=dp(normal,out+9),f=dp(normal,out+12),g=dp(normal,out+15);
    double numerator=e*G-2*f*F+g*E,H=numerator/(2*D),K=(e*g-f*f)/D;
    out[34]=H;out[35]=K;double spread=sqrt(fmax(0,H*H-K));out[36]=H-spread;out[37]=H+spread;
    const double view[3]={.3/sqrt(1.13),.2/sqrt(1.13),1/sqrt(1.13)};double viewdot=dp(normal,view);
    for(int k=0;k<3;k++)out[38+k]=2*viewdot*normal[k]-view[k];
    out[41]=cos(62.83185307179586*(out[38]+.2*out[39]+.1*out[40]));out[42]=1;out[43]=jac/sqrt(E*G);out[44]=out[36]*out[36]+out[37]*out[37];
    double hn[2],rn[2][3];
    for(int axis=0;axis<2;axis++){
        double*aa=out+(axis?12:9),*ab=out+(axis?15:12),c1[3],c2[3],nd[3];cross3(aa,b,c1);cross3(a,ab,c2);
        for(int k=0;k<3;k++)c1[k]+=c2[k];double projection=dp(c1,normal);
        for(int k=0;k<3;k++)nd[k]=(c1[k]-normal[k]*projection)/jac;
        double Ed=2*dp(a,aa),Fd=dp(aa,b)+dp(a,ab),Gd=2*dp(b,ab),Dd=Ed*G+E*Gd-2*F*Fd;
        double ed=dp(nd,out+9)+dp(normal,out+(axis?21:18));
        double fd=dp(nd,out+12)+dp(normal,out+(axis?24:21));
        double gd=dp(nd,out+15)+dp(normal,out+(axis?27:24));
        double Nd=ed*G+e*Gd-2*fd*F-2*f*Fd+gd*E+g*Ed;
        hn[axis]=(Nd*D-numerator*Dd)/(2*D*D);double vd=dp(nd,view);
        for(int k=0;k<3;k++)rn[axis][k]=2*(vd*normal[k]+viewdot*nd[k]);
    }
    out[45]=fmax(0,(G*hn[0]*hn[0]-2*F*hn[0]*hn[1]+E*hn[1]*hn[1])/D);
    out[46]=fmax(0,(G*dp(rn[0],rn[0])-2*F*dp(rn[0],rn[1])+E*dp(rn[1],rn[1]))/D);
}
extern "C" __global__ void surface_eval(const double*arena,int slot,int nv,int p,int q,const double*basis,int count,double*out){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<count)evaluate(arena+slot*SLOT_WORDS,nv,p,q,basis+67*k,out+48*k);
}
extern "C" __global__ void surface_summary(const double*grid,int count,double*out){
    if(threadIdx.x)return;double area=0,curv=0,reflection=0,mink=1e300,maxk=-1e300,regular=0,minjac=1;
    for(int k=0;k<count;k++){const double*r=grid+48*k;regular+=r[42];mink=fmin(mink,r[36]);maxk=fmax(maxk,r[37]);minjac=fmin(minjac,r[43]);area+=r[47];curv+=r[45]*r[47];reflection+=r[46]*r[47];}
    out[0]=count;out[1]=regular;out[2]=mink;out[3]=maxk;out[4]=area;out[5]=curv/fmax(area,1e-300);out[6]=reflection/fmax(area,1e-300);out[7]=minjac;
}
extern "C" __global__ void surface_clone(double*arena,int source,int dest,int words){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<words)arena[dest*SLOT_WORDS+k]=arena[source*SLOT_WORDS+k];}
extern "C" __global__ void surface_refine(double*arena,int source,int dest,const double*in){
    int k=blockIdx.x*blockDim.x+threadIdx.x,count=int(in[0]*in[1]),old=int(in[2]*in[3]);if(k>=count)return;
    double h[4]={0};for(int j=0;j<old;j++){const double*p=arena+source*SLOT_WORDS+4*j;double b=in[4+k*old+j]*p[3];for(int a=0;a<3;a++)h[a]+=b*p[a];h[3]+=b;}
    for(int a=0;a<3;a++)arena[dest*SLOT_WORDS+4*k+a]=h[a]/h[3];arena[dest*SLOT_WORDS+4*k+3]=h[3];
}
// Network input: header, metadata, projection, nullspace, seed compatibility, basis.
extern "C" __global__ void network_gather(const double*arena,const double*in,double*seed){
    int k=blockIdx.x*blockDim.x+threadIdx.x,N=int(in[1]);if(k>=N)return;
    for(int p=0;p<int(in[0]);p++){const double*m=in+16+8*p;int off=int(m[2]),n=int(m[3]*m[4]);if(k>=off&&k<off+n)for(int a=0;a<4;a++)seed[4*k+a]=arena[int(m[0])*SLOT_WORDS+4*(k-off)+a];}
}
extern "C" __global__ void network_project(const double*in,const double*seed,double*current){
    int k=blockIdx.x*blockDim.x+threadIdx.x,N=int(in[1]);if(k>=N)return;const double*P=in+16+8*int(in[0]);
    for(int a=0;a<3;a++){double v=0;for(int j=0;j<N;j++)v+=P[k*N+j]*seed[4*j+a];current[4*k+a]=v;}current[4*k+3]=seed[4*k+3];
}
extern "C" __global__ void network_prepare(const double*in,const double*current,const double*state,double*candidates){
    int k=blockIdx.x*blockDim.x+threadIdx.x,N=int(in[1]),df=int(in[2]);if(k>=N)return;
    const double*Q=in+16+8*int(in[0])+N*N;int round=int(state[0]),mode=df?round%(3*df):0;
    double amplitude=in[5]*pow(.5,df?round/(3*df):0);
    for(int c=0;c<3;c++)for(int a=0;a<4;a++)candidates[c*4*N+4*k+a]=current[4*k+a]+((df&&a<3&&a==mode%3)?(c==1?1:c==2?-1:0)*amplitude*Q[k*df+mode/3]:0);
}
extern "C" __global__ void network_eval(const double*in,const double*positions,int copies,double*grid){
    int k=blockIdx.x*blockDim.x+threadIdx.x,np=int(in[0]),N=int(in[1]),df=int(in[2]),M=int(in[3]),total=np*M;if(k>=copies*total)return;
    int c=k/total,p=(k%total)/M,t=k%M;const double*m=in+16+8*p,*basis=in+16+8*np+N*N+N*df+N*int(in[11]);
    evaluate(positions+c*4*N+4*int(m[2]),int(m[4]),int(m[5]),int(m[6]),basis+67*(p*M+t),grid+48*k);
}
extern "C" __global__ void network_score(const double*in,const double*seed,const double*reference,const double*candidates,const double*grid,double*scores){
    int c=blockIdx.x,k=threadIdx.x,N=int(in[1]),total=int(in[0]*in[3]);double area=0,curv=0,refl=0,fit=0,bad=0,maximum=0;
    for(int i=k;i<total;i+=64){const double*r=grid+48*(c*total+i),*b=reference+48*i;double dist=0;for(int a=0;a<3;a++)dist+=(r[a]-b[a])*(r[a]-b[a]);area+=r[47];curv+=r[45]*r[47];refl+=r[46]*r[47];fit+=dist*r[47];bad+=r[42]!=1;}
    for(int i=k;i<N;i+=64){double d=0;for(int a=0;a<3;a++){double value=candidates[c*4*N+4*i+a];if(!isfinite(value)||fabs(value)>1e6)bad+=1;double v=value-seed[4*i+a];d+=v*v;}maximum=fmax(maximum,sqrt(d));}
    const double*compat=in+16+8*int(in[0])+N*N+N*int(in[2]);
    for(int i=k;i<int(in[11]);i+=64)for(int a=0;a<3;a++){double mismatch=0;for(int j=0;j<N;j++)mismatch+=compat[i*N+j]*seed[4*j+a];if(fabs(mismatch)>1e-9)bad+=1;}
    __shared__ double sums[6][64];sums[0][k]=area;sums[1][k]=curv;sums[2][k]=refl;sums[3][k]=fit;sums[4][k]=bad;sums[5][k]=maximum;__syncthreads();
    for(int stride=32;stride;stride/=2){if(k<stride){for(int j=0;j<5;j++)sums[j][k]+=sums[j][k+stride];sums[5][k]=fmax(sums[5][k],sums[5][k+stride]);}__syncthreads();}
    if(k==0){double A=fmax(sums[0][0],1e-300),L=in[7],C=sums[1][0]/A,R=sums[2][0]/A,F=sums[3][0]/A;
        double objective=in[8]*C*L*L*L*L+in[9]*R*L*L+in[10]*F/fmax(in[6]*in[6],1e-30);
        double*out=scores+8*c;out[0]=(sums[4][0]||sums[5][0]>in[6]*(1+1e-10)||!isfinite(objective))?1e300:objective;
        out[1]=sums[5][0];out[2]=C;out[3]=R;out[4]=F;out[5]=total-sums[4][0];out[6]=sums[0][0];out[7]=0;}
}
extern "C" __global__ void network_choose(const double*in,const double*candidates,const double*scores,double*current,double*state,int initial){
    int k=blockIdx.x*blockDim.x+threadIdx.x,N=int(in[1]),best=0;
    if(!initial){if(scores[8]<scores[0])best=1;if(scores[16]<scores[8*best])best=2;}
    if(k<4*N)current[k]=candidates[best*4*N+k];
    if(k==0){if(initial){for(int a=0;a<8;a++)state[2+a]=scores[a];}else{state[0]+=1;state[1]+=best!=0;}for(int a=0;a<8;a++)state[10+a]=scores[8*best+a];}
}
extern "C" __global__ void network_publish(double*arena,const double*in,const double*current){
    int k=blockIdx.x*blockDim.x+threadIdx.x,N=int(in[1]);if(k>=N)return;
    for(int p=0;p<int(in[0]);p++){const double*m=in+16+8*p;int off=int(m[2]),n=int(m[3]*m[4]);if(k>=off&&k<off+n)for(int a=0;a<4;a++)arena[int(m[1])*SLOT_WORDS+4*(k-off)+a]=current[4*k+a];}
}
