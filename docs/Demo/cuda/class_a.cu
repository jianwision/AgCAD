// Batch polynomial Bezier jets / reflection diagnostics / explicit bounded fairing.
__device__ double choose(int n,int k){if(k<0||k>n)return 0;double x=1;for(int i=1;i<=k;i++)x=x*(n-i+1)/i;return x;}
__device__ double basis(int n,int i,double t){return i<0||i>n?0:choose(n,i)*pow(t,i)*pow(1-t,n-i);}
__device__ double derivative(int n,int i,int d,double t){
    double x=0,f=1;for(int k=0;k<d;k++)f*=n-k;
    for(int k=0;k<=d;k++)x+=((d-k)%2?-1:1)*choose(d,k)*basis(n-d,i-k,t);return f*x;
}
__device__ double dot3(const double* a,const double* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
extern "C" __global__ void class_a_batch(const double* input,double* output){
    const int c=blockIdx.x,k=threadIdx.x,samples=(int)input[3];
    const double* record=input+16+4360*c;const int nu=(int)record[0],nv=(int)record[1],iterations=(int)record[2];
    const double budget=record[3],step=record[4];const double* origin=record+8;
    const double* mask=origin+192;const double* h=mask+64;
    __shared__ double poles[192],trial[192];
    for(int a=0;a<3;a++)poles[3*k+a]=origin[3*k+a];__syncthreads();
    for(int round=0;round<iterations;round++){
        double delta[3];
        for(int a=0;a<3;a++){
            double gradient=0;for(int j=0;j<64;j++)gradient+=h[64*k+j]*poles[3*j+a];
            delta[a]=mask[k]?0:poles[3*k+a]-2*step*gradient-origin[3*k+a];
        }
        const double norm=sqrt(dot3(delta,delta));const double fraction=norm>budget?budget/fmax(norm,1e-300):1;
        for(int a=0;a<3;a++)trial[3*k+a]=origin[3*k+a]+fraction*delta[a];
        __syncthreads();for(int a=0;a<3;a++)poles[3*k+a]=trial[3*k+a];__syncthreads();
    }
    double* out=output+c*(192+48*samples*samples);
    for(int a=0;a<3;a++)out[3*k+a]=poles[3*k+a];
    const int orders[10][2]={{0,0},{1,0},{0,1},{2,0},{1,1},{0,2},{3,0},{2,1},{1,2},{0,3}};
    for(int sample=k;sample<samples*samples;sample+=64){
        double* row=out+192+48*sample;for(int i=0;i<48;i++)row[i]=0;
        double u=double(sample/samples)/(samples-1),v=double(sample%samples)/(samples-1);
        for(int d=0;d<10;d++)for(int i=0;i<nu;i++)for(int j=0;j<nv;j++){
            double b=derivative(nu-1,i,orders[d][0],u)*derivative(nv-1,j,orders[d][1],v);
            for(int a=0;a<3;a++)row[3*d+a]+=b*poles[24*i+3*j+a];
        }
        double* a=row+3;double* b=row+6;
        double cross[3]={a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
        double E=dot3(a,a),F=dot3(a,b),G=dot3(b,b),det=dot3(cross,cross),jac=sqrt(det);row[33]=jac;
        if(E<1e-24||G<1e-24||jac<1e-12*sqrt(E*G))continue;
        for(int axis=0;axis<3;axis++)row[30+axis]=cross[axis]/jac;
        double e=dot3(row+30,row+9),f=dot3(row+30,row+12),g=dot3(row+30,row+15);
        double H=(e*G-2*f*F+g*E)/(2*det),K=(e*g-f*f)/det;
        row[34]=H;row[35]=K;double spread=sqrt(fmax(0,H*H-K));row[36]=H-spread;row[37]=H+spread;
        const double view[3]={.3/sqrt(1.13),.2/sqrt(1.13),1/sqrt(1.13)};double d=dot3(row+30,view);
        for(int axis=0;axis<3;axis++)row[38+axis]=2*d*row[30+axis]-view[axis];
        row[41]=cos(62.83185307179586*(row[38]+.2*row[39]+.1*row[40]));row[42]=1;
        row[43]=jac/sqrt(E*G);row[44]=row[36]*row[36]+row[37]*row[37];
    }
}
