// D0-D4 diagnostic executor. f64 geometry, no learned feedback ABI implied.
// Orientation is the signed XY projection used by the original hood contract.
struct Header { char magic[8]; unsigned version,patches,systems,repeats,programs,rounds; };
static_assert(sizeof(Header)==32,"diagnostic header ABI");
__device__ double choose(unsigned n,unsigned k){
    double v=1.;for(unsigned i=1;i<=k;i++)v=v*(n+1-i)/i;return v;
}
__device__ void cubic(double t,double* b,double* d){
    double s=1-t;b[0]=s*s*s;b[1]=3*t*s*s;b[2]=3*t*t*s;b[3]=t*t*t;
    d[0]=-3*s*s;d[1]=3*s*s-6*t*s;d[2]=6*t*s-3*t*t;d[3]=3*t*t;
}
__device__ double jac_at(const double* p,double u,double v){
    double a[4],da[4],b[4],db[4];cubic(u,a,da);cubic(v,b,db);
    double ux=0,uy=0,vx=0,vy=0;
    for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++){
        ux+=da[i]*b[j]*p[12*i+3*j];uy+=da[i]*b[j]*p[12*i+3*j+1];
        vx+=a[i]*db[j]*p[12*i+3*j];vy+=a[i]*db[j]*p[12*i+3*j+1];
    }return ux*vy-uy*vx;
}
__device__ void jac_coefficients(const double* p,double* c){
    for(unsigned r=0;r<36;r++)c[r]=0.;
    for(unsigned i=0;i<3;i++)for(unsigned j=0;j<4;j++)
    for(unsigned a=0;a<4;a++)for(unsigned b=0;b<3;b++){
        unsigned u=12*i+3*j,v=12*a+3*b;
        double ux=3*(p[u+12]-p[u]),uy=3*(p[u+13]-p[u+1]);
        double vx=3*(p[v+3]-p[v]),vy=3*(p[v+4]-p[v+1]);
        double w=choose(2,i)*choose(3,a)/choose(5,i+a)*choose(3,j)*choose(2,b)/choose(5,j+b);
        c[6*(i+a)+j+b]+=w*(ux*vy-uy*vx);
    }
}
__device__ void split(const double* src,double* left,double* right,unsigned axis){
    for(unsigned line=0;line<6;line++){
        double a[6];for(unsigned i=0;i<6;i++)a[i]=src[axis==0?6*i+line:6*line+i];
        left[axis==0?line:6*line]=a[0];right[axis==0?30+line:6*line+5]=a[5];
        for(unsigned depth=1;depth<6;depth++){
            for(unsigned i=0;i<6-depth;i++)a[i]=.5*(a[i]+a[i+1]);
            left[axis==0?6*depth+line:6*line+depth]=a[0];
            right[axis==0?6*(5-depth)+line:6*line+5-depth]=a[5-depth];
        }
    }
}
__device__ bool hood_valid(const double* p){
    // Raw hood semantics, no dependence on the original constraint row compiler.
    for(unsigned i=0;i<192;i++)if(!isfinite(p[i])||fabs(p[i])>1e6)return false;
    for(unsigned pu=0;pu<2;pu++)for(unsigned pv=0;pv<2;pv++){
        unsigned base=48*(2*pu+pv);
        double xx[4]={double(pu*pu),pu*pu+2*pu/3.,pu*pu+4*pu/3.+1/3.,double((pu+1)*(pu+1))};
        double yy[4]={double(pv*pv),pv*pv+2*pv/3.,pv*pv+4*pv/3.+1/3.,double((pv+1)*(pv+1))};
        for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++){
            unsigned index=base+12*i+3*j;
            if(fabs(p[index]-(-1500+1500*(pu+i/3.)))>1e-7||fabs(p[index+1]-(-750+750*(pv+j/3.)))>1e-7)return false;
            if((pu==0&&i==0)||(pu==1&&i==3)||(pv==0&&j==0)||(pv==1&&j==3)){
                double z=240-60*(xx[i]-2*(pu+i/3.)+1)-40*(yy[j]-2*(pv+j/3.)+1);
                if(fabs(p[index+2]-z)>1e-7)return false;
            }
        }
    }
    for(unsigned axis=0;axis<2;axis++)for(unsigned side=0;side<2;side++)
    for(unsigned j=0;j<4;j++)for(unsigned d=0;d<3;d++)for(unsigned xyz=0;xyz<3;xyz++){
        unsigned pa=axis==0?side:2*side,pb=axis==0?2+side:2*side+1;
        double av=0,bv=0;const double weights[3][3]={{1,0,0},{3,-3,0},{6,-12,6}};
        for(unsigned k=0;k<=d;k++){
            unsigned ia=axis==0?12*(3-k)+3*j:12*j+3*(3-k);
            unsigned ib=axis==0?12*k+3*j:12*j+3*k;
            av+=weights[d][k]*p[48*pa+ia+xyz];
            bv+=weights[d][k]*p[48*pb+ib+xyz]*(d==1?-1:1);
        }
        if(fabs(av-bv)>1e-7)return false;
    }
    for(unsigned pu=0;pu<2;pu++)for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++)for(unsigned xyz=0;xyz<3;xyz++){
        double a=p[96*pu+12*i+3*j+xyz],b=p[96*pu+48+12*i+3*(3-j)+xyz];
        if(fabs(a-(xyz==1?-b:b))>1e-7)return false;
    }
    // Fixed XY implies a positive constant XY Jacobian on each patch.
    return true;
}
__device__ double probe_score(const double* q,const double* target){
    double s=0;for(unsigned j=0;j<6;j++)s+=(q[j]-target[j])*(q[j]-target[j]);return s;
}
extern "C" {
__global__ void diagnostic_geometry(const unsigned char* input,double* out){
    const Header* h=(const Header*)input;unsigned id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=h->patches)return;
    const double* p=(const double*)(input+32)+48*id;double* r=out+8*id;
    for(unsigned i=0;i<48;i++)if(!isfinite(p[i])){r[0]=-1;r[1]=0;r[2]=3;return;}
    double sampled=1e300,coeff[36];
    for(unsigned i=0;i<5;i++)for(unsigned j=0;j<5;j++)sampled=fmin(sampled,jac_at(p,i/4.,j/4.));
    r[0]=sampled;r[1]=sampled>1e-6?1:0;
    jac_coefficients(p,coeff);double magnitude=0;
    for(unsigned i=0;i<36;i++)magnitude=fmax(magnitude,fabs(coeff[i]));
    // Conservative numerical decision margin, not outward-rounded interval arithmetic.
    double margin=1e-12*fmax(magnitude,1.);r[5]=margin;
    double stack[9][36];unsigned depths[9];unsigned size=1,nodes=0;bool unknown=false;
    for(unsigned i=0;i<36;i++)stack[0][i]=coeff[i];depths[0]=0;
    while(size){
        --size;unsigned depth=depths[size];double a[36],lo=1e300,hi=-1e300;
        for(unsigned i=0;i<36;i++){a[i]=stack[size][i];lo=fmin(lo,a[i]);hi=fmax(hi,a[i]);}
        ++nodes;
        // Corners are actual surface values. A nonpositive witness rejects the patch.
        if(fmin(fmin(a[0],a[5]),fmin(a[30],a[35]))<1e-6-margin){r[2]=2;r[3]=nodes;return;}
        if(lo>1e-6+margin)continue;
        if(depth==8){unknown=true;continue;}
        double left[36],right[36];split(a,left,right,depth%2);
        for(unsigned i=0;i<36;i++){stack[size][i]=right[i];stack[size+1][i]=left[i];}
        depths[size]=depth+1;depths[size+1]=depth+1;size+=2;
    }
    r[2]=unknown?0:1;r[3]=nodes;
}
__global__ void diagnostic_solve(const unsigned char* input,double* out,unsigned method){
    const Header* h=(const Header*)input;unsigned job=blockIdx.x*blockDim.x+threadIdx.x;
    if(job>=h->systems)return;
    unsigned id=job;const double* s=(const double*)(input+32)+48*h->patches+49*id;
    double* r=out+8*h->patches+10*(2*job+method);double x[6]={0,0,0,0,0,0};
    for(unsigned i=0;i<10;i++)r[i]=0.;
    if(method==0){
        double a[6][7];for(unsigned i=0;i<6;i++){for(unsigned j=0;j<6;j++)a[i][j]=s[6*i+j];a[i][6]=s[36+i];}
        for(unsigned col=0;col<6;col++){
            unsigned pivot=col;for(unsigned i=col+1;i<6;i++)if(fabs(a[i][col])>fabs(a[pivot][col]))pivot=i;
            if(fabs(a[pivot][col])<1e-14){r[9]=-1;return;}
            for(unsigned j=col;j<7;j++){double v=a[col][j];a[col][j]=a[pivot][j];a[pivot][j]=v;}
            for(unsigned i=col+1;i<6;i++){double f=a[i][col]/a[col][col];for(unsigned j=col;j<7;j++)a[i][j]-=f*a[col][j];}
        }
        for(int i=5;i>=0;i--){double v=a[i][6];for(unsigned j=i+1;j<6;j++)v-=a[i][j]*x[j];x[i]=v/a[i][i];}
        r[7]=1;
    }else{
        for(unsigned i=0;i<6;i++)x[i]=s[43+i];
        for(unsigned step=0;step<128;step++){
            double g[6],hg[6],gg=0,ghg=0;
            for(unsigned i=0;i<6;i++){g[i]=-s[36+i];for(unsigned j=0;j<6;j++)g[i]+=s[6*i+j]*x[j];gg+=g[i]*g[i];}
            for(unsigned i=0;i<6;i++){hg[i]=0;for(unsigned j=0;j<6;j++)hg[i]+=s[6*i+j]*g[j];ghg+=g[i]*hg[i];}
            if(gg<1e-22)break;
            if(!(ghg>0)){r[9]=-1;return;}
            for(unsigned i=0;i<6;i++)x[i]-=gg/ghg*g[i];r[7]+=1;
        }
    }
    double value=s[42],gn=0;
    for(unsigned i=0;i<6;i++){
        double hx=0;for(unsigned j=0;j<6;j++)hx+=s[6*i+j]*x[j];
        value+=x[i]*(hx-2*s[36+i]);gn+=(hx-s[36+i])*(hx-s[36+i]);r[i]=x[i];
    }
    r[6]=value;r[8]=sqrt(gn);r[9]=1;
}
__global__ void diagnostic_compare(double* out,const double* previous,unsigned words){
    if(blockIdx.x||threadIdx.x)return;unsigned mismatches=0;
    for(unsigned i=0;i<words-1;i++)if(__double_as_longlong(out[i])!=__double_as_longlong(previous[i]))++mismatches;
    out[words-1]=mismatches;
}
__global__ void diagnostic_proposals(const unsigned char* input,double* out){
    const Header* h=(const Header*)input;unsigned id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=h->programs)return;
    const double* shared=(const double*)(input+32)+48*h->patches+49*h->systems;
    const double* x0=shared;const double* map=x0+192;const double* probes=map+1152;const double* weights=probes+1152;
    const double* program=weights+462+(13+7*h->rounds)*id;
    unsigned stride=404+20*h->rounds;
    double* r=out+8*h->patches+20*h->systems+stride*id;
    for(unsigned i=0;i<stride;i++)r[i]=0;
    const double* target=program;unsigned mode=(unsigned)program[12];
    double q[6],bestq[6],x[192],bestx[192];
    for(unsigned j=0;j<6;j++)q[j]=bestq[j]=program[6+j];
    for(unsigned i=0;i<192;i++){x[i]=x0[i];for(unsigned j=0;j<6;j++)x[i]+=map[6*i+j]*q[j];bestx[i]=x[i];}
    double score=probe_score(q,target),best=score;unsigned commits=0,rejects=0,worse=0,stopped=hood_valid(x)?0:3,executed=0,last=0;
    for(unsigned round=0;round<h->rounds;round++){
        double* trace=r+404+20*round;trace[0]=-1;trace[1]=5;
        trace[2]=trace[3]=trace[4]=score;trace[5]=best;
        for(unsigned j=0;j<6;j++)trace[6+j]=q[j];
        if(stopped)continue;
        ++executed;unsigned opcode;double values[6];
        if(mode==1){
            opcode=0;double hidden[24];
            for(unsigned k=0;k<24;k++){
                double v=weights[288+k];for(unsigned j=0;j<6;j++)v+=q[j]/10*weights[24*j+k]+target[j]/10*weights[24*(6+j)+k];hidden[k]=tanh(v);
            }
            for(unsigned j=0;j<6;j++){double v=weights[456+j];for(unsigned k=0;k<24;k++)v+=hidden[k]*weights[312+6*k+j];values[j]=10*v;}
        }else{
            const double* action=program+13+7*round;opcode=(unsigned)action[0];for(unsigned j=0;j<6;j++)values[j]=action[1+j];
        }
        trace[0]=opcode;for(unsigned j=0;j<6;j++)trace[12+j]=isfinite(values[j])?values[j]:0;
        unsigned reason=0;bool finite=true;for(unsigned j=0;j<6;j++)if(!isfinite(values[j]))finite=false;
        double trial[192],trialq[6],candidate=score;
        if(opcode>4)reason=1;
        else if(!finite)reason=2;
        else if(opcode==4){reason=4;stopped=1;}
        else{
            for(unsigned j=0;j<6;j++)trialq[j]=opcode==0?q[j]+values[j]:opcode==1?values[j]:bestq[j];
            if(opcode==3){
                for(unsigned i=0;i<192;i++)trial[i]=x[i];
                if(values[0]<0||values[0]>=192||floor(values[0])!=values[0])reason=1;
                else trial[(unsigned)values[0]]+=values[1];
                for(unsigned j=0;j<6;j++){trialq[j]=0;for(unsigned i=0;i<192;i++)trialq[j]+=probes[192*j+i]*(trial[i]-x0[i]);}
            }else if(opcode==2){for(unsigned i=0;i<192;i++)trial[i]=bestx[i];}
            else{for(unsigned i=0;i<192;i++){trial[i]=x0[i];for(unsigned j=0;j<6;j++)trial[i]+=map[6*i+j]*trialq[j];}}
            if(reason==0&&!hood_valid(trial))reason=3;
            if(reason==0){
                candidate=probe_score(trialq,target);if(!isfinite(candidate))reason=2;
                else{
                    // Admission depends on constraints, NEVER on objective improvement.
                    if(candidate>score+1e-10)++worse;
                    for(unsigned i=0;i<192;i++)x[i]=trial[i];for(unsigned j=0;j<6;j++)q[j]=trialq[j];
                    score=candidate;++commits;
                    if(score<best){best=score;for(unsigned i=0;i<192;i++)bestx[i]=x[i];for(unsigned j=0;j<6;j++)bestq[j]=q[j];}
                }
            }
        }
        if(reason!=0&&reason!=4)++rejects;last=reason;trace[1]=reason;trace[3]=candidate;trace[4]=score;trace[5]=best;
        for(unsigned j=0;j<6;j++)trace[6+j]=q[j];trace[18]=commits;trace[19]=stopped?0:1;
    }
    for(unsigned i=0;i<192;i++){r[i]=x[i];r[192+i]=bestx[i];}
    for(unsigned j=0;j<6;j++){r[384+j]=q[j];r[390+j]=bestq[j];}
    double stats[8]={double(commits),double(rejects),double(worse),double(stopped),double(executed),double(last),score,best};
    for(unsigned j=0;j<8;j++)r[396+j]=stats[j];
}
}
