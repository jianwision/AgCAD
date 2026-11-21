// Initial fixed-topology CAD harness. No learned policy is claimed here.
// Immutable shared template + per-candidate transactions, all in one CUDA graph.
struct Header {
    char magic[8];
    unsigned version,n,m,k,r,candidates,rounds,template_rev,coefficient_rev,goal_rev,trace_words,reserved;
    double lower,upper,gradient_tol,improvement,score_slack,cache_epsilon,jac_min,center,scale;
};
struct Request { double goal,initial; unsigned budget,goal_rev; };
struct Command { unsigned opcode,expected_revision,template_rev,coefficient_rev,goal_rev,index; double value; };
static_assert(sizeof(Header)==128,"header ABI");
static_assert(sizeof(Request)==24,"request ABI");
static_assert(sizeof(Command)==32,"command ABI");
enum { AUTO=0,SET_PARAMETER=1,EDIT_CONTROL_DELTA=2,NOOP=3,STOP=4 };
enum { ACCEPTED=0,CACHE_HIT=1,STOPPED=2,INACTIVE=3,BUDGET_EXHAUSTED=4,
       STALE_REVISION=5,NEED_RECOMPILE=6,STALE_GOAL=7,INVALID_OPCODE=8,
       INVALID_PARAMETER=9,NONFINITE_PARAMETER=10,HARD_FAILED=11,GEOMETRY_FAILED=12,
       NUMERICAL_FAILED=13,NO_IMPROVEMENT=14,INVALID_BASELINE=15 };
enum { STATUS=0,REV=1,BUDGET=2,GOAL=3,THETA=4,OBJECTIVE=5,HARD=6,JAC=7,GRAD=8,
       LAST=9,COMMITS=10,REJECTS=11,HITS=12,EVALS=13,FLAGS=14,ATTEMPTS=15,
       PENDING=16,TRIAL_THETA=17,OP=18,REASON=19,TRIAL_OBJ=20,TRIAL_HARD=21,
       TRIAL_JAC=22,TRIAL_GRAD=23,TRIAL_FLAGS=24,BEFORE_REV=25,BEFORE_BUDGET=26,
       BEFORE_THETA=27,BEFORE_OBJ=28,REQUESTED_OP=29,PARAM_FINITE=30 };
struct View {
    const Header* h;
    const double *z,*x0,*ymap,*a,*bmap,*tol,*hard,*weight,*b,*dmap;
    const Request* requests; const Command* commands;
    __device__ View(const unsigned char* input){
        h=(const Header*)input;const double* p=(const double*)(input+128);
        z=p;p+=(long)h->n*h->k;x0=p;p+=h->n;ymap=p;p+=2*h->k;
        a=p;p+=(long)h->r*h->n;bmap=p;p+=2*h->r;tol=p;p+=h->r;
        hard=p;p+=h->r;weight=p;p+=h->r;b=p;p+=(long)h->m*h->k;dmap=p;p+=2*h->m;
        requests=(const Request*)p;commands=(const Command*)(requests+h->candidates);
    }
    __device__ long stride() const{return 2L*h->n+2*h->k+32+(long)h->rounds*h->trace_words;}
};
__device__ double* state(double* w,const Header* h){return w+2*h->n+2*h->k;}
__device__ double sum256(double value){
    __shared__ double s[256];unsigned t=threadIdx.x;s[t]=value;__syncthreads();
    for(unsigned d=128;d;d/=2){if(t<d)s[t]+=s[t+d];__syncthreads();}
    double result=s[0];__syncthreads();return result;
}
__device__ double max256(double value){
    __shared__ double s[256];unsigned t=threadIdx.x;s[t]=value;__syncthreads();
    for(unsigned d=128;d;d/=2){if(t<d)s[t]=fmax(s[t],s[t+d]);__syncthreads();}
    double result=s[0];__syncthreads();return result;
}
__device__ void basis(double t,double* p,double* d){
    double s=1-t;p[0]=s*s*s;p[1]=3*t*s*s;p[2]=3*t*t*s;p[3]=t*t*t;
    d[0]=-3*s*s;d[1]=3*s*s-6*t*s;d[2]=6*t*s-3*t*t;d[3]=3*t*t;
}
extern "C" {
__global__ void harness_initialize(const unsigned char* input,double* out){
    View v(input);const Header* h=v.h;unsigned c=blockIdx.y,j=blockIdx.x*256+threadIdx.x;
    double* w=out+c*v.stride();double* s=state(w,h);double theta=v.requests[c].initial,t=(theta-h->center)/h->scale;
    if(j<h->n){double x=v.x0[j];for(unsigned l=0;l<h->k;l++)x+=v.z[(long)j*h->k+l]*(v.ymap[2*l]+t*v.ymap[2*l+1]);w[j]=x;w[h->n+j]=x;}
    if(j<h->k){double y=v.ymap[2*j]+t*v.ymap[2*j+1];w[2*h->n+j]=y;w[2*h->n+h->k+j]=y;}
    if(j==0){s[GOAL]=v.requests[c].goal;s[THETA]=theta;s[TRIAL_THETA]=theta;s[BUDGET]=v.requests[c].budget;s[OP]=SET_PARAMETER;}
}
__global__ void harness_propose(const unsigned char* input,double* out,unsigned round){
    View v(input);const Header* h=v.h;unsigned c=blockIdx.x*256+threadIdx.x;if(c>=h->candidates)return;
    double* w=out+c*v.stride();double* s=state(w,h);Command cmd=v.commands[(long)round*h->candidates+c];
    s[BEFORE_REV]=s[REV];s[BEFORE_BUDGET]=s[BUDGET];s[BEFORE_THETA]=s[THETA];s[BEFORE_OBJ]=s[OBJECTIVE];
    s[REQUESTED_OP]=cmd.opcode;s[OP]=cmd.opcode;s[PENDING]=0;s[REASON]=0;s[TRIAL_THETA]=s[THETA];
    s[TRIAL_OBJ]=s[OBJECTIVE];s[TRIAL_HARD]=s[HARD];s[TRIAL_JAC]=s[JAC];s[TRIAL_GRAD]=s[GRAD];s[TRIAL_FLAGS]=s[FLAGS];
    s[PARAM_FINITE]=isfinite(cmd.value)?1:0;
    if(s[STATUS]!=0){s[REASON]=s[STATUS]==3?INVALID_BASELINE:INACTIVE;return;}
    if(cmd.opcode==AUTO){cmd.opcode=fabs(s[THETA]-s[GOAL])<=1e-8&&((int)s[FLAGS]&1)?STOP:SET_PARAMETER;
        cmd.value=s[GOAL];cmd.expected_revision=(unsigned)s[REV];cmd.template_rev=h->template_rev;
        cmd.coefficient_rev=h->coefficient_rev;cmd.goal_rev=v.requests[c].goal_rev;s[OP]=cmd.opcode;s[PARAM_FINITE]=1;}
    if(cmd.opcode==STOP){
        if(cmd.expected_revision!=(unsigned)s[REV]){s[REASON]=STALE_REVISION;return;}
        if(cmd.template_rev!=h->template_rev||cmd.coefficient_rev!=h->coefficient_rev){s[REASON]=NEED_RECOMPILE;return;}
        if(cmd.goal_rev!=v.requests[c].goal_rev){s[REASON]=STALE_GOAL;return;}
        s[STATUS]=1;s[REASON]=STOPPED;return;}
    if(s[BUDGET]<=0){s[STATUS]=2;s[REASON]=BUDGET_EXHAUSTED;return;}
    s[BUDGET]-=1;s[ATTEMPTS]+=1;
    if(cmd.expected_revision!=(unsigned)s[REV]){s[REASON]=STALE_REVISION;return;}
    if(cmd.template_rev!=h->template_rev||cmd.coefficient_rev!=h->coefficient_rev){s[REASON]=NEED_RECOMPILE;return;}
    if(cmd.goal_rev!=v.requests[c].goal_rev){s[REASON]=STALE_GOAL;return;}
    if(cmd.opcode==NOOP){s[REASON]=CACHE_HIT;s[HITS]+=1;s[PENDING]=2;return;}
    if(cmd.opcode!=SET_PARAMETER&&cmd.opcode!=EDIT_CONTROL_DELTA){s[REASON]=INVALID_OPCODE;return;}
    if(!isfinite(cmd.value)){s[REASON]=NONFINITE_PARAMETER;return;}
    if(cmd.opcode==SET_PARAMETER){
        if(cmd.value<h->lower||cmd.value>h->upper){s[REASON]=INVALID_PARAMETER;return;}
        s[TRIAL_THETA]=cmd.value;
        if(fabs(cmd.value-s[THETA])<=h->cache_epsilon&&((int)s[FLAGS]&1)){s[REASON]=CACHE_HIT;s[HITS]+=1;s[PENDING]=2;return;}
    }else if(cmd.index>=h->n||fabs(cmd.value)>1e6){s[REASON]=INVALID_PARAMETER;return;}
    s[PENDING]=1;
}
__global__ void harness_construct(const unsigned char* input,double* out,unsigned round){
    View v(input);const Header* h=v.h;unsigned c=blockIdx.y,j=blockIdx.x*256+threadIdx.x;
    double* w=out+c*v.stride();double* s=state(w,h);if(s[PENDING]!=1)return;
    if((unsigned)s[OP]==SET_PARAMETER){double t=(s[TRIAL_THETA]-h->center)/h->scale;
        if(j<h->n){double x=v.x0[j];for(unsigned l=0;l<h->k;l++)x+=v.z[(long)j*h->k+l]*(v.ymap[2*l]+t*v.ymap[2*l+1]);w[h->n+j]=x;}
        if(j<h->k)w[2*h->n+h->k+j]=v.ymap[2*j]+t*v.ymap[2*j+1];
    }else{const Command& cmd=v.commands[(long)round*h->candidates+c];
        if(j<h->n)w[h->n+j]=w[j]+(j==cmd.index?cmd.value:0.);
        if(j<h->k)w[2*h->n+h->k+j]=w[2*h->n+j];}
}
__global__ void harness_verify(const unsigned char* input,double* out,unsigned baseline){
    View v(input);const Header* h=v.h;unsigned c=blockIdx.x,t=threadIdx.x;
    double* w=out+c*v.stride();double* s=state(w,h);if(!baseline&&s[PENDING]!=1)return;
    const double* x=w+h->n;const double* y=w+2*h->n+h->k;
    double gt=(s[GOAL]-h->center)/h->scale,pt=(s[TRIAL_THETA]-h->center)/h->scale;
    double objective=0,hard=0,invalid=0;
    for(unsigned i=t;i<h->r;i+=256){double value=0;for(unsigned j=0;j<h->n;j++)value+=v.a[(long)i*h->n+j]*x[j];
        double residual=value-v.bmap[2*i]-gt*v.bmap[2*i+1];
        if(!isfinite(residual)){invalid=1;continue;}
        if(v.hard[i])hard=fmax(hard,fabs(residual)/v.tol[i]);else objective+=v.weight[i]*residual*residual;}
    for(unsigned j=t;j<h->n;j+=256)if(!isfinite(x[j]))invalid=1;
    double jac=1e300;
    if(t<100){unsigned patch=t/25,sample=t%25;double bu[4],du[4],bv[4],dv[4];basis((sample/5)/4.,bu,du);basis((sample%5)/4.,bv,dv);
        double u[3]={0,0,0},vv[3]={0,0,0};
        for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++)for(unsigned axis=0;axis<3;axis++){
            double p=x[48*patch+12*i+3*j+axis];u[axis]+=du[i]*bv[j]*p;vv[axis]+=bu[i]*dv[j]*p;}
        jac=u[0]*vv[1]-u[1]*vv[0];if(!isfinite(jac)){invalid=1;jac=-1e300;}}
    double gn=0,rn=0;
    if((unsigned)s[OP]==SET_PARAMETER)for(unsigned j=t;j<h->k;j+=256){double gradient=0,rhs=0;
        for(unsigned i=0;i<h->m;i++){double value=0;for(unsigned l=0;l<h->k;l++)value+=v.b[(long)i*h->k+l]*y[l];
            double d=v.dmap[2*i]+pt*v.dmap[2*i+1];gradient+=v.b[(long)i*h->k+j]*(value-d);rhs+=v.b[(long)i*h->k+j]*d;}
        gn+=gradient*gradient;rn+=rhs*rhs;}
    objective=sum256(objective);hard=max256(hard);invalid=max256(invalid);jac=-max256(-jac);gn=sum256(gn);rn=sum256(rn);
    if(t==0){double gradient=sqrt(gn/fmax(rn,1.));unsigned flags=0;
        if(!invalid&&hard<=1.)flags|=2;if(!invalid&&jac>h->jac_min)flags|=4;
        if((unsigned)s[OP]==SET_PARAMETER&&!invalid&&isfinite(gradient)&&gradient<=h->gradient_tol)flags|=1;
        if(!isfinite(objective)||!isfinite(gradient)){flags=0;objective=1e300;gradient=1e300;}
        s[TRIAL_OBJ]=objective;s[TRIAL_HARD]=hard;s[TRIAL_JAC]=jac;s[TRIAL_GRAD]=gradient;s[TRIAL_FLAGS]=flags;s[EVALS]+=1;}
}
__global__ void harness_baseline(const unsigned char* input,double* out){
    View v(input);unsigned c=blockIdx.x*256+threadIdx.x;if(c>=v.h->candidates)return;
    double* s=state(out+c*v.stride(),v.h);s[OBJECTIVE]=s[TRIAL_OBJ];s[HARD]=s[TRIAL_HARD];s[JAC]=s[TRIAL_JAC];s[GRAD]=s[TRIAL_GRAD];s[FLAGS]=s[TRIAL_FLAGS];
    if((unsigned)s[FLAGS]!=7){s[STATUS]=3;s[LAST]=INVALID_BASELINE;}
}
__global__ void harness_decide(const unsigned char* input,double* out){
    View v(input);unsigned c=blockIdx.x*256+threadIdx.x;if(c>=v.h->candidates)return;
    double* s=state(out+c*v.stride(),v.h);
    if(s[PENDING]!=1){if(s[REASON]>=STALE_REVISION&&s[REASON]<=NONFINITE_PARAMETER)s[REJECTS]+=1;return;}
    unsigned flags=(unsigned)s[TRIAL_FLAGS],reason=ACCEPTED;
    if(!(flags&2))reason=HARD_FAILED;else if(!(flags&4))reason=GEOMETRY_FAILED;
    else if((unsigned)s[OP]==SET_PARAMETER&&!(flags&1))reason=NUMERICAL_FAILED;
    else if(!(s[TRIAL_OBJ]<s[OBJECTIVE]-v.h->improvement-v.h->score_slack*fmax(fabs(s[OBJECTIVE]),1.)))reason=NO_IMPROVEMENT;
    s[REASON]=reason;if(reason==ACCEPTED)s[PENDING]=3;else{s[REJECTS]+=1;s[PENDING]=0;}
}
__global__ void harness_commit(const unsigned char* input,double* out){
    View v(input);const Header* h=v.h;unsigned c=blockIdx.y,j=blockIdx.x*256+threadIdx.x;
    double* w=out+c*v.stride();double* s=state(w,h);if(s[PENDING]!=3)return;
    if(j<h->n)w[j]=w[h->n+j];if(j<h->k)w[2*h->n+j]=w[2*h->n+h->k+j];
    if(j==0){s[REV]+=1;s[COMMITS]+=1;s[THETA]=s[TRIAL_THETA];s[OBJECTIVE]=s[TRIAL_OBJ];s[HARD]=s[TRIAL_HARD];s[JAC]=s[TRIAL_JAC];s[GRAD]=s[TRIAL_GRAD];s[FLAGS]=s[TRIAL_FLAGS];}
}
__global__ void harness_trace(const unsigned char* input,double* out,unsigned round){
    View v(input);unsigned c=blockIdx.x*256+threadIdx.x;if(c>=v.h->candidates)return;
    double* s=state(out+c*v.stride(),v.h);double* trace=s+32+(long)round*v.h->trace_words;
    double values[24]={double(round),s[REQUESTED_OP],s[OP],s[REASON],s[BEFORE_REV],s[REV],s[BEFORE_BUDGET],s[BUDGET],s[BEFORE_THETA],s[TRIAL_THETA],s[THETA],s[BEFORE_OBJ],s[TRIAL_OBJ],s[OBJECTIVE],s[TRIAL_HARD],s[TRIAL_JAC],s[TRIAL_GRAD],s[PENDING]==2?1.:0.,s[PENDING]==3||(s[REASON]>=HARD_FAILED&&s[REASON]<=NO_IMPROVEMENT)?1.:0.,s[TRIAL_FLAGS],s[STATUS],s[ATTEMPTS],s[COMMITS],s[REJECTS]};
    for(unsigned i=0;i<24;i++)trace[i]=values[i];s[LAST]=s[REASON];
}
__global__ void harness_select(const unsigned char* input,double* out){
    if(blockIdx.x||threadIdx.x)return;View v(input);double best=1e300;int id=-1;
    for(unsigned c=0;c<v.h->candidates;c++){double* s=state(out+c*v.stride(),v.h);if(s[STATUS]!=3&&((unsigned)s[FLAGS]&6)==6&&s[OBJECTIVE]<best){best=s[OBJECTIVE];id=c;}}
    double* result=out+v.h->candidates*v.stride();result[0]=id;result[1]=best;result[2]=id>=0?1:0;result[3]=v.h->candidates;result[4]=0;
}
__global__ void harness_compare(const unsigned char* input,double* out,const double* previous){
    View v(input);long count=v.h->candidates*v.stride()+8;double mismatch=0;
    for(long i=threadIdx.x;i<count;i+=256)if(__double_as_longlong(out[i])!=__double_as_longlong(previous[i]))mismatch+=1;
    mismatch=sum256(mismatch);if(threadIdx.x==0)out[count-4]=mismatch;
}
}
