// Persistent geometry. Models choose programs; admission is constraint-only.
struct SceneHeader { char magic[8]; unsigned version,slots,budget,reserved; };
struct SessionCommand { char magic[8]; unsigned version,op,slot,other,revision,generation,steps,contract,flags,reserved; double values[12]; unsigned padding[4]; };
struct SessionAction { unsigned op,index; double values[6]; unsigned reserved[2]; };
static_assert(sizeof(SceneHeader)==24,"scene ABI");
static_assert(sizeof(SessionCommand)==160,"command ABI");
static_assert(sizeof(SessionAction)==64,"action ABI");
enum { ADD=0,SET=1,RAW=2,BEGIN=3,COMMIT=4,ABORT=5,STOP=6,MEASURE=7,BLEND_REFERENCE=8 };
enum { OK=0,UNSUPPORTED=1,NONFINITE=2,CONSTRAINT_FAILED=3,STOPPED=4,NOT_RUN=5,
       NO_DRAFT=6,OPEN_DRAFT=7,STALE_REVISION=8,STALE_HANDLE=9,INVALID_PARAMETER=10 };
enum { REV=0,USED=1,DRAFT=2,STOP_FLAG=3,LAST=4,COMMITS=5,REJECTS=6,UPHILL=7,
       SCORE=8,DRAFT_SCORE=9,GENERATION=10,LIVE=11,EXECUTED=12,LAST_FLAGS=13 };
const unsigned SLOT_WORDS=448;
__device__ double* meta(double* slot){return slot+402;}
__device__ void probes_of(const double* scene,const double* x,double* q){
    const double* x0=scene;const double* e=scene+192+1152;
    for(unsigned j=0;j<6;j++){q[j]=0;for(unsigned i=0;i<192;i++)q[j]+=e[192*j+i]*(x[i]-x0[i]);}
}
__device__ double score_of(const double* q,const double* target){
    double v=0;for(unsigned j=0;j<6;j++)v+=(q[j]-target[j])*(q[j]-target[j]);return v;
}
__device__ void summary(const double* slot,double* out){
    for(unsigned i=0;i<64;i++)out[i]=0;
    for(unsigned i=0;i<18;i++)out[i]=slot[384+i];
    for(unsigned i=0;i<16;i++)out[18+i]=slot[402+i];
    out[34]=hood_flags(slot);out[35]=slot[404]?hood_flags(slot+192):0;
    double error=0,draft_error=0;
    for(unsigned i=0;i<6;i++){
        error=fmax(error,fabs(slot[384+i]-slot[396+i]));
        draft_error=fmax(draft_error,fabs(slot[390+i]-slot[396+i]));
    }
    out[36]=error;out[37]=draft_error;
}
__device__ unsigned apply_action(const double* scene,double* slot,const SessionAction& action){
    double* s=meta(slot);double* current=slot+(s[DRAFT]?192:0);
    unsigned op=action.op;s[LAST_FLAGS]=0;
    for(unsigned i=0;i<6;i++)if(!isfinite(action.values[i]))return NONFINITE;
    if(op>BLEND_REFERENCE)return UNSUPPORTED;
    if(op==BEGIN){
        if(s[DRAFT])return OPEN_DRAFT;
        for(unsigned i=0;i<192;i++)slot[192+i]=slot[i];
        for(unsigned i=0;i<6;i++)slot[390+i]=slot[384+i];
        s[DRAFT]=1;s[DRAFT_SCORE]=s[SCORE];s[REV]+=1;return OK;
    }
    if(op==ABORT){if(!s[DRAFT])return NO_DRAFT;s[DRAFT]=0;s[REV]+=1;return OK;}
    if(op==STOP){if(s[DRAFT])return OPEN_DRAFT;s[STOP_FLAG]=1;return STOPPED;}
    if(op==MEASURE)return OK;
    if(op==COMMIT){
        if(!s[DRAFT])return NO_DRAFT;
        unsigned flags=hood_flags(slot+192);s[LAST_FLAGS]=flags;if(flags)return CONSTRAINT_FAILED;
        if(s[DRAFT_SCORE]>s[SCORE]+1e-10)s[UPHILL]+=1;
        for(unsigned i=0;i<192;i++)slot[i]=slot[192+i];
        for(unsigned i=0;i<6;i++)slot[384+i]=slot[390+i];
        s[SCORE]=s[DRAFT_SCORE];s[DRAFT]=0;s[REV]+=1;s[COMMITS]+=1;return OK;
    }
    double trial[192],q[6],delta[6];
    probes_of(scene,current,q);
    for(unsigned i=0;i<192;i++)trial[i]=current[i];
    if(op==RAW){
        if(action.index>=192)return INVALID_PARAMETER;
        trial[action.index]+=action.values[0];
    }else{
        if(op==BLEND_REFERENCE&&(action.values[0]<0||action.values[0]>1))return INVALID_PARAMETER;
        for(unsigned j=0;j<6;j++)delta[j]=op==ADD?action.values[j]:op==SET?action.values[j]-q[j]:-action.values[0]*q[j];
        const double* map=scene+192;
        for(unsigned i=0;i<192;i++)for(unsigned j=0;j<6;j++)trial[i]+=map[6*i+j]*delta[j];
    }
    unsigned flags=hood_flags(trial);s[LAST_FLAGS]=flags;
    // Drafts may temporarily violate geometric constraints, but not the declared
    // finite numerical domain. A failed COMMIT leaves the repairable draft alive.
    if((flags&1)||(!s[DRAFT]&&flags))return CONSTRAINT_FAILED;
    probes_of(scene,trial,q);double score=score_of(q,slot+396);
    if(!isfinite(score))return NONFINITE;
    for(unsigned i=0;i<192;i++)current[i]=trial[i];
    for(unsigned j=0;j<6;j++)slot[(s[DRAFT]?390:384)+j]=q[j];
    if(s[DRAFT]){s[DRAFT_SCORE]=score;s[REV]+=1;}
    else{if(score>s[SCORE]+1e-10)s[UPHILL]+=1;s[SCORE]=score;s[REV]+=1;s[COMMITS]+=1;}
    return OK;
}
extern "C" {
__global__ void session_execute(const unsigned char* input,const unsigned char* template_data,double* arena,double* out){
    if(blockIdx.x||threadIdx.x)return;
    const SessionCommand* command=(const SessionCommand*)input;
    const double* scene=(const double*)(template_data+24);
    double* slot=arena+SLOT_WORDS*command->slot;double* s=meta(slot);
    if(command->op==0){
        for(unsigned i=0;i<SLOT_WORDS;i++)slot[i]=0;
        const double* map=scene+192;
        for(unsigned i=0;i<192;i++){slot[i]=scene[i];for(unsigned j=0;j<6;j++)slot[i]+=map[6*i+j]*command->values[j];}
        probes_of(scene,slot,slot+384);
        for(unsigned j=0;j<6;j++){slot[390+j]=slot[384+j];slot[396+j]=command->values[6+j];}
        s[GENERATION]=command->generation;s[LIVE]=1;s[SCORE]=score_of(slot+384,slot+396);
        s[DRAFT_SCORE]=s[SCORE];s[LAST_FLAGS]=hood_flags(slot);s[LAST]=s[LAST_FLAGS]?CONSTRAINT_FAILED:OK;
        if(s[LAST_FLAGS])s[LIVE]=0;
    }else if(command->op==3){
        const double* source=arena+SLOT_WORDS*command->other;
        for(unsigned i=0;i<SLOT_WORDS;i++)slot[i]=0;
        for(unsigned i=0;i<192;i++)slot[i]=source[i];
        for(unsigned j=0;j<6;j++){slot[384+j]=slot[390+j]=source[384+j];slot[396+j]=source[396+j];}
        s[SCORE]=s[DRAFT_SCORE]=score_of(slot+384,slot+396);s[GENERATION]=command->generation;s[LIVE]=1;
    }else if(command->op==1||command->op==2){
        s[EXECUTED]=0;s[STOP_FLAG]=0;
        unsigned entrance=s[GENERATION]!=command->generation?STALE_HANDLE:s[REV]!=command->revision?STALE_REVISION:OK;
        bool halted=entrance!=OK;
        for(unsigned round=0;round<command->steps;round++){
            double* trace=out+64+26*round;for(unsigned i=0;i<26;i++)trace[i]=0;
            trace[0]=-1;trace[1]=entrance?entrance:NOT_RUN;
            if(halted)continue;
            SessionAction action;
            if(command->op==2){
                action.op=ADD;action.index=0;double hidden[24];
                const double* weights=scene+192+1152+1152;
                const double* q=slot+(s[DRAFT]?390:384);
                for(unsigned k=0;k<24;k++){
                    double v=weights[288+k];for(unsigned j=0;j<6;j++)v+=q[j]/10*weights[24*j+k]+slot[396+j]/10*weights[24*(6+j)+k];hidden[k]=tanh(v);
                }
                for(unsigned j=0;j<6;j++){double v=weights[456+j];for(unsigned k=0;k<24;k++)v+=hidden[k]*weights[312+6*k+j];action.values[j]=10*v;}
            }else action=((const SessionAction*)(input+160))[round];
            trace[0]=action.op;trace[2]=s[REV];trace[3]=s[DRAFT];trace[4]=s[SCORE];
            for(unsigned j=0;j<6;j++)trace[20+j]=isfinite(action.values[j])?action.values[j]:0;
            unsigned reason=apply_action(scene,slot,action);s[LAST]=reason;s[USED]+=1;s[EXECUTED]+=1;
            if(reason!=OK&&reason!=STOPPED)s[REJECTS]+=1;
            trace[1]=reason;trace[5]=s[REV];trace[6]=s[SCORE];trace[7]=s[DRAFT_SCORE];
            for(unsigned j=0;j<6;j++){trace[8+j]=slot[384+j];trace[14+j]=slot[390+j];}
            if(reason==STOPPED||(reason!=OK&&!(command->flags&1)))halted=true;
        }
        if(entrance)s[LAST]=entrance;
    }else if(command->op==6){s[LIVE]=0;}
    summary(slot,out);
    if(command->op==5){const double* source=slot+((command->flags&1)?192:0);for(unsigned i=0;i<192;i++)out[64+i]=source[i];}
}
}
