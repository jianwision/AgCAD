// Trained 5->24(tanh)->25 policy. Weights and input already reside on device.
// Four instruction slots, eight doubles each: opcode, parameters, reserved.
// 0=NOOP, 1=NURBS_PATCH, 2=MIRROR_Y, 3=REVOLVE_Y, 4=TESSELLATE.
// packet[32] is a reduced failure code; no host observes intermediate commands.
extern "C" __global__ void infer_commands(const double* weights,const double* input,double* packet){
    if(blockIdx.x||threadIdx.x)return;
    double hidden[24],out[25];
    for(int i=0;i<5;++i)if(!isfinite(input[i])||fabs(input[i])>1.00001){packet[32]=1;return;}
    for(int j=0;j<24;++j){double a=weights[120+j];for(int i=0;i<5;++i)a+=input[i]*weights[24*i+j];hidden[j]=tanh(a);}
    for(int j=0;j<25;++j){double a=weights[744+j];for(int i=0;i<24;++i)a+=hidden[i]*weights[144+25*i+j];out[j]=a;}
    for(int slot=0;slot<4;++slot){int best=0;for(int k=1;k<5;++k)if(out[5*slot+k]>out[5*slot+best])best=k;packet[8*slot]=best;}
    for(int j=0;j<5;++j)if(!isfinite(out[20+j])){packet[32]=2;return;}
    if(packet[0]!=1||packet[8]!=2||(packet[16]!=0&&packet[16]!=3)||packet[24]!=4){packet[32]=3;return;}
    for(int j=0;j<3;++j){if(out[20+j]<.7||out[20+j]>1.3){packet[32]=4;return;}packet[1+j]=out[20+j];}
    packet[4]=fmin(1.,fmax(0.,out[23]));packet[17]=out[24];
    if(packet[17]<.8||packet[17]>1.2){packet[32]=5;return;}
    packet[33]=4;
}
