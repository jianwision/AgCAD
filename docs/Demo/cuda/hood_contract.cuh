// Fixed-hood v1 raw semantic contract, independent of the affine row compiler.
// Bit flags: numerical domain, XY, perimeter, C2 seam, symmetry.
__device__ unsigned hood_flags(const double* p){
    unsigned flags=0;
    for(unsigned i=0;i<192;i++)if(!isfinite(p[i])||fabs(p[i])>1e6)return 1;
    for(unsigned pu=0;pu<2;pu++)for(unsigned pv=0;pv<2;pv++){
        unsigned base=48*(2*pu+pv);
        double xx[4]={double(pu*pu),pu*pu+2*pu/3.,pu*pu+4*pu/3.+1/3.,double((pu+1)*(pu+1))};
        double yy[4]={double(pv*pv),pv*pv+2*pv/3.,pv*pv+4*pv/3.+1/3.,double((pv+1)*(pv+1))};
        for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++){
            unsigned index=base+12*i+3*j;
            if(fabs(p[index]-(-1500+1500*(pu+i/3.)))>1e-7||fabs(p[index+1]-(-750+750*(pv+j/3.)))>1e-7)flags|=2;
            if((pu==0&&i==0)||(pu==1&&i==3)||(pv==0&&j==0)||(pv==1&&j==3)){
                double z=240-60*(xx[i]-2*(pu+i/3.)+1)-40*(yy[j]-2*(pv+j/3.)+1);
                if(fabs(p[index+2]-z)>1e-7)flags|=4;
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
        if(fabs(av-bv)>1e-7)flags|=8;
    }
    for(unsigned pu=0;pu<2;pu++)for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++)for(unsigned xyz=0;xyz<3;xyz++){
        double a=p[96*pu+12*i+3*j+xyz],b=p[96*pu+48+12*i+3*(3-j)+xyz];
        if(fabs(a-(xyz==1?-b:b))>1e-7)flags|=16;
    }
    return flags;
}
