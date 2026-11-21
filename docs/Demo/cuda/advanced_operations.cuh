// Bounded advanced surface operations. All poles are nonrational and frames
// use an orthonormal local profile. Closed paths include a duplicate endpoint.
__device__ bool sweep_sections(double* s,const P3* path,const P3* tangent,const P3* guide,
    const P3* sections,int rows,int corners,const double* scales,bool closed,P3 initial){
    if(rows<2||rows>32||corners<3||corners>31)return false;
    P3 ns[32],ts[32];double lengths[32];lengths[0]=0;
    for(int i=0;i<rows;++i){
        if(!finite_point(path[i])||!finite_point(tangent[i])||scalar(tangent[i],tangent[i])<1e-20||!isfinite(scales[i])||scales[i]<=0)return false;
        ts[i]=unit(tangent[i]);if(i){double d=sqrt(scalar(minus(path[i],path[i-1]),minus(path[i],path[i-1])));if(d<1e-12)return false;lengths[i]=lengths[i-1]+d;if(scalar(ts[i],ts[i-1])<-1+1e-10)return false;}
        P3 n=i?ns[i-1]:initial;
        if(guide)n=minus(guide[i],path[i]);
        else if(i){P3 axis=vector_cross(ts[i-1],ts[i]);double sine=sqrt(scalar(axis,axis));if(sine>1e-12)n=rotate_axis(n,p3(0,0,0),axis,atan2(sine,scalar(ts[i-1],ts[i])));}
        n=minus(n,times(ts[i],scalar(n,ts[i])));if(scalar(n,n)<1e-20)return false;ns[i]=unit(n);
        for(int j=0;j<corners;++j){P3 q=sections[i*corners+j];if(!finite_point(q)||fabs(q.z)>1e-12)return false;}
    }
    if(closed){
        if(guide||scalar(minus(path[0],path[rows-1]),minus(path[0],path[rows-1]))>1e-18||scalar(ts[0],ts[rows-1])<1-1e-10)return false;
        if(fabs(scales[0]-scales[rows-1])>1e-12)return false;
        for(int j=0;j<corners;++j)if(scalar(minus(sections[j],sections[(rows-1)*corners+j]),minus(sections[j],sections[(rows-1)*corners+j]))>1e-18)return false;
        double correction=atan2(scalar(ts[0],vector_cross(ns[rows-1],ns[0])),scalar(ns[rows-1],ns[0]));
        for(int i=0;i<rows;++i)ns[i]=rotate_axis(ns[i],p3(0,0,0),ts[i],correction*lengths[i]/lengths[rows-1]);
    }
    setup_surface(s,rows,corners+1,1,1);
    for(int i=0;i<rows+2;++i)s[16+i]=i<2?0:i>=rows?1:lengths[i-1]/lengths[rows-1];
    for(int j=0;j<corners+3;++j)s[56+j]=j<2?0:j>=corners+1?1:double(j-1)/corners;
    for(int i=0;i<rows;++i){P3 b=vector_cross(ts[i],ns[i]);for(int j=0;j<=corners;++j){P3 q=sections[i*corners+j%corners];set_pole(s,i,j,plus(path[i],times(plus(times(ns[i],q.x),times(b,q.y)),scales[i])));}}
    return true;
}

// Match target U=0 against source U=1 for compatible bicubic Bezier patches.
// Scale=1 gives C1/C2 (therefore G1/G2) with the same parameter orientation.
__device__ bool match_bezier_boundary(const P3* source,P3* target,int order,double scale){
    if(order<1||order>2||!isfinite(scale)||scale<=0)return false;
    for(int j=0;j<4;++j){target[j]=source[12+j];target[4+j]=plus(target[j],times(minus(source[12+j],source[8+j]),scale));
        if(order==2)target[8+j]=plus(minus(times(target[4+j],2),target[j]),times(plus(minus(source[12+j],times(source[8+j],2)),source[4+j]),scale*scale));}
    return true;
}

// Joint projected-gradient bending-energy fairing of a 7x4 bicubic net.
// Outer boundary locked; orthogonal C2 projection after every step. Bounded baseline,
// not an unconstrained Class-A variational optimizer.
__device__ bool fair_patch_pair(P3* p,int iterations,double alpha){
    if(iterations<0||iterations>64||alpha<0||alpha>.5)return false;
    for(int j=0;j<=3;j+=3){P3 a=minus(times(p[12+j],2),p[8+j]);P3 b=plus(minus(times(p[16+j],2),p[12+j]),plus(minus(p[12+j],times(p[8+j],2)),p[4+j]));if(scalar(minus(a,p[16+j]),minus(a,p[16+j]))>1e-20||scalar(minus(b,p[20+j]),minus(b,p[20+j]))>1e-20)return false;}
    for(int it=0;it<=iterations;++it){
        if(it){P3 g[28];for(int k=0;k<28;++k)g[k]=p3(0,0,0);
            for(int i=1;i<6;++i)for(int j=0;j<4;++j){int k=4*i+j;P3 d=plus(minus(p[k-4],times(p[k],2)),p[k+4]);g[k-4]=plus(g[k-4],times(d,2));g[k]=minus(g[k],times(d,4));g[k+4]=plus(g[k+4],times(d,2));}
            for(int i=0;i<7;++i)for(int j=1;j<3;++j){int k=4*i+j;P3 d=plus(minus(p[k-1],times(p[k],2)),p[k+1]);g[k-1]=plus(g[k-1],times(d,2));g[k]=minus(g[k],times(d,4));g[k+1]=plus(g[k+1],times(d,2));}
            for(int i=1;i<6;++i)for(int j=1;j<3;++j){int k=4*i+j;p[k]=minus(p[k],times(g[k],alpha/64.));}
        }
        for(int j=1;j<3;++j){P3 c1=times(plus(minus(p[8+j],times(p[12+j],2)),p[16+j]),1./6.);P3 c2=times(plus(minus(times(p[8+j],2),p[4+j]),minus(p[20+j],times(p[16+j],2))),.1);
            p[4+j]=plus(p[4+j],c2);p[8+j]=minus(p[8+j],plus(c1,times(c2,2)));p[12+j]=plus(p[12+j],times(c1,2));p[16+j]=plus(minus(p[16+j],c1),times(c2,2));p[20+j]=minus(p[20+j],c2);}
    }return true;
}

const int ADV_SURF_COUNT=9;
const int ADV_SAMPLE_OFFSET=ADV_SURF_COUNT*SURF_WORDS;
const int ADV_AUX_OFFSET=ADV_SAMPLE_OFFSET+ADV_SURF_COUNT*81*9;
extern "C" __global__ void build_advanced_surfaces(double* arena,unsigned offset){
    double* a=arena+offset;
    if(blockIdx.x||threadIdx.x)return;
    for(int id=0;id<4;++id){P3 path[25],tangent[25],guide[25],profile[25*8];double scales[25];
        for(int i=0;i<25;++i){double u=i/24.,q=6.283185307179586*u;
            if(id==3){path[i]=p3((1+.15*cos(3*q))*cos(q),(1+.15*cos(3*q))*sin(q),.2*sin(2*q));tangent[i]=p3(-3*.15*sin(3*q)*cos(q)-(1+.15*cos(3*q))*sin(q),-3*.15*sin(3*q)*sin(q)+(1+.15*cos(3*q))*cos(q),.4*cos(2*q));}
            else{path[i]=p3(2*u,.2*sin(q*.5),.3*u*u);tangent[i]=p3(2,.1*6.283185307179586*cos(q*.5),.6*u);}
            guide[i]=plus(path[i],p3(.1*sin(q),.2*cos(q),1));scales[i]=id==1?1+.35*sin(q*.5):1;
            for(int j=0;j<8;++j){double v=6.283185307179586*j/8;profile[8*i+j]=p3(.15*cos(v)*(id==2?1+.4*u:1),.15*sin(v)*(id==2?1-.3*u:1),0);}
        }
        if(!sweep_sections(a+id*SURF_WORDS,path,tangent,id==0?guide:nullptr,profile,25,8,scales,id==3,p3(0,0,1)))a[id*SURF_WORDS+11]=1;
    }
    P3 left[16],right[16];
    for(int i=0;i<4;++i)for(int j=0;j<4;++j){left[4*i+j]=p3(i/3.,j/3.,.16*i*i/9.+.05*sin(double(j)));right[4*i+j]=p3(1+i/3.,j/3.,.4+.2*cos(double(i+j)));}
    make_bezier_patch(a+4*SURF_WORDS,left,4,4); // source
    for(int order=1;order<=2;++order){P3 matched[16];for(int k=0;k<16;++k)matched[k]=right[k];match_bezier_boundary(left,matched,order,1);make_bezier_patch(a+(4+order)*SURF_WORDS,matched,4,4);}
    P3 pair[28];for(int i=0;i<7;++i)for(int j=0;j<4;++j)pair[4*i+j]=p3(i/3.,j/3.,.1*(i/3.)*(i/3.)+.015*j*(3-j)*sin(double(i*j)));
    for(int j=0;j<4;++j){pair[16+j]=minus(times(pair[12+j],2),pair[8+j]);pair[20+j]=plus(minus(times(pair[16+j],2),pair[12+j]),plus(minus(pair[12+j],times(pair[8+j],2)),pair[4+j]));}
    for(int k=0;k<28;++k){a[ADV_AUX_OFFSET+8+3*k]=pair[k].x;a[ADV_AUX_OFFSET+9+3*k]=pair[k].y;a[ADV_AUX_OFFSET+10+3*k]=pair[k].z;}
    bool fair_ok=fair_patch_pair(pair,12,.3);P3 pa[16],pb[16];for(int i=0;i<4;++i)for(int j=0;j<4;++j){pa[4*i+j]=pair[4*i+j];pb[4*i+j]=pair[4*(i+3)+j];}
    make_bezier_patch(a+7*SURF_WORDS,pa,4,4);make_bezier_patch(a+8*SURF_WORDS,pb,4,4);
    if(!fair_ok){a[7*SURF_WORDS+11]=1;a[8*SURF_WORDS+11]=1;}
    a[ADV_AUX_OFFSET]=!match_bezier_boundary(left,right,3,1);a[ADV_AUX_OFFSET+1]=!fair_patch_pair(pair,65,.3);
}
extern "C" __global__ void sample_advanced_surfaces(double* arena,unsigned offset){
    double* a=arena+offset;
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=ADV_SURF_COUNT*81)return;int id=k/81,i=(k%81)/9,j=k%9;double* s=a+id*SURF_WORDS;
    Jet z=nurbs_surface(s+96,int(s[0]),int(s[1]),int(s[2]),int(s[3]),s+16,s+56,i/8.,j/8.);
    P3 p[3]={z.p,z.du,z.dv};for(int n=0;n<3;++n){a[ADV_SAMPLE_OFFSET+9*k+3*n]=p[n].x;a[ADV_SAMPLE_OFFSET+9*k+3*n+1]=p[n].y;a[ADV_SAMPLE_OFFSET+9*k+3*n+2]=p[n].z;}
}
