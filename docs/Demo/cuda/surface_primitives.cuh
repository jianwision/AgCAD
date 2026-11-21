// GPU geometry primitives. Double precision; millimetres. Degrees <= 5,
// at most 32 poles per parameter direction. No topology is inferred here.
struct P3 { double x,y,z; };
__device__ P3 p3(double x,double y,double z){return {x,y,z};}
__device__ P3 plus(P3 a,P3 b){return p3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ P3 minus(P3 a,P3 b){return p3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ P3 times(P3 a,double k){return p3(a.x*k,a.y*k,a.z*k);}
__device__ double scalar(P3 a,P3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ P3 vector_cross(P3 a,P3 b){return p3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ P3 unit(P3 a){return times(a,1.0/sqrt(fmax(1e-30,scalar(a,a))));}
__device__ P3 mix_point(P3 a,P3 b,double t){return plus(times(a,1-t),times(b,t));}
__device__ P3 mirror_y(P3 p){return p3(p.x,-p.y,p.z);}
__device__ P3 affine(P3 p,const double* m){return p3(m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11]);}
__device__ P3 bezier(const P3* poles,int degree,double t){P3 d[6];for(int i=0;i<=degree;++i)d[i]=poles[i];for(int k=degree;k>0;--k)for(int i=0;i<k;++i)d[i]=mix_point(d[i],d[i+1],t);return d[0];}
__device__ P3 ruled(P3 a,P3 b,double v){return mix_point(a,b,v);}
__device__ P3 revolve_y(P3 profile,double angle){return p3(profile.x*cos(angle),profile.y,profile.x*sin(angle));}
__device__ P3 sweep_frame(P3 origin,P3 normal,P3 binormal,double x,double y){return plus(origin,plus(times(normal,x),times(binormal,y)));}
__device__ P3 coons(P3 c0,P3 c1,P3 d0,P3 d1,P3 p00,P3 p10,P3 p01,P3 p11,double u,double v){return minus(plus(mix_point(c0,c1,v),mix_point(d0,d1,u)),mix_point(mix_point(p00,p10,u),mix_point(p01,p11,u),v));}
// Quintic Hermite blend: endpoint position, first and second derivatives.
// Equal parameter scales are the caller's responsibility. This is a curve
// construction primitive, not a general Class-A surface-blend solver.
__device__ P3 blend_c2(P3 a,P3 da,P3 dda,P3 b,P3 db,P3 ddb,double t){
    P3 p[6];p[0]=a;p[1]=plus(a,times(da,.2));p[2]=plus(times(dda,.05),minus(times(p[1],2),p[0]));
    p[5]=b;p[4]=minus(b,times(db,.2));p[3]=plus(times(ddb,.05),minus(times(p[4],2),p[5]));return bezier(p,5,t);
}

// Cox-de Boor basis plus analytic first/second derivatives.
__device__ void basis(int count,int degree,const double* k,double u,double* n,double* d,double* dd){
    double a[40]={0},b[40]={0},c[40]={0};int intervals=count+degree;
    for(int i=0;i<intervals;++i)a[i]=(u>=k[i]&&u<k[i+1])?1.0:0.0;
    if(u==k[intervals])a[count-1]=1.0;
    for(int p=1;p<=degree;++p){
        for(int i=0;i<intervals-p;++i){
            double l=k[i+p]-k[i],r=k[i+p+1]-k[i+1];
            n[i]=(l>0?(u-k[i])/l*a[i]:0)+(r>0?(k[i+p+1]-u)/r*a[i+1]:0);
            d[i]=(l>0?p/l*a[i]:0)-(r>0?p/r*a[i+1]:0);
            dd[i]=(l>0?p/l*b[i]:0)-(r>0?p/r*b[i+1]:0);
        }
        for(int i=0;i<intervals-p;++i){a[i]=n[i];b[i]=d[i];c[i]=dd[i];}
    }
    for(int i=0;i<count;++i){n[i]=a[i];d[i]=b[i];dd[i]=c[i];}
}
struct Jet {P3 p,du,dv,duu,duv,dvv;};
// Poles are homogeneous (x,y,z,w), with ordinary xyz coordinates, U-major.
__device__ Jet nurbs_surface(const double* poles,int nu,int nv,int pu,int pv,const double* ku,const double* kv,double u,double v){
    double un[40]={0},ud[40]={0},udd[40]={0},vn[40]={0},vd[40]={0},vdd[40]={0};
    basis(nu,pu,ku,u,un,ud,udd);basis(nv,pv,kv,v,vn,vd,vdd);
    P3 a[6];double w[6]={0};for(int z=0;z<6;++z)a[z]=p3(0,0,0);
    for(int i=0;i<nu;++i)for(int j=0;j<nv;++j){
        const double* cp=poles+4*(i*nv+j);P3 p=p3(cp[0],cp[1],cp[2]);
        double f[6]={un[i]*vn[j],ud[i]*vn[j],un[i]*vd[j],udd[i]*vn[j],ud[i]*vd[j],un[i]*vdd[j]};
        for(int z=0;z<6;++z){double q=f[z]*cp[3];a[z]=plus(a[z],times(p,q));w[z]+=q;}
    }
    Jet s;s.p=times(a[0],1/w[0]);s.du=times(minus(a[1],times(s.p,w[1])),1/w[0]);s.dv=times(minus(a[2],times(s.p,w[2])),1/w[0]);
    s.duu=times(minus(minus(a[3],times(s.du,2*w[1])),times(s.p,w[3])),1/w[0]);
    s.duv=times(minus(minus(minus(a[4],times(s.du,w[2])),times(s.dv,w[1])),times(s.p,w[4])),1/w[0]);
    s.dvv=times(minus(minus(a[5],times(s.dv,2*w[2])),times(s.p,w[5])),1/w[0]);return s;
}
__device__ P3 surface_normal(Jet s){return unit(vector_cross(s.du,s.dv));}
__device__ P3 offset_point(Jet s,double distance){return plus(s.p,times(surface_normal(s),distance));}
__device__ bool trim_polygon(double u,double v,const double* uv,int count){bool inside=false;for(int i=0,j=count-1;i<count;j=i++){double ax=uv[2*i],ay=uv[2*i+1],bx=uv[2*j],by=uv[2*j+1];if((ay>v)!=(by>v)&&u<(bx-ax)*(v-ay)/(by-ay)+ax)inside=!inside;}return inside;}
