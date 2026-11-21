// Parameterized control-net builders used by the batch and available to CUDA
// recipes. Include after surface_primitives.cuh and surface_records.cuh.
__device__ bool make_nurbs_patch(double* s,int nu,int nv,int pu,int pv,const double* ku,const double* kv,const double* xyzw){
    if(nu<2||nv<2||nu>32||nv>32||pu<1||pv<1||pu>5||pv>5||pu>=nu||pv>=nv)return false;
    setup_surface(s,nu,nv,pu,pv);for(int i=0;i<nu+pu+1;++i){if(!isfinite(ku[i])||(i&&ku[i]<ku[i-1]))return false;s[16+i]=ku[i];}
    for(int i=0;i<nv+pv+1;++i){if(!isfinite(kv[i])||(i&&kv[i]<kv[i-1]))return false;s[56+i]=kv[i];}
    for(int i=0;i<nu*nv;++i){for(int k=0;k<4;++k)if(!isfinite(xyzw[4*i+k]))return false;if(xyzw[4*i+3]<=0)return false;for(int k=0;k<4;++k)s[96+4*i+k]=xyzw[4*i+k];}
    s[4]=ku[pu];s[5]=ku[nu];s[6]=kv[pv];s[7]=kv[nv];return ku[pu]<ku[nu]&&kv[pv]<kv[nv];
}
__device__ void make_plane(double* s,P3 origin,P3 u,P3 v){setup_surface(s,2,2,1,1);for(int i=0;i<2;++i)for(int j=0;j<2;++j)set_pole(s,i,j,plus(origin,plus(times(u,i),times(v,j))));}
__device__ bool make_bezier_patch(double* s,const P3* poles,int nu,int nv){if(nu<2||nv<2||nu>6||nv>6)return false;setup_surface(s,nu,nv,nu-1,nv-1);for(int i=0;i<nu;++i)for(int j=0;j<nv;++j)set_pole(s,i,j,poles[i*nv+j]);return true;}
__device__ bool make_ruled(double* s,const P3* a,const P3* b,int n,int degree,const double* knots){if(n>32||degree<1||degree>5||degree>=n)return false;setup_surface(s,2,n,1,degree);for(int i=0;i<n+degree+1;++i)s[56+i]=knots[i];s[6]=knots[degree];s[7]=knots[n];for(int j=0;j<n;++j){set_pole(s,0,j,a[j]);set_pole(s,1,j,b[j]);}return true;}
__device__ bool make_extruded(double* s,const double* cp,int n,int degree,const double* knots,P3 direction){if(n>32||degree<1||degree>5||degree>=n)return false;setup_surface(s,n,2,degree,1);for(int i=0;i<n+degree+1;++i)s[16+i]=knots[i];s[4]=knots[degree];s[5]=knots[n];for(int i=0;i<n;++i)for(int j=0;j<2;++j)set_pole(s,i,j,plus(p3(cp[4*i],cp[4*i+1],cp[4*i+2]),times(direction,j)),cp[4*i+3]);return true;}
// Rational revolution about Y, with <=4 arcs of <=90 degrees. NURBS U is not
// linear in rotation angle. Profile poles are (radius, axial_y, 0, weight).
__device__ bool make_revolved(double* s,const double* cp,int n,int degree,const double* knots,double start,double end){
    if(n>32||degree<1||degree>5||degree>=n||fabs(end-start)<1e-12||fabs(end-start)>6.283185307179587)return false;
    int arcs=int(ceil(fabs(end-start)/1.5707963267948966)),nu=2*arcs+1;setup_surface(s,nu,n,2,degree);
    int k=16;s[k++]=0;s[k++]=0;s[k++]=0;for(int i=1;i<arcs;++i){s[k++]=double(i)/arcs;s[k++]=double(i)/arcs;}s[k++]=1;s[k++]=1;s[k++]=1;
    for(int j=0;j<n+degree+1;++j)s[56+j]=knots[j];s[6]=knots[degree];s[7]=knots[n];
    for(int i=0;i<nu;++i){double a=start+(end-start)*i/(nu-1),w=i%2?cos((end-start)/(2*arcs)):1;for(int j=0;j<n;++j){double r=cp[4*j];set_pole(s,i,j,p3(r*cos(a)/w,cp[4*j+1],r*sin(a)/w),w*cp[4*j+3]);}}return true;
}
__device__ bool make_translation_sweep(double* s,const double* a,int nu,int pu,const double* ku,const double* b,int nv,int pv,const double* kv){
    if(nu>32||nv>32||pu<1||pv<1||pu>5||pv>5||pu>=nu||pv>=nv)return false;setup_surface(s,nu,nv,pu,pv);
    for(int i=0;i<nu+pu+1;++i)s[16+i]=ku[i];for(int j=0;j<nv+pv+1;++j)s[56+j]=kv[j];s[4]=ku[pu];s[5]=ku[nu];s[6]=kv[pv];s[7]=kv[nv];
    for(int i=0;i<nu;++i)for(int j=0;j<nv;++j)set_pole(s,i,j,p3(a[4*i]+b[4*j],a[4*i+1]+b[4*j+1],a[4*i+2]+b[4*j+2]),a[4*i+3]*b[4*j+3]);return true;
}
// Compatible nonrational sections. Natural cubic interpolation through 2..11
// section rows, <=32 poles per section. No CPU solve and no fitted approximation.
__device__ bool make_loft(double* s,const P3* sections,int rows,int nv,int degree,const double* knots){
    if(rows<2||rows>11||nv>32||degree<1||degree>5||degree>=nv)return false;int nu=3*(rows-1)+1;setup_surface(s,nu,nv,3,degree);
    int k=16;for(int i=0;i<4;++i)s[k++]=0;for(int r=1;r<rows-1;++r)for(int i=0;i<3;++i)s[k++]=double(r)/(rows-1);for(int i=0;i<4;++i)s[k++]=1;
    for(int j=0;j<nv+degree+1;++j)s[56+j]=knots[j];s[6]=knots[degree];s[7]=knots[nv];
    for(int j=0;j<nv;++j){P3 rhs[11],m[11];double c[11];for(int i=0;i<rows;++i)rhs[i]=times(minus(sections[(i==rows-1?i:i+1)*nv+j],sections[(i==0?i:i-1)*nv+j]),3);
        c[0]=.5;m[0]=times(rhs[0],.5);for(int i=1;i<rows;++i){double d=(i==rows-1?2:4)-c[i-1];c[i]=i<rows-1?1/d:0;m[i]=times(minus(rhs[i],m[i-1]),1/d);}for(int i=rows-2;i>=0;--i)m[i]=minus(m[i],times(m[i+1],c[i]));
        for(int i=0;i<rows-1;++i){P3 a=sections[i*nv+j],b=sections[(i+1)*nv+j];set_pole(s,3*i,j,a);set_pole(s,3*i+1,j,plus(a,times(m[i],1./3)));set_pole(s,3*i+2,j,minus(b,times(m[i+1],1./3)));set_pole(s,3*i+3,j,b);}}
    return true;
}
__device__ bool make_coons(double* s,const P3* c0,const P3* c1,const P3* d0,const P3* d1){
    P3 diffs[4]={minus(c0[0],d0[0]),minus(c0[3],d1[0]),minus(c1[0],d0[3]),minus(c1[3],d1[3])};for(int i=0;i<4;++i)if(scalar(diffs[i],diffs[i])>1e-20)return false;
    setup_surface(s,4,4,3,3);for(int i=0;i<4;++i)for(int j=0;j<4;++j)set_pole(s,i,j,coons(c0[i],c1[i],d0[j],d1[j],c0[0],c0[3],c1[0],c1[3],i/3.,j/3.));return true;
}
__device__ bool make_hermite_blend(double* s,int grade,const P3* a,const P3* b,const P3* da,const P3* db,const P3* dda,const P3* ddb,int nv,int degree,const double* knots){
    if((grade!=1&&grade!=2)||nv>32||degree<1||degree>5||degree>=nv)return false;int n=grade==1?4:6;setup_surface(s,n,nv,n-1,degree);for(int j=0;j<nv+degree+1;++j)s[56+j]=knots[j];s[6]=knots[degree];s[7]=knots[nv];
    for(int j=0;j<nv;++j){set_pole(s,0,j,a[j]);set_pole(s,n-1,j,b[j]);if(grade==1){set_pole(s,1,j,plus(a[j],times(da[j],1./3)));set_pole(s,2,j,minus(b[j],times(db[j],1./3)));}else{P3 p=plus(a[j],times(da[j],.2)),q=minus(b[j],times(db[j],.2));set_pole(s,1,j,p);set_pole(s,2,j,plus(times(dda[j],.05),minus(times(p,2),a[j])));set_pole(s,4,j,q);set_pole(s,3,j,plus(times(ddb[j],.05),minus(times(q,2),b[j])));}}return true;
}
__device__ bool make_offset(double* s,int base,double distance){if(base<0||base>=SURF_COUNT||!isfinite(distance))return false;s[8]=1;s[9]=base;s[10]=distance;s[4]=s[6]=0;s[5]=s[7]=1;return true;}
