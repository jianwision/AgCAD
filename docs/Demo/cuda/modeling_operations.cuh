// Local-coordinate deformations. Caller keeps radius/scale/Jacobian nonsingular.
__device__ P3 bend_point(P3 p,double radius){double a=p.x/radius;return p3((radius+p.z)*sin(a),p.y,(radius+p.z)*cos(a)-radius);}
__device__ P3 twist_point(P3 p,double rate){double a=rate*p.z;return p3(p.x*cos(a)-p.y*sin(a),p.x*sin(a)+p.y*cos(a),p.z);}
__device__ P3 taper_point(P3 p,double rate){double s=1+rate*p.z;return p3(s*p.x,s*p.y,p.z);}

// Closed rectangular prism with stations along an arbitrary orthonormal frame.
__device__ bool make_prism_stations(GMesh m,int count,double length,double hx,double hy,P3 axis,P3 x,P3 y){
    if(count<2||count>64||length<=0||hx<=0||hy<=0){m.a[2]=201;return false;}
    P3 corners[4]={p3(-hx,-hy,0),p3(hx,-hy,0),p3(hx,hy,0),p3(-hx,hy,0)};
    for(int i=0;i<count;++i)for(int j=0;j<4;++j)add_vertex(m,plus(times(axis,length*(double(i)/(count-1)-.5)),plus(times(x,corners[j].x),times(y,corners[j].y))));
    for(int i=0;i<count-1;++i)for(int j=0;j<4;++j)quad(m,4*i+j,4*i+(j+1)%4,4*(i+1)+(j+1)%4,4*(i+1)+j);
    int lo=add_vertex(m,times(axis,-length*.5)),hi=add_vertex(m,times(axis,length*.5));
    for(int j=0;j<4;++j){add_triangle(m,lo,(j+1)%4,j);add_triangle(m,hi,4*(count-1)+j,4*(count-1)+(j+1)%4);}return true;
}

// Discrete parallel transport. Profile is strictly convex CCW in local XY,
// contains its origin and has no repeated endpoint; tangent reversals rejected.
__device__ bool make_path_sweep(double* s,GMesh mesh,const P3* path,const P3* tangents,int count,const P3* profile,int sides,P3 initial_normal){
    if(count<2||count>32||sides<3||sides>31)return false;
    for(int j=0;j<sides;++j){if(!finite_point(profile[j])||fabs(profile[j].z)>1e-12||profile[j].x*profile[(j+1)%sides].y-profile[j].y*profile[(j+1)%sides].x<=1e-12)return false;}
    for(int i=0;i<count;++i){if(!finite_point(path[i])||!finite_point(tangents[i])||scalar(tangents[i],tangents[i])<1e-20)return false;if(i&&scalar(unit(tangents[i-1]),unit(tangents[i]))<-1+1e-10)return false;}
    P3 t=unit(tangents[0]),n=minus(initial_normal,times(t,scalar(initial_normal,t)));
    if(scalar(n,n)<1e-20)return false;n=unit(n);
    setup_surface(s,count,sides+1,1,1);
    for(int i=0;i<count+2;++i)s[16+i]=i<2?0:i>=count?1:double(i-1)/(count-1);
    for(int i=0;i<sides+3;++i)s[56+i]=i<2?0:i>=sides+1?1:double(i-1)/sides;
    for(int i=0;i<count;++i){P3 next=unit(tangents[i]);double cosine=fmin(1.,fmax(-1.,scalar(t,next)));
        if(cosine<-1+1e-10)return false;P3 axis=vector_cross(t,next);double sine=sqrt(scalar(axis,axis));
        if(sine>1e-12)n=rotate_axis(n,p3(0,0,0),axis,atan2(sine,cosine));
        t=next;n=unit(minus(n,times(t,scalar(n,t))));P3 b=vector_cross(t,n);
        for(int j=0;j<=sides;++j){P3 p=plus(path[i],plus(times(n,profile[j%sides].x),times(b,profile[j%sides].y)));set_pole(s,i,j,p);if(j<sides)add_vertex(mesh,p);}
    }
    for(int i=0;i<count-1;++i)for(int j=0;j<sides;++j)quad(mesh,i*sides+j,i*sides+(j+1)%sides,(i+1)*sides+(j+1)%sides,(i+1)*sides+j);
    int start=add_vertex(mesh,path[0]),end=add_vertex(mesh,path[count-1]);
    for(int j=0;j<sides;++j){add_triangle(mesh,start,(j+1)%sides,j);add_triangle(mesh,end,(count-1)*sides+j,(count-1)*sides+(j+1)%sides);}return true;
}

// Natural cubic compatible-section loft with centroid-chord parameters.
__device__ bool make_chordal_loft(double* s,const P3* sections,int rows,int nv,int degree,const double* knots){
    if(rows<2||rows>11||nv<2||nv>32||degree<1||degree>5||degree>=nv)return false;
    double t[11],h[10];P3 centers[11];t[0]=0;
    for(int i=0;i<rows;++i){centers[i]=p3(0,0,0);for(int j=0;j<nv;++j){if(!finite_point(sections[i*nv+j]))return false;centers[i]=plus(centers[i],times(sections[i*nv+j],1./nv));}
        if(i){P3 d=minus(centers[i],centers[i-1]);double q=sqrt(scalar(d,d));if(q<1e-12)return false;t[i]=t[i-1]+q;}}
    for(int i=1;i<rows;++i)t[i]/=t[rows-1]; // Denominator remains unchanged until last iteration.
    for(int i=0;i<rows-1;++i)h[i]=t[i+1]-t[i];
    setup_surface(s,3*(rows-1)+1,nv,3,degree);int k=16;for(int j=0;j<4;++j)s[k++]=0;
    for(int i=1;i<rows-1;++i)for(int j=0;j<3;++j)s[k++]=t[i];for(int j=0;j<4;++j)s[k++]=1;
    for(int j=0;j<nv+degree+1;++j)s[56+j]=knots[j];s[6]=knots[degree];s[7]=knots[nv];
    for(int j=0;j<nv;++j){P3 m[11];double upper[11];m[0]=p3(0,0,0);upper[0]=0;
        for(int i=1;i<rows-1;++i){double diag=2*(h[i-1]+h[i])-h[i-1]*upper[i-1];upper[i]=h[i]/diag;
            P3 rhs=times(minus(times(minus(sections[(i+1)*nv+j],sections[i*nv+j]),1/h[i]),times(minus(sections[i*nv+j],sections[(i-1)*nv+j]),1/h[i-1])),6);
            m[i]=times(minus(rhs,times(m[i-1],h[i-1])),1/diag);}
        m[rows-1]=p3(0,0,0);for(int i=rows-2;i>0;--i)m[i]=minus(m[i],times(m[i+1],upper[i]));
        for(int i=0;i<rows-1;++i){P3 a=sections[i*nv+j],b=sections[(i+1)*nv+j],slope=times(minus(b,a),1/h[i]);P3 da=minus(slope,times(plus(times(m[i],2),m[i+1]),h[i]/6)),db=plus(slope,times(plus(m[i],times(m[i+1],2)),h[i]/6));set_pole(s,3*i,j,a);set_pole(s,3*i+1,j,plus(a,times(da,h[i]/3)));set_pole(s,3*i+2,j,minus(b,times(db,h[i]/3)));set_pole(s,3*i+3,j,b);}}
    return true;
}

// One Catmull-Clark step on a closed manifold quad mesh. Fresh output required.
__device__ bool catmull_clark(GMesh dst,const P3* p,int nv,const int* quads,int nf){
    if(nv<4||nv>128||nf<4||nf>128)return false;
    P3 fc[128];int ea[256],eb[256],ef0[256],ef1[256],edges[512],ne=0;
    for(int f=0;f<nf;++f){fc[f]=p3(0,0,0);for(int j=0;j<4;++j){int a=quads[4*f+j],b=quads[4*f+(j+1)%4];if(a<0||a>=nv||b<0||b>=nv||a==b)return false;fc[f]=plus(fc[f],times(p[a],.25));if(a>b){int v=a;a=b;b=v;}int e=-1;for(int k=0;k<ne;++k)if(ea[k]==a&&eb[k]==b)e=k;
            if(e<0){if(ne==256)return false;e=ne++;ea[e]=a;eb[e]=b;ef0[e]=f;ef1[e]=-1;}else{if(ef1[e]>=0)return false;ef1[e]=f;}edges[4*f+j]=e;}}
    for(int e=0;e<ne;++e)if(ef1[e]<0)return false;
    for(int v=0;v<nv;++v){P3 fsum=p3(0,0,0),rsum=fsum;int faces=0,valence=0;
        for(int f=0;f<nf;++f)for(int j=0;j<4;++j)if(quads[4*f+j]==v){fsum=plus(fsum,fc[f]);++faces;}
        for(int e=0;e<ne;++e)if(ea[e]==v||eb[e]==v){rsum=plus(rsum,mix_point(p[ea[e]],p[eb[e]],.5));++valence;}
        if(valence<3||valence!=faces)return false;add_vertex(dst,times(plus(plus(times(fsum,1./faces),times(rsum,2./valence)),times(p[v],valence-3)),1./valence));}
    for(int e=0;e<ne;++e)add_vertex(dst,times(plus(plus(p[ea[e]],p[eb[e]]),plus(fc[ef0[e]],fc[ef1[e]])),.25));
    for(int f=0;f<nf;++f)add_vertex(dst,fc[f]);
    for(int f=0;f<nf;++f)for(int j=0;j<4;++j)quad(dst,quads[4*f+j],nv+edges[4*f+j],nv+ne+f,nv+edges[4*f+(j+3)%4]);return true;
}
// One original Loop subdivision step, closed manifold triangular input only.
__device__ bool loop_subdivide(GMesh dst,const P3* p,int nv,const int* ids,int nt){
    if(nv<4||nv>128||nt<4||nt>256)return false;int ea[384],eb[384],eo0[384],eo1[384],edges[768],ne=0;
    for(int f=0;f<nt;++f)for(int j=0;j<3;++j){int a=ids[3*f+j],b=ids[3*f+(j+1)%3],o=ids[3*f+(j+2)%3];if(a<0||a>=nv||b<0||b>=nv||o<0||o>=nv||a==b||a==o||b==o)return false;if(a>b){int t=a;a=b;b=t;}int e=-1;for(int k=0;k<ne;++k)if(ea[k]==a&&eb[k]==b)e=k;
        if(e<0){if(ne==384)return false;e=ne++;ea[e]=a;eb[e]=b;eo0[e]=o;eo1[e]=-1;}else{if(eo1[e]>=0)return false;eo1[e]=o;}edges[3*f+j]=e;}
    for(int e=0;e<ne;++e)if(eo1[e]<0)return false;
    for(int v=0;v<nv;++v){P3 sum=p3(0,0,0);int n=0;for(int e=0;e<ne;++e)if(ea[e]==v||eb[e]==v){sum=plus(sum,p[ea[e]==v?eb[e]:ea[e]]);++n;}if(n<3)return false;double q=.375+.25*cos(6.283185307179586/n),beta=(.625-q*q)/n;add_vertex(dst,plus(times(p[v],1-n*beta),times(sum,beta)));}
    for(int e=0;e<ne;++e)add_vertex(dst,plus(times(plus(p[ea[e]],p[eb[e]]),.375),times(plus(p[eo0[e]],p[eo1[e]]),.125)));
    for(int f=0;f<nt;++f){int a=ids[3*f],b=ids[3*f+1],c=ids[3*f+2],ab=nv+edges[3*f],bc=nv+edges[3*f+1],ca=nv+edges[3*f+2];add_triangle(dst,a,ab,ca);add_triangle(dst,ab,b,bc);add_triangle(dst,ca,bc,c);add_triangle(dst,ab,bc,ca);}return true;
}
__device__ bool resample_grid(GMesh m,const P3* points,int nu,int nv,int us,int vs){
    if(nu<2||nv<2||nu>32||nv>32||us<1||vs<1||(us+1)*(vs+1)>MV||2*us*vs>MT)return false;
    for(int i=0;i<=us;++i)for(int j=0;j<=vs;++j){double u=double(i)*(nu-1)/us,v=double(j)*(nv-1)/vs;int a=min(int(u),nu-2),b=min(int(v),nv-2);add_vertex(m,mix_point(mix_point(points[a*nv+b],points[(a+1)*nv+b],u-a),mix_point(points[a*nv+b+1],points[(a+1)*nv+b+1],u-a),v-b));}
    for(int i=0;i<us;++i)for(int j=0;j<vs;++j)quad(m,(vs+1)*i+j,(vs+1)*(i+1)+j,(vs+1)*(i+1)+j+1,(vs+1)*i+j+1);return true;
}
__device__ bool resample_polyline(P3* out,int samples,const P3* points,int count){
    if(count<2||count>128||samples<2||samples>4096)return false;double length[128];length[0]=0;
    for(int i=1;i<count;++i){P3 d=minus(points[i],points[i-1]);double h=sqrt(scalar(d,d));if(!isfinite(h)||h<=1e-12)return false;length[i]=length[i-1]+h;}
    for(int i=0;i<samples;++i){double l=length[count-1]*i/(samples-1);int k=0;while(k<count-2&&length[k+1]<l)++k;out[i]=mix_point(points[k],points[k+1],(l-length[k])/(length[k+1]-length[k]));}return true;
}
