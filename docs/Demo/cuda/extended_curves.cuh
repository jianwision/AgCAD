// Curve records: kind/count/degree/domain at 0..4, a/b/c at 5..7,
// origin/X/Y/Z at 8..19, status at 20, reference at 21, offset at 22,
// ordinary XYZW poles at 24..151, full knots at 152..191.
const int EC_COUNT=9,EC_WORDS=192,EC_SAMPLES=513,EC_OP_WORDS=256;
__device__ P3 ec_vec(const double* s,int k){return p3(s[k],s[k+1],s[k+2]);}
__device__ void ec_set(double* s,int k,P3 v){s[k]=v.x;s[k+1]=v.y;s[k+2]=v.z;}
__device__ bool finite_point(P3 p){return isfinite(p.x)&&isfinite(p.y)&&isfinite(p.z);}
__device__ bool make_conic_curve(double* s,int kind,P3 origin,P3 x,P3 y,double a,double b,double c,double first,double last){
    if(kind<1||kind>5||!finite_point(origin)||!finite_point(x)||!finite_point(y)||!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(first)||!isfinite(last)||first>=last)return false;
    if(fabs(scalar(x,x)-1)>1e-10||fabs(scalar(y,y)-1)>1e-10||fabs(scalar(x,y))>1e-10)return false;
    if(a<=0||(kind==2&&b<=0)||(kind==4&&fmin(a+b*first,a+b*last)<=0))return false;
    s[0]=kind;s[3]=first;s[4]=last;s[5]=a;s[6]=b;s[7]=c;s[20]=0;
    ec_set(s,8,origin);ec_set(s,11,x);ec_set(s,14,y);ec_set(s,17,vector_cross(x,y));return true;
}
__device__ bool make_hermite_curve(double* s,P3 a,P3 b,P3 da,P3 db){
    if(!finite_point(a)||!finite_point(b)||!finite_point(da)||!finite_point(db))return false;
    s[0]=0;s[1]=4;s[2]=3;s[3]=0;s[4]=1;s[20]=0;
    P3 cp[4]={a,plus(a,times(da,1./3)),minus(b,times(db,1./3)),b};
    for(int i=0;i<4;++i){ec_set(s,24+4*i,cp[i]);s[27+4*i]=1;}
    for(int i=0;i<8;++i)s[152+i]=i<4?0:1;return true;
}
__device__ bool make_three_point_arc(double* s,P3 a,P3 b,P3 c){
    if(!finite_point(a)||!finite_point(b)||!finite_point(c))return false;
    P3 u=minus(b,a),v=minus(c,a),n=vector_cross(u,v);double nn=scalar(n,n);
    if(nn<1e-20)return false;
    P3 center=plus(a,times(plus(times(vector_cross(v,n),scalar(u,u)),times(vector_cross(n,u),scalar(v,v))),.5/nn));
    P3 x=unit(minus(a,center)),y=vector_cross(unit(n),x);P3 end=minus(c,center),mid=minus(b,center);
    double last=atan2(scalar(end,y),scalar(end,x)),middle=atan2(scalar(mid,y),scalar(mid,x));
    if(last<=0)last+=6.283185307179586;if(middle<=0)middle+=6.283185307179586;
    if(middle>=last)return false;
    return make_conic_curve(s,5,center,x,y,sqrt(scalar(minus(a,center),minus(a,center))),0,0,0,last);
}
// Natural cubic interpolation at equally spaced parameters, 2..11 points.
__device__ bool make_interpolating_curve(double* s,const P3* points,int count){
    if(count<2||count>11)return false;for(int i=0;i<count;++i)if(!finite_point(points[i]))return false;
    P3 rhs[11],m[11];double c[11];
    for(int i=0;i<count;++i)rhs[i]=times(minus(points[i==count-1?i:i+1],points[i==0?i:i-1]),3);
    c[0]=.5;m[0]=times(rhs[0],.5);
    for(int i=1;i<count;++i){double d=(i==count-1?2:4)-c[i-1];c[i]=i<count-1?1/d:0;m[i]=times(minus(rhs[i],m[i-1]),1/d);}
    for(int i=count-2;i>=0;--i)m[i]=minus(m[i],times(m[i+1],c[i]));
    s[0]=6;s[1]=3*(count-1)+1;s[2]=3;s[3]=0;s[4]=1;s[20]=0;
    for(int i=0;i<count-1;++i){P3 cp[4]={points[i],plus(points[i],times(m[i],1./3)),minus(points[i+1],times(m[i+1],1./3)),points[i+1]};for(int j=0;j<4;++j){int k=24+4*(3*i+j);ec_set(s,k,cp[j]);s[k+3]=1;}}
    int k=152;for(int j=0;j<4;++j)s[k++]=0;for(int i=1;i<count-1;++i)for(int j=0;j<3;++j)s[k++]=double(i)/(count-1);for(int j=0;j<4;++j)s[k++]=1;return true;
}
__device__ CurveJet analytic_curve_jet(const double* s,double t){
    int kind=int(s[0]);if(kind==0||kind==6)return nurbs_curve(s+24,int(s[1]),int(s[2]),s+152,t);
    double a=s[5],b=s[6],c=s[7],px=0,py=0,pz=0,dx=0,dy=0,dz=0,ddx=0,ddy=0;
    if(kind==1){px=t*t/(4*a);py=t;dx=t/(2*a);dy=1;ddx=1/(2*a);}
    if(kind==2){px=a*cosh(t);py=b*sinh(t);dx=a*sinh(t);dy=b*cosh(t);ddx=px;ddy=py;}
    if(kind==3||kind==5){px=a*cos(t);py=a*sin(t);dx=-a*sin(t);dy=a*cos(t);ddx=-px;ddy=-py;if(kind==3){pz=c*t/6.283185307179586;dz=c/6.283185307179586;}}
    if(kind==4){double r=a+b*t;px=r*cos(t);py=r*sin(t);dx=b*cos(t)-r*sin(t);dy=b*sin(t)+r*cos(t);ddx=-2*b*sin(t)-r*cos(t);ddy=2*b*cos(t)-r*sin(t);}
    P3 o=ec_vec(s,8),x=ec_vec(s,11),y=ec_vec(s,14),z=ec_vec(s,17);CurveJet q;
    q.p=plus(o,plus(plus(times(x,px),times(y,py)),times(z,pz)));
    q.d=plus(plus(times(x,dx),times(y,dy)),times(z,dz));q.dd=plus(times(x,ddx),times(y,ddy));return q;
}
__device__ CurveJet reverse_curve_jet(const double* base,double t){CurveJet q=analytic_curve_jet(base,base[3]+base[4]-t);q.d=times(q.d,-1);return q;}
__device__ P3 planar_curve_offset(CurveJet jet,P3 normal,double distance){return plus(jet.p,times(unit(vector_cross(normal,jet.d)),distance));}
// Composite five-point Gauss-Legendre integration, no host-side length query.
__device__ double curve_arc_length(const double* s,double first,double last,int panels){
    const double x[5]={-.906179845938664,-.538469310105683,0,.538469310105683,.906179845938664};
    const double w[5]={.236926885056189,.478628670499366,.568888888888889,.478628670499366,.236926885056189};
    double total=0,h=(last-first)/panels;
    for(int i=0;i<panels;++i)for(int j=0;j<5;++j){CurveJet q=analytic_curve_jet(s,first+h*(i+.5+.5*x[j]));total+=w[j]*sqrt(scalar(q.d,q.d))*.5*h;}return total;
}
__device__ double curve_distance_parameter(const double* s,double fraction){
    double a=s[3],b=s[4],length=curve_arc_length(s,a,b,64),target=fraction*length;
    for(int i=0;i<42;++i){double m=(a+b)*.5;if(curve_arc_length(s,s[3],m,32)<target)a=m;else b=m;}return (a+b)*.5;
}
__device__ double curve_distance_squared(const double* s,P3 p,double t){P3 d=minus(analytic_curve_jet(s,t).p,p);return scalar(d,d);}
// Bounded nearest-point search: coarse basin selection followed by golden section.
// Not a proof of the global minimum for arbitrary oscillatory/self-crossing curves.
__device__ double closest_curve_parameter(const double* s,P3 p){
    double first=s[3],h=(s[4]-first)/128,best=1e300;int k=0;
    for(int i=0;i<=128;++i){double d=curve_distance_squared(s,p,first+i*h);if(d<best){best=d;k=i;}}
    double a=first+max(0,k-1)*h,b=first+min(128,k+1)*h,g=.61803398874989485;
    double l=b-g*(b-a),r=a+g*(b-a),fl=curve_distance_squared(s,p,l),fr=curve_distance_squared(s,p,r);
    for(int i=0;i<64;++i){if(fl<fr){b=r;r=l;fr=fl;l=b-g*(b-a);fl=curve_distance_squared(s,p,l);}else{a=l;l=r;fl=fr;r=a+g*(b-a);fr=curve_distance_squared(s,p,r);}}
    double t=(a+b)*.5;if(curve_distance_squared(s,p,first)<curve_distance_squared(s,p,t))t=first;if(curve_distance_squared(s,p,s[4])<curve_distance_squared(s,p,t))t=s[4];return t;
}
