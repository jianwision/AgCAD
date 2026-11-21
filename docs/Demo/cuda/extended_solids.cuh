// Solid definitions are CPU-CAD topology inputs and GPU mesh-construction inputs.
// kind/parameters at 0..4, origin at 8, frame X/Y/Z at 11/14/17, status at 20.
const int ES_COUNT=10,ES_WORDS=64;
__device__ bool make_solid_definition(double* s,int kind,double a,double b,double c,double d,P3 origin,P3 x,P3 y){
    if(kind<0||kind>=ES_COUNT||!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(d)||!finite_point(origin)||!finite_point(x)||!finite_point(y))return false;
    if(fabs(scalar(x,x)-1)>1e-10||fabs(scalar(y,y)-1)>1e-10||fabs(scalar(x,y))>1e-10||a<=0)return false;
    if((kind==1||kind==2||kind==3||kind==4||kind==5||kind==6||kind==7||kind==8||kind==9)&&b<=0)return false;
    if((kind==1||kind==4||kind==9)&&c<=0)return false;
    if(kind==5&&a<=b)return false;
    if((kind==7||kind==8)&&(d<3||d>64||d!=int(d)))return false;
    if(kind==9&&(d<=0||d>=a))return false;
    s[0]=kind;s[1]=a;s[2]=b;s[3]=c;s[4]=d;s[20]=0;ec_set(s,8,origin);ec_set(s,11,x);ec_set(s,14,y);ec_set(s,17,vector_cross(x,y));return true;
}
__device__ bool make_capsule(GMesh m,double radius,double height,int segments,int hemisphere_rings){
    int n=segments,r=hemisphere_rings,rows=2*r;if(radius<=0||height<=0||n<3||r<2||2+rows*n>MV||2*rows*n>MT){m.a[2]=110;return false;}
    const double pi=3.14159265358979323846;add_vertex(m,p3(0,0,height+radius));
    for(int i=0;i<rows;++i){double angle=(i<r?i+1:i)*pi/(2*r),z=(i<r?height:0)+radius*cos(angle);for(int j=0;j<n;++j)add_vertex(m,p3(radius*sin(angle)*cos(2*pi*j/n),radius*sin(angle)*sin(2*pi*j/n),z));}
    int south=add_vertex(m,p3(0,0,-radius));for(int j=0;j<n;++j){int k=(j+1)%n;add_triangle(m,0,1+j,1+k);add_triangle(m,south,1+(rows-1)*n+k,1+(rows-1)*n+j);}
    for(int i=0;i<rows-1;++i)for(int j=0;j<n;++j)quad(m,1+i*n+j,1+(i+1)*n+j,1+(i+1)*n+(j+1)%n,1+i*n+(j+1)%n);return true;
}
__device__ bool make_regular_prism(GMesh m,double radius,double height,int sides){
    if(radius<=0||height<=0||sides<3||sides>64){m.a[2]=110;return false;}P3 p[64];for(int i=0;i<sides;++i){double a=6.283185307179586*i/sides;p[i]=p3(radius*cos(a),radius*sin(a),0);}return extrude_polygon(m,p,sides,p3(0,0,height));
}
__device__ bool make_regular_pyramid(GMesh m,double radius,double height,int sides){
    if(radius<=0||height<=0||sides<3||sides>64){m.a[2]=110;return false;}P3 p[64];for(int i=0;i<sides;++i){double a=6.283185307179586*i/sides;p[i]=p3(radius*cos(a),radius*sin(a),0);}
    if(!triangulate_polygon(m,p,sides))return false;for(int t=0;t<triangles(m);++t){int k=8+6*MV+3*t;double v=m.a[k+1];m.a[k+1]=m.a[k+2];m.a[k+2]=v;}
    int apex=add_vertex(m,p3(0,0,height));for(int i=0;i<sides;++i)add_triangle(m,i,(i+1)%sides,apex);return true;
}
__device__ bool make_wedge(GMesh m,double width,double depth,double height,double top_width){
    if(width<=0||depth<=0||height<=0||top_width<=0||top_width>=width){m.a[2]=110;return false;}
    if(!make_box(m,p3(0,0,0),p3(width,depth,height)))return false;for(int i=4;i<8;++i){P3 p=vertex(m,i);p.x=p.x?top_width:0;set_vertex(m,i,p);}return true;
}
__device__ bool solid_mesh(const double* s,GMesh mesh){
    int kind=int(s[0]);double a=s[1],b=s[2],c=s[3],d=s[4];bool ok=false;
    if(kind==0)ok=make_uv_sphere(mesh,p3(0,0,0),a,48,24);
    if(kind==1){ok=make_uv_sphere(mesh,p3(0,0,0),1,48,24);for(int i=0;i<vertices(mesh);++i){P3 p=vertex(mesh,i);set_vertex(mesh,i,p3(p.x*a,p.y*b,p.z*c));}}
    if(kind==2||kind==3||kind==4)ok=make_frustum(mesh,p3(0,0,0),a,kind==2?a:kind==3?0:b,kind==4?c:b,48);
    if(kind==5)ok=make_torus(mesh,p3(0,0,0),a,b,48,24);
    if(kind==6)ok=make_capsule(mesh,a,b,48,8);
    if(kind==7)ok=make_regular_prism(mesh,a,b,int(d));
    if(kind==8)ok=make_regular_pyramid(mesh,a,b,int(d));
    if(kind==9)ok=make_wedge(mesh,a,b,c,d);
    if(!ok)return false;
    P3 o=ec_vec(s,8),x=ec_vec(s,11),y=ec_vec(s,14),z=ec_vec(s,17);
    for(int i=0;i<vertices(mesh);++i){P3 p=vertex(mesh,i);set_vertex(mesh,i,plus(o,plus(plus(times(x,p.x),times(y,p.y)),times(z,p.z))));}mesh_normals(mesh);return true;
}
