// Packed extension partition, appended to the original batch arena.
const int EC_SAMPLE_OFFSET=EC_COUNT*EC_WORDS;
const int EC_OP_OFFSET=EC_SAMPLE_OFFSET+EC_COUNT*EC_SAMPLES*9;
const int ES_DEF_OFFSET=EC_OP_OFFSET+EC_OP_WORDS;
const int ES_MESH_OFFSET=ES_DEF_OFFSET+ES_COUNT*ES_WORDS;
extern "C" __global__ void build_extension_curves(double* packet,unsigned offset){
    double* a=packet+offset;int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=EC_COUNT)return;double* s=a+id*EC_WORDS;
    P3 o=p3(.1,.2,.3),x=p3(.6,.8,0),y=p3(-.8,.6,0);bool ok=true;
    if(id==0)ok=make_hermite_curve(s,p3(0,0,0),p3(2,.3,.5),p3(1,2,0),p3(1,-1,.6));
    if(id==1)ok=make_conic_curve(s,1,o,x,y,.7,0,0,-1.2,1.4);
    if(id==2)ok=make_conic_curve(s,2,o,x,y,1.2,.55,0,-.9,.9);
    if(id==3)ok=make_conic_curve(s,3,o,x,y,.8,0,.7,-.4,12.866370614359172);
    if(id==4)ok=make_conic_curve(s,4,o,x,y,.4,.08,0,.2,12.566370614359172);
    if(id==5){P3 origin=p3(.3,-.2,.5),xx=p3(.6,.8,0),yy=p3(0,0,1),p[3];double t[3]={.2,1.4,3.7};for(int i=0;i<3;++i)p[i]=ellipse_point(origin,xx,yy,2,2,t[i]);ok=make_three_point_arc(s,p[0],p[1],p[2]);}
    if(id==6){P3 p[5]={p3(0,0,0),p3(.4,.8,.2),p3(1,.3,.6),p3(1.6,1,.4),p3(2,0,.8)};ok=make_interpolating_curve(s,p,5);}
    if(id==7||id==8){s[0]=id;s[3]=id==7?0:-1.2;s[4]=id==7?1:1.4;s[21]=id==7?6:1;s[22]=.15;ec_set(s,17,p3(0,0,1));}
    if(!ok)s[20]=1;
}
extern "C" __global__ void sample_extension_curves(double* packet,unsigned offset){
    double* a=packet+offset;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=EC_COUNT*EC_SAMPLES)return;
    int id=i/EC_SAMPLES,k=i%EC_SAMPLES;const double* s=a+id*EC_WORDS;double t=s[3]+(s[4]-s[3])*k/(EC_SAMPLES-1);CurveJet q;
    if(id==7)q=reverse_curve_jet(a+int(s[21])*EC_WORDS,t);
    else if(id==8){q=analytic_curve_jet(a+int(s[21])*EC_WORDS,t);q.p=planar_curve_offset(q,ec_vec(s,17),s[22]);q.d=q.dd=p3(0,0,0);}
    else q=analytic_curve_jet(s,t);
    double* out=a+EC_SAMPLE_OFFSET+9*i;ec_set(out,0,q.p);ec_set(out,3,q.d);ec_set(out,6,q.dd);
}
extern "C" __global__ void extension_curve_ops(double* packet,unsigned offset){
    double* a=packet+offset,*out=a+EC_OP_OFFSET;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<65){const double* s=a+EC_WORDS;double t=curve_distance_parameter(s,i/64.);ec_set(out,32+3*i,analytic_curve_jet(s,t).p);}
    if(i<8){const double* s=a+(i==7?6:i)*EC_WORDS;out[i]=curve_arc_length(s,s[3],s[4],64);}
    if(i==0){P3 query=p3(.8,-.3,.5);const double* s=a+EC_WORDS;double t=closest_curve_parameter(s,query);P3 p=analytic_curve_jet(s,t).p;out[8]=t;out[9]=sqrt(curve_distance_squared(s,query,t));ec_set(out,10,p);
        double temp[EC_WORDS],spec[ES_WORDS];P3 p0=p3(0,0,0),x=p3(1,0,0),y=p3(0,1,0);
        out[13]=!make_three_point_arc(temp,p0,x,times(x,2));
        out[14]=!make_conic_curve(temp,1,p0,x,y,-1,0,0,0,1);
        out[15]=!make_conic_curve(temp,1,p0,x,x,1,0,0,0,1);
        out[16]=!make_interpolating_curve(temp,&p0,12);
        out[17]=!make_solid_definition(spec,0,-1,0,0,0,p0,x,y);
        out[18]=!make_solid_definition(spec,5,.5,1,0,0,p0,x,y);
        out[19]=!make_solid_definition(spec,7,1,1,0,3.5,p0,x,y);
        out[20]=!make_solid_definition(spec,9,1,1,1,2,p0,x,y);
    }
}
extern "C" __global__ void build_extension_solids(double* packet,unsigned offset){
    double* a=packet+offset;int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=ES_COUNT)return;
    double* s=a+ES_DEF_OFFSET+id*ES_WORDS;GMesh mesh={a+ES_MESH_OFFSET+id*MESH_WORDS};
    double p[10][4]={{1.2,0,0,0},{1.4,.8,1.1,0},{.8,2.3,0,0},{1,2.4,0,0},{1.1,.5,1.9,0},{1.4,.35,0,0},{.65,1.5,0,0},{1,1.8,0,6},{1.1,1.7,0,5},{2,1.2,1.6,.7}};
    if(!make_solid_definition(s,id,p[id][0],p[id][1],p[id][2],p[id][3],p3(.1,-.2,.3),p3(.6,.8,0),p3(-.48,.36,.8))||!solid_mesh(s,mesh))s[20]=1;
}
