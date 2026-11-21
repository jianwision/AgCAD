// Infrastructure stays fixed; model.cuh replaces the marker below via Rust.
struct V3 { float x, y, z; };
__device__ V3 v(float x, float y, float z) { return {x, y, z}; }
__device__ V3 add(V3 a, V3 b) { return v(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ V3 sub(V3 a, V3 b) { return v(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ V3 mul(V3 a, float s) { return v(a.x*s, a.y*s, a.z*s); }
__device__ float dot(V3 a, V3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ V3 cross(V3 a, V3 b) { return v(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ float len(V3 a) { return sqrtf(dot(a,a)); }
__device__ V3 norm(V3 a) { return mul(a, 1.0f/fmaxf(len(a), 1e-20f)); }

// __MODEL_SOURCE__

extern "C" __global__ void sample_field(float* field, unsigned nodes,
    float bound, float step, float radius, float hole_radius, unsigned* packet) {
    unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= nodes*nodes*nodes) return;
    unsigned x = i%nodes, y = (i/nodes)%nodes, z = i/(nodes*nodes);
    float value=model(v(-bound+step*x, -bound+step*y, -bound+step*z), radius, hole_radius);
    field[i] = value;
    if (!isfinite(value)) atomicAdd(packet+2,1u);
    unsigned n=nodes-1;
    if ((x==0 || y==0 || z==0 || x==n || y==n || z==n) && value<=0.0f)
        atomicAdd(packet+1,1u);
}

extern "C" __global__ void gather_probes(const float* field, unsigned count, unsigned* packet) {
    unsigned q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q<64) packet[16+q]=__float_as_uint(field[(q*2654435761u)%count]);
}

__device__ V3 interpolate(V3 a, V3 b, float fa, float fb) {
    // Canonical endpoint order makes shared-edge vertices bit-identical.
    if (a.x>b.x || (a.x==b.x && (a.y>b.y || (a.y==b.y && a.z>b.z)))) {
        V3 temp=a; a=b; b=temp;
        float ftemp=fa; fa=fb; fb=ftemp;
    }
    if (fa==0.0f) return a;
    if (fb==0.0f) return b;
    float t=fa/(fa-fb);
    return add(a, mul(sub(b,a),t));
}

__device__ void emit_triangle(V3 a, V3 b, V3 c, V3 outward,
    unsigned* packet, unsigned capacity) {
    V3 normal=cross(sub(b,a),sub(c,a));
    if (dot(normal,normal)==0.0f) return;
    if (dot(normal,outward)<0.0f) { V3 temp=b; b=c; c=temp; }
    unsigned index=atomicAdd(packet,1u);
    if (index>=capacity) return;
    unsigned* dst=packet+80+9*index;
    dst[0]=__float_as_uint(a.x); dst[1]=__float_as_uint(a.y); dst[2]=__float_as_uint(a.z);
    dst[3]=__float_as_uint(b.x); dst[4]=__float_as_uint(b.y); dst[5]=__float_as_uint(b.z);
    dst[6]=__float_as_uint(c.x); dst[7]=__float_as_uint(c.y); dst[8]=__float_as_uint(c.z);
}

extern "C" __global__ void polygonize(const float* field, unsigned n,
    float bound, float step, unsigned* packet, unsigned capacity) {
    unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=n*n*n) return;
    unsigned x=cell%n, y=(cell/n)%n, z=cell/(n*n), nodes=n+1;
    const unsigned dx[8]={0,1,1,0,0,1,1,0};
    const unsigned dy[8]={0,0,1,1,0,0,1,1};
    const unsigned dz[8]={0,0,0,0,1,1,1,1};
    const int tets[6][4]={{0,5,1,6},{0,1,2,6},{0,2,3,6},
                         {0,3,7,6},{0,7,4,6},{0,4,5,6}};
    V3 points[8]; float values[8];
    bool negative=false, positive=false;
    for (int i=0;i<8;++i) {
        unsigned ix=x+dx[i], iy=y+dy[i], iz=z+dz[i];
        points[i]=v(-bound+step*ix,-bound+step*iy,-bound+step*iz);
        values[i]=field[ix+nodes*(iy+nodes*iz)];
        negative|=values[i]<0.0f; positive|=values[i]>=0.0f;
    }
    if (!negative || !positive) return;
    for (int t=0;t<6;++t) {
        int in[4], out[4], ni=0, no=0;
        V3 inside=v(0,0,0), outside=v(0,0,0);
        for (int k=0;k<4;++k) {
            int j=tets[t][k];
            if (values[j]<0.0f) { in[ni++]=j; inside=add(inside,points[j]); }
            else { out[no++]=j; outside=add(outside,points[j]); }
        }
        if (ni==0 || ni==4) continue;
        V3 direction=sub(mul(outside,1.0f/no),mul(inside,1.0f/ni));
        if (ni==1 || ni==3) {
            int pivot=ni==1?in[0]:out[0];
            int* others=ni==1?out:in;
            V3 a=interpolate(points[pivot],points[others[0]],values[pivot],values[others[0]]);
            V3 b=interpolate(points[pivot],points[others[1]],values[pivot],values[others[1]]);
            V3 c=interpolate(points[pivot],points[others[2]],values[pivot],values[others[2]]);
            emit_triangle(a,b,c,direction,packet,capacity);
        } else {
            V3 a=interpolate(points[in[0]],points[out[0]],values[in[0]],values[out[0]]);
            V3 b=interpolate(points[in[0]],points[out[1]],values[in[0]],values[out[1]]);
            V3 c=interpolate(points[in[1]],points[out[0]],values[in[1]],values[out[0]]);
            V3 d=interpolate(points[in[1]],points[out[1]],values[in[1]],values[out[1]]);
            emit_triangle(a,b,c,direction,packet,capacity);
            emit_triangle(b,d,c,direction,packet,capacity);
        }
    }
}

__device__ V3 surface_normal(V3 p, float radius, float hole) {
    float e=radius*0.0002f;
    return norm(v(
        model(add(p,v(e,0,0)),radius,hole)-model(sub(p,v(e,0,0)),radius,hole),
        model(add(p,v(0,e,0)),radius,hole)-model(sub(p,v(0,e,0)),radius,hole),
        model(add(p,v(0,0,e)),radius,hole)-model(sub(p,v(0,0,e)),radius,hole)));
}

extern "C" __global__ void render(unsigned* packet, unsigned pixel_offset,
    unsigned width, float radius, float hole) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=width*width) return;
    float sx=(2.0f*(i%width+0.5f)/width-1.0f);
    float sy=(1.0f-2.0f*(i/width+0.5f)/width);
    V3 eye=mul(v(2.6f,-3.2f,2.7f),radius);
    V3 forward=norm(mul(eye,-1.0f));
    V3 right=norm(cross(forward,v(0,0,1))), up=cross(right,forward);
    V3 ray=norm(add(mul(forward,3.0f),add(mul(right,sx),mul(up,sy))));
    float t=0.0f, epsilon=radius*0.00008f;
    bool hit=false; V3 p;
    for (int k=0;k<384 && t<radius*10.0f;++k) {
        p=add(eye,mul(ray,t));
        float d=model(p,radius,hole);
        if (d<epsilon) { hit=true; break; }
        t+=fmaxf(d,epsilon*0.5f);
    }
    V3 color=v(0.89f+0.04f*sy,0.93f+0.03f*sy,0.97f+0.02f*sy);
    if (hit) {
        V3 normal=surface_normal(p,radius,hole);
        V3 light=norm(v(-0.5f,-0.65f,1.0f));
        float diffuse=fmaxf(0.0f,dot(normal,light));
        float ao=1.0f;
        for (int j=1;j<=5;++j) {
            float h=radius*0.025f*j;
            ao-=fmaxf(0.0f,h-model(add(p,mul(normal,h)),radius,hole))/h*0.09f;
        }
        ao=fmaxf(0.3f,ao);
        float spec=powf(fmaxf(0.0f,dot(normal,norm(sub(light,ray)))),64.0f);
        float inner=fabsf(sqrtf(p.x*p.x+p.y*p.y)-hole)<fabsf(len(p)-radius)?1.0f:0.0f;
        V3 base=inner>0.5f?v(0.75f,0.43f,0.16f):v(0.055f,0.38f,0.59f);
        color=add(mul(base,(0.20f+0.80f*diffuse)*ao),v(spec*0.55f,spec*0.55f,spec*0.55f));
    }
    unsigned red=(unsigned)(255.0f*sqrtf(fminf(1.0f,fmaxf(0.0f,color.x))));
    unsigned green=(unsigned)(255.0f*sqrtf(fminf(1.0f,fmaxf(0.0f,color.y))));
    unsigned blue=(unsigned)(255.0f*sqrtf(fminf(1.0f,fmaxf(0.0f,color.z))));
    packet[pixel_offset+i]=red|(green<<8)|(blue<<16)|(255u<<24);
}
