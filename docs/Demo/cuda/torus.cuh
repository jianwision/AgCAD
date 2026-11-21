// Alternative runtime model. The domain is still controlled by --radius.
__device__ float model(V3 p, float radius, float hole_radius) {
    float radial = sqrtf(p.x*p.x+p.y*p.y) - radius*0.65f;
    return sqrtf(radial*radial+p.z*p.z) - radius*0.22f;
}
