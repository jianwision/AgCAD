// LLM-generated geometry goes here. This file is read and compiled at runtime.
// Coordinates and both parameters are in millimetres. Negative = inside.
// Keep the field conservative (1-Lipschitz) for the sphere-tracing preview.
// max/min Boolean combinations preserve the sign/zero set, but the result need
// not be an exact signed distance everywhere.
__device__ float model(V3 p, float radius, float hole_radius) {
    float sphere = len(p) - radius;
    float cylinder = sqrtf(p.x * p.x + p.y * p.y) - hole_radius;
    return fmaxf(sphere, -cylinder);
}
