// AgCAD OptiX shim: one triangle-mesh closest-hit ray-query pipeline. C API, no exceptions.
// Device programs come in as PTX (compiled by the caller with NVRTC).
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cstring>
#include <vector>

namespace {
struct Params { OptixTraversableHandle gas; CUdeviceptr rays; CUdeviceptr hits; unsigned n; unsigned pad; };
struct Record { alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
struct Shim {
    OptixDeviceContext ctx = nullptr;
    OptixModule mod = nullptr;
    OptixProgramGroup rg = nullptr, ms = nullptr, hg = nullptr;
    OptixPipeline pipe = nullptr;
    CUdeviceptr sbt_mem = 0, params = 0;
    OptixShaderBindingTable sbt = {};
    std::vector<CUdeviceptr> gas_bufs;
};
int fail(OptixResult r) { return r == OPTIX_SUCCESS ? 0 : 1000 + (int)r; }
int cufail(CUresult r) { return r == CUDA_SUCCESS ? 0 : 2000 + (int)r; }
}

#define TRY(x) do { int _e = (x); if (_e) return _e; } while (0)

extern "C" int agcad_optix_create(void* cu_ctx, const char* ptx, size_t ptx_len, void** out) {
    TRY(fail(optixInit()));
    Shim* s = new Shim();
    OptixDeviceContextOptions o = {};
    TRY(fail(optixDeviceContextCreate((CUcontext)cu_ctx, &o, &s->ctx)));
    OptixModuleCompileOptions mo = {};
    mo.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions po = {};
    po.usesMotionBlur = 0;
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues = 5;
    po.numAttributeValues = 2;
    po.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[4096]; size_t ll = sizeof(log);
    TRY(fail(optixModuleCreate(s->ctx, &mo, &po, ptx, ptx_len, log, &ll, &s->mod)));
    OptixProgramGroupOptions go = {};
    OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = s->mod; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = s->mod; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = s->mod; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup groups[3];
    ll = sizeof(log);
    TRY(fail(optixProgramGroupCreate(s->ctx, d, 3, &go, log, &ll, groups)));
    s->rg = groups[0]; s->ms = groups[1]; s->hg = groups[2];
    OptixPipelineLinkOptions lo = {};
    lo.maxTraceDepth = 1;
    ll = sizeof(log);
    TRY(fail(optixPipelineCreate(s->ctx, &po, &lo, groups, 3, log, &ll, &s->pipe)));
    TRY(fail(optixPipelineSetStackSize(s->pipe, 0, 0, 2048, 1)));
    // shader binding table: one header-only record per group
    Record rec[3];
    for (int i = 0; i < 3; ++i) TRY(fail(optixSbtRecordPackHeader(groups[i], &rec[i])));
    TRY(cufail(cuMemAlloc(&s->sbt_mem, sizeof(rec))));
    TRY(cufail(cuMemcpyHtoD(s->sbt_mem, rec, sizeof(rec))));
    s->sbt.raygenRecord = s->sbt_mem;
    s->sbt.missRecordBase = s->sbt_mem + sizeof(Record); s->sbt.missRecordStrideInBytes = sizeof(Record); s->sbt.missRecordCount = 1;
    s->sbt.hitgroupRecordBase = s->sbt_mem + 2 * sizeof(Record); s->sbt.hitgroupRecordStrideInBytes = sizeof(Record); s->sbt.hitgroupRecordCount = 1;
    TRY(cufail(cuMemAlloc(&s->params, sizeof(Params))));
    *out = s;
    return 0;
}

extern "C" int agcad_optix_build_gas(void* h, unsigned long long verts, unsigned n_verts, unsigned long long idx, unsigned n_tris, unsigned long long* out_gas) {
    Shim* s = (Shim*)h;
    OptixBuildInput in = {};
    in.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    CUdeviceptr vb = (CUdeviceptr)verts;
    unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    in.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    in.triangleArray.vertexStrideInBytes = 12;
    in.triangleArray.numVertices = n_verts;
    in.triangleArray.vertexBuffers = &vb;
    in.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    in.triangleArray.indexStrideInBytes = 12;
    in.triangleArray.numIndexTriplets = n_tris;
    in.triangleArray.indexBuffer = (CUdeviceptr)idx;
    in.triangleArray.flags = &flags;
    in.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sz;
    TRY(fail(optixAccelComputeMemoryUsage(s->ctx, &ao, &in, 1, &sz)));
    CUdeviceptr tmp, outb;
    TRY(cufail(cuMemAlloc(&tmp, sz.tempSizeInBytes)));
    TRY(cufail(cuMemAlloc(&outb, sz.outputSizeInBytes)));
    OptixTraversableHandle handle;
    TRY(fail(optixAccelBuild(s->ctx, 0, &ao, &in, 1, tmp, sz.tempSizeInBytes, outb, sz.outputSizeInBytes, &handle, nullptr, 0)));
    TRY(cufail(cuStreamSynchronize(0)));
    cuMemFree(tmp);
    s->gas_bufs.push_back(outb);
    *out_gas = (unsigned long long)handle;
    return 0;
}

extern "C" int agcad_optix_trace(void* h, unsigned long long gas, unsigned long long rays, unsigned long long hits, unsigned n, void* stream) {
    Shim* s = (Shim*)h;
    if (n == 0) return 0;
    Params p = { (OptixTraversableHandle)gas, (CUdeviceptr)rays, (CUdeviceptr)hits, n, 0 };
    TRY(cufail(cuMemcpyHtoDAsync(s->params, &p, sizeof(p), (CUstream)stream)));
    TRY(fail(optixLaunch(s->pipe, (CUstream)stream, s->params, sizeof(Params), &s->sbt, n, 1, 1)));
    return cufail(cuStreamSynchronize((CUstream)stream));
}

extern "C" void agcad_optix_destroy(void* h) {
    Shim* s = (Shim*)h;
    if (!s) return;
    for (CUdeviceptr b : s->gas_bufs) cuMemFree(b);
    if (s->params) cuMemFree(s->params);
    if (s->sbt_mem) cuMemFree(s->sbt_mem);
    if (s->pipe) optixPipelineDestroy(s->pipe);
    if (s->hg) optixProgramGroupDestroy(s->hg);
    if (s->ms) optixProgramGroupDestroy(s->ms);
    if (s->rg) optixProgramGroupDestroy(s->rg);
    if (s->mod) optixModuleDestroy(s->mod);
    if (s->ctx) optixDeviceContextDestroy(s->ctx);
    delete s;
}
