#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <stdexcept>
#include <vector>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;
struct Vertex {
  float xyz[3], normal[3];
  uint32_t face;
};
struct Camera {
  float yaw = .64f, pitch = .62f, zoom = 1, pan_x = 0, pan_y = 0;
  uint32_t sequence = 0;
};
struct Uniforms {
  float rows[3][4];
  uint32_t selected, skybox, pad[2];
};

static Uniforms transform(const Camera &c, uint32_t selected) {
  float s = std::sin(c.yaw), k = std::cos(c.yaw), h = std::sin(c.pitch),
        v = std::cos(c.pitch);
  float x = c.zoom * 9.f / 16, y = c.zoom;
  return {{{k * x, s * x, 0, c.pan_x},
           {-s * h * y, k * h * y, v * y, c.pan_y},
           {-s * v * .25f, k * v * .25f, -h * .25f, .5f}},
          selected, 0,
          {}};
}
static void check(HRESULT r, const char *what) {
  if (FAILED(r))
    throw std::runtime_error(what);
}

class Renderer {
  ComPtr<ID3D11VertexShader> vs;
  ComPtr<ID3D11PixelShader> ps;
  ComPtr<ID3D11VertexShader> sky_vs;
  ComPtr<ID3D11PixelShader> sky_ps;
  ComPtr<ID3D11InputLayout> layout;
  ComPtr<ID3D11Buffer> uniforms, vertices, marker_buffer;
  uint64_t uploaded_marker = 0;
  ComPtr<ID3D11DepthStencilView> depth;
  ComPtr<ID3D11RasterizerState> raster;
  ComPtr<ID3D11ShaderResourceView> palette;
  uint64_t palette_version = 0;
  uint64_t uploaded = 0;
  static ComPtr<ID3DBlob> compile(const char *source, const char *entry,
                                  const char *target) {
    ComPtr<ID3DBlob> shader, error;
    check(D3DCompile(source, strlen(source), nullptr, nullptr, nullptr, entry,
                     target, 0, 0, &shader, &error),
          "CAD shader compilation failed");
    return shader;
  }
  void initialize(ID3D11Device *d) {
    const char *source = R"(
cbuffer View { float4 right, up, depth; uint selected, skybox; };
struct SkyOutput { float4 p:SV_POSITION; float2 uv:TEXCOORD0; };
SkyOutput skyVertex(uint id:SV_VertexID) {
    SkyOutput o;
    o.uv=float2(id==2 ? 3 : -1,id==1 ? 3 : -1);
    o.p=float4(o.uv,0,1);
    return o;
}
// Z-up world directions. Colors and exposure are evaluated in scene-linear space.
struct Environment {
    float3 lightDir, lightColor, ground, horizon, zenith;
    float haze, sunRadius, exposure;
};
Environment preset() {
    Environment e;
    if (skybox == 1) {
        e.lightDir=normalize(float3(-.62,-.45,.68)); e.lightColor=float3(1.1,1.08,1.05);
        e.ground=float3(.075,.085,.10); e.horizon=float3(.49,.53,.58); e.zenith=float3(.23,.26,.30);
        e.haze=.34; e.sunRadius=0; e.exposure=.93;
    } else if (skybox == 2) {
        e.lightDir=normalize(float3(-.41,.78,.47)); e.lightColor=float3(1.6,1.48,1.28);
        e.ground=float3(.075,.093,.068); e.horizon=float3(.86,1.0,1.10); e.zenith=float3(.075,.29,.70);
        e.haze=.18; e.sunRadius=.015; e.exposure=1.05;
    } else {
        e.lightDir=normalize(float3(.48,.87,.12)); e.lightColor=float3(1.9,.89,.40);
        e.ground=float3(.086,.047,.064); e.horizon=float3(1.03,.46,.29); e.zenith=float3(.105,.14,.36);
        e.haze=.30; e.sunRadius=.021; e.exposure=.90;
    }
    return e;
}
float softbox(float3 ray, float3 direction, float2 halfSize, float blur) {
    float3 side=normalize(cross(float3(0,0,1),direction));
    float facing=dot(ray,direction);
    float2 uv=abs(float2(dot(ray,side),dot(ray,cross(direction,side)))) / max(facing,.001);
    float2 mask=1-smoothstep(halfSize-blur,halfSize+blur,uv);
    return saturate(facing*12)*mask.x*mask.y;
}
float3 radiance(Environment e, float3 ray, float roughness) {
    float blur=roughness*roughness;
    float skyWeight=smoothstep(-.018-blur*.3,.018+blur*.3,ray.z);
    float3 pole=lerp(e.ground,e.zenith,skyWeight);
    float air=exp2(-abs(ray.z)/(e.haze+.22*blur));
    float3 color=lerp(pole,e.horizon,air);
    if (skybox==1) {
        float width=.025+.6*blur;
        float key=softbox(ray,e.lightDir,float2(.52,.27),width);
        float strip=softbox(ray,normalize(float3(.81,.42,.40)),float2(.15,.62),width);
        return color+(3*key+2*strip)*e.lightColor/(1+5*blur);
    }
    // Chord distance is stable at the center of a small angular disk.
    float distanceToSun=length(ray-e.lightDir);
    float radius=e.sunRadius;
    float spread=radius+.32*blur;
    float disk=(1-smoothstep(.92*spread,1.08*spread,distanceToSun))*radius*radius/(spread*spread);
    float halo=exp2(-distanceToSun/(.07+.4*e.haze+.12*blur));
    return color+skyWeight*e.lightColor*(6*disk+.69*halo/(1+blur));
}
float3 displayColor(float3 light, float exposure) {
    float3 mapped=1-exp2(-max(light,0)*exposure);
    return max(1.055*pow(mapped,1/2.4)-.055,0);
}
float4 skyPixel(SkyOutput o):SV_TARGET {
    float3 ray=normalize(normalize(depth.xyz)+o.uv.x*(16.0/9)*normalize(right.xyz)+o.uv.y*normalize(up.xyz));
    Environment e=preset();
    return float4(displayColor(radiance(e,ray,0),e.exposure),1);
}
struct Finish { float4 color; float4 surface; };
StructuredBuffer<Finish> finishes:register(t0);
struct Input { float3 p:POSITION; float3 n:NORMAL; uint face:FACE; };
struct Output { float4 p:SV_POSITION; float3 n:NORMAL; nointerpolation uint face:FACE; };
Output vertex(Input i) {
    Output o; float4 p=float4(i.p,1); o.p=float4(dot(p,right),dot(p,up),dot(p,depth),1);
    if (i.face >= 0x80000000u) {
        float2 pixelSize = float2(2.0/1920, 2.0/1080);
        float2 offset = i.n.xy;
        if (i.face != 0x80000000u) {
            float3 span = i.n - i.p;
            float2 dir = float2(dot(span,right.xyz),dot(span,up.xyz)) / pixelSize;
            offset = float2(-dir.y,dir.x) / max(length(dir),1e-6);
            offset *= i.face == 0x80000001u ? 3.0 : -3.0;
        }
        o.p.xy += offset * pixelSize;
        o.p.z -= 0.00002;
    }
    o.n=i.n; o.face=i.face; return o;
}
float4 pixel(Output o):SV_TARGET {
    if (o.face >= 0x80000000u) return float4(1,.66,.12,1);
    Finish f=finishes[o.face]; float3 n=normalize(o.n), v=-normalize(depth.xyz);
    if(dot(n,v)<0) n=-n;
    float rough=f.surface.x, metal=f.surface.y;
    float3 color;
    if (skybox==0) {
        float3 l=normalize(float3(.32,-.45,.83)), h=normalize(l+v);
        color=f.color.rgb*(.34+.66*saturate(dot(n,l)))*(1-.36*metal);
        color+=lerp(.12.xxx,f.color.rgb,metal)*pow(saturate(dot(n,h)),lerp(100,4,rough));
    } else {
        Environment e=preset();
        float3 albedo=pow(saturate(f.color.rgb),2.2);
        float3 ambient=lerp(e.ground,e.zenith,.5+.5*n.z)+.28*e.horizon;
        float3 diffuse=albedo*(1-metal)*(ambient+e.lightColor*saturate(dot(n,e.lightDir)));
        float3 f0=lerp(.04.xxx,albedo,metal);
        float3 fresnel=f0+(max(1-rough,f0)-f0)*pow(1-saturate(dot(n,v)),5);
        float3 reflection=radiance(e,reflect(-v,n),rough);
        color=displayColor(diffuse+fresnel*reflection,e.exposure);
    }
    if(o.face==selected || f.surface.w>0.5) color=lerp(color,float3(1,.63,.16),.32);
    return float4(saturate(color),1);
})";
    auto v = compile(source, "vertex", "vs_5_0"),
         p = compile(source, "pixel", "ps_5_0");
    auto sv = compile(source, "skyVertex", "vs_5_0"), sp = compile(source, "skyPixel", "ps_5_0");
    check(d->CreateVertexShader(sv->GetBufferPointer(), sv->GetBufferSize(), nullptr, &sky_vs), "sky vertex shader");
    check(d->CreatePixelShader(sp->GetBufferPointer(), sp->GetBufferSize(), nullptr, &sky_ps), "sky pixel shader");
    check(d->CreateVertexShader(v->GetBufferPointer(), v->GetBufferSize(),
                                nullptr, &vs),
          "vertex shader");
    check(d->CreatePixelShader(p->GetBufferPointer(), p->GetBufferSize(),
                               nullptr, &ps),
          "pixel shader");
    D3D11_INPUT_ELEMENT_DESC fields[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,
         D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"FACE", 0, DXGI_FORMAT_R32_UINT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA,
         0}};
    check(d->CreateInputLayout(fields, 3, v->GetBufferPointer(),
                               v->GetBufferSize(), &layout),
          "vertex layout");
    D3D11_BUFFER_DESC b = {};
    b.ByteWidth = sizeof(Uniforms);
    b.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    check(d->CreateBuffer(&b, nullptr, &uniforms), "camera buffer");
    D3D11_TEXTURE2D_DESC t = {};
    t.Width = 1920;
    t.Height = 1080;
    t.MipLevels = t.ArraySize = t.SampleDesc.Count = 1;
    t.Format = DXGI_FORMAT_D32_FLOAT;
    t.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    ComPtr<ID3D11Texture2D> texture;
    check(d->CreateTexture2D(&t, nullptr, &texture), "depth texture");
    check(d->CreateDepthStencilView(texture.Get(), nullptr, &depth),
          "depth view");
    D3D11_RASTERIZER_DESC rs = {};
    rs.FillMode = D3D11_FILL_SOLID;
    rs.CullMode = D3D11_CULL_NONE;
    rs.DepthClipEnable = TRUE;
    check(d->CreateRasterizerState(&rs, &raster), "rasterizer");
  }

public:
  void draw(ID3D11Device *d, ID3D11DeviceContext *ctx,
            ID3D11RenderTargetView *target, uint64_t scene,
            const std::vector<Vertex> &mesh, const Uniforms &u,
            const std::vector<std::array<float, 8>> &colors, uint64_t version,
            const std::vector<Vertex> &marker, uint64_t marker_id) {
    if (!vs)
      initialize(d);
    if (!mesh.empty() && uploaded != scene) {
      D3D11_BUFFER_DESC b = {};
      b.ByteWidth = static_cast<UINT>(mesh.size() * sizeof(Vertex));
      b.BindFlags = D3D11_BIND_VERTEX_BUFFER;
      b.Usage = D3D11_USAGE_IMMUTABLE;
      D3D11_SUBRESOURCE_DATA data = {mesh.data(), 0, 0};
      check(d->CreateBuffer(&b, &data, vertices.ReleaseAndGetAddressOf()),
            "mesh upload");
      uploaded = scene;
    }
    if (!colors.empty() && palette_version != version) {
      D3D11_BUFFER_DESC b = {};
      b.ByteWidth = static_cast<UINT>(colors.size() * sizeof(colors[0]));
      b.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      b.Usage = D3D11_USAGE_IMMUTABLE;
      b.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      b.StructureByteStride = sizeof(colors[0]);
      ComPtr<ID3D11Buffer> buf;
      D3D11_SUBRESOURCE_DATA data = {colors.data(), 0, 0};
      check(d->CreateBuffer(&b, &data, &buf), "material buffer");
      check(d->CreateShaderResourceView(buf.Get(), nullptr,
                                        palette.ReleaseAndGetAddressOf()),
            "material view");
      palette_version = version;
    }
    ctx->PSSetShaderResources(0, 1, palette.GetAddressOf());
    float bg[] = {.052f, .074f, .092f, 1};
    ctx->ClearRenderTargetView(target, bg);
    ctx->ClearDepthStencilView(depth.Get(), D3D11_CLEAR_DEPTH, 1, 0);
    ctx->OMSetRenderTargets(1, &target, depth.Get());
    ctx->OMSetDepthStencilState(nullptr, 0);
    ctx->OMSetBlendState(nullptr, nullptr, ~0u);
    D3D11_VIEWPORT vp = {0, 0, 1920, 1080, 0, 1};
    ctx->RSSetViewports(1, &vp);
    ctx->RSSetState(raster.Get());
    UINT stride = sizeof(Vertex), offset = 0;
    ctx->IASetVertexBuffers(0, 1, vertices.GetAddressOf(), &stride, &offset);
    ctx->IASetInputLayout(layout.Get());
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->UpdateSubresource(uniforms.Get(), 0, nullptr, &u, 0, 0);
    ctx->VSSetConstantBuffers(0, 1, uniforms.GetAddressOf());
    ctx->PSSetConstantBuffers(0, 1, uniforms.GetAddressOf());
    if (u.skybox != 0) {
      ctx->OMSetRenderTargets(1, &target, nullptr);
      ctx->IASetInputLayout(nullptr);
      ctx->VSSetShader(sky_vs.Get(), nullptr, 0);
      ctx->PSSetShader(sky_ps.Get(), nullptr, 0);
      ctx->Draw(3, 0);
      ctx->OMSetRenderTargets(1, &target, depth.Get());
      ctx->IASetInputLayout(layout.Get());
    }
    if (mesh.empty()) {
      ctx->OMSetRenderTargets(0, nullptr, nullptr);
      return;
    }
    ctx->VSSetShader(vs.Get(), nullptr, 0);
    ctx->PSSetShader(ps.Get(), nullptr, 0);
    ctx->Draw(static_cast<UINT>(mesh.size()), 0);
    if (uploaded_marker != marker_id) {
      marker_buffer.Reset();
      if (!marker.empty()) {
        D3D11_BUFFER_DESC b = {};
        b.ByteWidth = static_cast<UINT>(marker.size() * sizeof(Vertex));
        b.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        b.Usage = D3D11_USAGE_IMMUTABLE;
        D3D11_SUBRESOURCE_DATA data = {marker.data(), 0, 0};
        check(d->CreateBuffer(&b, &data, &marker_buffer), "selection upload");
      }
      uploaded_marker = marker_id;
    }
    if (marker_buffer) {
      ctx->IASetVertexBuffers(0, 1, marker_buffer.GetAddressOf(), &stride, &offset);
      ctx->Draw(static_cast<UINT>(marker.size()), 0);
    }
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
  }
};
