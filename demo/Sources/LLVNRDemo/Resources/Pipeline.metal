#include <metal_stdlib>
using namespace metal;

struct Parameters {
    uint width, height, packedWidth, packedHeight;
    uint mapWidth, mapHeight, reset, fullRange;
    uint originX, originY, matrix601, reserved;
    float dnr, fusion, historyWeight, split;
};

kernel void pack420(texture2d<float,access::read> y [[texture(0)]],
                    texture2d<float,access::read> uv [[texture(1)]],
                    device half *current [[buffer(0)]],
                    device half *history [[buffer(1)]],
                    constant Parameters &p [[buffer(2)]], uint2 xy [[thread_position_in_grid]]) {
    if(xy.x>=p.packedWidth || xy.y>=p.packedHeight) return;
    int2 source=int2(xy*2)-int2(p.originX,p.originY);
    uint2 s=uint2(clamp(source,int2(0),int2(p.width-1,p.height-1)));
    uint2 s1=uint2(clamp(source+int2(1),int2(0),int2(p.width-1,p.height-1)));
    const float scaleY=p.fullRange ? 1.0 : 255.0/219.0;
    const float offsetY=p.fullRange ? 0.0 : 16.0/219.0;
    float4 luma=float4(y.read(s).r,y.read(uint2(s1.x,s.y)).r,
                      y.read(uint2(s.x,s1.y)).r,y.read(s1).r)*scaleY-offsetY;
    uint2 cs=uint2(clamp(source/2,int2(0),int2(p.width/2-1,p.height/2-1)));
    float2 chroma=(uv.read(cs).rg*255.0-128.0)/(p.fullRange ? 255.0 : 224.0);
    uint i=xy.y*p.packedWidth+xy.x;
    uint plane=p.packedWidth*p.packedHeight;
    for(uint c=0;c<6;c++) {
        half value=half(c<4 ? luma[c] : chroma[c-4]);
        current[i+c*plane]=value;
        if(p.reset) history[i+c*plane]=value;
    }
}

kernel void fillMaps(device half *dnr [[buffer(0)]], device half *fusion [[buffer(1)]],
                     constant Parameters &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.mapWidth*p.mapHeight) return;
    dnr[i]=half(p.dnr); fusion[i]=half(p.fusion);
}

kernel void updateHistory(device half *history [[buffer(0)]], device const half *state [[buffer(1)]],
                          constant Parameters &p [[buffer(2)]], device half *previewHistory [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if(i>=6*p.packedWidth*p.packedHeight) return;
    // FP32 arithmetic then FP16 storage matches the validated host policy.
    if(p.reserved) previewHistory[i]=history[i];
    history[i]=half((1.0-p.historyWeight)*float(history[i])+p.historyWeight*float(state[i]));
}

kernel void unpack420(device const half *out [[buffer(0)]],
                      constant Parameters &p [[buffer(1)]],
                      texture2d<float,access::write> result [[texture(0)]],
                      texture2d<float,access::read> sourceY [[texture(1)]],
                      texture2d<float,access::read> sourceUV [[texture(2)]],
                      texture2d<float,access::write> original [[texture(3)]],
                      uint2 xy [[thread_position_in_grid]]) {
    if(xy.x>=p.width || xy.y>=p.height) return;
    uint2 modelXY=xy+uint2(p.originX,p.originY);
    uint index=(modelXY.y/2)*p.packedWidth+modelXY.x/2;
    uint plane=p.packedWidth*p.packedHeight;
    uint phase=(modelXY.y%2)*2+modelXY.x%2;
    float y=float(out[index+phase*plane]);
    float cb=float(out[index+4*plane]), cr=float(out[index+5*plane]);
    // Preserve Y' domain. bt709 primaries/sRGB transfer is set on the layer.
    float3 rgb = p.matrix601 ? float3(y+1.402*cr,y-0.344136*cb-0.714136*cr,y+1.772*cb)
                            : float3(y+1.5748*cr,y-0.187324*cb-0.468124*cr,y+1.8556*cb);
    result.write(float4(clamp(rgb,0.0,1.0),1.0),xy);
    float sy=sourceY.read(xy).r;
    sy=p.fullRange ? sy : (sy*255.0-16.0)/219.0;
    float2 suv=(sourceUV.read(xy/2).rg*255.0-128.0)/(p.fullRange ? 255.0 : 224.0);
    float3 srgb=p.matrix601 ? float3(sy+1.402*suv.y,sy-0.344136*suv.x-0.714136*suv.y,sy+1.772*suv.x)
                           : float3(sy+1.5748*suv.y,sy-0.187324*suv.x-0.468124*suv.y,sy+1.8556*suv.x);
    original.write(float4(clamp(srgb,0.0,1.0),1.0),xy);
}

struct VertexOut { float4 position [[position]]; float2 uv; };
vertex VertexOut displayVertex(uint id [[vertex_id]]) {
    float2 pos[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    VertexOut v;v.position=float4(pos[id],0,1);v.uv=float2((pos[id].x+1)*.5,(1-pos[id].y)*.5);return v;
}
struct DisplayParameters {
    uint mode;
    float split, zoom, centerX, centerY;
};
fragment float4 displayFragment(VertexOut in [[stage_in]],texture2d<float> input [[texture(0)]],
                                texture2d<float> original [[texture(1)]],
                                constant DisplayParameters &p [[buffer(0)]]) {
    constexpr sampler s(filter::linear,address::clamp_to_edge);
    if(p.mode==1) return original.sample(s,in.uv);
    if(p.mode==2) {
        if(abs(in.uv.x-p.split)<.001) return float4(1,1,1,1);
        return in.uv.x<p.split ? original.sample(s,in.uv) : input.sample(s,in.uv);
    }
    if(p.mode==3) {
        if(abs(in.uv.x-.5)<.0006) return float4(.4,.4,.4,1);
        float2 paneUV=float2(fract(in.uv.x*2),in.uv.y);
        // Each pane fills half the 16:9 view with the same 8:9 source crop.
        // Shared zoom and center keep both sides aligned without stretching.
        float2 crop=float2(p.centerX,p.centerY)+(paneUV-.5)*float2(.5,1)/p.zoom;
        return in.uv.x<.5 ? original.sample(s,crop) : input.sample(s,crop);
    }
    return input.sample(s,in.uv);
}
