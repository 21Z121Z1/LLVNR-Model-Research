// Compiled together with Pipeline.metal, which defines Parameters.
struct VideoTileParameters {
    uint width, height;
    int x, y;
};

kernel void packVideo420(texture2d<float,access::read> y [[texture(0)]],
                         texture2d<float,access::read> uv [[texture(1)]],
                         device half *current [[buffer(0)]], device half *history [[buffer(1)]],
                         constant Parameters &p [[buffer(2)]],
                         constant VideoTileParameters &tile [[buffer(3)]],
                         uint2 xy [[thread_position_in_grid]]) {
    if(xy.x>=p.packedWidth || xy.y>=p.packedHeight) return;
    int2 source=int2(xy*2)-int2(p.originX,p.originY)+int2(tile.x,tile.y);
    uint2 a=uint2(clamp(source,int2(0),int2(tile.width-1,tile.height-1)));
    uint2 b=uint2(clamp(source+1,int2(0),int2(tile.width-1,tile.height-1)));
    // AVAssetReader supplies P010: 10-bit code values in the high ten bits.
    float4 luma=float4(y.read(a).r,y.read(uint2(b.x,a.y)).r,
                       y.read(uint2(a.x,b.y)).r,y.read(b).r)*65535.0/64.0;
    luma=(luma-64.0)/876.0;
    uint2 c=uint2(clamp(source/2,int2(0),int2(tile.width/2-1,tile.height/2-1)));
    float2 chroma=(uv.read(c).rg*65535.0/64.0-512.0)/896.0;
    uint i=xy.y*p.packedWidth+xy.x, plane=p.packedWidth*p.packedHeight;
    for(uint channel=0;channel<6;channel++) {
        half value=half(channel<4 ? luma[channel]:chroma[channel-4]);
        current[i+channel*plane]=value;
        if(p.reset) history[i+channel*plane]=value;
    }
}

kernel void clearVideo(texture2d<float,access::write> accumulation [[texture(0)]],
                       uint2 xy [[thread_position_in_grid]]) {
    if(xy.x<accumulation.get_width() && xy.y<accumulation.get_height()) accumulation.write(float4(0),xy);
}

kernel void accumulateVideoTile(device const half *out [[buffer(0)]],
                                constant Parameters &p [[buffer(1)]],
                                constant VideoTileParameters &tile [[buffer(2)]],
                                texture2d<float,access::read_write> accumulation [[texture(0)]],
                                uint2 xy [[thread_position_in_grid]]) {
    uint2 canvas=uint2(p.packedWidth*2,p.packedHeight*2);
    if(any(xy>=canvas)) return;
    int2 location=int2(xy)-int2(p.originX,p.originY)+int2(tile.x,tile.y);
    if(any(location<0) || location.x>=int(tile.width) || location.y>=int(tile.height)) return;
    uint plane=p.packedWidth*p.packedHeight, index=(xy.y/2)*p.packedWidth+xy.x/2;
    uint phase=(xy.y%2)*2+xy.x%2;
    float3 value=float3(out[index+phase*plane],out[index+4*plane],out[index+5*plane]);
    float2 leading=float2(xy)/max(float2(p.originX,p.originY),1.0);
    float2 trailing=float2(canvas-1-xy)/max(float2(canvas)-float2(p.originX+p.width,p.originY+p.height),1.0);
    float2 edge=clamp(min(leading,trailing),0.0,1.0);
    float weight=edge.x*edge.y;
    uint2 target=uint2(location);
    accumulation.write(accumulation.read(target)+float4(value*weight,weight),target);
}

float3 videoValue(texture2d<float,access::read> accumulation,uint2 xy) {
    float4 value=accumulation.read(xy);
    return value.xyz/max(value.w,1e-8);
}

kernel void finishVideo420(texture2d<float,access::read> accumulation [[texture(0)]],
                           texture2d<float,access::write> y [[texture(1)]],
                           texture2d<float,access::write> uv [[texture(2)]],
                           uint2 xy [[thread_position_in_grid]]) {
    if(xy.x>=y.get_width() || xy.y>=y.get_height()) return;
    float3 value=videoValue(accumulation,xy);
    float code=round(clamp(value.x*876.0+64.0,0.0,1023.0));
    y.write(float4(code*64.0/65535.0),xy);
    if((xy.x%2)==0 && (xy.y%2)==0) {
        float2 chroma=(value.yz+videoValue(accumulation,xy+uint2(1,0)).yz+
                       videoValue(accumulation,xy+uint2(0,1)).yz+videoValue(accumulation,xy+1).yz)*.25;
        float2 codes=round(clamp(chroma*896.0+512.0,0.0,1023.0));
        uv.write(float4(codes*64.0/65535.0,0,1),xy/2);
    }
}
