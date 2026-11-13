#!/usr/bin/env python3
"""Lower precision-sensitive recovered operators to explicit FP32 Metal kernels."""
import json
import math
from pathlib import Path

import numpy as np

from portable_graph import build_ane
from probe_execution_barriers import partition


def export_prefix(plan, coefficients, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prefix,body=partition(plan,coefficients,[["temporal_band0","state_yuv"]])
    body.pop("execution_profile",None)
    shapes={name:list(t.shape) for name,t in build_ane(prefix,coefficients).items()}
    blob=bytearray();functions=[];records=[]
    def weight(value):
        offset=len(blob)//2
        blob.extend(np.asarray(value,dtype="<f2").tobytes())
        return offset
    def number(value):return float(np.float32(value)).hex()+"f"
    def source(shape,c="c",y="y",x="x"):
        return f"(({c if shape[1]!=1 else '0'})*{shape[2]}+({y if shape[2]!=1 else '0'}))*{shape[3]}+({x if shape[3]!=1 else '0'})"
    for node in prefix["nodes"]:
        name,op=node["name"],node["op"]
        shape=shapes[name];count=math.prod(shape)
        if op=="input":
            records.append({"name":name,"shape":shape,"inputs":[],"input":True});continue
        inputs=node["inputs"];args=[shapes[n] for n in inputs]
        read=[f"float({chr(97+i)}[{source(s)}])" for i,s in enumerate(args)]
        lines=[]
        if op=="conv2d":
            entry=coefficients[node["group"]];w=entry["weights"];b=entry["bias"]
            co,ci,kh,kw=w.shape;off=weight(w);boff=weight(b) if b is not None else None
            stride,pad=node["stride"],node["pad"];ih,iw=args[0][2:]
            lines += ["float value=0.0f;",f"for(int ky=0;ky<{kh};ky++) for(int kx=0;kx<{kw};kx++) {{",
                      f"int sy=y*{stride}+ky-{pad}, sx=x*{stride}+kx-{pad};"]
            if node.get("edge"):
                lines += [f"sy=clamp(sy,0,{ih-1});sx=clamp(sx,0,{iw-1});"]
            else:lines += [f"if(sy<0 || sy>={ih} || sx<0 || sx>={iw}) continue;"]
            lines += ["float product=0.0f;",f"for(int ic=0;ic<{ci};ic++) {{",
                      f"float wx=float(w[{off}+((c*{ci}+ic)*{kh}+ky)*{kw}+kx]);",
                      f"product=product+wx*float(a[(ic*{ih}+sy)*{iw}+sx]);","}","value=value+product;","}"]
            if boff is not None:lines += [f"value=value+float(w[{boff}+c]);"]
            if node.get("activation")=="relu":lines += ["value=max(value,0.0f);"]
            if node.get("activation")=="sigmoid":
                table=weight(np.frombuffer(bytes.fromhex(node["activation_lut_hex"]),dtype="<f2"))
                lines += [f"value=lut_sigmoid(value,w+{table});"]
        elif op in ("pool","max_pool_same"):
            k=node["kernel"];stride=node.get("stride",1);edge=node.get("edge",False)
            border=k//2 if edge or op=="max_pool_same" else 0
            mode=node.get("mode",2);ih,iw=args[0][2:]
            lines += ["float value="+("-INFINITY;" if mode==2 else "0.0f;"),
                      f"for(int ky=0;ky<{k};ky++) for(int kx=0;kx<{k};kx++) {{",
                      f"int sy=y*{stride}+ky-{border},sx=x*{stride}+kx-{border};"]
            if edge:lines += [f"sy=clamp(sy,0,{ih-1});sx=clamp(sx,0,{iw-1});"]
            else:lines += [f"if(sy<0||sy>={ih}||sx<0||sx>={iw})continue;"]
            lines += [f"float sample=float(a[(c*{ih}+sy)*{iw}+sx]);",
                      "value=max(value,sample);" if mode==2 else "value=value+sample;","}"]
            if mode!=2:lines += [f"value=value/{number(k*k)};"]
            if "alpha" in node:lines += [f"value=value*{number(node['alpha'])}+{number(node['beta'])};"]
        elif op=="concat":
            lines += ["float value=0.0f;"]
            first=0
            for i,s in enumerate(args):
                lines += [f"if(c>={first} && c<{first+s[1]}) value=float({chr(97+i)}[{source(s,c=f'(c-{first})')}]);"]
                first+=s[1]
        elif op=="nearest2":lines += [f"float value=float(a[{source(args[0],y='y/2',x='x/2')}]);"]
        elif op=="match_spatial":
            lines += [f"float value=float(a[{source(args[0],y=f'min(y,{args[0][2]-1})',x=f'min(x,{args[0][3]-1})')}]);"]
        elif op=="channel":lines += [f"float value=float(a[{source(args[0],c=str(node['index']))}]);"]
        else:
            expr={"affine":lambda:f"{read[0]}*{number(node['alpha'])}+{number(node['beta'])}",
                  "clipped_affine":lambda:f"clamp({read[0]}*{number(node['alpha'])}+{number(node['beta'])},{number(node['minimum'])},{number(node['maximum'])})",
                  "clip":lambda:f"clamp({read[0]},{number(node['minimum'])},{number(node['maximum'])})",
                  "absdiff":lambda:f"abs({read[0]}-{read[1]})","add":lambda:f"{read[0]}+{read[1]}",
                  "mul":lambda:f"{read[0]}*{read[1]}","maximum":lambda:f"max({read[0]},{read[1]})"}
            if op=="affine_sigmoid":
                table=weight(np.frombuffer(bytes.fromhex(node["activation_lut_hex"]),dtype="<f2"))
                lines += [f"float value=lut_sigmoid(({read[0]}+{number(node['bias'])})*{number(node['scale'])},w+{table});"]
            elif op in expr:lines += ["float value="+expr[op]()+";"]
            else:raise ValueError(f"unimplemented precision prefix operator {op}")
        function=f"prefix_{len(functions)}"
        function_source=f'''kernel void {function}(device const half*a[[buffer(0)]],device const half*b[[buffer(1)]],
device const half*d[[buffer(3)]],
device const half*w[[buffer(4)]],device half*out[[buffer(5)]],uint index[[thread_position_in_grid]]){{
if(index>={count})return;int x=index%{shape[3]},y=(index/{shape[3]})%{shape[2]},c=index/{shape[2]*shape[3]};
{chr(10).join(lines)}
out[index]=half(value);
}}'''
        # Buffer 2 is named c_input so the channel coordinate can remain c.
        function_source=function_source.replace('device const half*d[[buffer(3)]]','device const half*c_input[[buffer(2)]],device const half*d[[buffer(3)]]')
        function_source=function_source.replace('float(c[','float(c_input[')
        functions.append(function_source)
        records.append({"name":name,"function":function,"shape":shape,"inputs":inputs,"input":False})
    header='''#include <metal_stdlib>
using namespace metal;
float lut_sigmoid(float x,device const half*t){
 float p=(clamp(x,-8.0f,8.0f)+8.0f)*2.0f;
 int i=min(int(p),31);
 float v=float(t[4+i])+(p-float(i))*(float(t[5+i])-float(t[4+i]));
 if(x< -8.0f)v=x*float(t[37])+float(t[38]);
 if(x>8.0f)v=x*float(t[39])+float(t[40]);
 if(x<float(t[0]))v=float(t[2]);if(x>float(t[1]))v=float(t[3]);return v;
}
'''
    (folder/"prefix.metal").write_text(header+"\n".join(functions)+"\n")
    (folder/"prefix_weights.bin").write_bytes(blob)
    manifest={"version":1,"nodes":records,"outputs":prefix["outputs"],"arithmetic":"FP32 per original operator, FP16 nearest-even storage between operators; fast math disabled"}
    (folder/"prefix.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return prefix,body
