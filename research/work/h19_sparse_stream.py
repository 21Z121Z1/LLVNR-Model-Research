#!/usr/bin/env python3
"""Count-first sparse streams with 16-bit masks and even output packing."""
import numpy as np
from h19_sparse_extended import tiles_for_shape, position


def positions(packing, shape, parity=True):
    cin,kh,kw = shape
    slots = max(2, (packing+1)&~1)
    result = []
    for ci,by,bx in tiles_for_shape(cin,kh,kw):
        for tap in range(4):
            channel,y,x = position(ci,by,bx,tap,kh,kw,parity)
            for output in range(slots):
                result.append((output,channel,y,x) if output<packing and channel<cin and y<kh and x<kw else None)
    return result


def decode(raw,packing,shape,has_bias,parity=True):
    order = positions(packing,shape,parity)
    mask_bytes = ((len(order)+15)//16)*2
    prefix = packing*2 if has_bias else 0
    code = int.from_bytes(raw[prefix:prefix+2],"little")
    used = prefix+3+code
    size = (used+63)&~63
    if len(raw)<size or code<mask_bytes or (code-mask_bytes)%2 or raw[prefix+2]:
        raise ValueError("invalid stream sparse header")
    expected = (code-mask_bytes)//2
    if expected>packing*np.prod(shape):
        raise ValueError("stream sparse count exceeds source")
    weights = np.zeros((packing,*shape),np.float16)
    bias = np.frombuffer(raw[:prefix],dtype='<f2').copy() if has_bias else None
    cursor,count = prefix+3,0
    for start in range(0,len(order),16):
        if count==expected:break
        mask = int.from_bytes(raw[cursor:cursor+2],"little")
        cursor += 2
        for bit in range(16):
            if not mask&(1<<bit):continue
            if start+bit>=len(order) or order[start+bit] is None or count>=expected or cursor+2>used:
                raise ValueError("stream sparse selected padded coefficient")
            weights[order[start+bit]] = np.frombuffer(raw[cursor:cursor+2],dtype='<f2')[0]
            cursor += 2
            count += 1
    if count!=expected or any(raw[cursor:size]) or not np.isfinite(weights).all():
        raise ValueError("stream sparse count or padding mismatch")
    return weights,bias,size


def encode(weights,bias,parity=True):
    packing,*shape = weights.shape
    order = positions(packing,shape,parity)
    mask_bytes = ((len(order)+15)//16)*2
    nonzero = int(np.count_nonzero(weights))
    raw = bytearray(np.asarray(bias,dtype='<f2').tobytes() if bias is not None else b'')
    raw.extend((mask_bytes+2*nonzero).to_bytes(2,'little'))
    raw.append(0)
    consumed = 0
    for start in range(0,len(order),16):
        if consumed==nonzero:break
        mask,payload = 0,bytearray()
        for bit in range(16):
            index = order[start+bit] if start+bit<len(order) else None
            if index is not None and weights[index]!=0:
                mask |= 1<<bit
                payload.extend(np.asarray(weights[index],dtype='<f2').tobytes())
                consumed += 1
        raw.extend(mask.to_bytes(2,'little'))
        raw.extend(payload)
    return bytes(raw).ljust((len(raw)+63)&~63,b'\0')
