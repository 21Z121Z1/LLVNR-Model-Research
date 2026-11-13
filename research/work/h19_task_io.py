import struct,sys
p=sys.argv[1] if len(sys.argv)>1 else 'work/ota_extract/llv_model_2304/24C5037i__iPhone19,2_3_7/System/Library/PrivateFrameworks/ISPKit.framework/V63/LowLightVideo/LLV-HD-Wide-2x-v0.3.0_2304x1296.bundle/H19.bundle/main/main_ane/model.hwx'
b=open(p,'rb').read(); d=b[0x14000:0x14000+0x2259c]; w=list(struct.unpack('<%dI'%(len(d)//4),d)); pos=4; out=[]
while pos<len(w):
 while pos<len(w) and w[pos]==0:pos+=1
 if pos>=len(w):break
 h=w[pos]; n=(h>>16)&0x7ff
 if not n or pos+n>len(w):break
 td=w[pos:pos+n]; regs={}; packets=[]; i=9
 while i<len(td):
  x=td[i]
  if x==0 or (x==1 and td[i+1:i+4]==[0,0,0]) or (not(x>>31) and ((x>>26)&31) and not((x>>29)&1)):break
  a=x&0x7fff;f=(x>>15)&0xffff
  if x>>31: addrs=[a]+[a+1+k for k in range(16) if f>>k&1]
  else: addrs=[a+k for k in range((f&0x3f)+1)]
  vals=td[i+1:i+1+len(addrs)]
  if len(vals)<len(addrs):break
  for aa,vv in zip(addrs,vals):regs[aa]=vv
  if (x>>29)&1: packets.append((a,f,vals))
  i+=1+len(addrs)
 out.append((pos,n,h,regs,packets));pos+=n
print('idx typ bytes W H C  oW oH oC  src1       src2       dst        kbase      flags')
for j,(pos,n,h,r,pk) in enumerate(out):
 typ=(r.get(0xf,0)>>4)&15
 def val(reg):
  for a,f,vals in pk:
   if a==reg:return vals[0] if vals else None
  return None
 # AP values are the packet values; use raw reg if already set
 src1=val(0x1346) if val(0x1346) is not None else r.get(0x1346)
 src2=val(0x134c) if val(0x134c) is not None else r.get(0x134c)
 dst=val(0x1444) if val(0x1444) is not None else r.get(0x1444)
 kb=val(0x1544) if val(0x1544) is not None else r.get(0x1544)
 print(f'{j:3d} {typ:3d} {n*4:5x} {r.get(1,0):4d} {r.get(2,0):4d} {r.get(3,0):3d}  {r.get(5,0):4d} {r.get(6,0):4d} {r.get(7,0):3d}  {src1!s:10} {src2!s:10} {dst!s:10} {kb!s:10} {r.get(0xf,0):08x}')
