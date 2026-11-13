import struct,sys
p=sys.argv[1] if len(sys.argv)>1 else 'work/ota_extract/llv_model_2304/24C5037i__iPhone19,2_3_7/System/Library/PrivateFrameworks/ISPKit.framework/V63/LowLightVideo/LLV-HD-Wide-2x-v0.3.0_2304x1296.bundle/H19.bundle/main/main_ane/model.hwx'
b=open(p,'rb').read();d=b[0x14000:0x14000+0x2259c];w=list(struct.unpack('<%dI'%(len(d)//4),d));pos=4;tasks=[]
while pos<len(w):
 while pos<len(w) and w[pos]==0:pos+=1
 if pos>=len(w):break
 h=w[pos];n=(h>>16)&0x7ff
 if not n or pos+n>len(w):break
 td=w[pos:pos+n];regs={};ap=[];i=9
 while i<len(td):
  x=td[i]
  if x==0 or (x==1 and td[i+1:i+4]==[0,0,0]) or (not(x>>31) and ((x>>26)&31) and not((x>>29)&1)):break
  a=x&0x7fff;f=(x>>15)&0xffff
  if x>>31:addrs=[a]+[a+1+k for k in range(16) if f>>k&1]
  else:addrs=[a+k for k in range((f&0x3f)+1)]
  vals=td[i+1:i+1+len(addrs)]
  if len(vals)<len(addrs):break
  for aa,vv in zip(addrs,vals):regs[aa]=vv
  if (x>>29)&1:ap.append((a,x,vals))
  i+=1+len(addrs)
 tasks.append((pos,n,h,regs,ap));pos+=n
def first(reg):
 for a,x,vs in tasks[cur][4]:
  if a==reg:return vs[0] if vs else None
 return None
print('tasks',len(tasks))
readers={};writers={}
for cur,(pos,n,h,r,ap) in enumerate(tasks):
 src1=first(0x1346);src2=first(0x134c);dst=first(0x1444);kb=first(0x1544) if first(0x1544) is not None else first(0x158e)
 typ=(r.get(0xf,0)>>4)&15
 if src1 is not None:readers.setdefault(src1,[]).append((cur,'s1'))
 if src2 is not None:readers.setdefault(src2,[]).append((cur,'s2'))
 if dst is not None:writers.setdefault(dst,[]).append(cur)
 print(f'{cur:3d} t{typ} src1={src1!s:>9} src2={src2!s:>9} dst={dst!s:>9} kb={kb!s:>9}')
print('\nfirst reads with no earlier write')
for off,rs in sorted(readers.items()):
 first_i=rs[0][0]
 prior=[x for x in writers.get(off,[]) if x<first_i]
 if not prior:print(hex(off),rs[:8],'writers',writers.get(off,[])[:8])
