import numpy as np
stops=20*1.5**np.arange(30)
N=4
def grid(stop): return np.unique(np.r_[0,np.geomspace(max(1e-9,stop*1e-9),stop,320),np.linspace(0,stop,320)[1:]])
def exact(om):
 s=(1+1j)*np.sqrt(om); y=np.ones_like(s);y[om>0]=np.tanh(s[om>0])/s[om>0];return 1/y
def unpack(q): return np.exp(q[0]),np.exp(q[1:5]),np.exp(q[5:])
def model(q,om):
 G,R,L=unpack(q);return 1/(G+np.sum(1/(R[:,None]+1j*om[None,:]*L[:,None]),axis=0))
def obj(q,om,ze):
 z=model(q,om);e=(z-ze)/np.abs(ze);r=np.r_[e.real,e.imag,30*(z[0].real-1)];return float(np.mean(r*r))
def nm(x0,om,ze,step=.3,maxiter=10000):
 n=len(x0);S=np.vstack([x0,*[x0+np.eye(n)[i]*step for i in range(n)]]);V=np.array([obj(x,om,ze) for x in S])
 for it in range(maxiter):
  ix=np.argsort(V);S=S[ix];V=V[ix]
  if np.std(V)<1e-20 and np.max(np.ptp(S,axis=0))<1e-8:break
  c=S[:-1].mean(0);xr=2*c-S[-1];fr=obj(xr,om,ze)
  if fr<V[0]:
   xe=c+2*(xr-c);fe=obj(xe,om,ze);S[-1],V[-1]=(xe,fe) if fe<fr else (xr,fr)
  elif fr<V[-2]:S[-1],V[-1]=xr,fr
  else:
   xc=c+(.5*(xr-c) if fr<V[-1] else .5*(S[-1]-c));fc=obj(xc,om,ze)
   if fc<min(fr,V[-1]):S[-1],V[-1]=xc,fc
   else:S[1:]=S[0]+.5*(S[1:]-S[0]);V[1:]=[obj(x,om,ze) for x in S[1:]]
 return S[np.argmin(V)]
# independent Foster starts keep generation deterministic and avoid cross-band local minima
n=np.arange(1,N+1);R0=np.pi**2*(2*n-1)**2/8;L0=np.ones(N);G0=1-np.sum(1/R0);base=np.log(np.r_[G0,R0,L0])
rows=[]
for i,stop in enumerate(stops):
 om=grid(stop);ze=exact(om);q=nm(base,om,ze,step=.4,maxiter=12000)
 z=model(q,om);rel=np.abs((z-ze)/ze);G,R,L=unpack(q);rows.append((G,R,L,rel.max(),np.sqrt(np.mean(rel**2)),z[0].real))
 print(i,f'{stop:.10g}',f'max={rel.max():.6g}',f'rms={np.sqrt(np.mean(rel**2)):.6g}',flush=True)
for field in ['g4','r1_4','l1_4','r2_4','l2_4','r3_4','l3_4','r4_4','l4_4']:
 vals=[]
 for G,R,L,*_ in rows:
  vals.append({'g4':G,'r1_4':R[0],'l1_4':L[0],'r2_4':R[1],'l2_4':L[1],'r3_4':R[2],'l3_4':L[2],'r4_4':R[3],'l4_4':L[3]}[field])
 print('double '+field+'[30]={'+','.join(f'{v:.12g}' for v in vals)+'};')
