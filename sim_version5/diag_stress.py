import sys, io, numpy as np
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
from dataclasses import replace
from config import OP_B
from channel import ChannelSimulator
from run_col import run_col_aif
MC,T=3,16; HALF=slice(T//2,None)
for rho in (0.9,0.7):
    op=replace(OP_B,rho=rho,delta_max=7)
    accB={'r':[],'m':[]}; accA={'r':[],'m':[]}
    for s in range(MC):
        sim=ChannelSimulator(Nx=op.Nx,Ny=op.Ny,Wx=op.Wx,Wy=op.Wy,K=op.K,rho=rho,beta=op.beta,seed=400+s)
        H=sim.generate(T)
        for tag,R,acc in [('B',op.R(),accB),('A',op.R_block(),accA)]:
            r=run_col_aif(op,H,np.random.default_rng(500+s),R=R)
            acc['r'].append(r['rate'][HALF].mean()); acc['m'].append(r['move'][HALF].mean())
    rB,mB=np.mean(accB['r']),np.mean(accB['m']); rA,mA=np.mean(accA['r']),np.mean(accA['m'])
    oB,oA=rB-op.eta_mv*mB, rA-op.eta_mv*mA
    print(f"rho={rho}: B rate {rB:.2f} move {mB:.2f} obj {oB:.2f} | A rate {rA:.2f} move {mA:.2f} obj {oA:.2f} | B-A obj {oB-oA:+.3f}")
