import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_cell_state(state, colormap=None, title=None, save_path=None, show=True):
    fig, ax = plt.subplots()

    p = ax.pcolormesh(state.cpu(), cmap=colormap, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(p, ax=ax)
    if title is not None: plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    if show: plt.show()
    
# iterace
def time_step(S, beta, gamma, mu, N, V):
    # mu used as conv filter  dim:= (out, in, h, w)
    dim = (S.shape[0], S.shape[1]) # (32,32)
    
    S_past = S[:,:,0]
    I_past = S[:,:,1]
    R_past = S[:,:,2]
    S_new = torch.zeros(S.shape).to(device)
    
    # SUM -> conv 
    # NI = torch.stack((N, S[:,:,1])) #stack N + I tensors
    NI = N * S[:,:,1]
    NI = NI[None, :]
    sum = F.conv2d(NI, mu, stride=1, padding='same') 
    #   S
    S_new[:,:,0] = S_past - beta * S_past * I_past - S_past * sum
    #   I
    S_new[:,:,1] = (1-gamma) * I_past + beta * S_past * I_past + S_past * sum
    #   R
    S_new[:,:,2] = R_past + gamma * I_past
    
    return S_new

def cellular_sim(mu, fmask, T=4, beta=0.01, gamma=0.001, V = [(-1,0),(0,1),(1,0),(0,-1)]):
    dim = (32,32)
    N = torch.ones(dim).to(device)# --> calc automat. from initial SIR vals for each cell
    mu_dim = (dim[0], dim[1], len(V))    #mu param for each neighbor for each cell

    S = torch.ones((dim[0], dim[1], 3)).to(device) #state consisting of SIR values
    S[:,:,1] = fmask
    S[:,:,0] -= S[:,:,1]
    S[:,:,2] *= 0

    for t in range(T):
        S = time_step(S, beta, gamma, mu, N, V)
    plot_cell_state(S[:,:,1],  colormap='viridis', title=f"t={t}")
    return S


def start_sim():
    #sim setup
    T = 20
    out_dir_path = 'sim_torch/' 
 
    # sim params
    dim = (32, 32)
    beta = 0.01
    gamma = 0.1 
    V = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]    

    out_p = Path.cwd()/out_dir_path
    out_p.mkdir(exist_ok=True)
    
    N = torch.ones(dim) 
    N = N.to(device)
   
    #mu param for each neighbor for each cell
    mu = torch.ones((1,1,3,3))* 0.2
    # mu = torch.randn((1,1,3,3))
    mu[0,0,1,1] = 0
    mu = mu.to(device)

    S = torch.ones((dim[0], dim[1], 3)).to(device) #state consisting of SIR values
    S[:,:,0] *= N
    S[:,:,1] *= 0
    S[:,:,2] *= 0

    # init infection
    inf_idx = (5,5)
    infected_count = 1

    S[inf_idx[0],inf_idx[1],1] = infected_count
    S[inf_idx[0],inf_idx[1],0] = N[inf_idx] - infected_count 

    # RUN SIM
    for t in range(T):
        S = time_step(S, beta, gamma, mu, N, V)
        plot_cell_state(S[:,:,1],  colormap='viridis', title=f"Infected t={t}", show=True)
        
if __name__=='__main__':
    start_sim()