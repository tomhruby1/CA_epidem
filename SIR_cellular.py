import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cell_state(state, colormap=None, title=None, save_path=None, show=True):
    fig, ax = plt.subplots()
    p = ax.pcolormesh(S[:,:,1], cmap=colormap, rasterized=True, vmin=0, vmax=5)
    fig.colorbar(p, ax=ax)
    if title is not None: plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    if show: plt.show()

def time_step(S, beta, gamma, mu, N, V):
    for i in range(dim[0]):
        for j in range(dim[1]):
            S_past = S[i,j][0]
            I_past = S[i,j][1]
            sum = 0
            # calculate the connection sum
            for v in V:
                v = np.asarray(v)
                ij = np.asarray([i,j])
                # considering only cells in the grid TODO: maybe add padding around?
                if (ij + v >= 0).all() and ((ij + v)[0] < dim[0]) and ((ij + v)[1] < dim[1]): 
                    sum += N[tuple(ij + v)] * mu[i,j,V.index(tuple(v))] * S[tuple(ij+v)][1] 
            #   S
            S[i,j][0] += -beta * S_past * I_past - S_past * sum
            #   I
            S[i,j][1] = (1-gamma)*I_past + beta * S_past * I_past + S_past * sum
            #   R
            S[i,j][2] += gamma * I_past
    return S

if __name__=='__main__':
    #sim setup
    T = 20
    out_dir_path = 'sim1/' 
 
    # sim params
    dim = (15, 15)
    beta = 0.01
    gamma = 0.1 
    V = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]    

    out_p = Path.cwd()/out_dir_path
    out_p.mkdir(exist_ok=True)
    
    N = np.ones(dim) * 5 # --> calc automat. from initial SIR vals for each cell
    mu_dim = (dim[0], dim[1], 8)    #mu param for each neighbor for each cell
    mu = np.ones(mu_dim) * 0.005

    S = np.ones((dim[0], dim[1], 3)) #state consisting of SIR values
    S[:,:,0] *= N
    S[:,:,1] *= 0
    S[:,:,2] *= 0

    # init infection
    inf_idx = (2,2)
    infected_count = 1

    S[inf_idx[0],inf_idx[1],1] = infected_count
    S[inf_idx[0],inf_idx[1],0] = N[inf_idx] - infected_count 

    # RUN SIM
    for t in range(T):
        S = time_step(S, beta, gamma, mu, N, V)
        
        save_name = 'step_'+(3-len(str(t)))*'0'+str(t)+'.png'
        plot_cell_state(S[:,:,1],  colormap='viridis', title=f"Infected t={t}", 
                        save_path=str(out_p/save_name), show=False)