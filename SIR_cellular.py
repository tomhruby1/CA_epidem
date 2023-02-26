import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path


def init_state(N_pop, V, virulence, gamma, connection, movement, infected_coords, infected_count):
    dim = N_pop.shape
    mu = np.ones((dim[0], dim[1], len(V))) * connection * movement * virulence
    
    S = np.ones((dim[0], dim[1], 3)) #state consisting of SIR values
    S[:,:,0] *= 1
    S[:,:,1] *= 0
    S[:,:,2] *= 0
    S[infected_coords[0], infected_coords[1], 1] = infected_count / N_pop[infected_coords] 
    S[infected_coords[0], infected_coords[1], 0] = 1 - infected_count / N_pop[infected_coords] 

    print(f"Init spread with {infected_count} infected individuals ({S[infected_coords[0], infected_coords[1], 1] * 100}% local pop.) at cell {infected_coords}.")

    return S, mu

def plot_cell_state(state, colormap=None, title=None, save_path=None, show=True):
    fig, ax = plt.subplots()

    p = ax.pcolormesh(state, cmap=colormap, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(p, ax=ax)
    if title is not None: plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    if show: plt.show()

def time_step(S, beta, gamma, mu, N, V):
    dim = (S.shape[0], S.shape[1])
    #TODO: remove deepcopy
    S_past = copy.deepcopy(S[:,:,0])
    I_past = copy.deepcopy(S[:,:,1])
    for i in range(dim[0]):
        for j in range(dim[1]):
            if N[i,j] < 1.0: #skip if population zero 
                continue
            sum = 0
            # calculate the connection sum
            for v in V:
                v = np.asarray(v)
                ij = np.asarray([i,j])
                # considering only cells in the grid TODO: maybe add padding around?
                if (ij + v >= 0).all() and ((ij + v)[0] < dim[0]) and ((ij + v)[1] < dim[1]): 
                    sum += N[tuple(ij + v)]/N[tuple(ij)] * mu[i,j,V.index(tuple(v))] * I_past[tuple(ij+v)] 
            #   S
            S[i,j][0] += -beta * S_past[i,j] * I_past[i,j] - S_past[i,j] * sum
            #   I
            S[i,j][1] = (1-gamma) * I_past[i,j] + beta * S_past[i,j] * I_past[i,j] + S_past[i,j] * sum
            #   R
            S[i,j][2] += gamma * I_past[i,j]
    return S


if __name__=='__main__':
    #sim setup
    T = 20
    out_dir_path = 'sim_std/' 
 
    # sim params
    dim = (32, 32)
    beta = 0.01
    gamma = 0.1 
    V = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]    

    out_p = Path.cwd()/out_dir_path
    out_p.mkdir(exist_ok=True)
    
    N = np.ones(dim)  # --> calc automat. from initial SIR vals for each cell
    mu_dim = (dim[0], dim[1], len(V))    #mu param for each neighbor for each cell
    mu = np.ones(mu_dim) * 0.2

    S = np.ones((dim[0], dim[1], 3)) #state consisting of SIR values
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
        plot_cell_state(S[:,:,1],  colormap='viridis', title=f"Infected t={t}", show=False)