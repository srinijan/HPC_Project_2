from mpi4py import MPI 
import numpy as np



def rij(ij):
    '''
    input: x,y,z coordinates of each particle 
    output: the distance between each particle 
    '''
    xi = ij[0][0]
    yi = ij[0][1]
    zi = ij[0][2]
    for x in range(1, ij.shape[0]):
        xj = ij[x][0]
        yj = ij[x][1]
        zj = ij[x][2]
        rij = np.sqrt(np.square(xi-xj) + np.square(yi-yj) + np.square(zi-zj))
        if x == 1:
            rij_array = rij
        else:
            rij_array = np.append(rij_array, rij)
    return rij_array

def avg_rij(rij_array):
    '''
    input: the distance between each particle 
    output: the average distance between each particle 
    '''
    avg_rij = np.mean(rij_array)
    return avg_rij

def vij(rij):
    '''
    input: the distance between each particle 
    output: Lenard Jones potential between each particle 
    '''
    length = rij.size
    if length == 1:
        rij_12 = np.power(rij, 12)
        rij_6 = np.power(rij, 6)
        vij_array = np.subtract(np.divide(1,rij_12), np.divide(2,rij_6))
    else: 
        for x in range (length):
            rij_12 = np.power(rij[x], 12)
            rij_6 = np.power(rij[x], 6)
            vij = np.subtract(np.divide(1,rij_12), np.divide(2,rij_6))
            if x==0:
                vij_array = vij 
            else:
                vij_array = np.append(vij_array, vij)
    return vij_array

def vi(vij):
    '''
    input: the distance between each particle 
    output: Total energy for each particle 
    '''
    vi = np.multiply(0.5, np.sum(vij))
    return vi

def fi(vi, mean_rij, ij):
    '''
    input: Total energy for each particle, the average distance between each particle, and the x,y,z coordinates of each particle 
    output: Total force on each particle
    '''
    delta = 0.1*mean_rij
    # Partial derivative of V with respect to x
    xi = ij[0][0] + delta 
    yi = ij[0][1]
    zi = ij[0][2]
    Vx = delta_calculations(xi, yi, zi, ij) - vi
    Vx = -1*Vx
    # Partial derivative of V with respect to y
    xi = ij[0][0] 
    yi = ij[0][1] + delta 
    zi = ij[0][2]
    Vy = delta_calculations(xi, yi, zi, ij) - vi
    Vy = -1*Vy
    # Partial derivative of V with respect to z
    xi = ij[0][0] 
    yi = ij[0][1]
    zi = ij[0][2] + delta
    Vz = delta_calculations(xi, yi, zi, ij) - vi
    Vz = -1*Vz
    Fi = np.array([Vx, Vy, Vz])
    return Fi
    
def delta_calculations(xi, yi, zi, ij):
    '''
    input: x,y,z coordinates for the particles, and the x,y,z coordinates in the matrix 
    output: delta calculations needed for the gradient functions. Assists the force calculation function 
    '''
    for x in range (1, ij.shape[0]):
        xj = ij[x][0]
        yj = ij[x][1]
        zj = ij[x][2]
        rij = np.sqrt(np.square(xi-xj) + np.square(yi-yj) + np.square(zi-zj))
        if x == 1:
            rij_array = rij
        else:
            rij_array = np.append(rij_array, rij)
    length = rij.size
    if length == 1:
        rij_12 = np.power(rij, 12)
        rij_6 = np.power(rij, 6)
        vij_array = np.subtract(np.divide(1,rij_12), np.divide(2,rij_6))
        vi = np.multiply(0.5, np.sum(vij_array))
    else: 
        for x in range (length):
            rij_12 = np.power(rij_array[x], 12)
            rij_6 = np.power(rij_array[x], 6)
            vij = np.subtract(np.divide(1,rij_12), np.divide(2,rij_6))
            if x==0:
                vij_array = vij 
            else:
                vij_array = np.append(vij_array, vij)
        vi = np.multiply(0.5, np.sum(vij_array))
    return vi 
    
    

# Driver
count = 1
N = 1792
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
recv_count = (N/num_proc)/2
start = MPI.Wtime()
if rank == 0:
    ij = np.random.uniform(0, 10, size = (N,3))
    for x in range(N-1):
        if x!=0:
            ij = np.delete(ij, 0, 0)
        if(count<=num_proc-1):
            comm.send(ij, dest = count, tag = 0)
            count+=1
        else:
            count = 1
            Rij = rij(ij)
            Avg_rij = avg_rij(Rij)
            Vij = vij(Rij)
            Vi = vi(Vij)
            Fi = fi(Vi, Avg_rij, ij)
            i = x+1
            text = "Force vector from processor 0\n {}"
            print(text.format(Fi))
                
else:
    for x in range(N/num_proc):
        ij = comm.recv(source = 0, tag = 0)
        Rij = rij(ij)
        Avg_rij = avg_rij(Rij)
        Vij = vij(Rij)
        Vi = vi(Vij)
        Fi = fi(Vi, Avg_rij, ij)
        text = "Force vector from processor {}\n {}"
        print(text.format(rank, Fi))

end = MPI.Wtime()
time = end - start 

text = "Total Runtime for {} particles using {} processors: {}"
print(text.format(N, num_proc, time))








