import numpy as np
import FFTHelperFuncs
from mpi4py_fft import newDistArray

def MPIderiv2(comm,var,dim):
    """Returns first derivative (2-point central finite difference)
    Robust version: assumes a 1D slab decomposition along axis 0 and uses
    explicit neighbor ranks (rank±1 mod size) to avoid mis-inferred
    topologies on uneven splits. Intended for the transposed layout where
    axis0 is the slabbed (Y) direction.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
         
    sl_m1 = slice(None,-2,None)
    sl_p1 = slice(2,None,None)
    
    # Correctly calculate ds for the specific dimension using global shape
    # This fixes the bug where L_N was calculated incorrectly for odd-sized grids in parallel
    N_global = FFTHelperFuncs.FFT.global_shape()
    if dim < 0 or dim > 2:
        print("watch out for dimension!")
        return np.zeros_like(var)
        
    # ds represents 2*dx (central difference denominator)
    # assuming physical box length L=1.0
    ds = 2.0 / float(N_global[dim])

    if dim == 0:
        next_proc = (rank + 1) % size
        prev_proc = (rank - 1 + size) % size
        leftSlice = comm.sendrecv(sendobj=var[-1:,:,:],dest=next_proc,source=prev_proc)
        rightSlice = comm.sendrecv(sendobj=var[:1,:,:],dest=prev_proc,source=next_proc)
        tmp = np.concatenate((leftSlice,var,rightSlice),axis=0)
        p1 = tmp[sl_p1,:,:]
        m1 = tmp[sl_m1,:,:]
    elif dim == 1:
        tmp = np.concatenate((var[:,-1:,:],var,var[:,:1,:]),axis=1)
        p1 = tmp[:,sl_p1,:]
        m1 = tmp[:,sl_m1,:]
    elif dim == 2:
        tmp = np.concatenate((var[:,:,-1:],var,var[:,:,:1]),axis=2)
        p1 = tmp[:,:,sl_p1]
        m1 = tmp[:,:,sl_m1]
            
    return np.array((p1 - m1)/ds)

def MPIXdotGradYScalar(comm,X,Y):
    """ returns  (X . grad) Y with axis0=Y, axis1=X (transposed layout) 
    X[0]=Ux, X[1]=Uy, X[2]=Uz
    """
    
    # Ux * dY/dx (dim 1) + Uy * dY/dy (dim 0) + Uz * dY/dz (dim 2)
    return X[0] * MPIderiv2(comm,Y,1) + X[1] * MPIderiv2(comm,Y,0) + X[2] * MPIderiv2(comm,Y,2)

def MPIXdotGradY(comm,X,Y):
    """ returns  (X . grad) Y
    """
    
    res = np.zeros_like(X)
    for i in range(3):
        res[i] = MPIXdotGradYScalar(comm, X, Y[i])
        
    return res

def MPIdivX(comm,X):
    """
    Spectral divergence: div X = i k · F(X)
    FIXED for Transposed Data: Axis0=Y, Axis1=X
    """
    FFT = FFTHelperFuncs.FFT
    # Unpack based on Transposed Layout: (k_y, k_x, k_z)
    ky, kx, kz = FFTHelperFuncs.local_wavenumbermesh
    
    fx = FFT.forward(X[0], newDistArray(FFT, rank=0)) # F(Ux)
    fy = FFT.forward(X[1], newDistArray(FFT, rank=0)) # F(Uy)
    fz = FFT.forward(X[2], newDistArray(FFT, rank=0)) # F(Uz)
    
    # div = i(kx*Fx + ky*Fy + kz*Fz)
    div_k = 1j * (kx * fx + ky * fy + kz * fz)
    
    div = FFT.backward(div_k, newDistArray(FFT, False, rank=0)).real
    return div

def MPIdivXY(comm,X,Y):
    """ returns  pd_j X_j Y_i
    """
    res = np.zeros_like(Y)
    
    for i in range(3):
        # d/dx(Ux*Yi) + d/dy(Uy*Yi) + d/dz(Uz*Yi)
        res[i] = MPIderiv2(comm,X[0]*Y[i],1) + MPIderiv2(comm,X[1]*Y[i],0) + MPIderiv2(comm,X[2]*Y[i],2)
    
    return res

def MPIgradX(comm,X):
    """
    Spectral gradient: grad X = i k * F(X)
    FIXED for Transposed Data: Axis0=Y, Axis1=X
    Returns [d/dx, d/dy, d/dz]
    """
    FFT = FFTHelperFuncs.FFT
    # Unpack based on Transposed Layout: (k_y, k_x, k_z)
    ky, kx, kz = FFTHelperFuncs.local_wavenumbermesh
    
    fx = FFT.forward(X, newDistArray(FFT, rank=0))
    
    # Gradient vector: [d/dx, d/dy, d/dz]
    gx = FFT.backward(1j * kx * fx, newDistArray(FFT, False, rank=0)).real
    gy = FFT.backward(1j * ky * fx, newDistArray(FFT, False, rank=0)).real
    gz = FFT.backward(1j * kz * fx, newDistArray(FFT, False, rank=0)).real
    return np.array([gx, gy, gz])

def MPIrotX(comm,X):
    """
    Spectral curl: curl X = i k x F(X)
    FIXED for Transposed Data: Axis0=Y, Axis1=X
    """
    FFT = FFTHelperFuncs.FFT
    # Unpack based on Transposed Layout: (k_y, k_x, k_z)
    ky, kx, kz = FFTHelperFuncs.local_wavenumbermesh
    
    fx = FFT.forward(X[0], newDistArray(FFT, rank=0)) # Ux
    fy = FFT.forward(X[1], newDistArray(FFT, rank=0)) # Uy
    fz = FFT.forward(X[2], newDistArray(FFT, rank=0)) # Uz
    
    # Curl X: dy*Uz - dz*Uy
    cx = FFT.backward(1j * (ky * fz - kz * fy), newDistArray(FFT, False, rank=0)).real
    
    # Curl Y: dz*Ux - dx*Uz
    cy = FFT.backward(1j * (kz * fx - kx * fz), newDistArray(FFT, False, rank=0)).real
    
    # Curl Z: dx*Uy - dy*Ux
    cz = FFT.backward(1j * (kx * fy - ky * fx), newDistArray(FFT, False, rank=0)).real
    
    return np.array([cx, cy, cz])