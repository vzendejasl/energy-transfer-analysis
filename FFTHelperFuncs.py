"""
BSD 2-Clause License
Author: Lisandro Dalcin and Mikael Mortensen
Contact:    dalcinl@gmail.com or mikaem@math.uio.no

Copyright (c) 2017, Lisandro Dalcin and Mikael Mortensen. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import sys
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
comm  = MPI.COMM_WORLD

FFT = None
local_wavenumbermesh = None
local_shape = None

def setup_fft(res, dtype=np.complex128):
    """ Setup shared FFT object and properties
        res - linear resolution
    """

    global FFT
    global local_wavenumbermesh
    global local_shape

    if comm.Get_rank() == 0:
        print("""!!! WARNING - CURRENT PITFALLS !!!
        - data units are ignored
        - data is assumed to live on a 3d uniform grid with L = 1
        - for the FFT L = 2 pi is implicitly assumed to work with integer wavenumbers
        """)

    time_start = MPI.Wtime()

    N = np.array([res, res, res], dtype=int)
    # using L = 2pi as we work (e.g. when binning) with integer wavenumbers
    L = np.array([2*np.pi, 2*np.pi, 2*np.pi], dtype=float)
    
    # --- CRITICAL FIX: FORCE 1D SLAB DECOMPOSITION ---
    # We pass grid=(size, 1, 1) to ensure only the first axis is split.
    # This prevents the dimension mismatch in MPIderivHelperFuncs.py
    size = comm.Get_size()
    FFT = PFFT(comm, N, axes=(0,1,2), collapse=False, dtype=dtype, grid=(size, 1, 1))

    local_wavenumbermesh = get_local_wavenumbermesh(FFT, L)
    local_shape = newDistArray(FFT,False).shape

    time_elapsed = MPI.Wtime() - time_start
    time_elapsed = comm.gather(time_elapsed)

    if comm.Get_rank() == 0:
        print("Setup up FFT and wavenumbers done in %.3g +/- %.3g" %
            (np.mean(time_elapsed), np.std(time_elapsed)))
        sys.stdout.flush()


# from
# https://bitbucket.org/mpi4py/mpi4py-fft/raw/67dfed980115108c76abb7e865860b5da98674f9/examples/spectral_dns_solver.py
# with modification for complex numbers
def get_local_wavenumbermesh(FFT, L):
    """Returns local wavenumber mesh."""
    # CRITICAL FIX 1: Get the slice for the TRANSFORMED (Spectral) array.
    s = FFT.local_slice(True) 
    
    N = FFT.global_shape()
    # Set wavenumbers in grid
    if FFT.dtype() == np.complex128:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N]
    else:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))
    
    # Scale by domain size early
    Lp = 2*np.pi/L
    k = [(ki * Lp[i]).astype(float) for i, ki in enumerate(k)]

    # CRITICAL FIX 2: Handle Spectral Transpose in Parallel
    # When using Slab decomposition (grid=[size,1,1]) on axis 0, mpi4py-fft 
    # transposes the output spectral array to shape (N1, N0, N2).
    # We must permute our 'k' vectors to match this transposed layout (1, 0, 2)
    # so that the correct wavenumber matches the correct axis.
    if comm.Get_size() > 1:
        # Permute k: Global 1 (X), Global 0 (Y), Global 2 (Z)
        k_ordered = [k[1], k[0], k[2]]
    else:
        k_ordered = k

    # Create meshgrids using the aligned k vectors
    K = [ki[si] for ki, si in zip(k_ordered, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    
    # Broadcast to full local shape
    Ks_broad = [np.broadcast_to(k_i, FFT.shape(True)) for k_i in Ks]
    
    # CRITICAL FIX 3: Return in Global Order
    # Ks_broad is currently ordered by [Output_Ax0, Output_Ax1, Output_Ax2].
    # In parallel, this is [Global_X, Global_Y, Global_Z].
    # FlowAnalysis expects [Global_Y, Global_X, Global_Z] (k0, k1, k2).
    if comm.Get_size() > 1:
        return [Ks_broad[1], Ks_broad[0], Ks_broad[2]]
    else:
        return Ks_broad