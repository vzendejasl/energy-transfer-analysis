import numpy as np
import FFTHelperFuncs
from mpi4py_fft import newDistArray
import time
import pickle
import sys
import h5py
import os
from scipy.stats import binned_statistic
from math import ceil
from MPIderivHelperFuncs import MPIderiv2, MPIXdotGradY, MPIdivX, MPIdivXY, MPIgradX, MPIrotX

class FlowAnalysis:
    
    def __init__(self, MPI, args, fields):
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.res = args['res']

        self.comm.Barrier()
        time_start = MPI.Wtime()

        if args['extrema_file'] is not None:
            try:
                self.global_min_max = pickle.load(open(args['extrema_file'], 'rb'))
                if self.rank == 0:
                    print("Successfully loaded global extrema dict: %s" %
                          self.global_min_maxFile)
            except FileNotFoundError:
                raise SystemExit('Extrema file not found: ', args['extrema_file'])
        else:
            self.global_min_max = {}

        self.rho = fields['rho']
        self.U = fields['U']
        self.B = fields['B']
        self.Acc = fields['Acc']
        self.P = fields['P']
        self.eos = args['eos']
        self.gamma = args['gamma']
        self.has_b_fields = args['b']
        self.kernels= args['kernels']

        if self.rank == 0:
            self.outfile_path = args['outfile']

        # maximum integer wavenumber
        k_max_int = ceil(self.res*0.5*np.sqrt(3.))
        self.k_bins = np.linspace(0.5,k_max_int + 0.5,k_max_int+1)
        # if max k does not fall in last bin, remove last bin
        if self.res*0.5*np.sqrt(3.) < self.k_bins[-2]:
            self.k_bins = self.k_bins[:-1]

        self.FFT = FFTHelperFuncs.FFT
        self.localK = FFTHelperFuncs.local_wavenumbermesh
        self.localKmag = np.linalg.norm(self.localK,axis=0)

        self.localKunit = np.copy(self.localK)
        # fixing division by 0 for harmonic part
        if self.rank == 0:
            if self.localKmag[0,0,0] == 0.:
                self.localKmag[0,0,0] = 1.
            else:
                raise SystemExit("[0] kmag not zero where it's supposed to be")
        else:
            if (self.localKmag == 0.).any():
                raise SystemExit("[%d] kmag not zero where it's supposed to be" % self.rank)

        self.localKunit /= self.localKmag
        if self.rank == 0:
            self.localKmag[0,0,0] = 0.

    def calculate_filtering_spectrum(self, kernel):
        """
        following Sadek and Aluie (2018) for calculating kinetic energy density filtering spectrum spectrum of kinetic energy
        kernel - one of the following kernel types: Box, Gauss, Sharp
        """
        # set up array of filters widths
        delta_x = 1/self.res
        m = np.arange(self.res/2 - 1, 0, -1) # array of integer number of grid cells (see Sadek and Aluie eqn. 31)
        k_lm = lm = np.zeros(int(self.res/2)-1)
        lm = np.array(2*m*delta_x) # array of length scales (see Sadek and Aluie eqn. 31)
        k_lm = 1/lm # array of wavenumbers (see Sadek and Aluie eqn. 32)

        momentum = self.rho * self.U
        epsilon = np.zeros(len(k_lm))
        N = self.res * self.res * self.res
        
        for i, k in enumerate(k_lm):
            # set up for calculating cumulative spectrum (epsilon) for each filter length scale
            energy = newDistArray(self.FFT,False,rank=1)
            FT_momentum = newDistArray(self.FFT,rank=1)
            momentum_filtered = newDistArray(self.FFT,False,rank=1)
            FT_rho = newDistArray(self.FFT,rank=0)
            rho_filtered = newDistArray(self.FFT,False,rank=0)

            # calculate the cumulative spectrum (epsilon) for each filter length scale
            # (see equations 27 and 6 of Sadek and Aluie)
            FT_rho = self.FFT.forward(self.rho, FT_rho)
            for j in range(3):
                FT_momentum[j] = self.FFT.forward(momentum[j], FT_momentum[j])

            filter_width = self.res/(2*k)
            FT_G = self.Kernel(filter_width, kernel)  # calculate convolution kernel
            # calculate filtered values of momentum and density
            rho_filtered = (self.FFT.backward(FT_G * FT_rho, rho_filtered)).real
            for j in range(3):
                momentum_filtered[j] = (self.FFT.backward(FT_G * FT_momentum[j], momentum_filtered[j])).real

            # compute the average of momentum^2 / density
            filtered_kin_energy = 0.5 * np.sum(np.abs(momentum_filtered**2) / np.abs(rho_filtered), axis=0)
            total_kin_energy = (self.comm.allreduce(np.sum(filtered_kin_energy)))
            # average energy
            epsilon[i] = total_kin_energy / N

        # calculate the energy spectrum for each filter length scale (see equation 33 of Sadek and Aluie)
        KinEnergy = np.zeros(len(k_lm)-1)
        for i in range(len(k_lm) - 1):
            KinEnergy[i] = (epsilon[i+1] - epsilon[i])/(k_lm[i+1] - k_lm[i])

        if self.rank == 0:
            self.outfile.require_dataset('Filtered_' + kernel + '_KinEnergy/PowSpec/Bins', (1,len(k_lm)-1), dtype='f')[0] = k_lm[1:]
            self.outfile.require_dataset('Filtered_' + kernel + '_KinEnergy/PowSpec/Full', (1,len(k_lm)-1), dtype='f')[0] = KinEnergy
            self.outfile.require_dataset('Filtered_' + kernel + '_KinEnergy/PowSpec/low_pass_energies', (1,len(k_lm)), dtype='f')[0] = epsilon

    def run_analysis(self):

        rho = self.rho
        U = self.U
        B = self.B
        Acc = self.Acc
        P = self.P

        # Debug: confirm that all ranks reached analysis with consistent shapes
        print(f"[rank {self.rank}] Entering run_analysis: U shape={U.shape}, rho shape={rho.shape}, FFT local_shape={FFTHelperFuncs.local_shape}")
        sys.stdout.flush()
        
        if self.rank == 0:
            self.outfile = h5py.File(self.outfile_path, "w")

        if self.rank == 0:
            print("[rank 0] Computing vector_power_spectrum for: u, rhoU, rhoThirdU")
            sys.stdout.flush()
        self.vector_power_spectrum('u',U)
        self.vector_power_spectrum('rhoU',np.sqrt(rho)*U)
        self.vector_power_spectrum('rhoThirdU',rho**(1./3.)*U)

        if self.rank == 0:
            print("[rank 0] Starting Helmholtz decomposition (u_s / u_c)")
            sys.stdout.flush()
        vec_harm, vec_sol, vec_dil = self.decompose_vector(U)
        if self.rank == 0:
            print("[rank 0] Computing vector_power_spectrum for: u_s, u_c, rhou_s, rhou_c")
            sys.stdout.flush()
        self.vector_power_spectrum('u_s',vec_sol)
        self.vector_power_spectrum('u_c',vec_dil)
        self.vector_power_spectrum('rhou_s',np.sqrt(rho)*vec_sol)
        self.vector_power_spectrum('rhou_c',np.sqrt(rho)*vec_dil)
        del vec_harm, vec_sol, vec_dil

        if self.rank == 0:
            print("[rank 0] Finished Helmholtz spectra, moving to scalar stats/spectra")
            sys.stdout.flush()
        self.get_and_write_statistics_to_file(rho,"rho")
        self.get_and_write_statistics_to_file(np.log(rho),"lnrho")
        self.scalar_power_spectrum('rho',rho)
        self.scalar_power_spectrum('lnrho',np.log(rho))

        V2 = np.sum(U**2.,axis=0)
        self.get_and_write_statistics_to_file(np.sqrt(V2),"u")

        self.get_and_write_statistics_to_file(0.5 * rho * V2,"KinEnDensity")
        self.get_and_write_statistics_to_file(0.5 * V2,"KinEnSpecific")

        if Acc is not None:
            Amag = np.sqrt(np.sum(Acc**2.,axis=0))
            self.get_and_write_statistics_to_file(Amag,"a")
            self.vector_power_spectrum('a',Acc)
            self.scalar_power_spectrum('a_x',Acc[0,:,:,:])
            self.scalar_power_spectrum('a_y',Acc[1,:,:,:])
            self.scalar_power_spectrum('a_z',Acc[2,:,:,:])

            vec_harm, vec_sol, vec_dil = self.decompose_vector(Acc)
            self.vector_power_spectrum('a_s',vec_sol)
            self.vector_power_spectrum('a_c',vec_dil)
            self.get_and_write_statistics_to_file(np.sqrt(np.sum(vec_sol**2.,axis=0)),"a_s_mag")
            self.get_and_write_statistics_to_file(np.sqrt(np.sum(vec_dil**2.,axis=0)),"a_c_mag")
            del vec_harm, vec_sol, vec_dil

            corrUA = self.get_corr_coeff(np.sqrt(V2),Amag)
            if self.rank == 0:
                self.outfile.require_dataset('U-A/corr', (1,), dtype='f')[0] = corrUA

            UHarm, USol, UDil = self.decompose_vector(U)
            self.get_and_write_statistics_to_file(
                np.sum(Acc*U,axis=0)/(
                    np.linalg.norm(Acc,axis=0)*np.linalg.norm(U,axis=0)),"Angle_u_a")
            self.get_and_write_statistics_to_file(
                np.sum(Acc*USol,axis=0)/(
                    np.linalg.norm(Acc,axis=0)*np.linalg.norm(USol,axis=0)),"Angle_uSol_a")
            self.get_and_write_statistics_to_file(
                np.sum(Acc*UDil,axis=0)/(
                    np.linalg.norm(Acc,axis=0)*np.linalg.norm(UDil,axis=0)),"Angle_uDil_a")

        DivU = MPIdivX(self.comm,U)
        self.get_and_write_statistics_to_file(np.abs(DivU),"AbsDivU")
        self.get_and_write_statistics_to_file(np.sqrt(np.sum(MPIrotX(self.comm,U)**2.,axis=0)),"AbsRotU")

        if self.eos == 'adiabatic':
            self.gamma = self.gamma
            if self.rank == 0:
                print("Using gamma = %.3f for adiabatic EOS" % self.gamma)
            
            c_s2 = self.gamma * P / rho
            Ms2 = V2 / (c_s2)
            self.get_and_write_statistics_to_file(np.sqrt(Ms2),"Ms")
            self.get_2d_hist('rho-P',rho,P)
            self.get_and_write_statistics_to_file(P,"P")
            self.get_and_write_statistics_to_file(np.log10(P),"log10P")
            T = P / (self.gamma - 1.0) /rho
            self.get_and_write_statistics_to_file(T,"T")
            self.get_and_write_statistics_to_file(np.log10(T),"log10T")
            self.get_2d_hist('rho-T',rho,T)

            K = T/rho**(2./3.)
            self.get_2d_hist('T-K',T,K)
            self.get_2d_hist('rho-K',rho,K)

            self.co_spectrum('PD',P,DivU)

            self.scalar_power_spectrum('eint',np.sqrt(rho*c_s2))

        if self.kernels is not None:
            if self.rank == 0:
                print("Using the following kernels for the filtering spectrum: ", self.kernels)
            for kernel in self.kernels:
                if self.rank == 0:
                    print("on kernel: ", kernel)
                self.calculate_filtering_spectrum(kernel)
        
        if not self.has_b_fields:
            if self.rank == 0:
                self.outfile.close()
            return

        B2 = np.sum(B**2.,axis=0)
        self.get_and_write_statistics_to_file(np.sqrt(B2),"B")
        self.get_and_write_statistics_to_file(0.5 * B2,"MagEnDensity")

        if self.eos == 'adiabatic':
            TotPres = P + B2/2.
            corrPBcomp = self.get_corr_coeff(P,np.sqrt(B2)/rho**(2./3.))
            self.get_2d_hist('P-B',P,np.sqrt(B2))
            self.get_2d_hist('P-MagEnDensity',P,0.5 * B2)
            
            plasmaBeta = 2.* P / B2
        elif self.eos == 'isothermal':
            if self.rank == 0:
                print('Warning: assuming c_s = 1 for isothermal EOS')
            TotPres = rho + B2/2.
            corrPBcomp = self.get_corr_coeff(rho,np.sqrt(B2)/rho**(2./3.))
            
            plasmaBeta = 2.*rho/B2
        else:
            raise SystemExit('Unknown EOS', self.eos)

        self.get_and_write_statistics_to_file(TotPres,"TotPres")
        self.get_and_write_statistics_to_file(plasmaBeta,"plasmabeta")
        self.get_and_write_statistics_to_file(np.log10(plasmaBeta),"log10plasmabeta")

        if self.rank == 0:
            self.outfile.require_dataset('P-Bcomp/corr', (1,), dtype='f')[0] = corrPBcomp

        AlfMach2 = V2*rho/B2
        AlfMach = np.sqrt(AlfMach2)

        self.get_and_write_statistics_to_file(AlfMach,"AlfvenicMach")

        self.vector_power_spectrum('B',B)

        # this is cheap... and only works for pencil decomp in z axis
        # np.sum is required for slabs with width > 1
        if rho.shape[-1] != self.res:
            raise SystemExit('Calculation of dispersion measures only works for pencils')

        # using subcomms here so that the pencil based slices are not getting mixed
        DM = FFTHelperFuncs.FFT.subcomm[0].allreduce(np.sum(rho,axis=0))/float(self.res)
        RM = FFTHelperFuncs.FFT.subcomm[0].allreduce(np.sum(B[0]*rho,axis=0))/float(self.res)
        # using chunks so that each process only contributes it's own chunk of data
        # as all processes have the full information after the allreduce
        chunkSize = DM.shape[0] // FFTHelperFuncs.FFT.subcomm[0].Get_size()
        startIdx = FFTHelperFuncs.FFT.subcomm[0].Get_rank() * chunkSize
        endIdx = (FFTHelperFuncs.FFT.subcomm[0].Get_rank() + 1) * chunkSize
        self.get_and_write_statistics_to_file(DM[startIdx:endIdx,:],"DM_x")
        self.get_and_write_statistics_to_file(np.log(DM[startIdx:endIdx,:]),"lnDM_x")
        self.get_and_write_statistics_to_file(RM[startIdx:endIdx,:],"RM_x")
        self.get_and_write_statistics_to_file(RM[startIdx:endIdx,:]/DM[startIdx:endIdx,:],"LOSB_x")

        # using subcomms here so that the pencil based slices are not getting mixed
        DM = FFTHelperFuncs.FFT.subcomm[1].allreduce(np.sum(rho,axis=1))/float(self.res)
        RM = FFTHelperFuncs.FFT.subcomm[1].allreduce(np.sum(B[1]*rho,axis=1))/float(self.res)
        # using chunks so that each process only contributes it's own chunk of data
        # as all processes have the full information after the allreduce
        chunkSize = DM.shape[1] // FFTHelperFuncs.FFT.subcomm[1].Get_size()
        startIdx = FFTHelperFuncs.FFT.subcomm[1].Get_rank() * chunkSize
        endIdx = (FFTHelperFuncs.FFT.subcomm[1].Get_rank() + 1) * chunkSize
        self.get_and_write_statistics_to_file(DM[:,startIdx:endIdx],"DM_y")
        self.get_and_write_statistics_to_file(np.log(DM[:,startIdx:endIdx]),"lnDM_y")
        self.get_and_write_statistics_to_file(RM[:,startIdx:endIdx],"RM_y")
        self.get_and_write_statistics_to_file(RM[:,startIdx:endIdx]/DM[:,startIdx:endIdx],"LOSB_y")

        DM = np.mean(rho,axis=2)
        RM = np.mean(B[2]*rho,axis=2)
        self.get_and_write_statistics_to_file(DM,"DM_z")
        self.get_and_write_statistics_to_file(np.log(DM),"lnDM_z")
        self.get_and_write_statistics_to_file(RM,"RM_z")
        self.get_and_write_statistics_to_file(RM/DM,"LOSB_z")

        corrRhoB = self.get_corr_coeff(rho,np.sqrt(B2))
        if self.rank == 0:
            self.outfile.require_dataset('rho-B/corr', (1,), dtype='f')[0] = corrRhoB

        if self.eos == 'adiabatic':
            rhoToGamma = rho**self.gamma
            corrRhoToGammaB = self.get_corr_coeff(rhoToGamma,np.sqrt(B2))
            if self.rank == 0:
                self.outfile.require_dataset('rhoToGamma-B/corr', (1,), dtype='f')[0] = corrRhoToGammaB

            self.get_2d_hist('rhoToGamma-B',rhoToGamma,np.sqrt(B2))
            self.get_2d_hist('rhoToGamma-MagEnDensity',rhoToGamma,0.5 * B2)

        self.get_2d_hist('rho-B',rho,np.sqrt(B2))
        self.get_2d_hist('log10rho-B',np.log10(rho),np.sqrt(B2))

        if self.rank == 0:
            self.outfile.close()

    def Kernel(self,DELTA,KERNEL,factor=None):
        """
        This function creates the convolution kernel for use with a particular filtering scale
        (see equation 6 of "Extracting the spectrum of a flow by spatial filtering" by Sadek and Aluie (2018))
        For an introduction to scale filtering, see chapter 2 of "Large Eddy Simulation for Incompressible Flows" by Pierre Sagaut.
        The three classical filters used here are described in section 1.5 of that chapter.
        KERNEL - the chosen type of convolution kernel (Box, Sharp, or Gaussian)
        DELTA  - filter width (in grid cells): filter_width = self.res/(2k) where k is the wavenumber associated with the filter.
        factor - (optional) multiplicative factor of the filter width
        """
        k = self.localKmag
        pi = np.pi

        if factor is None:
            factor = 1
        else:
            factor = np.int(factor)

        # NOTE: Box Kernel needs to be updated for parallel computing (currently non-functional)
        if KERNEL == "Box":   # aka top hat filter
            sys.exit("Box kernel is currently non-functional")
            localKern = np.zeros((self.res,self.res,self.res))
            for i in range(-factor*np.int(DELTA)//2,factor*np.int(DELTA)//2+1):
                for j in range(-factor*np.int(DELTA)//2,factor*np.int(DELTA)//2+1):
                    for k in range(-factor*np.int(DELTA)//2,factor*np.int(DELTA)//2+1):
                        localFac = 1.
                        if np.abs(i) == factor*np.int(DELTA)//2:
                            localFac *= 0.5
                        if np.abs(j) == factor*np.int(DELTA)//2:
                            localFac *= 0.5
                        if np.abs(k) == factor*np.int(DELTA)//2:
                            localFac *= 0.5 
            localKern[i,j,k] = localFac / float(factor*np.int(DELTA))**3.
            return fftn(localKern)
        elif KERNEL == "Sharp":
            localKern = np.ones_like(k)
            localKern[k > np.float(self.res)/(2. * factor * np.float(DELTA))] = 0. # Remove small scales
            return localKern
        elif KERNEL == "Gauss":
            return np.exp(-(pi * factor * DELTA/self.res * k)**2. /6.)
        else:
            sys.exit("Unknown kernel used")

    def get_rotation_free_vec_field(self, vec):
        """
        Calculates the dilatational (compressive) component using spectral projection.
        Corrects for the Transposed Data Layout (Y, X, Z).
        
        Math: u_dil = k * (k . u) / k^2
        """
        # 1. Forward FFT
        FT_vec = newDistArray(self.FFT, rank=1)
        for i in range(3):
            FT_vec[i] = self.FFT.forward(vec[i], FT_vec[i])

        # 2. Get local wavenumbers and Map to Physical Axes
        # Data is (Y, X, Z) -> localK is (ky, kx, kz)
        ky = self.localK[0]
        kx = self.localK[1]
        kz = self.localK[2]
        
        # 3. Calculate k^2
        k2 = kx**2 + ky**2 + kz**2

        # 4. Safe Inverse of k^2 (Fixes the AssertionError)
        # Instead of masking the Distributed Array, we mask the local 'inv_k2' array
        inv_k2 = np.zeros_like(k2)
        mask = k2 > 0
        inv_k2[mask] = 1.0 / k2[mask]

        # 5. Project: (k . u) / k^2
        # Correct Dot Product: kx*Ux + ky*Uy + kz*Uz
        k_dot_u = kx * FT_vec[0] + ky * FT_vec[1] + kz * FT_vec[2]
        
        factor = k_dot_u * inv_k2

        FT_dil = newDistArray(self.FFT, rank=1)
        FT_dil[0] = factor * kx # Proj x
        FT_dil[1] = factor * ky # Proj y
        FT_dil[2] = factor * kz # Proj z

        # 6. Inverse FFT
        dil = newDistArray(self.FFT, False, rank=1)
        for i in range(3):
            dil[i] = self.FFT.backward(FT_dil[i], dil[i]).real

        return dil

    def normalized_spectrum(self,k,quantity):
        """ 
        Calculate normalized power spectra with robust error handling
        """
        histSum = binned_statistic(k,quantity,bins=self.k_bins,statistic='sum')[0]
        kSum = binned_statistic(k,k,bins=self.k_bins,statistic='sum')[0]
        histCount = np.histogram(k,bins=self.k_bins)[0]

        totalHistSum = self.comm.reduce(histSum.astype(np.float64))
        totalKSum = self.comm.reduce(kSum.astype(np.float64))
        totalHistCount = self.comm.reduce(histCount.astype(np.float64))

        if self.rank == 0:
            # Only fail if ALL bins are empty (would be a real problem)
            if (totalHistCount == 0.).all():
                print("ERROR: All histogram bins are empty!")
                sys.exit(1)

            # Handle empty bins safely
            valid_bins = totalHistCount > 0
            n_empty = np.sum(~valid_bins)
            if n_empty > 0 and self.rank == 0:
                print(f"Note: {n_empty}/{len(totalHistCount)} bins are empty (normal for high-k)")

            # Initialize all arrays
            centeredK = np.zeros_like(totalHistCount, dtype=float)
            valsShell = np.zeros_like(totalHistCount, dtype=float)  
            valsVol = np.zeros_like(totalHistCount, dtype=float)

            # Only compute for non-empty bins
            if np.any(valid_bins):
                centeredK[valid_bins] = totalKSum[valid_bins] / totalHistCount[valid_bins]
                valsShell[valid_bins] = 4. * np.pi * centeredK[valid_bins]**2. * (totalHistSum[valid_bins] / totalHistCount[valid_bins])
                valsVol[valid_bins] = 4. * np.pi / 3. * (self.k_bins[1:][valid_bins]**3. - self.k_bins[:-1][valid_bins]**3.) * (totalHistSum[valid_bins] / totalHistCount[valid_bins])

            # Raw counts (always safe)
            valsNoNorm = totalHistSum.copy()

            return [centeredK, valsShell, valsVol, valsNoNorm]
        else:
            return None
    
    def decompose_vector(self, vec):
        """ decomposed input vector into harmonic, rotational and compressive part
        """

        N = float(self.comm.allreduce(vec.size))
        total = self.comm.allreduce(np.sum(vec,axis=(1,2,3)))
        # dividing by N/3 as the FFTs are per dimension, i.e., normal is N^3 but N is 3N^3
        harm = total / (N/3)

        # Spectral Helmholtz projection for compressive part
        dil = self.spectral_dil(vec)
        sol = vec - harm.reshape((3,1,1,1)) - dil

        return harm, sol, dil

    def spectral_dil(self, vec):
            """
            Compute compressive (irrotational) component using spectral projection.
            FIXED for Transposed Data (Axis0=Y, Axis1=X) and Distributed Arrays.
            """
            # 1. Map FFT axes to Physical Wavenumbers
            # FFT returns (k0, k1, k2). Since we transposed data to (Y, X, Z):
            # k0 = ky, k1 = kx, k2 = kz
            ky_grid, kx_grid, kz_grid = FFTHelperFuncs.local_wavenumbermesh
            
            # 2. Compute k^2
            k2 = kx_grid**2 + ky_grid**2 + kz_grid**2
    
            # 3. Create Safe Inverse (Fixes AssertionError)
            # We cannot use boolean masking on Distributed Arrays directly.
            # Instead, we create a local numpy array for the inverse factor.
            inv_k2 = np.zeros_like(k2)
            nonzero_mask = k2 > 0
            inv_k2[nonzero_mask] = 1.0 / k2[nonzero_mask]
    
            # 4. Forward FFT of Velocity
            FT_vec = newDistArray(self.FFT, rank=1)
            for i in range(3):
                FT_vec[i] = self.FFT.forward(vec[i], FT_vec[i])
    
            # 5. Compute Divergence (k dot u)
            # Correct Physics: kx*Ux + ky*Uy + kz*Uz
            # Note: FT_vec[0] is Ux, FT_vec[1] is Uy
            k_dot_u = kx_grid * FT_vec[0] + ky_grid * FT_vec[1] + kz_grid * FT_vec[2]
    
            # 6. Compute Scalar Factor for Projection
            factor = k_dot_u * inv_k2
    
            # 7. Compute Dilatational Component in Fourier Space
            # vec_dil = factor * vector_k
            FT_dil = newDistArray(self.FFT, rank=1)
            FT_dil[0] = factor * kx_grid # Ux component uses kx
            FT_dil[1] = factor * ky_grid # Uy component uses ky
            FT_dil[2] = factor * kz_grid # Uz component uses kz
    
            # 8. Backward FFT
            dil = newDistArray(self.FFT, False, rank=1)
            for i in range(3):
                dil[i] = self.FFT.backward(FT_dil[i], dil[i]).real
    
            return dil

    def scalar_power_spectrum(self,name,field):

        FT_field = newDistArray(self.FFT)
        FT_field = self.FFT.forward(field, FT_field)

        # Note: FFT normalization follows mpi4py-fft plan (no extra scaling here).
        FT_fieldAbs2 = np.abs(FT_field)**2.
        PS_Full = self.normalized_spectrum(self.localKmag.reshape(-1),FT_fieldAbs2.reshape(-1))

        if self.rank == 0:
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full

# e.g. (38) in https://arxiv.org/pdf/1101.0150.pdf
    def co_spectrum(self,name,fieldA,fieldB):

        FT_fieldA = newDistArray(self.FFT)
        FT_fieldA = self.FFT.forward(fieldA, FT_fieldA)

        FT_fieldB = newDistArray(self.FFT)
        FT_fieldB = self.FFT.forward(fieldB, FT_fieldB)

        FT_CoSpec = FT_fieldA * np.conj(FT_fieldB)
        PS_Abs = self.normalized_spectrum(self.localKmag.reshape(-1),np.abs(FT_CoSpec).reshape(-1))
        PS_Real = self.normalized_spectrum(self.localKmag.reshape(-1),np.real(FT_CoSpec).reshape(-1))

        if self.rank == 0:
            self.outfile.require_dataset(name + '/CoSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/CoSpec/Abs', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Abs
            self.outfile.require_dataset(name + '/CoSpec/Real', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Real

    def vector_power_spectrum(self, name, vec):
        print(f"[rank {self.rank}] vector_power_spectrum start: {name}, local vec shape={vec.shape}, expected FFT local_shape={FFTHelperFuncs.local_shape}")
        sys.stdout.flush()
        vec = np.ascontiguousarray(vec)
        # Real-space kinetic energy density (for Parseval check)
        real_ke = 0.5 * self.comm.allreduce(np.sum(vec**2.)) / float(self.res**3)
        real_ke = float(np.real(real_ke))

        FT_vec = newDistArray(self.FFT,rank=1)
        for i in range(3):
            FT_vec[i] = self.FFT.forward(vec[i], FT_vec[i])
        print(f"[rank {self.rank}] vector_power_spectrum after forward: {name}")
        sys.stdout.flush()

        # Note: FFT normalization follows mpi4py-fft plan (no extra scaling here).
        FT_vecAbs2 = np.linalg.norm(FT_vec,axis=0)**2.
        PS_Full = self.normalized_spectrum(self.localKmag.reshape(-1),FT_vecAbs2.reshape(-1))
        print(f"[rank {self.rank}] vector_power_spectrum after normalized_spectrum: {name}")
        sys.stdout.flush()
        
        totPowFull = self.comm.allreduce(np.sum(FT_vecAbs2))
        totPowFull = float(np.real(totPowFull))
        # Spectral kinetic energy density assuming unnormalized forward FFT:
        # sum_x |u|^2 = (1/N^3) * sum_k |U|^2, so mean(0.5|u|^2) = 0.5 * sum_k|U|^2 / N^6
        spec_ke = float(np.real(0.5 * totPowFull))
        if self.rank == 0:
            print(f"[rank 0] vector_power_spectrum after totals: {name}, totPowFull={totPowFull:.6e}")
            sys.stdout.flush()
        
        if self.rank == 0:
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full
            self.outfile.require_dataset(name + '/PowSpec/TotFull', (1,), dtype='f8')[0] = float(totPowFull)
            self.outfile.require_dataset(name + '/PowSpec/TotKE_real', (1,), dtype='f8')[0] = float(real_ke)
            self.outfile.require_dataset(name + '/PowSpec/TotKE_spec', (1,), dtype='f8')[0] = float(spec_ke)
            # Light-weight Parseval check
            print(f"[{name}] KE(real)={real_ke:.6e}, KE(spec)={spec_ke:.6e}, ratio={spec_ke/real_ke if real_ke!=0 else np.nan:.3f}")

        # Spectral Helmholtz projection for Dil/Sol spectra
        # local_wavenumbermesh is already aligned with FFT/local data ordering
        kx, ky, kz = FFTHelperFuncs.local_wavenumbermesh
        k2 = kx**2 + ky**2 + kz**2
        mask = k2 != 0.0

        FT_Dil = newDistArray(self.FFT, rank=1)
        for i in range(3):
            FT_Dil[i].fill(0.0)
        kdotv = kx * FT_vec[0] + ky * FT_vec[1] + kz * FT_vec[2]
        k2_safe = k2.copy()
        k2_safe[~mask] = 1.0  # avoid divide by zero
        proj = (kdotv / k2_safe) * mask  # set k=0 to zero
        for i, kcomp in enumerate([kx, ky, kz]):
            FT_Dil[i][...] = kcomp * proj
        FT_DilAbs2 = np.linalg.norm(FT_Dil,axis=0)**2.
        PS_Dil = self.normalized_spectrum(self.localKmag.reshape(-1),FT_DilAbs2.reshape(-1))
        
        FT_Sol = FT_vec - FT_Dil
        # remove harmonic (k=0) from solenoidal component
        if self.rank == 0:
            FT_Sol[:,0,0,0] = 0.0
        FT_SolAbs2 = np.linalg.norm(FT_Sol,axis=0)**2.
        PS_Sol = self.normalized_spectrum(self.localKmag.reshape(-1),FT_SolAbs2.reshape(-1))

        totPowDil = float(np.real(self.comm.allreduce(np.sum(FT_DilAbs2))))
        totPowSol = float(np.real(self.comm.allreduce(np.sum(FT_SolAbs2))))
        totPowHarm = float(np.real(np.linalg.norm(FT_vec[:,0,0,0],axis=0)**2.))

        if self.rank == 0:
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full
            self.outfile.require_dataset(name + '/PowSpec/Dil', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Dil
            self.outfile.require_dataset(name + '/PowSpec/Sol', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Sol
            self.outfile.require_dataset(name + '/PowSpec/TotSol', (1,), dtype='f')[0] = totPowSol
            self.outfile.require_dataset(name + '/PowSpec/TotDil', (1,), dtype='f')[0] = totPowDil
            self.outfile.require_dataset(name + '/PowSpec/TotHarm', (1,), dtype='f')[0] = totPowHarm

        print(f"[rank {self.rank}] vector_power_spectrum end: {name}")
        sys.stdout.flush()
        # Ensure temporaries are released before next spectrum to avoid memory pressure
        del FT_vec, FT_vecAbs2, PS_Full, FT_Dil, FT_DilAbs2, PS_Dil, FT_Sol, FT_SolAbs2, PS_Sol

    def get_and_write_statistics_to_file(self,field,name,bounds=None):
        """
            field - 3d scalar field to get statistics from
            name - human readable name of the field
            bounds - tuple, lower and upper bound for histogram, if None then min/max
        """

        N = float(self.comm.allreduce(field.size))
        total = self.comm.allreduce(np.sum(field))
        mean = total / N

        totalSqrd = self.comm.allreduce(np.sum(field**2.))
        rms = np.sqrt(totalSqrd / N)

        var = self.comm.allreduce(np.sum((field - mean)**2.)) / (N - 1.)

        stddev = np.sqrt(var)

        # old
        #skew = self.comm.allreduce(np.sum((field - mean)**3. / stddev**3.)) / N
        #kurt = self.comm.allreduce(np.sum((field - mean)**4. / stddev**4.)) / N - 3.

        if stddev > 1e-15:  # Avoid division by zero for constant fields
            skew = self.comm.allreduce(np.sum((field - mean)**3. / stddev**3.)) / N
            kurt = self.comm.allreduce(np.sum((field - mean)**4. / stddev**4.)) / N - 3.
        else:
            skew = 0.0  # Skewness is undefined for constant fields
            kurt = 0.0  # Kurtosis is undefined for constant fields


        globMin = self.comm.allreduce(np.min(field),op=self.MPI.MIN)
        globMax = self.comm.allreduce(np.max(field),op=self.MPI.MAX)

        globAbsMin = self.comm.allreduce(np.min(np.abs(field)),op=self.MPI.MIN)
        globAbsMax = self.comm.allreduce(np.max(np.abs(field)),op=self.MPI.MAX)

        if self.rank == 0:
            self.outfile.require_dataset(name + '/moments/mean', (1,), dtype='f')[0] = mean
            self.outfile.require_dataset(name + '/moments/rms', (1,), dtype='f')[0] = rms
            self.outfile.require_dataset(name + '/moments/var', (1,), dtype='f')[0] = var
            self.outfile.require_dataset(name + '/moments/stddev', (1,), dtype='f')[0] = stddev
            self.outfile.require_dataset(name + '/moments/skew', (1,), dtype='f')[0] = skew
            self.outfile.require_dataset(name + '/moments/kurt', (1,), dtype='f')[0] = kurt
            self.outfile.require_dataset(name + '/moments/min', (1,), dtype='f')[0] = globMin
            self.outfile.require_dataset(name + '/moments/max', (1,), dtype='f')[0] = globMax
            self.outfile.require_dataset(name + '/moments/absmin', (1,), dtype='f')[0] = globAbsMin
            self.outfile.require_dataset(name + '/moments/absmax', (1,), dtype='f')[0] = globAbsMax

        if bounds is None:
            bounds = [globMin,globMax]
            HistBins = "Snap"
        else:
            HistBins = "Sim"

        Bins = np.linspace(bounds[0],bounds[1],129)
        hist = np.histogram(field.reshape(-1),bins=Bins)[0]
        totalHist = self.comm.allreduce(hist)

        if self.rank == 0:
            tmp  = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,129), dtype='f')
            tmp[0] = Bins
            tmp[1,:-1] = totalHist.astype(float)

        if name in self.global_min_max.keys():
            Bins = np.linspace(self.global_min_max[name][0],self.global_min_max[name][1],129)
            hist = np.histogram(field.reshape(-1),bins=Bins)[0]
            totalHist = self.comm.allreduce(hist)

            HistBins = 'globalMinMax'

            if self.rank == 0:
                tmp  = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,129), dtype='f')
                tmp[0] = Bins
                tmp[1,:-1] = totalHist.astype(float)


    def get_corr_coeff(self,X,Y):
        N = float(self.comm.allreduce(X.size))
        meanX = self.comm.allreduce(np.sum(X)) / N
        meanY = self.comm.allreduce(np.sum(Y)) / N

        stdX = np.sqrt(self.comm.allreduce(np.sum((X - meanX)**2.)))
        stdY = np.sqrt(self.comm.allreduce(np.sum((Y - meanY)**2.)))

        cov = self.comm.allreduce(np.sum( (X - meanX)*(Y - meanY)))

        return cov / (stdX*stdY)

    def get_2d_hist(self,name,X,Y,bounds = None):
        if bounds is None:
            globXMin = self.comm.allreduce(np.min(X),op=self.MPI.MIN)
            globXMax = self.comm.allreduce(np.max(X),op=self.MPI.MAX)
            
            globYMin = self.comm.allreduce(np.min(Y),op=self.MPI.MIN)
            globYMax = self.comm.allreduce(np.max(Y),op=self.MPI.MAX)

            bounds = [[globXMin,globXMax],[globYMin,globYMax]]
            HistBins = "Snap"
        else:
            HistBins = "Sim"

        XBins = np.linspace(bounds[0][0],bounds[0][1],129)
        YBins = np.linspace(bounds[1][0],bounds[1][1],129)

        hist = np.histogram2d(X.reshape(-1),Y.reshape(-1),bins=[XBins,YBins])[0]
        totalHist = self.comm.allreduce(hist)

        if self.rank == 0:
            tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,129), dtype='f')
            tmp[0] = XBins
            tmp[1] = YBins
            tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/counts', (128,128), dtype='f')
            tmp[:,:] = totalHist.astype(float)

        Xname, Yname = name.split('-')
        if Xname in self.global_min_max.keys() and Yname in self.global_min_max.keys():
            XBins = np.linspace(self.global_min_max[Xname][0],self.global_min_max[Xname][1],129)
            YBins = np.linspace(self.global_min_max[Yname][0],self.global_min_max[Yname][1],129)

            hist = np.histogram2d(X.reshape(-1),Y.reshape(-1),bins=[XBins,YBins])[0]
            totalHist = self.comm.allreduce(hist)

            HistBins = 'globalMinMax'

            if self.rank == 0:
                tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,129), dtype='f')
                tmp[0] = XBins
                tmp[1] = YBins
                tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/counts', (128,128), dtype='f')
                tmp[:,:] = totalHist.astype(float)

    def run_test(self):
        """ simple tests (to be expanded and separated)
        """
        # using velocity field for no reason, but it's probably most likely to be avail
        vec = self.U

        # TESTING VECTOR DECOMPOSITION
        vec_harm, vec_sol, vec_dil = self.decompose_vector(vec)
        np.testing.assert_allclose(vec,
                                   vec_harm.reshape((3,1,1,1)) + vec_sol + vec_dil,
                                   err_msg="Mismatch bw/ decomposed and original vector.")

        msg = "solenoidal part is not divergence free"
        assert np.sum(np.abs(MPIdivX(self.comm, vec_sol)))/vec_sol.size/3 < 1e-13, msg
        
        msg = "compressive part is not rotation free"
        assert np.sum(np.linalg.norm(MPIrotX(self.comm, vec_dil),axis=0))/vec_dil.size/3 < 1e-13, msg
