# Sample run 

mpirun -n 8 python run_analysis.py --res 128 --type flow --data_type FiniteElement   --data_path ../mfem/miniapps/navier/SamplePointsVelocity_Re400NumPtsPerDir8RefLv2P4/cycle_7999/SampledData7999.txt   --eos isothermal --outfile flow_output.hdf5

# And then run 

python plot_flow_analysis.py --file flow_output.hdf5 --moments-fields AbsDivU AbsRotU u

# Energy Transfer Analsyis
mpirun -n 8 python run_analysis.py   --res 128   --type transfer   --data_type FiniteElement   --data_path ../mfem/miniapps/navier/SamplePointsVelocity_Re400NumPtsPerDir8RefLv2P4/cycle_7999/SampledData7999.txt   --eos isothermal   --terms UU   --binning log   --outfile transfer_output.pkl

python plot_transfer_hydro.py transfer_output.pkl --show-plot

python plot_txt_spectra.py spectrum.txt ../mfem/miniapps/navier/SamplePointsVelocity_Re400NumPtsPerDir8RefLv2P4/cycle_7999/energy_spectrum_step_7999.txt 



# Running test
# In the testing directory run

mpirun -np 8 python ../run_analysis.py --terms All FU PU BUPbb UBPbb --res 128 --data_path DD0024/data0024 --data_type Enzo --binning test --type transfer --outfile test.pkl --eos adiabatic --gamma 1.0001  -forced -b

python runTest.py 0024-All-Forc-Pres-testing-128_HDF-gold.pkl test.pkl
