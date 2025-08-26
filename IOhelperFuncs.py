import yt
import numpy as np
from mpi4py import MPI
from mpi4py_fft import newDistArray
import FFTHelperFuncs
import sys
import h5py

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_fields(args):
    """
    Read all fields of a simulation snapshot

    args : forwarded command line arguments from the main script
    """
    # data dictionary
    fields = {
        'B' : None,
        'Acc' : None,
        'P' : None,
    }
    pressField = None
    magFields = None
    accFields = None

    time_start = MPI.Wtime()
    
    if args['data_type'] == 'FiniteElement':
        return read_finite_element_data(args)
    elif args['data_type'] == 'Enzo':
        rhoField = "Density"
        velFields = ["x-velocity","y-velocity","z-velocity"]
        if args['b']:
            magFields = ["Bx","By","Bz"]
        if args['forced']:
            accFields = ['x-acceleration','y-acceleration','z-acceleration']

        readAllFieldsWithYT(fields, args['data_path'], args['res'],
                            rhoField, velFields, magFields,
                            accFields, pressField)

    elif args['data_type'][:8] == 'AthenaPP':
        rhoField = ('athena_pp', 'rho')
        velFields = [('athena_pp', 'vel1'), ('athena_pp', 'vel2'), ('athena_pp', 'vel3')]
        if args['b']:
            magFields = [('athena_pp', 'Bcc1'), ('athena_pp', 'Bcc2'), ('athena_pp', 'Bcc3')]
        if args['forced']:
            accFields = [('athena_pp', 'acceleration_x'),
                         ('athena_pp', 'acceleration_y'),
                         ('athena_pp', 'acceleration_z')]

        if args['eos'] == 'adiabatic':
            pressField = ('athena_pp', 'press')

        if 'HDF' in args['data_type']:
            readAllFieldsWithHDF(fields,'./Turb.prim.' + args['data_path'], args['res'],
                                rhoField, velFields, magFields,
                                None, pressField,'F',use_athena_hdf=True)
            readAllFieldsWithHDF(fields,'./Turb.acc.' + args['data_path'], args['res'],
                                None, None, None,
                                accFields, None,'F',use_athena_hdf=True)
        else:
            readAllFieldsWithYT(fields,'./Turb.prim.' + args['data_path'], args['res'],
                                rhoField, velFields, magFields,
                                None, pressField)
            readAllFieldsWithYT(fields,'./Turb.acc.' + args['data_path'], args['res'],
                                None, None, None,
                                accFields, None)



    elif args['data_type'] == 'AthenaHDFC':
        rhoField = 'density'
        velFields = ['velocity_x', 'velocity_y', 'velocity_z']
        if args['b']:
            magFields = ['cell_centered_B_x', 'cell_centered_B_y', 'cell_centered_B_z']
        if args['forced']:
            accFields = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        if args['eos'] == 'adiabatic':
            pressField = 'pressure'

        order = 'C'

        fields  = readAllFieldsWithHDF(args['data_path'], args['res'],
                                       rhoField, velFields, magFields,
                                       accFields, pressField,order)

    elif args['data_type'] == 'Athena':
        rhoField = 'density'
        velFields = ['velocity_x', 'velocity_y', 'velocity_z']
        if args['b']:
            magFields = ['cell_centered_B_x', 'cell_centered_B_y', 'cell_centered_B_z']
        if args['forced']:
            accFields = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        if args['eos'] == 'adiabatic':
            pressField = 'pressure'

        order = 'C'

        readAllFieldsWithYT(fields, args['data_path'], args['res'],
                            rhoField, velFields, magFields,
                            accFields, pressField)

    else:
        raise SystemExit('Unknown data type: ', data_type)

    time_elapsed = MPI.Wtime() - time_start
    time_elapsed = comm.gather(time_elapsed)

    if comm.Get_rank() == 0:
        print("Reading data done in %.3g +/- %.3g" %
            (np.mean(time_elapsed), np.std(time_elapsed)))
        sys.stdout.flush()

    return fields

def readAllFieldsWithYT(fields,loadPath,Res,
    rhoField,velFields,magFields,accFields,pressField=None):
    """
    Reads all fields using the yt frontend. Data is read in parallel.

    """
    pencil_shape = FFTHelperFuncs.local_shape
    if (np.array(FFTHelperFuncs.FFT.global_shape(), dtype=int) % pencil_shape != 0).any():
        raise SystemExit(
            'Data cannot be split evenly among processes. ' +
            'Abort (for now) - fix me!')

    ds = yt.load(loadPath)
    left_edge = ds.domain_left_edge
    right_edge = ds.domain_right_edge

    n_proc = np.array(FFTHelperFuncs.FFT.global_shape(), dtype=int) // pencil_shape
    gid_x_s = rank // n_proc[1] * pencil_shape[0] # global x start index
    gid_y_s = rank % n_proc[1] * pencil_shape[1] # global y start index

    start_pos = left_edge
    start_pos[0] += gid_x_s / Res * (right_edge[0] - left_edge[0])
    start_pos[1] += gid_y_s / Res * (right_edge[1] - left_edge[1])
    if rank == 0:
        print("Loading "+ loadPath)
        print("Chunk dimensions = ", pencil_shape)


    ad = ds.h.covering_grid(level=0, left_edge=start_pos,dims=FFTHelperFuncs.local_shape)

    if rhoField is not None:
        fields['rho'] = ad[rhoField].d

    if pressField is not None:
        fields['P'] = ad[pressField].d
    else:
        if rank == 0:
            print("WARNING: assuming isothermal EOS with c_s = 1, i.e. P = rho")
        fields['P'] = fields['rho']
    
    if velFields is not None:
        U = np.zeros((3,) + pencil_shape,dtype=np.float64)
        U[0] = ad[velFields[0]].d
        U[1] = ad[velFields[1]].d
        U[2] = ad[velFields[2]].d
        fields['U'] = U

    if magFields is not None:
        B = np.zeros((3,) + pencil_shape,dtype=np.float64)
        B[0] = ad[magFields[0]].d
        B[1] = ad[magFields[1]].d
        B[2] = ad[magFields[2]].d
        fields['B'] = B

    if accFields is not None:
        Acc = np.zeros((3,) + pencil_shape,dtype=np.float64)
        Acc[0] = ad[accFields[0]].d
        Acc[1] = ad[accFields[1]].d
        Acc[2] = ad[accFields[2]].d
        fields['Acc'] = Acc


def readOneFieldWithHDF(loadPath,FieldName,Res,order):

    if order == 'F':
        Filename = loadPath + '/' + FieldName + '-' + str(Res) + '.hdf5'

        if rank == 0:

            h5Data = h5py.File(Filename, 'r')[FieldName]
        
            tmp = np.float64(h5Data[0,:,:,:]).T.reshape((size,int(Res/size),Res,Res))

            data = comm.scatter(tmp)

        else:
            data = comm.scatter(None)

    elif order == 'C':
        Filename = loadPath + '/' + FieldName + '-' + str(Res) + '-C.hdf5'
        
        chunkSize = Res/size
        startIdx = int(rank * chunkSize)
        endIdx = int((rank + 1) * chunkSize)
        if endIdx == Res:                
            endIdx = None
        
        h5Data = h5py.File(Filename, 'r')[FieldName]
        data = np.float64(h5Data[0,startIdx:endIdx,:,:])

    if rank == 0:
        print("[%03d] done reading %s" % (rank,FieldName))

    return np.ascontiguousarray(data)

def readOneFieldWithAthenaPPHDF(loadPath,FieldName,Res,order):
    """
    reading (K-)Athena++ HDF data dumps
    """

    # stripping the yt field type
    FieldName = FieldName[1]

    tmp = np.zeros(FFTHelperFuncs.local_shape, dtype=np.float64)
    loc_slc = tmp.shape

    n_proc = np.array(FFTHelperFuncs.FFT.global_shape(), dtype=int) // loc_slc

    if h5py.h5.get_config().mpi:
        h5py_kwargs = {
            'driver' : 'mpio',
            'comm' : comm,
            }
    else:
        h5py_kwargs = {}

    with h5py.File(loadPath,'r', **h5py_kwargs) as f:
        mb_size = f.attrs['MeshBlockSize']
        rg_size = f.attrs['RootGridSize']

        if 'rho' == FieldName:
            field_idx = 0
            ds_name = 'prim'
        elif 'press' == FieldName:
            field_idx = 1
            ds_name = 'prim'
        elif 'vel' in FieldName:
            field_idx = 1 + int(FieldName[-1])
            ds_name = 'prim'
        elif 'Bcc' in FieldName:
            field_idx = int(FieldName[-1]) - 1
            ds_name = 'B'
        elif 'acc' in FieldName:
            # translate from ..._x, _y, _z to index 0, 1, 2
            field_idx = ord(FieldName[-1]) - 120
            ds_name = 'hydro'
        else:
            raise SystemExit(
                'Unknown field: ', FieldName)


        if not ((loc_slc[0] % mb_size[0] == 0 or mb_size[0] % loc_slc[0] == 0) and
                (loc_slc[1] % mb_size[1] == 0 or mb_size[1] % loc_slc[1] == 0)):
            raise SystemExit(
                'Error: local data size  ', loc_slc,
                'cannot be matched to meshblock size of ', mb_size)

        gid_x_s = rank // n_proc[1] * tmp.shape[0] # global x start index
        gid_x_e = rank // n_proc[1] * tmp.shape[0] + tmp.shape[0] # global x end index
        gid_y_s = rank % n_proc[1] * tmp.shape[1] # global y start index
        gid_y_e = rank % n_proc[1] * tmp.shape[1] + tmp.shape[1] # global y end index

        log_loc_all = np.copy(f['LogicalLocations']) # all logical meshblock locations

        for i, loc in enumerate(log_loc_all):
            gid_mb = loc * mb_size # index of meshblock
            # make sure meshblock belong to this MPI proc
            if not ((gid_mb[0] <= gid_x_s < gid_mb[0] + mb_size[0] or
                     gid_x_s <= gid_mb[0] < gid_x_e) and
                    (gid_mb[1] <= gid_y_s < gid_mb[1] + mb_size[1] or
                     gid_y_s <= gid_mb[1] < gid_y_e)):
                continue

            try:
                data = f[ds_name][field_idx,i,:,:,:] # actual meshblock data
            except KeyError:
                raise SystemExit(
                    'Cannot find data in dataset: ', ds_name, field_idx
                    )

            # if local x pencil dim smaller than a meshblock use entire local pencil
            if mb_size[0] >= loc_slc[0]:
                loc_x_s = 0
                loc_x_e = tmp.shape[0]
            else:
                loc_x_s = loc[0]*mb_size[0] - gid_x_s
                loc_x_e = (loc[0]+1)*mb_size[0] - gid_x_s
            sl_x = slice(gid_x_s % mb_size[0],gid_x_s % mb_size[0] + tmp.shape[0])

            # if local y pencil dim smaller than a meshblock use entire local pencil
            if mb_size[1] >= loc_slc[1]:
                loc_y_s = 0
                loc_y_e = tmp.shape[1]
            else:
                loc_y_s = loc[1]*mb_size[1] - gid_y_s
                loc_y_e = (loc[1] + 1)*mb_size[1] - gid_y_s
            sl_y = slice(gid_y_s % mb_size[1],gid_y_s % mb_size[1] + tmp.shape[1])

            tmp[loc_x_s: loc_x_e,
                loc_y_s: loc_y_e,
                loc[2]*mb_size[2] : (loc[2] + 1)*mb_size[2]] = data.T[sl_x,sl_y,:]

    return np.ascontiguousarray(np.float64(tmp))

def readAllFieldsWithHDF(fields,loadPath,Res,
    rhoField,velFields,magFields,accFields,pField,order,use_athena_hdf=False):
    """
    Reads all fields using the HDF5. Data is read in parallel.

    """

    FinalShape = FFTHelperFuncs.local_shape

    if (FinalShape[0] * FFTHelperFuncs.FFT.subcomm[0].Get_size() != Res or
        FinalShape[1] * FFTHelperFuncs.FFT.subcomm[1].Get_size() != Res):
        print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
        sys.exit(1)

    if order is not "C" and order is not "F":
        print("For safety reasons you have to specify the order (row or column major) for your data.")
        sys.exit(1)

    if use_athena_hdf:
        readOneFieldWithX = readOneFieldWithAthenaPPHDF
    else:
        readOneFieldWithX = readOneFieldWithHDF

    if rhoField is not None:
        fields['rho'] = readOneFieldWithX(loadPath,rhoField,Res,order)

    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = readOneFieldWithX(loadPath,velFields[0],Res,order)
        U[1] = readOneFieldWithX(loadPath,velFields[1],Res,order)
        U[2] = readOneFieldWithX(loadPath,velFields[2],Res,order)
        fields['U'] = U

    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = readOneFieldWithX(loadPath,magFields[0],Res,order)
        B[1] = readOneFieldWithX(loadPath,magFields[1],Res,order)
        B[2] = readOneFieldWithX(loadPath,magFields[2],Res,order)
        fields['B'] = B

    if accFields is not None:
        Acc = np.zeros((3,) + FinalShape,dtype=np.float64)  
        Acc[0] = readOneFieldWithX(loadPath,accFields[0],Res,order)
        Acc[1] = readOneFieldWithX(loadPath,accFields[1],Res,order)
        Acc[2] = readOneFieldWithX(loadPath,accFields[2],Res,order)
        fields['Acc'] = Acc

    if pField is not None:
        fields['P'] = readOneFieldWithX(loadPath,pField,Res,order)
#TODO FIXME for new layout
#    else:
#        # CAREFUL assuming isothermal EOS here with c_s = 1 -> P = rho in code units
#        if rank == 0:
#            print("WARNING: remember assuming isothermal EOS with c_s = 1, i.e. P = rho")
#        P = rho

# Add this function to IOhelperFuncs.py

def read_finite_element_data(args):
    """
    Read finite element data from custom format
    """
    import re
    
    data_path = args['data_path']
    
    print(f"Reading FiniteElement data from: {data_path}")
    
    # Parse header
    with open(data_path, 'r') as f:
        header_lines = [next(f) for _ in range(6)]
    
    # Extract cycle and time 
    cycle = 0
    time = 0.0
    for line in header_lines:
        if 'Cycle' in line:
            cycle_match = re.search(r'Cycle\s*[:=]\s*(\d+)', line)
            if cycle_match:
                cycle = int(cycle_match.group(1))
        if 'Time' in line:
            time_match = re.search(r'Time\s*[:=]\s*([0-9.eE+-]+)', line)
            if time_match:
                time = float(time_match.group(1))
    
    print(f"Cycle: {cycle}, Time: {time}")
    
    # Load data (skip 5 lines - the format shows 6 header lines but skip_header=5)
    data = np.genfromtxt(data_path, delimiter=None, skip_header=5)
    
    if data.shape[1] != 6:
        raise ValueError(f"Expected 6 columns (x,y,z,vx,vy,vz), got {data.shape[1]}")
    
    # Extract coordinates and velocities
    xpos, ypos, zpos = data[:, 0], data[:, 1], data[:, 2]
    velx, vely, velz = data[:, 3], data[:, 4], data[:, 5]
    
    # Round coordinates to handle precision issues
    xpos_rounded = np.round(xpos, decimals=10)
    ypos_rounded = np.round(ypos, decimals=10) 
    zpos_rounded = np.round(zpos, decimals=10)
    
    # Get unique coordinates and determine grid
    x_unique = np.sort(np.unique(xpos_rounded))
    y_unique = np.sort(np.unique(ypos_rounded))
    z_unique = np.sort(np.unique(zpos_rounded))
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    
    # Check if resolution matches expected
    expected_res = args['res']
    if max(nx, ny, nz) != expected_res:
        print(f"WARNING: Grid size {max(nx,ny,nz)} doesn't match --res {expected_res}")
        print(f"Consider using --res {max(nx,ny,nz)}")
    
    # Create velocity grids
    velx_grid = np.full((nx, ny, nz), np.nan)
    vely_grid = np.full((nx, ny, nz), np.nan)
    velz_grid = np.full((nx, ny, nz), np.nan)
    
    # Create coordinate-to-index mappings
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)} 
    z_idx = {val: i for i, val in enumerate(z_unique)}
    
    # Fill grids
    for i in range(len(xpos)):
        try:
            xi = x_idx[xpos_rounded[i]]
            yi = y_idx[ypos_rounded[i]]
            zi = z_idx[zpos_rounded[i]]
            velx_grid[xi, yi, zi] = velx[i]
            vely_grid[xi, yi, zi] = vely[i] 
            velz_grid[xi, yi, zi] = velz[i]
        except KeyError as e:
            print(f"Warning: Could not map point {i} with coords {xpos_rounded[i], ypos_rounded[i], zpos_rounded[i]}")
    
    # Check for missing data
    n_nan = np.sum(np.isnan(velx_grid))
    if n_nan > 0:
        print(f"Warning: {n_nan}/{nx*ny*nz} grid points have no data")
    
    # Handle NaNs and infinities
    velx_grid = np.nan_to_num(velx_grid, nan=0.0, posinf=0.0, neginf=0.0)
    vely_grid = np.nan_to_num(vely_grid, nan=0.0, posinf=0.0, neginf=0.0)
    velz_grid = np.nan_to_num(velz_grid, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute some basic diagnostics
    tke_physical = 0.5 * np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    max_vel = np.sqrt(np.max(velx_grid**2 + vely_grid**2 + velz_grid**2))
    print(f"Total Kinetic Energy: {tke_physical:.6e}")
    print(f"Maximum velocity magnitude: {max_vel:.6e}")
    
    # Create fields dictionary
    fields = {}
    
    # Velocity field - NOTE: Order is [3, nx, ny, nz] for vector fields
    fields['U'] = np.array([velx_grid, vely_grid, velz_grid])
    
    # Density field (assume uniform for now)
    fields['rho'] = np.ones((nx, ny, nz), dtype=np.float64)
    
    # Pressure (derive from isothermal EOS: P = rho * c_s^2, assuming c_s = 1)
    if args['eos'] == 'isothermal':
        fields['P'] = fields['rho'].copy()  # P = rho when c_s = 1
    elif args['eos'] == 'adiabatic':
        # For adiabatic without temperature data, set P = None
        # Could derive from energy if available
        fields['P'] = None
        print("Warning: No pressure data available for adiabatic EOS")
    
    # No magnetic field data
    fields['B'] = None
    
    # No external forcing/acceleration
    fields['Acc'] = None
    
    print("Successfully created fields dictionary")
    print(f"  U shape: {fields['U'].shape}")
    print(f"  rho shape: {fields['rho'].shape}")

    # Add this after creating the velocity grids in read_finite_element_data():

    # Debug: Check for problematic values
    print(f"Velocity grid stats:")
    for i, name in enumerate(['vx', 'vy', 'vz']):
        field = [velx_grid, vely_grid, velz_grid][i]
        n_nan = np.sum(np.isnan(field))
        n_inf = np.sum(np.isinf(field))
        n_finite = np.sum(np.isfinite(field))
        print(f"  {name}: NaN={n_nan}, Inf={n_inf}, Finite={n_finite}, Min={np.nanmin(field):.6e}, Max={np.nanmax(field):.6e}")

    print(f"Density grid stats:")
    n_nan = np.sum(np.isnan(fields['rho']))
    n_inf = np.sum(np.isinf(fields['rho']))
    print(f"  rho: NaN={n_nan}, Inf={n_inf}, Min={np.nanmin(fields['rho']):.6e}, Max={np.nanmax(fields['rho']):.6e}")
        
    return fields