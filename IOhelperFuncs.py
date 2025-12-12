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

    start_pos = left_edge.copy()
    start_pos[0] += gid_x_s / Res * (right_edge[0] - left_edge[0])
    start_pos[1] += gid_y_s / Res * (right_edge[1] - left_edge[1])
    if rank == 0:
        print("Loading "+ loadPath)
        print("Chunk dimensions = ", pencil_shape)


    ad = ds.covering_grid(level=0, left_edge=start_pos,dims=FFTHelperFuncs.local_shape)

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

    if order != "C" and order != "F":
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

def read_data_file_chunked(filename, chunk_size=5_000_000):
    """
    Read velocity data file in chunks (pandas) to keep memory use modest.
    Returns (velx_grid, vely_grid, velz_grid, x_unique, y_unique, z_unique).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required for FiniteElement chunked reading; install it and retry.") from exc

    print(f"Reading data from: {filename} (chunked, size={chunk_size})")
    sys.stdout.flush()
    reader = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=6,  # header has 6 lines
        header=None,
        names=["x", "y", "z", "vx", "vy", "vz"],
        chunksize=chunk_size,
        engine="python",
        on_bad_lines="skip",
    )

    xpos_list, ypos_list, zpos_list = [], [], []
    velx_list, vely_list, velz_list = [], [], []

    for chunk in reader:
        # Force numeric conversion; coerce non-numeric to NaN then drop if any
        xp_raw = pd.to_numeric(chunk["x"], errors="coerce").to_numpy()
        yp_raw = pd.to_numeric(chunk["y"], errors="coerce").to_numpy()
        zp_raw = pd.to_numeric(chunk["z"], errors="coerce").to_numpy()
        vx_raw = pd.to_numeric(chunk["vx"], errors="coerce").to_numpy()
        vy_raw = pd.to_numeric(chunk["vy"], errors="coerce").to_numpy()
        vz_raw = pd.to_numeric(chunk["vz"], errors="coerce").to_numpy()

        # Drop rows with NaNs (e.g., malformed lines)
        mask = ~(np.isnan(xp_raw) | np.isnan(yp_raw) | np.isnan(zp_raw) |
                 np.isnan(vx_raw) | np.isnan(vy_raw) | np.isnan(vz_raw))
        if not np.any(mask):
            continue
        xp_raw, yp_raw, zp_raw = xp_raw[mask], yp_raw[mask], zp_raw[mask]
        vx_raw, vy_raw, vz_raw = vx_raw[mask], vy_raw[mask], vz_raw[mask]

        xp = np.round(xp_raw, 10)
        yp = np.round(yp_raw, 10)
        zp = np.round(zp_raw, 10)
        vx = vx_raw
        vy = vy_raw
        vz = vz_raw

        xpos_list.append(xp)
        ypos_list.append(yp)
        zpos_list.append(zp)
        velx_list.append(vx)
        vely_list.append(vy)
        velz_list.append(vz)

    if not xpos_list:
        raise SystemExit(f"No data found in {filename}")

    total_pts = sum(arr.size for arr in xpos_list)
    print(f"  Total data points: {total_pts}")

    xpos = np.empty(total_pts, dtype=xpos_list[0].dtype)
    ypos = np.empty(total_pts, dtype=ypos_list[0].dtype)
    zpos = np.empty(total_pts, dtype=zpos_list[0].dtype)
    velx = np.empty(total_pts, dtype=velx_list[0].dtype)
    vely = np.empty(total_pts, dtype=vely_list[0].dtype)
    velz = np.empty(total_pts, dtype=velz_list[0].dtype)

    offset = 0
    for xp, yp, zp, vx, vy, vz in zip(
            xpos_list, ypos_list, zpos_list, velx_list, vely_list, velz_list):
        n = xp.size
        xpos[offset:offset+n] = xp
        ypos[offset:offset+n] = yp
        zpos[offset:offset+n] = zp
        velx[offset:offset+n] = vx
        vely[offset:offset+n] = vy
        velz[offset:offset+n] = vz
        offset += n

    # Free chunk lists
    del xpos_list, ypos_list, zpos_list, velx_list, vely_list, velz_list

    x_unique = np.unique(xpos)
    y_unique = np.unique(ypos)
    z_unique = np.unique(zpos)
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    print(f"  Grid dimensions: {nx} x {ny} x {nz}")

    expected_num_points = nx * ny * nz
    if total_pts != expected_num_points:
        print(f"  Warning: Actual points ({total_pts}) != expected ({expected_num_points})")

    velx_grid = np.zeros((nx, ny, nz))
    vely_grid = np.zeros((nx, ny, nz))
    velz_grid = np.zeros((nx, ny, nz))

    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}

    for i in range(total_pts):
        xi = x_idx[xpos[i]]
        yi = y_idx[ypos[i]]
        zi = z_idx[zpos[i]]
        velx_grid[xi, yi, zi] = velx[i]
        vely_grid[xi, yi, zi] = vely[i]
        velz_grid[xi, yi, zi] = velz[i]

    return velx_grid, vely_grid, velz_grid, x_unique, y_unique, z_unique

def distribute_velocity_field(U_global, args):
    """
    Broadcast the global velocity field (shape (3, Nx, Ny, Nz)) from rank 0
    and slice it according to FFT.local_slice(). Verbose debug output.
    """
    if rank == 0:
        print(f"[rank {rank}] distribute: global shape {U_global.shape}, dtype={U_global.dtype}")
        sys.stdout.flush()

    # Broadcast shape/dtype
    shape = comm.bcast(U_global.shape if rank == 0 else None, root=0)
    dtype = comm.bcast(U_global.dtype if rank == 0 else None, root=0)
    if rank != 0:
        U_global = np.empty(shape, dtype=dtype)
    comm.Barrier()
    comm.Bcast(U_global, root=0)
    comm.Barrier()

    # Slice according to FFT decomposition
    if args['type'] == 'transfer':
        slc = FFTHelperFuncs.FFT.local_slice(False)
    else:
        slc = FFTHelperFuncs.FFT.local_slice()

    if rank == 0:
        print(f"[rank {rank}] FFT local_slice={slc}, FFT local_shape={FFTHelperFuncs.local_shape}")
        sys.stdout.flush()

    U_local = U_global[(slice(None),) + slc]

    # Ensure shape matches expected local_shape; permute spatial axes if needed
    target_shape = FFTHelperFuncs.local_shape
    lengths = [s.stop - s.start for s in slc]
    
    # Check if permutation is required to match FFT slab layout
    if tuple(lengths) != tuple(target_shape):
        import itertools
        perm = None
        for p in itertools.permutations([0, 1, 2]):
            if [lengths[i] for i in p] == list(target_shape):
                perm = p
                break
        if perm is None:
            raise SystemExit(f"Cannot match local slices {lengths} to target shape {target_shape}")
        
        # perm is purely spatial (0,1,2). We need to account for vector dim at index 0.
        # So spatial 0 becomes index 1, spatial 1 becomes index 2, etc.
        axes_order = (0,) + tuple(1 + i for i in perm)
        U_local = np.transpose(U_local, axes_order)
        print(f"[rank {rank}] Permuting local velocity axes with order {axes_order} to match target {target_shape}")
        sys.stdout.flush()
    else:
        print(f"[rank {rank}] Using local slab lengths {lengths} matching target {target_shape}")
        sys.stdout.flush()

    # CRITICAL FIX: Ensure the array is contiguous in memory. 
    # Transpose only creates a view; MPI/FFT libraries often require contiguous C-ordered memory.
    return np.ascontiguousarray(U_local)

def read_finite_element_data(args):
    """
    Read finite element data from custom format (Robust MPI Version)
    """
    import re
    
    data_path = args['data_path']
    
    # --- STEP 1: Rank 0 Reads Metadata ---
    cycle = 0
    time = 0.0
    dims = (0, 0, 0)
    
    if rank == 0:
        print(f"Reading FiniteElement data from: {data_path}")
        # Parse header
        with open(data_path, 'r') as f:
            header_lines = [next(f) for _ in range(6)]
        
        for line in header_lines:
            if 'Cycle' in line:
                m = re.search(r'Cycle\s*[:=]\s*(\d+)', line)
                if m: cycle = int(m.group(1))
            if 'Time' in line:
                m = re.search(r'Time\s*[:=]\s*([0-9.eE+-]+)', line)
                if m: time = float(m.group(1))
        
        print(f"Cycle: {cycle}, Time: {time}")
        
        # Load heavy data
        velx_grid, vely_grid, velz_grid, x_unique, y_unique, z_unique = read_data_file_chunked(data_path)
        nx, ny, nz = velx_grid.shape
        dims = (nx, ny, nz)
        print(f"Grid dimensions: {nx} x {ny} x {nz}")
        sys.stdout.flush()
    else:
        # Other ranks stay empty
        velx_grid = vely_grid = velz_grid = None
    
    # --- STEP 2: Broadcast Metadata to everyone ---
    # Everyone waits here until Rank 0 is done reading
    dims = comm.bcast(dims, root=0)
    cycle = comm.bcast(cycle, root=0)
    time = comm.bcast(time, root=0)
    nx, ny, nz = dims
    
    # Check resolution
    expected_res = args['res']
    if rank == 0 and max(nx, ny, nz) != expected_res:
        print(f"WARNING: Grid size {max(nx,ny,nz)} doesn't match --res {expected_res}")

    comm.Barrier() # Ensure everyone is ready for distribution

    if rank == 0:
        print(f"[rank {rank}] Header read complete. Starting distribution...")
        sys.stdout.flush()

    # --- STEP 3: Distribute Data ---
    # Rank 0 packs the data, everyone calls distribute
    if rank == 0:
        U_global = np.array([velx_grid, vely_grid, velz_grid], dtype=np.float64)
    else:
        U_global = None

    U_local = distribute_velocity_field(U_global,args)
    
    # --- STEP 4: Create Fields ---
    fields = {}
    fields['U'] = U_local
    fields['rho'] = np.ones(FFTHelperFuncs.local_shape, dtype=np.float64)

    if args['eos'] == 'isothermal':
        fields['P'] = fields['rho'].copy()
    elif args['eos'] == 'adiabatic':
        fields['P'] = None
    
    fields['B'] = None
    fields['Acc'] = None

    if rank == 0:
        print("Successfully created fields dictionary")
        sys.stdout.flush()
        
    return fields
