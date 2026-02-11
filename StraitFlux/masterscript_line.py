import xarray as xa 
import pandas as pd
import numpy as np
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('skipping matplotlib')
import sys
from functools import partial
import time
try:
    from dask.diagnostics import ProgressBar
except ImportError:
    print('skipping dask import')

from xmip.preprocessing import rename_cmip6, promote_empty_dims, broadcast_lonlat, correct_coordinates
import StraitFlux.preprocessing as prepro
import StraitFlux.functions as func
from StraitFlux.indices import check_availability_indices, prepare_indices

def save_section_flux_detailed(udata, vdata, Tdata, Sdata, out_u, out_v, out_u_vz, 
                                 indices, sign_v, min_x, min_y, product, strait, 
                                 model, time_start, time_end, path_save):
    """
    Save detailed section flux for density-space analysis.
    Output: Normal flux, T, S at each section point (U/V faces) for later density binning.
    """
    
    # Extract section points from indices
    indi1 = indices.indices[:,2][indices.indices[:,3]!=0] # meridional
    indi2 = indices.indices[:,3][indices.indices[:,3]!=0]
    
    # Build section point lists
    npt_u = len(out_u)
    npt_v = len(indi1) - 1 if len(indi1) > 0 else 0
    
    nt, nlev = udata.shape[0], udata.shape[1]
    
    # Extract U-face section flux: qn_u(time, lev, npt_u) [m³/s]
    if npt_u > 0:
        qn_u_list, lon_u_list, lat_u_list = [], [], []
        for l in range(npt_u):
            iy, ix = int(out_u_vz[l,1]-min_y+1), int(out_u_vz[l,0]-min_x+1)
            sign = -1 if out_u_vz[l][2] == -1 else 1
            qn_u_list.append(udata[:, :, iy, ix].values * sign)
            lon_u_list.append(udata.lon[iy, ix].values)
            lat_u_list.append(udata.lat[iy, ix].values)
        qn_u = np.stack(qn_u_list, axis=2)
        lon_u, lat_u = np.array(lon_u_list), np.array(lat_u_list)
    else:
        qn_u = np.zeros((nt, nlev, 0))
        lon_u, lat_u = np.array([]), np.array([])
    
    # Extract V-face section flux: qn_v(time, lev, npt_v) [m³/s]
    if npt_v > 0:
        qn_v_list, lon_v_list, lat_v_list = [], [], []
        for m in range(npt_v):
            iy, ix = int(indi2[m]-min_y+1), int(indi1[m]-min_x+1)
            qn_v_list.append(vdata[:, :, iy, ix].values * sign_v[m])
            lon_v_list.append(vdata.lon[iy, ix].values)
            lat_v_list.append(vdata.lat[iy, ix].values)
        qn_v = np.stack(qn_v_list, axis=2)
        lon_v, lat_v = np.array(lon_v_list), np.array(lat_v_list)
    else:
        qn_v = np.zeros((nt, nlev, 0))
        lon_v, lat_v = np.array([]), np.array([])
    
    # Interpolate T/S to U/V faces (flux-consistent positions)
    if product in ['volume', 'heat', 'salt']:
        
        T_raw = Tdata.thetao.values  # (time, lev, y, x)
        S_raw = Sdata.so.values if Sdata is not None else None

        # T at U-faces
        if npt_u > 0:
            T_u_list = []
            for l in range(npt_u):
                iy, ix = int(out_u_vz[l,1]-min_y+1), int(out_u_vz[l,0]-min_x+1)
                T_u_list.append(T_raw[:, :, iy, ix])  
            T_u = np.stack(T_u_list, axis=2)
        else:
            T_u = np.full((nt, nlev, 0), np.nan)

        # S at U-faces
        if Sdata is not None and npt_u > 0:
            S_u_list = []
            for l in range(npt_u):
                iy, ix = int(out_u_vz[l,1]-min_y+1), int(out_u_vz[l,0]-min_x+1)
                S_u_list.append(S_raw[:, :, iy, ix])
            S_u = np.stack(S_u_list, axis=2)
        else:
            S_u = np.full((nt, nlev, npt_u), np.nan)

        # V-faces 
        if npt_v > 0:
            T_v_list = []
            for m in range(npt_v):
                iy, ix = int(indi2[m]-min_y+1), int(indi1[m]-min_x+1)
                T_v_list.append(T_raw[:, :, iy, ix])
            T_v = np.stack(T_v_list, axis=2)
        else:
            T_v = np.full((nt, nlev, 0), np.nan)

        if Sdata is not None and npt_v > 0:
            S_v_list = []
            for m in range(npt_v):
                iy, ix = int(indi2[m]-min_y+1), int(indi1[m]-min_x+1)
                S_v_list.append(S_raw[:, :, iy, ix])
            S_v = np.stack(S_v_list, axis=2)
        else:
            S_v = np.full((nt, nlev, npt_v), np.nan)

    
    # Calculate verification variable: Q_level(time, lev)
    Q_level = qn_u.sum(axis=2) + qn_v.sum(axis=2)
    
    # Build Dataset
    ds_section = xa.Dataset(
        {
            'qn_u': (['time', 'lev', 'npt_u'], qn_u),
            'qn_v': (['time', 'lev', 'npt_v'], qn_v),
            'T_u': (['time', 'lev', 'npt_u'], T_u),
            'T_v': (['time', 'lev', 'npt_v'], T_v),
            'S_u': (['time', 'lev', 'npt_u'], S_u),
            'S_v': (['time', 'lev', 'npt_v'], S_v),
            'Q_level': (['time', 'lev'], Q_level),
            'lon_u': (['npt_u'], lon_u),
            'lat_u': (['npt_u'], lat_u),
            'lon_v': (['npt_v'], lon_v),
            'lat_v': (['npt_v'], lat_v),
        },
        coords={
            'time': udata.time,
            'lev': udata.lev,
            'npt_u': np.arange(npt_u),
            'npt_v': np.arange(npt_v),
        }
    )
    
    # Add metadata
    ds_section['qn_u'].attrs = {'long_name': 'Normal volume flux at U-face section points', 'units': 'm³/s'}
    ds_section['qn_v'].attrs = {'long_name': 'Normal volume flux at V-face section points', 'units': 'm³/s'}
    # σ0/σ2 is insensitive to horizontal half-cell offset. The effect of lateral half-cell offset (~several kilometres) on temperature/salinity is typically < 0.1°C/0.1 PSU in non-frontal zones, with an impact on density of ~0.01 kg/m³. But much efficient than interpolation
    ds_section['T_u'].attrs = {'long_name': 'Temperature at U-face (flux-consistent)', 'units': '°C', 'note': 'No interpolation applied'}
    ds_section['T_v'].attrs = {'long_name': 'Temperature at V-face (flux-consistent)', 'units': '°C', 'note': 'No interpolation applied'}
    ds_section['S_u'].attrs = {'long_name': 'Salinity at U-face (flux-consistent)', 'units': 'PSU'}
    ds_section['S_v'].attrs = {'long_name': 'Salinity at V-face (flux-consistent)', 'units': 'PSU'}
    ds_section['Q_level'].attrs = {'long_name': 'Total flux per level (verification)', 'units': 'm³/s'}
    ds_section['lev'].attrs = {'long_name': 'Depth (T-point center)', 'units': 'm', 'positive': 'down'}
    
    # Save
    outfile = f"{path_save}{strait}_section_detailed_{model}_{time_start}-{time_end}.nc"
    ds_section.to_netcdf(outfile)
    print(f"✓ Saved detailed section flux: {outfile}")



def transports(product,strait,model,time_start,time_end,file_u,file_v,file_t,file_z, file_zu=None, file_zv=None,mesh_dxv=0, mesh_dyu=0,coords=0,set_latlon=False,lon_p=0,lat_p=0,file_s='',file_sic='',file_sit='',Arakawa='',rho=1026,cp=3996, Tref=0,path_save='',path_indices='',path_mesh='',saving=True):

    '''Calculation of Transports using line integration

    INPUT Parameters:
    product (str): volume, heat, salt or ice
    strait (str): desired oceanic strait, either pre-defined from indices file or new
    model (str): desired CMIP6 model or reanalysis
    time_start (str or int): starting year
    time_end (str or int): ending year
    file_u (str OR ): path + filename(s) of u field(s); use ice velocities (ui) for ice transports; (multiple files possible, use *; must be possible to combine files over time coordinate)
    file_v (str): path + filename(s) of v field(s); use ice velocities (vi) for ice transports; (multiple files possible, use *)
    file_t (str): path + filename(s) of temperature field(s); (multiple files possible, use *)
    file_z (str): path + filename(s) of cell thickness field(s); (multiple files possible, use *)

    OPTIONAL:
    mesh_dxu/mesh_dyv (array): arrays containing the exact grid cell dimensions at northern and eastern grid cell faces of u and v cells (dxv and dyu); if not supplied will be calculated
    coords (tuple): coordinates for strait, if not pre-defined: (latitude_start,longitude_start,latitude_end,longitude_end)
    set_latlon: set True if you wish to pass arrays of latitudes and longitudes
    lon (array): longitude coordinates for strait, if not pre-defined. (range -180 to 180; same length as lat needed!)
    lat (array): latitude coordinates for strait, if not pre-defined. (range -90 to 90; same length as lon needed!)
    file_s (str): only needed for salinity transports; path + filename(s) of salinity field(s); (multiple files possible, use *)
    file_sic (str): only needed for ice transports; path + filename(s) of sea ice concentration field(s); (multiple files possible, use *)
    file_sit (str): only needed for ice transports; path + filename(s) of sea ice thickness field(s); (multiple files possible, use *)
    Arakawa (str): Arakawa-A, Arakawa-B or Arakawa-C; only needed if automatic check fails
    rho (int or array): default = 1026 kg/m3
    cp (int or array): default = 3996 J/(kgK)
    Tref (int or array): default = 0°C
    path_save (str): path to save transport data
    path_indices (str): path to save indices data
    path_mesh (str): path to save mesh data


    RETURNS:
    volume, heat, salt or ice transports through specified strait for specified model

    '''


    if product == 'ice':
        partial_func = partial(prepro._preprocess1i)
    else:
        partial_func = partial(prepro._preprocess1)


    try:
        indices=xa.open_dataset(path_indices+model+'_'+strait+'_indices.nc')
    except OSError:
        print('calc indices')
        print('read and load files for indices')
        # Use only first file for indices calculation
        if isinstance(file_t, list):
            file_t_first = [file_t[0]]
            file_u_first = [file_u[0]]
            file_v_first = [file_v[0]]
            file_sit_first = [file_sit[0]] if file_sit else file_sit
        else:
            file_t_first = file_t
            file_u_first = file_u
            file_v_first = file_v
            file_sit_first = file_sit

        if product == 'ice':
            ti = xa.open_mfdataset(file_sit_first, preprocess=partial_func).isel(time=0)
        else:
            ti = xa.open_mfdataset(file_t_first, preprocess=partial_func).isel(time=0)

        ui = xa.open_mfdataset(file_u_first, preprocess=partial_func).isel(time=0)
        vi = xa.open_mfdataset(file_v_first, preprocess=partial_func).isel(time=0)
        try:
            with ProgressBar():
                ti=ti.load()
                ui=ui.load()
                vi=vi.load()
        except NameError:
            ti=ti.load()
            ui=ui.load()
            vi=vi.load()
        indices,line = check_availability_indices(ti,strait,model,coords,lon_p,lat_p,set_latlon)
        i2=indices.indices.where(indices.indices!=0)
        try:
            if product == 'ice':
                plt.pcolormesh(ti.sithick/ti.sithick,cmap='tab20c')
            else:
                plt.pcolormesh((ti.thetao/ti.thetao),cmap='tab20c')
            plt.scatter(i2[:,2],i2[:,3],color='tab:red',s=0.1,marker='x')
            plt.scatter(i2[:,0],i2[:,1],color='tab:red',s=0.1,marker='x')
            plt.title(model+'_'+strait,fontsize=14)
            plt.ylabel('y',fontsize=14)
            plt.xlabel('x',fontsize=14)
            plt.savefig(path_save+strait+'_'+model+'_indices.png')
            plt.close()
            print(f" Saved indices plot: {path_save+strait+'_'+model+'_indices.png'}")
        except:
            print('skipping Plot')
        out_u,out_v,out_u_vz = prepare_indices(indices)
        if product == 'ice':
            func.check_indices(indices,out_u,out_v,ti,ui,vi,strait,model,path_save)
        else:
            func.check_indices(indices,out_u,out_v,ti,ui,vi,strait,model,path_save)
        if saving == True:
            indices.to_netcdf(path_indices+model+'_'+strait+'_indices.nc')

    out_u,out_v,out_u_vz = prepare_indices(indices)
    if Arakawa in ['Arakawa-A','Arakawa-B','Arakawa-C']:
        grid=Arakawa
    elif Arakawa == '':
        try:
            file = open(path_mesh+model+'grid.txt', 'r')
            grid= file.read()
        except OSError:
            try:
                grid = func.check_Arakawa(ui,vi,ti,model)
                if saving == True:
                    with open(path_mesh+model+'grid.txt', 'w') as f:
                        f.write(grid)
            except NameError:
                print('read and load files for grid check')
                if isinstance(file_t, list):
                    file_t_first = [file_t[0]]
                    file_u_first = [file_u[0]]
                    file_v_first = [file_v[0]]
                    file_sit_first = [file_sit[0]] if file_sit else file_sit
                else:
                    file_t_first = file_t
                    file_u_first = file_u
                    file_v_first = file_v
                    file_sit_first = file_sit
                ti = xa.open_mfdataset(file_t_first, preprocess=partial_func).isel(time=0)
                ui = xa.open_mfdataset(file_u_first, preprocess=partial_func).isel(time=0)
                vi = xa.open_mfdataset(file_v_first, preprocess=partial_func).isel(time=0)
                try:
                    with ProgressBar():
                        ti=ti.load()
                        ui=ui.load()
                        vi=vi.load()
                except NameError:
                    ti=ti.load()
                    ui=ui.load()
                    vi=vi.load()
                grid = func.check_Arakawa(ui,vi,ti,model)
                if saving == True:
                    with open(path_mesh+model+'grid.txt', 'w') as f:
                        f.write(grid)
    else:
        print('grid not known')
        sys.exit()


    min_x=np.nanmin((min(out_u[:,0],default=np.nan),min(out_v[:,0],default=np.nan)))
    max_x=np.nanmax((max(out_u[:,0],default=np.nan),max(out_v[:,0],default=np.nan)))
    min_y=np.nanmin((min(out_u[:,1],default=np.nan),min(out_v[:,1],default=np.nan)))
    max_y=np.nanmax((max(out_u[:,1],default=np.nan),max(out_v[:,1],default=np.nan)))

    if min_x == -1:
        min_x = 0
        max_x = max_x + 1
    partial_func2 = partial(prepro._preprocess2,lon_bnds=(int(min_x)-1,int(max_x)+1),lat_bnds=(int(min_y)-1,int(max_y)+1))
    try:
        mu=xa.open_mfdataset(path_mesh+'mesh_dyu_'+model+'.nc', preprocess=partial_func2)
        mv=xa.open_mfdataset(path_mesh+'mesh_dxv_'+model+'.nc', preprocess=partial_func2)
    except:
        if isinstance(mesh_dxv, xa.Dataset):
            original_var_name = list(mesh_dxv.data_vars)[0]
            mesh_dxv.rename({original_var_name: "dxv"}).to_netcdf(path_mesh+'mesh_dxv_'+model+'.nc')
            mesh_dyu.rename({original_var_name: "dyu"}).to_netcdf(path_mesh+'mesh_dyu_'+model+'.nc')
            mu=xa.open_mfdataset(path_mesh+'mesh_dyu_'+model+'.nc', preprocess=partial_func2)
            mv=xa.open_mfdataset(path_mesh+'mesh_dxv_'+model+'.nc', preprocess=partial_func2)
        elif isinstance(mesh_dxv, xa.DataArray):
            mesh_dxv.to_dataset(name="dxv").to_netcdf(path_mesh+'mesh_dxv_'+model+'.nc')
            mesh_dyu.to_dataset(name='dyu').to_netcdf(path_mesh+'mesh_dyu_'+model+'.nc')
            mu=xa.open_mfdataset(path_mesh+'mesh_dyu_'+model+'.nc', preprocess=partial_func2)
            mv=xa.open_mfdataset(path_mesh+'mesh_dxv_'+model+'.nc', preprocess=partial_func2)
        else:       
            print('meshes not supplied or not DataArray/DataSet: calc horizontal meshes')
            try:
                mu,mv = prepro.calc_dxdy(model,ui,vi,path_mesh)
                mu=xa.open_mfdataset(path_mesh+'mesh_dyu_'+model+'.nc', preprocess=partial_func2)
                mv=xa.open_mfdataset(path_mesh+'mesh_dxv_'+model+'.nc', preprocess=partial_func2) 
            except NameError:
                print('read and load files for mesh')
                # ui = xa.open_mfdataset(file_u, preprocess=partial_func).isel(time=0)
                # vi = xa.open_mfdataset(file_v, preprocess=partial_func).isel(time=0)
                if isinstance(file_t, list):
                    file_u_first = [file_u[0]]
                    file_v_first = [file_v[0]]
                else:
                    file_u_first = file_u
                    file_v_first = file_v
                ui = xa.open_mfdataset(file_u_first, preprocess=partial_func).isel(time=0)
                vi = xa.open_mfdataset(file_v_first, preprocess=partial_func).isel(time=0)
                
                try:
                    with ProgressBar():
                        ui=ui.load()
                        vi=vi.load()
                except NameError:
                    ui=ui.load()
                    vi=vi.load()
                mu,mv = prepro.calc_dxdy(model,ui,vi,path_mesh)
                mu=xa.open_mfdataset(path_mesh+'mesh_dyu_'+model+'.nc', preprocess=partial_func2)
                mv=xa.open_mfdataset(path_mesh+'mesh_dxv_'+model+'.nc', preprocess=partial_func2) 


    print('read t, u and v fields')
    partial_func = partial(prepro._preprocess2,lon_bnds=(int(min_x)-1,int(max_x)+1),lat_bnds=(int(min_y)-1,int(max_y)+1))
    t = xa.open_mfdataset(file_t, preprocess=partial_func,chunks={'time':1})
    u = xa.open_mfdataset(file_u, preprocess=partial_func,chunks={'time':1})
    v = xa.open_mfdataset(file_v, preprocess=partial_func,chunks={'time':1})
    if product == 'ice':
        sit = xa.open_mfdataset(file_sit, preprocess=partial_func,chunks={'time':1})
        sic = xa.open_mfdataset(file_sic, preprocess=partial_func,chunks={'time':1})
    if 'time' in t.dims and t.dims['time'] > 1:
        t=t.sel(time=slice(str(time_start),str(time_end)))
        u=u.sel(time=slice(str(time_start),str(time_end)))
        v=v.sel(time=slice(str(time_start),str(time_end)))
        if product == 'ice':
            sit=sit.sel(time=slice(str(time_start),str(time_end)))
            sic=sic.sel(time=slice(str(time_start),str(time_end)))
    elif 'time' not in t.dims:
        t=t.expand_dims(dim={"time": 1})
        u=u.expand_dims(dim={"time": 1})
        v=v.expand_dims(dim={"time": 1})
        if product == 'ice':
            sit=sit.expand_dims(dim={"time": 1})
            sic=sic.expand_dims(dim={"time": 1})

    if product in ['volume','heat','salt']:
        deltaz = xa.open_mfdataset(file_z, preprocess=partial_func,chunks={'time':1})[['thkcello']]
        if 'time' in deltaz.dims:
            deltaz=deltaz.sel(time=slice(str(time_start),str(time_end)))
    mu=mu.sel(x=slice(int(min_x)-1,int(max_x)+1),y=slice(int(min_y)-1,int(max_y)+1)).load()
    mv=mv.sel(x=slice(int(min_x)-1,int(max_x)+1),y=slice(int(min_y)-1,int(max_y)+1)).load()

    print('load t, u and v fields')
    try:
        with ProgressBar():
            t=t.load()
            u=u.load()
            v=v.load()
            if product in ['volume','heat','salt']:
                deltaz=deltaz.load()
    except NameError:
        t=t.load()
        u=u.load()
        v=v.load()
        if product in ['volume','heat','salt']:
            deltaz=deltaz.load()

    if product in ['volume','heat','salt']:
        # Check if file_zu and file_zv are provided
        if file_zu and file_zv:
            try:
                dzu3 = xa.open_mfdataset(file_zu, preprocess=partial_func, chunks={'time': 1})['thkcello']
                dzv3 = xa.open_mfdataset(file_zv, preprocess=partial_func, chunks={'time': 1})['thkcello']
            except Exception as e:
                print(f"Error opening file_zu or file_zv: {e}")
                dzu3, dzv3 = func.calc_dz_faces(deltaz, grid, model, path_mesh)
        else:
            dzu3, dzv3 = func.calc_dz_faces(deltaz, grid, model, path_mesh)
        
    trans = xa.Dataset({'tot_'+product+'_flux':(('time'),np.array(np.zeros(t.time.size)))},coords=dict(time=t.time))
    sign_v=[]
    indi=indices.indices[:,2][indices.indices[:,3]!=0]
    for ind in range(len(indi)-1):
        if indi[ind]<indi[ind+1]:
            sign_v=np.append(sign_v,1)
        elif indi[ind]>=indi[ind+1]:
            if indi[ind+1] in [1,0,-1]:
                sign_v=np.append(sign_v,1)
            else:
                sign_v=np.append(sign_v,-1)
    try:
        sign_v=np.append(sign_v,sign_v[-1])
    except IndexError:
        pass


    print(' ...calculating transport')
    trans_arr = []
    udata = u.uo
    vdata = v.vo
    Tdata = t

    # if product == 'salt':
    Sdata = xa.open_mfdataset(file_s, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))


    if product in ['volume','heat','salt']:
        udata,vdata2,dzu3,dzv3,mu2,mv2 = func.transform_Arakawa(grid,mu,mv,deltaz,dzu3,dzv3,udata,vdata)


    if product == 'volume':
        print('calc u')
        udata=udata*mu2.dyu.values*dzu3.values
        print('calc v')
        vdata=vdata*mv2.dxv.values*dzv3.values


    if product == 'heat':
        print('rolling T')
        Tudata = func.interp_TS(Tdata.thetao,'x')
        Tvdata = func.interp_TS(Tdata.thetao,'y')
        print('calc u')
        udata=udata*mu2.dyu.values*dzu3.values*(Tudata.values-Tref)
        print('calc v')
        vdata=vdata*mv2.dxv.values*dzv3.values*(Tvdata.values-Tref)



    if product == 'salt':
        print('rolling S')
        Sudata = func.interp_TS(Sdata.so,'x')
        Svdata = func.interp_TS(Sdata.so,'y')
        print('calc u') 
        udata=udata*mu2.dyu.values*dzu3.values*Sudata.values
        print('calc v')
        vdata=vdata*mv2.dxv.values*dzv3.values*Svdata.values

    if product == 'ice':
        if sic.siconc.max() > 1:
            sic['siconc'] = sic.siconc/100
        print('calc u')
        udata=udata*mu.dyu.values*sit.sithick.values*sic.siconc.values
        print('calc v')
        vdata=vdata*mv.dxv.values*sit.sithick.values*sic.siconc.values

    udata = udata.fillna(0.)
    vdata = vdata.fillna(0.)
    print('calc line')
    if product in ['volume','heat','salt']:
        save_section_flux_detailed(
            udata, vdata, Tdata, Sdata, out_u, out_v, out_u_vz, 
            indices, sign_v, min_x, min_y, product, strait, 
            model, time_start, time_end, path_save
        )
    
        udata = udata.sum(dim='lev')
        vdata = vdata.sum(dim='lev')
    datau = xa.Dataset({'inte':(('time','y','x'),udata.data)},coords=({'time':('time',udata.time.data),'x':('x',udata.x.data),'y':('y',udata.y.data)}))
    datav = xa.Dataset({'inte':(('time','y','x'),vdata.data)},coords=({'time':('time',vdata.time.data),'x':('x',vdata.x.data),'y':('y',vdata.y.data)}))
    pointsu = np.zeros(datau.inte.shape)
    data1u = datau.inte

    for l in range(len(out_u)):
        if out_u_vz[l][2] == -1:
            pointsu[:,int(out_u_vz[l,1]-min_y+1),int(out_u_vz[l,0]-min_x+1)] = data1u[:,int(out_u_vz[l,1]-min_y+1),int(out_u_vz[l,0]-min_x+1)] * (-1)
        else:
            pointsu[:,int(out_u_vz[l,1]-min_y+1),int(out_u_vz[l,0]-min_x+1)] = data1u[:,int(out_u_vz[l,1]-min_y+1),int(out_u_vz[l,0]-min_x+1)]

    pointsv = np.zeros(datav.inte.shape)
    data1v = datav.inte

    indi1=indices.indices[:,2][indices.indices[:,3]!=0]
    indi2=indices.indices[:,3][indices.indices[:,3]!=0]

    for m in range(len(indi1)-1):
        pointsv[:,int(indi2[m]-min_y+1),int(indi1[m]-min_x+1)] = data1v[:,int(indi2[m]-min_y+1),int(indi1[m]-min_x+1)] * sign_v[m]

    vp = xa.Dataset({'v':(('time','y','x'),pointsv)},coords=datav.coords)
    up = xa.Dataset({'u':(('time','y','x'),pointsu)},coords=datau.coords)
    ges_l = datau.copy()
    ges_l['inte'] = vp['v'] + up['u']
    if product == 'heat':
        ges_l['inte'] = ges_l['inte'] * rho * cp
    elif product == 'salt':
        ges_l['inte'] = ges_l['inte'] * rho
    # ges_l.to_netcdf(model+'_'+strait+'_test.nc')
    summ = ges_l.sum(dim=['x','y'])
    summ = summ.inte
    trans_arr = np.append(trans_arr,summ)

    trans[model]= (['time'],trans_arr)
    trans = trans.drop_vars('tot_'+product+'_flux')
    trans.to_netcdf(path_save+strait+'_'+product+'_'+model+'_'+str(time_start)+'-'+str(time_end)+'.nc')
    print('- save net transport (LM):', path_save+strait+'_'+product+'_'+model+'_'+str(time_start)+'-'+str(time_end)+'.nc')
    

    return trans
