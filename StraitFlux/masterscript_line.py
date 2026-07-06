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
from StraitFlux.indices import available_sections as _available_sections
from StraitFlux.indices import check_availability_indices, prepare_indices




def transports(product,strait,model,time_start,time_end,file_u,file_v,file_t,file_z, file_zu=None, file_zv=None,mesh_dxv=0, mesh_dyu=0,coords=0,set_latlon=False,lon_p=0,lat_p=0,file_s='',file_sic='',file_sit='',file_tracer='',tracer_var='tracer',tracer_units='',Arakawa='',rho=1026,cp=3996, Tref=0,path_save='',path_indices='',path_mesh='',saving=True,user_rename_dict=None):

    '''Calculation of Transports using line integration

    INPUT Parameters:
    product (str): volume, heat, salt, tracer or ice
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
    file_tracer (str): only needed for tracer transports; path + filename(s) of tracer field(s); (multiple files possible, use *)
    tracer_var (str): only needed for tracer transports; variable name of the tracer after preprocessing
    tracer_units (str): optional units of the tracer concentration; output units are tracer_units * m3 s-1
    Arakawa (str): Arakawa-A, Arakawa-B or Arakawa-C; only needed if automatic check fails
    rho (int or array): default = 1026 kg/m3
    cp (int or array): default = 3996 J/(kgK)
    Tref (int or array): default = 0°C
    path_save (str): path to save transport data
    path_indices (str): path to save indices data
    path_mesh (str): path to save mesh data


    RETURNS:
    volume, heat, salt, tracer or ice transports through specified strait for specified model

    '''


    if product == 'ice':
        partial_func = partial(prepro._preprocess1i, user_rename_dict=user_rename_dict)
    else:
        partial_func = partial(prepro._preprocess1, user_rename_dict=user_rename_dict)


    try:
        indices=xa.open_dataset(path_indices+model+'_'+strait+'_indices.nc')
    except OSError:
        print('calc indices')
        print('read and load files for indices')
        if product == 'ice':
            ti = xa.open_mfdataset(file_sit, preprocess=partial_func).isel(time=0)
        else:
            ti = xa.open_mfdataset(file_t, preprocess=partial_func).isel(time=0)
        ui = xa.open_mfdataset(file_u, preprocess=partial_func).isel(time=0)
        vi = xa.open_mfdataset(file_v, preprocess=partial_func).isel(time=0)
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
                ti = xa.open_mfdataset(file_t, preprocess=partial_func).isel(time=0)
                ui = xa.open_mfdataset(file_u, preprocess=partial_func).isel(time=0)
                vi = xa.open_mfdataset(file_v, preprocess=partial_func).isel(time=0)
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
        raise ValueError(
            "grid not known. Please pass "
            "Arakawa='Arakawa-A', 'Arakawa-B', or 'Arakawa-C'."
        )


    min_x=np.nanmin((min(out_u[:,0],default=np.nan),min(out_v[:,0],default=np.nan)))
    max_x=np.nanmax((max(out_u[:,0],default=np.nan),max(out_v[:,0],default=np.nan)))
    min_y=np.nanmin((min(out_u[:,1],default=np.nan),min(out_v[:,1],default=np.nan)))
    max_y=np.nanmax((max(out_u[:,1],default=np.nan),max(out_v[:,1],default=np.nan)))

    if min_x == -1:
        min_x = 0
        max_x = max_x + 1
    partial_func2 = partial(prepro._preprocess2,lon_bnds=(int(min_x)-1,int(max_x)+1),lat_bnds=(int(min_y)-1,int(max_y)+1),user_rename_dict=user_rename_dict)
    if (
        not isinstance(mesh_dxv, (xa.Dataset, xa.DataArray))
        or not isinstance(mesh_dyu, (xa.Dataset, xa.DataArray))
    ):
        if "ui" not in locals() or "vi" not in locals():
            print("read and load files for mesh calculation")
            ui = xa.open_mfdataset(file_u, preprocess=partial_func).isel(time=0)
            vi = xa.open_mfdataset(file_v, preprocess=partial_func).isel(time=0)
            try:
                with ProgressBar():
                    ui = ui.load()
                    vi = vi.load()
            except NameError:
                ui = ui.load()
                vi = vi.load()    
    
    
    mu, mv = func._load_or_prepare_mesh(
        model=model,
        mesh_dxv=mesh_dxv,
        mesh_dyu=mesh_dyu,
        ui=ui if "ui" in locals() else None,
        vi=vi if "vi" in locals() else None,
        path_mesh=path_mesh,
        preprocess_func=None,
        saving=saving,
    )


    print('read t, u and v fields')
    partial_func = partial(prepro._preprocess2,lon_bnds=(int(min_x)-1,int(max_x)+1),lat_bnds=(int(min_y)-1,int(max_y)+1),user_rename_dict=user_rename_dict)
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

    if product in ['volume','heat','salt','tracer']:
        deltaz_ds = xa.open_mfdataset(file_z, preprocess=partial_func,chunks={'time':1})
        deltaz = _get_vertical_thickness_dataset(deltaz_ds, ['thkcello'], label='T-cell thickness')
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
            if product in ['volume','heat','salt','tracer']:
                deltaz=deltaz.load()
    except NameError:
        t=t.load()
        u=u.load()
        v=v.load()
        if product in ['volume','heat','salt','tracer']:
            deltaz=deltaz.load()

    if product in ['volume','heat','salt','tracer']:
        # Check if file_zu and file_zv are provided
        if file_zu and file_zv:
            try:
                dzu_ds = xa.open_mfdataset(file_zu, preprocess=partial_func, chunks={'time': 1})
                dzv_ds = xa.open_mfdataset(file_zv, preprocess=partial_func, chunks={'time': 1})
                dzu3 = _get_vertical_thickness_var(dzu_ds, ['thkcello_u', 'thkcello'], label='U-face thickness')
                dzv3 = _get_vertical_thickness_var(dzv_ds, ['thkcello_v', 'thkcello'], label='V-face thickness')
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

    if product == 'salt':
        Sdata = xa.open_mfdataset(file_s, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))

    if product == 'tracer':
        if file_tracer in ['', None]:
            raise ValueError("product='tracer' requires file_tracer to be supplied.")
        Cdata = xa.open_mfdataset(file_tracer, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))
        if tracer_var in Cdata:
            Cfield = Cdata[tracer_var]
        elif len(Cdata.data_vars) == 1:
            Cfield = Cdata[list(Cdata.data_vars)[0]]
        else:
            raise KeyError(
                f"Could not find tracer_var='{tracer_var}' in tracer file. "
                f"Available variables are {list(Cdata.data_vars)}."
            )


    if product in ['volume','heat','salt','tracer']:
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

    if product == 'tracer':
        print('rolling tracer')
        Cudata = func.interp_TS(Cfield,'x')
        Cvdata = func.interp_TS(Cfield,'y')
        print('calc u')
        udata=udata*mu2.dyu.values*dzu3.values*Cudata.values
        print('calc v')
        vdata=vdata*mv2.dxv.values*dzv3.values*Cvdata.values

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
    if product in ['volume','heat','salt','tracer']:
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
    #ges_l.to_netcdf(model+'_'+strait+'_test.nc')
    summ = ges_l.sum(dim=['x','y'])
    summ = summ.inte
    trans_arr = np.append(trans_arr,summ)

    trans[model]= (['time'],trans_arr)
    if product == 'tracer':
        trans[model].attrs['tracer_var'] = tracer_var
        if tracer_units not in ['', None]:
            trans[model].attrs['units'] = tracer_units + ' m3 s-1'
    trans = trans.drop_vars('tot_'+product+'_flux')
    trans.to_netcdf(path_save+strait+'_'+product+'_'+model+'_'+str(time_start)+'-'+str(time_end)+'.nc')

    return trans








# -----------------------------------------------------------------------------
# Extended zig-zag line diagnostics: volume, heat, MOC and optional density space
# -----------------------------------------------------------------------------



def _normalize_monthly_time(obj, normalize_time="MS", time_dim="time"):
    """Normalize monthly time coordinates so variables with different monthly
    timestamp conventions align correctly.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Object whose time coordinate should be normalized.
    normalize_time : {"MS", "ME", None}
        "MS" sets all timestamps to month start.
        "ME" sets all timestamps to month end.
        None/False/"" leaves the object unchanged.
    time_dim : str
        Name of the time dimension/coordinate.
    """
    if obj is None or normalize_time in [None, False, ""]:
        return obj
    if time_dim not in obj.dims and time_dim not in obj.coords:
        return obj

    time = pd.to_datetime(obj[time_dim].values)

    if normalize_time == "MS":
        new_time = time.to_period("M").to_timestamp(how="start")
    elif normalize_time == "ME":
        new_time = time.to_period("M").to_timestamp(how="end")
    else:
        raise ValueError("normalize_time must be 'MS', 'ME', None, False, or ''.")

    return obj.assign_coords({time_dim: new_time})



def _prepare_vertical_metric_time(obj, template, time_start, time_end, normalize_time="MS", time_dim="time"):
    """Prepare vertical thickness time coordinates.

    Static mesh files often contain a dummy singleton time coordinate
    (for example time_counter=0 -> 1970-01-01). In that case, dropping
    the time dimension is correct: the metric should broadcast over the
    velocity/temperature time axis.

    Time-varying VVL metrics, on the other hand, should keep their time
    dimension, be selected to the requested interval, and be normalized
    consistently with the model fields.
    """
    if obj is None:
        return obj

    if time_dim not in obj.dims:
        return obj

    # Singleton mesh time: keep it only if it genuinely matches a singleton
    # calculation period. Otherwise treat it as static geometry and drop it.
    if obj.sizes[time_dim] == 1:
        try:
            obj_time = pd.to_datetime(obj[time_dim].values)
            start = pd.to_datetime(str(time_start))
            end = pd.to_datetime(str(time_end))
            inside_requested_period = (obj_time[0] >= start) and (obj_time[0] <= end)
        except Exception:
            inside_requested_period = False

        template_has_same_single_time = (
            template is not None
            and time_dim in template.dims
            and template.sizes[time_dim] == 1
            and inside_requested_period
        )

        if template_has_same_single_time:
            obj = _normalize_monthly_time(obj, normalize_time=normalize_time, time_dim=time_dim)
            return obj.assign_coords({time_dim: template[time_dim]})

        return obj.isel({time_dim: 0}, drop=True)

    # True time-varying metric. Select and normalize like the prognostic fields.
    obj = func._select_time_if_present(obj, time_start, time_end)
    obj = _normalize_monthly_time(obj, normalize_time=normalize_time, time_dim=time_dim)

    # If the number of selected metric times matches the template, use the
    # template timestamps to avoid harmless timestamp convention mismatches.
    if template is not None and time_dim in template.dims and time_dim in obj.dims:
        if obj.sizes[time_dim] == template.sizes[time_dim]:
            obj = obj.assign_coords({time_dim: template[time_dim]})

    return obj


def _get_vertical_thickness_var(obj, candidates, label="vertical thickness"):
    """Return the first available vertical thickness variable from a Dataset/DataArray.

    This supports both the new explicit names
    ``thkcello`` / ``thkcello_u`` / ``thkcello_v`` and older separate files
    where the only variable is still called ``thkcello``.
    """
    if isinstance(obj, xa.DataArray):
        return obj

    if not isinstance(obj, xa.Dataset):
        raise TypeError(f"{label} must be an xarray Dataset or DataArray.")

    for name in candidates:
        if name in obj:
            return obj[name]

    if len(obj.data_vars) == 1:
        return obj[list(obj.data_vars)[0]]

    raise KeyError(
        f"Could not find {label}. Tried {candidates}. "
        f"Available variables are {list(obj.data_vars)}."
    )


def _get_vertical_thickness_dataset(obj, candidates, label="vertical thickness"):
    da = _get_vertical_thickness_var(obj, candidates, label=label)
    return da.to_dataset(name="thkcello")


def _overturning_core(
    strait,
    model,
    time_start,
    time_end,
    indices,
    grid,
    t,
    u,
    v,
    deltaz,
    mu,
    mv,
    dzu3,
    dzv3,
    min_x,
    min_y,
    sdata=None,
    rho=1026.0,
    cp=3996.0,
    Tref=0.0,
    apply_barotropic_correction=False,
    transport_target_m3s=None,
    sigmin=23.0,
    sigstp=0.02,
    nbins=270,
    salt_is_SA=False,
    temp_is_CT=False,
    return_diagnostics=False,
    saving=True,
    path_save="",
):
    """
    Shared implementation for path-based and Dataset-based overturning diagnostics.

    Expects all input fields to be preprocessed, subset to the section domain,
    time-selected, and loaded if desired. The public wrappers are responsible for
    opening/preparing data; this function only performs the physical diagnostics.
    """

    has_salinity = sdata is not None

    # Transform A/B grids to C-grid face velocities/thicknesses, as in transports().
    udata, vdata, dzu3, dzv3, mu2, mv2 = func.transform_Arakawa(
        grid, mu, mv, deltaz, dzu3, dzv3, u.uo, v.vo
    )

    # Convert global StraitFlux indices to local subset positions.
    indices_local = func.shift_indices_to_local(indices, min_x, min_y, pad=1)
    arr = indices_local.indices.values.copy()

    iu = arr[:, 0].astype(int)
    ju = arr[:, 1].astype(int)
    iv = arr[:, 2].astype(int)
    jv = arr[:, 3].astype(int)
    sign_u = arr[:, 4].astype(float)

    u_mask = ~(np.all(np.isnan(arr[:, 0:2]) | (arr[:, 0:2] == 0), axis=1))
    v_mask = ~(np.all(np.isnan(arr[:, 2:4]) | (arr[:, 2:4] == 0), axis=1))

    sign_v = func.calc_sign_v(indices)

    Tu = func.interp_TS(t.thetao, "x") - Tref
    Tv = func.interp_TS(t.thetao, "y") - Tref
    if has_salinity:
        Su = func.interp_TS(sdata.so, "x")
        Sv = func.interp_TS(sdata.so, "y")

    # Representative section position for GSW pressure / Absolute Salinity conversion.
    if np.any(u_mask):
        first_i = int(iu[u_mask][0])
        first_j = int(ju[u_mask][0])
        section_lon = float(u.lon.isel(y=first_j, x=first_i).values)
        section_lat = float(u.lat.isel(y=first_j, x=first_i).values)
    elif np.any(v_mask):
        first_i = int(iv[v_mask][0])
        first_j = int(jv[v_mask][0])
        section_lon = float(v.lon.isel(y=first_j, x=first_i).values)
        section_lat = float(v.lat.isel(y=first_j, x=first_i).values)
    else:
        raise ValueError("No valid section elements found in index file.")

    transport_list = []
    temp_list = []
    salt_list = []
    area_list = []
    face_type = []

    # u-face contributions
    for i, j, sg in zip(iu[u_mask], ju[u_mask], sign_u[u_mask]):
        area = func._strip_nondim_coords(mu2.dyu.isel(y=j, x=i) * dzu3.isel(y=j, x=i))
        tr = func._strip_nondim_coords(sg * udata.isel(y=j, x=i) * area)
        transport_list.append(tr)
        temp_list.append(func._strip_nondim_coords(Tu.isel(y=j, x=i)))
        area_list.append(area)
        face_type.append("u")
        if has_salinity:
            salt_list.append(func._strip_nondim_coords(Su.isel(y=j, x=i)))

    # v-face contributions
    for i, j, sg in zip(iv[v_mask], jv[v_mask], sign_v):
        area = func._strip_nondim_coords(mv2.dxv.isel(y=j, x=i) * dzv3.isel(y=j, x=i))
        tr = func._strip_nondim_coords(sg * vdata.isel(y=j, x=i) * area)
        transport_list.append(tr)
        temp_list.append(func._strip_nondim_coords(Tv.isel(y=j, x=i)))
        area_list.append(area)
        face_type.append("v")
        if has_salinity:
            salt_list.append(func._strip_nondim_coords(Sv.isel(y=j, x=i)))

    if len(transport_list) == 0:
        raise ValueError("No valid section elements found in index file.")

    transport_raw = xa.concat(transport_list, dim="section_element")
    temp_face = xa.concat(temp_list, dim="section_element")
    area_face = xa.concat(area_list, dim="section_element")
    if has_salinity:
        salt_face = xa.concat(salt_list, dim="section_element")

    sec_el = np.arange(transport_raw.sizes["section_element"])
    face_type_arr = np.asarray(face_type)
    for da in [transport_raw, temp_face, area_face]:
        da.coords["section_element"] = sec_el
        da.coords["face_type"] = ("section_element", face_type_arr)
    if has_salinity:
        salt_face.coords["section_element"] = sec_el
        salt_face.coords["face_type"] = ("section_element", face_type_arr)

    transport_raw = func._transpose_section(transport_raw)
    temp_face = func._transpose_section(temp_face)
    area_face = func._transpose_section(area_face)

    if has_salinity:
        salt_face = func._transpose_section(salt_face)

    if "time" in area_face.dims and area_face.sizes["time"] == 1:
        area_face = area_face.isel(time=0, drop=True)

    wet_bool = np.isfinite(transport_raw) & np.isfinite(area_face) & (area_face > 0)

    transport_raw = transport_raw.where(wet_bool, 0.0)
    area_face = area_face.where(wet_bool, 0.0)
    temp_face = temp_face.where(wet_bool)

    if has_salinity:
        salt_face = salt_face.where(wet_bool)

    # Optional barotropic correction.
    sum_dims = [d for d in ["lev", "section_element"] if d in transport_raw.dims]
    volume_raw_m3s = transport_raw.sum(dim=sum_dims)
    area_wet_m2 = area_face.sum(dim=sum_dims)

    if apply_barotropic_correction:
        if transport_target_m3s is None:
            target = xa.zeros_like(volume_raw_m3s)
        elif isinstance(transport_target_m3s, xa.DataArray):
            target = transport_target_m3s
            if "time" in volume_raw_m3s.dims and "time" not in target.dims:
                target = target.broadcast_like(volume_raw_m3s)
        else:
            target = xa.DataArray(
                transport_target_m3s,
                coords=volume_raw_m3s.coords,
                dims=volume_raw_m3s.dims,
            )

        v_barotropic = xa.where(area_wet_m2 > 0, (target - volume_raw_m3s) / area_wet_m2, 0.0)
        transport = transport_raw + v_barotropic.broadcast_like(transport_raw) * area_face
    else:
        target = xa.zeros_like(volume_raw_m3s)
        v_barotropic = xa.zeros_like(volume_raw_m3s)
        transport = transport_raw

    # Depth-space diagnostics.
    volume_transport_Sv = (transport.sum(dim=sum_dims) / 1e6).rename("volume_transport_Sv")

    V_depth = transport.sum(dim="section_element")
    A_depth = area_face.sum(dim="section_element")
    T_area_depth = (temp_face * area_face).sum(dim="section_element")

    overturning_depth_layer_Sv = (V_depth / 1e6).rename("overturning_depth_layer_Sv")
    overturning_depth_streamfunction_Sv = (
        overturning_depth_layer_Sv.cumsum(dim="lev")
    ).rename("overturning_depth_streamfunction_Sv")
    MOC_depth_Sv = overturning_depth_streamfunction_Sv.max(dim="lev").rename("MOC_depth_Sv")

    heat_total_PW = (rho * cp * (transport * temp_face).sum(dim=sum_dims) / 1e15).rename("heat_total_PW")
    heat_overturning_depth_PW = (
        rho * cp * xa.where(A_depth > 0, V_depth * T_area_depth / A_depth, 0.0).sum(dim="lev") / 1e15
    ).rename("heat_overturning_depth_PW")
    heat_gyre_depth_PW = (heat_total_PW - heat_overturning_depth_PW).rename("heat_gyre_depth_PW")

    out_vars = dict(
        volume_transport_Sv=volume_transport_Sv,
        overturning_depth_layer_Sv=overturning_depth_layer_Sv,
        overturning_depth_streamfunction_Sv=overturning_depth_streamfunction_Sv,
        MOC_depth_Sv=MOC_depth_Sv,
        heat_total_PW=heat_total_PW,
        heat_overturning_depth_PW=heat_overturning_depth_PW,
        heat_gyre_depth_PW=heat_gyre_depth_PW,
    )

    if apply_barotropic_correction:
        out_vars["volume_transport_raw_Sv"] = (volume_raw_m3s / 1e6).rename("volume_transport_raw_Sv")
        out_vars["volume_transport_target_Sv"] = (target / 1e6).rename("volume_transport_target_Sv")
        out_vars["barotropic_correction_ms"] = v_barotropic.rename("barotropic_correction_ms")

    # Optional salinity and density-space diagnostics.
    if has_salinity:
        salt_total = (rho * (transport * salt_face).sum(dim=sum_dims)).rename("salt_total")
        S_area_depth = (salt_face * area_face).sum(dim="section_element")
        salt_overturning_depth = (
            rho * xa.where(A_depth > 0, V_depth * S_area_depth / A_depth, 0.0).sum(dim="lev")
        ).rename("salt_overturning_depth")
        salt_gyre_depth = (salt_total - salt_overturning_depth).rename("salt_gyre_depth")
        out_vars.update(
            dict(
                salt_total=salt_total,
                salt_overturning_depth=salt_overturning_depth,
                salt_gyre_depth=salt_gyre_depth,
            )
        )

        try:
            import gsw
        except ImportError as exc:
            raise ImportError("Density-space MOC requires the gsw package.") from exc

        depth = transport["lev"].astype("float64")
        p_1d = xa.apply_ufunc(
            gsw.p_from_z,
            -depth,
            section_lat,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        p = p_1d.broadcast_like(transport)

        if salt_is_SA:
            SA = salt_face
        else:
            lon_sec = xa.zeros_like(transport.isel(lev=0)) + section_lon
            lat_sec = xa.zeros_like(transport.isel(lev=0)) + section_lat
            SA = xa.apply_ufunc(
                gsw.SA_from_SP,
                salt_face,
                p,
                lon_sec,
                lat_sec,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )

        if temp_is_CT:
            CT = temp_face
        else:
            CT = xa.apply_ufunc(
                gsw.CT_from_t,
                SA,
                temp_face,
                p,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )

        sigma0 = xa.apply_ufunc(
            gsw.sigma0,
            SA,
            CT,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        edges = sigmin + sigstp * np.arange(nbins + 1)
        centers = edges[:-1] + 0.5 * sigstp
        ibin = xa.apply_ufunc(
            np.floor,
            (sigma0 - sigmin) / sigstp,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        ).astype("int32")
        valid_sigma = wet_bool & np.isfinite(sigma0)
        ibin = xa.where(valid_sigma & (ibin >= 0) & (ibin < nbins), ibin, -1)

        A_density = func._bincount_timewise(area_face.where(valid_sigma, 0.0), ibin, centers, nbins, name="area_density_m2")
        V_density = func._bincount_timewise(transport.where(valid_sigma, 0.0), ibin, centers, nbins, name="V_density")
        T_area_density = func._bincount_timewise((CT * area_face).where(valid_sigma, 0.0), ibin, centers, nbins, name="T_area_density")
        S_area_density = func._bincount_timewise((SA * area_face).where(valid_sigma, 0.0), ibin, centers, nbins, name="S_area_density")

        overturning_density_bin_Sv = (V_density / 1e6).rename("overturning_density_bin_Sv")
        overturning_density_streamfunction_Sv = (
            overturning_density_bin_Sv.cumsum(dim="density_bin")
        ).rename("overturning_density_streamfunction_Sv")
        MOC_density_Sv = overturning_density_streamfunction_Sv.max(dim="density_bin").rename("MOC_density_Sv")

        heat_overturning_density_PW = (
            rho * cp * xa.where(A_density > 0, V_density * T_area_density / A_density, 0.0).sum(dim="density_bin") / 1e15
        ).rename("heat_overturning_density_PW")
        heat_gyre_density_PW = (heat_total_PW - heat_overturning_density_PW).rename("heat_gyre_density_PW")

        salt_overturning_density = (
            rho * xa.where(A_density > 0, V_density * S_area_density / A_density, 0.0).sum(dim="density_bin")
        ).rename("salt_overturning_density")
        salt_gyre_density = (salt_total - salt_overturning_density).rename("salt_gyre_density")

        out_vars.update(
            dict(
                overturning_density_bin_Sv=overturning_density_bin_Sv,
                overturning_density_streamfunction_Sv=overturning_density_streamfunction_Sv,
                MOC_density_Sv=MOC_density_Sv,
                heat_overturning_density_PW=heat_overturning_density_PW,
                heat_gyre_density_PW=heat_gyre_density_PW,
                salt_overturning_density=salt_overturning_density,
                salt_gyre_density=salt_gyre_density,
            )
        )

        if return_diagnostics:
            out_vars.update(
                dict(
                    sigma0_face=sigma0.rename("sigma0_face"),
                    CT_face=CT.rename("CT_face"),
                    SA_face=SA.rename("SA_face"),
                    density_bin_index=ibin.rename("density_bin_index"),
                    area_density_m2=A_density.rename("area_density_m2"),
                )
            )

    if return_diagnostics:
        out_vars.update(
            dict(
                transport_face_raw_m3s=transport_raw.rename("transport_face_raw_m3s"),
                transport_face_m3s=transport.rename("transport_face_m3s"),
                area_face_m2=area_face.rename("area_face_m2"),
                temperature_face=temp_face.rename("temperature_face"),
            )
        )
        if has_salinity:
            out_vars["salinity_face"] = salt_face.rename("salinity_face")

    out = xa.Dataset(out_vars)

    units = {
        "volume_transport_Sv": "Sv",
        "volume_transport_raw_Sv": "Sv",
        "volume_transport_target_Sv": "Sv",
        "barotropic_correction_ms": "m s-1",
        "overturning_depth_layer_Sv": "Sv",
        "overturning_depth_streamfunction_Sv": "Sv",
        "MOC_depth_Sv": "Sv",
        "overturning_density_bin_Sv": "Sv",
        "overturning_density_streamfunction_Sv": "Sv",
        "MOC_density_Sv": "Sv",
        "heat_total_PW": "PW",
        "heat_overturning_depth_PW": "PW",
        "heat_gyre_depth_PW": "PW",
        "heat_overturning_density_PW": "PW",
        "heat_gyre_density_PW": "PW",
        "transport_face_raw_m3s": "m3 s-1",
        "transport_face_m3s": "m3 s-1",
        "area_face_m2": "m2",
        "area_density_m2": "m2",
        "sigma0_face": "kg m-3 - 1000",
        "salt_total": "kg s-1",
        "salt_overturning_depth": "kg s-1",
        "salt_gyre_depth": "kg s-1",
        "salt_overturning_density": "kg s-1",
        "salt_gyre_density": "kg s-1",
    }
    for key, unit in units.items():
        if key in out:
            out[key].attrs["units"] = unit

    out.attrs["barotropic_correction_applied"] = str(apply_barotropic_correction)
    out.attrs["density_space_computed"] = str(has_salinity)
    if has_salinity:
        out.attrs["section_lon_for_gsw"] = section_lon
        out.attrs["section_lat_for_gsw"] = section_lat
    out.attrs["method"] = "StraitFlux zig-zag line indices with extended overturning diagnostics"

    if saving:
        suffix = "_adjusted" if apply_barotropic_correction else ""
        out.to_netcdf(
            path_save
            + strait
            + "_overturning_"
            + model
            + "_"
            + str(time_start)
            + "-"
            + str(time_end)
            + suffix
            + ".nc"
        )

    return out



def transports_overturning(
    strait,
    model,
    time_start,
    time_end,
    file_u,
    file_v,
    file_t,
    file_z,
    file_s="",
    file_zu=None,
    file_zv=None,
    mesh_dxv=0,
    mesh_dyu=0,
    coords=0,
    set_latlon=False,
    lon_p=0,
    lat_p=0,
    Arakawa="",
    rho=1026.0,
    cp=3996.0,
    Tref=0.0,
    path_save="",
    path_indices="",
    path_mesh="",
    saving=True,
    apply_barotropic_correction=False,
    transport_target_m3s=None,
    sigmin=23.0,
    sigstp=0.02,
    nbins=270,
    salt_is_SA=False,
    temp_is_CT=False,
    return_diagnostics=False,
    user_rename_dict=None,
    normalize_time="MS",
):
    """
    Extended StraitFlux line calculation using existing zig-zag indices.

    Path-based version. Opens files with preprocess=... so the expensive spatial
    subsetting happens as early as possible.
    """

    partial_func1 = partial(prepro._preprocess1, user_rename_dict=user_rename_dict)

    print("read and load surface fields for indices/grid check")
    ti = xa.open_mfdataset(file_t, preprocess=partial_func1).isel(time=0)
    ui = xa.open_mfdataset(file_u, preprocess=partial_func1).isel(time=0)
    vi = xa.open_mfdataset(file_v, preprocess=partial_func1).isel(time=0)
    ti, ui, vi = func._load_with_progress(ti, ui, vi)

    indices = func._load_or_calculate_indices(
        strait,
        model,
        ti,
        ui,
        vi,
        coords=coords,
        lon_p=lon_p,
        lat_p=lat_p,
        set_latlon=set_latlon,
        path_indices=path_indices,
        path_save=path_save,
        saving=saving,
    )

    grid = func._determine_grid(model, Arakawa, ti, ui, vi, path_mesh=path_mesh, saving=saving)

    out_u, out_v, _ = prepare_indices(indices)
    xmin, xmax, ymin, ymax, min_x, min_y = func.calc_section_bounds(out_u, out_v, pad=1)

    partial_func2 = partial(prepro._preprocess2, lon_bnds=(xmin, xmax), lat_bnds=(ymin, ymax), user_rename_dict=user_rename_dict)

    # Horizontal mesh metrics.
    if (
        not isinstance(mesh_dxv, (xa.Dataset, xa.DataArray))
        or not isinstance(mesh_dyu, (xa.Dataset, xa.DataArray))
    ):
        if "ui" not in locals() or "vi" not in locals():
            print("read and load files for mesh calculation")
            ui = xa.open_mfdataset(file_u, preprocess=partial_func1).isel(time=0)
            vi = xa.open_mfdataset(file_v, preprocess=partial_func1).isel(time=0)
            try:
                with ProgressBar():
                    ui = ui.load()
                    vi = vi.load()
            except NameError:
                ui = ui.load()
                vi = vi.load()    
    
    
    mu, mv = func._load_or_prepare_mesh(
        model=model,
        mesh_dxv=mesh_dxv,
        mesh_dyu=mesh_dyu,
        ui=ui if "ui" in locals() else None,
        vi=vi if "vi" in locals() else None,
        path_mesh=path_mesh,
        preprocess_func=None,
        saving=saving,
    )

    print("read t, u, v and optional s fields")
    t = xa.open_mfdataset(file_t, preprocess=partial_func2, chunks={"time": 1})
    u = xa.open_mfdataset(file_u, preprocess=partial_func2, chunks={"time": 1})
    v = xa.open_mfdataset(file_v, preprocess=partial_func2, chunks={"time": 1})
    deltaz_ds = xa.open_mfdataset(file_z, preprocess=partial_func2, chunks={"time": 1})
    deltaz = _get_vertical_thickness_dataset(deltaz_ds, ["thkcello"], label="T-cell thickness")

    has_salinity = file_s not in ["", None]
    sdata = xa.open_mfdataset(file_s, preprocess=partial_func2, chunks={"time": 1}) if has_salinity else None

    t = func._select_time_if_present(t, time_start, time_end)
    u = func._select_time_if_present(u, time_start, time_end)
    v = func._select_time_if_present(v, time_start, time_end)
    if has_salinity:
        sdata = func._select_time_if_present(sdata, time_start, time_end)

    t = func._ensure_time_axis(t)
    u = func._ensure_time_axis(u, template=t)
    v = func._ensure_time_axis(v, template=t)
    if has_salinity:
        sdata = func._ensure_time_axis(sdata, template=t)

    # Monthly products can use slightly different timestamps for different variables
    # even when they represent the same month. Normalize them before xarray alignment.
    try:
        t = _normalize_monthly_time(t, normalize_time=normalize_time)
        u = _normalize_monthly_time(u, normalize_time=normalize_time)
        v = _normalize_monthly_time(v, normalize_time=normalize_time)
        if has_salinity:
            sdata = _normalize_monthly_time(sdata, normalize_time=normalize_time)
    except:
        print('skipping time normalization!!!')

    # Vertical metrics can be static mesh fields with dummy time_counter=0,
    # or true time-varying VVL fields. Handle both cases explicitly.
    deltaz = _prepare_vertical_metric_time(
        deltaz, template=t, time_start=time_start, time_end=time_end, normalize_time=normalize_time
    )

    mu = mu.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
    mv = mv.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

    print("load final section subset")
    to_load = [t, u, v, deltaz, mu, mv] + ([sdata] if has_salinity else [])
    loaded = func._load_with_progress(*to_load)
    if has_salinity:
        t, u, v, deltaz, mu, mv, sdata = loaded
    else:
        t, u, v, deltaz, mu, mv = loaded

    if file_zu and file_zv:
        try:
            dzu_ds = xa.open_mfdataset(file_zu, preprocess=partial_func2, chunks={"time": 1})
            dzv_ds = xa.open_mfdataset(file_zv, preprocess=partial_func2, chunks={"time": 1})
            dzu3 = _get_vertical_thickness_var(dzu_ds, ["thkcello_u", "thkcello"], label="U-face thickness")
            dzv3 = _get_vertical_thickness_var(dzv_ds, ["thkcello_v", "thkcello"], label="V-face thickness")
            dzu3 = _prepare_vertical_metric_time(
                dzu3, template=u, time_start=time_start, time_end=time_end, normalize_time=normalize_time
            )
            dzv3 = _prepare_vertical_metric_time(
                dzv3, template=v, time_start=time_start, time_end=time_end, normalize_time=normalize_time
            )
            dzu3, dzv3 = func._load_with_progress(dzu3, dzv3)
        except Exception as e:
            print(f"Error opening file_zu or file_zv: {e}")
            dzu3, dzv3 = func.calc_dz_faces(deltaz, grid, model, path_mesh)
    else:
        dzu3, dzv3 = func.calc_dz_faces(deltaz, grid, model, path_mesh)

    dzu3 = _prepare_vertical_metric_time(
        dzu3, template=u, time_start=time_start, time_end=time_end, normalize_time=normalize_time
    )
    dzv3 = _prepare_vertical_metric_time(
        dzv3, template=v, time_start=time_start, time_end=time_end, normalize_time=normalize_time
    )

    if "lev" in deltaz.dims and "lev" in t.dims:
        deltaz = deltaz.assign_coords(lev=t.lev)

    if "lev" in dzu3.dims and "lev" in u.dims:
        dzu3 = dzu3.assign_coords(lev=u.lev)

    if "lev" in dzv3.dims and "lev" in v.dims:
        dzv3 = dzv3.assign_coords(lev=v.lev)

    return _overturning_core(
        strait=strait,
        model=model,
        time_start=time_start,
        time_end=time_end,
        indices=indices,
        grid=grid,
        t=t,
        u=u,
        v=v,
        deltaz=deltaz,
        mu=mu,
        mv=mv,
        dzu3=dzu3,
        dzv3=dzv3,
        min_x=min_x,
        min_y=min_y,
        sdata=sdata,
        rho=rho,
        cp=cp,
        Tref=Tref,
        apply_barotropic_correction=apply_barotropic_correction,
        transport_target_m3s=transport_target_m3s,
        sigmin=sigmin,
        sigstp=sigstp,
        nbins=nbins,
        salt_is_SA=salt_is_SA,
        temp_is_CT=temp_is_CT,
        return_diagnostics=return_diagnostics,
        saving=saving,
        path_save=path_save,
    )


def transports_overturning_ds(
    strait,
    model,
    time_start,
    time_end,
    ds_u,
    ds_v,
    ds_t,
    ds_z,
    ds_s=None,
    ds_zu=None,
    ds_zv=None,
    mesh_dxv=0,
    mesh_dyu=0,
    ds_indices=None,
    coords=0,
    set_latlon=False,
    lon_p=0,
    lat_p=0,
    Arakawa="",
    rho=1026.0,
    cp=3996.0,
    Tref=0.0,
    path_save="",
    path_indices="",
    path_mesh="",
    saving=True,
    apply_barotropic_correction=False,
    transport_target_m3s=None,
    sigmin=23.0,
    sigstp=0.02,
    nbins=270,
    salt_is_SA=False,
    temp_is_CT=False,
    return_diagnostics=False,
    preprocess=True,
    load=True,
    user_rename_dict=None,
    normalize_time="MS",
):
    """
    Dataset-based version of transports_overturning().

    Intended for already-open xarray datasets, for example from Pangeo/intake/zarr.
    """

    print("prepare surface fields for indices/grid check")
    ti = func._sf_preprocess1_ds(ds_t, preprocess=preprocess, user_rename_dict=user_rename_dict)
    ui = func._sf_preprocess1_ds(ds_u, preprocess=preprocess, user_rename_dict=user_rename_dict)
    vi = func._sf_preprocess1_ds(ds_v, preprocess=preprocess, user_rename_dict=user_rename_dict)

    if load:
        ti, ui, vi = func._load_with_progress(ti, ui, vi)

    if ds_indices is not None:
        indices = ds_indices
    else:
        indices = func._load_or_calculate_indices(
            strait,
            model,
            ti,
            ui,
            vi,
            coords=coords,
            lon_p=lon_p,
            lat_p=lat_p,
            set_latlon=set_latlon,
            path_indices=path_indices,
            path_save=path_save,
            saving=saving,
        )

    grid = func._determine_grid(model, Arakawa, ti, ui, vi, path_mesh=path_mesh, saving=saving)

    out_u, out_v, _ = prepare_indices(indices)
    xmin, xmax, ymin, ymax, min_x, min_y = func.calc_section_bounds(out_u, out_v, pad=1)
    lon_bnds = (xmin, xmax)
    lat_bnds = (ymin, ymax)

    print("prepare and subset t, u, v, z and optional s fields")
    t = func._sf_preprocess2_ds(ds_t, lon_bnds, lat_bnds, preprocess=preprocess, user_rename_dict=user_rename_dict)
    u = func._sf_preprocess2_ds(ds_u, lon_bnds, lat_bnds, preprocess=preprocess, user_rename_dict=user_rename_dict)
    v = func._sf_preprocess2_ds(ds_v, lon_bnds, lat_bnds, preprocess=preprocess, user_rename_dict=user_rename_dict)
    deltaz_ds = func._sf_preprocess2_ds(ds_z, lon_bnds, lat_bnds, preprocess=preprocess, user_rename_dict=user_rename_dict)
    deltaz = _get_vertical_thickness_dataset(deltaz_ds, ["thkcello"], label="T-cell thickness")

    has_salinity = ds_s is not None
    sdata = func._sf_preprocess2_ds(ds_s, lon_bnds, lat_bnds, preprocess=preprocess, user_rename_dict=user_rename_dict) if has_salinity else None

    t = func._select_time_if_present(t, time_start, time_end)
    u = func._select_time_if_present(u, time_start, time_end)
    v = func._select_time_if_present(v, time_start, time_end)
    if has_salinity:
        sdata = func._select_time_if_present(sdata, time_start, time_end)

    t = func._ensure_time_axis(t)
    u = func._ensure_time_axis(u, template=t)
    v = func._ensure_time_axis(v, template=t)
    if has_salinity:
        sdata = func._ensure_time_axis(sdata, template=t)

    t = _normalize_monthly_time(t, normalize_time=normalize_time)
    u = _normalize_monthly_time(u, normalize_time=normalize_time)
    v = _normalize_monthly_time(v, normalize_time=normalize_time)
    if has_salinity:
        sdata = _normalize_monthly_time(sdata, normalize_time=normalize_time)

    deltaz = _prepare_vertical_metric_time(
        deltaz, template=t, time_start=time_start, time_end=time_end, normalize_time=normalize_time
    )

    # Horizontal metrics.
    preprocess_mesh = lambda ds: func._sf_preprocess2_ds(
        ds,
        lon_bnds,
        lat_bnds,
        preprocess=preprocess,
        user_rename_dict=user_rename_dict,
    )
    mu, mv = func._load_or_prepare_mesh(
        model=model,
        mesh_dxv=mesh_dxv,
        mesh_dyu=mesh_dyu,
        ui=ui,
        vi=vi,
        path_mesh=path_mesh,
        preprocess_func=None,
        saving=saving,
    )
    mu = mu.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
    mv = mv.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

    if load:
        print("load final section subset")
        to_load = [t, u, v, deltaz, mu, mv] + ([sdata] if has_salinity else [])
        loaded = func._load_with_progress(*to_load)
        if has_salinity:
            t, u, v, deltaz, mu, mv, sdata = loaded
        else:
            t, u, v, deltaz, mu, mv = loaded

    # Vertical face thicknesses.
    if ds_zu is not None and ds_zv is not None:
        dzu_ds = func._sf_preprocess2_ds(ds_zu, lon_bnds, lat_bnds, preprocess=preprocess, user_rename_dict=user_rename_dict)
        dzv_ds = func._sf_preprocess2_ds(ds_zv, lon_bnds, lat_bnds, preprocess=preprocess, user_rename_dict=user_rename_dict)
        dzu3 = _get_vertical_thickness_var(dzu_ds, ["thkcello_u", "thkcello"], label="U-face thickness")
        dzv3 = _get_vertical_thickness_var(dzv_ds, ["thkcello_v", "thkcello"], label="V-face thickness")
        dzu3 = _prepare_vertical_metric_time(
            dzu3, template=u, time_start=time_start, time_end=time_end, normalize_time=normalize_time
        )
        dzv3 = _prepare_vertical_metric_time(
            dzv3, template=v, time_start=time_start, time_end=time_end, normalize_time=normalize_time
        )
        if load:
            dzu3, dzv3 = func._load_with_progress(dzu3, dzv3)
    else:
        dzu3, dzv3 = func.calc_dz_faces(deltaz, grid, model, path_mesh)
        dzu3 = _prepare_vertical_metric_time(
            dzu3, template=u, time_start=time_start, time_end=time_end, normalize_time=normalize_time
        )
        dzv3 = _prepare_vertical_metric_time(
            dzv3, template=v, time_start=time_start, time_end=time_end, normalize_time=normalize_time
        )

    if "lev" in deltaz.dims and "lev" in t.dims:
        deltaz = deltaz.assign_coords(lev=t.lev)

    if "lev" in dzu3.dims and "lev" in u.dims:
        dzu3 = dzu3.assign_coords(lev=u.lev)

    if "lev" in dzv3.dims and "lev" in v.dims:
        dzv3 = dzv3.assign_coords(lev=v.lev)

    return _overturning_core(
        strait=strait,
        model=model,
        time_start=time_start,
        time_end=time_end,
        indices=indices,
        grid=grid,
        t=t,
        u=u,
        v=v,
        deltaz=deltaz,
        mu=mu,
        mv=mv,
        dzu3=dzu3,
        dzv3=dzv3,
        min_x=min_x,
        min_y=min_y,
        sdata=sdata,
        rho=rho,
        cp=cp,
        Tref=Tref,
        apply_barotropic_correction=apply_barotropic_correction,
        transport_target_m3s=transport_target_m3s,
        sigmin=sigmin,
        sigstp=sigstp,
        nbins=nbins,
        salt_is_SA=salt_is_SA,
        temp_is_CT=temp_is_CT,
        return_diagnostics=return_diagnostics,
        saving=saving,
        path_save=path_save,
    )


def _dens0_unesco(S, T):
    """UNESCO 1983 density at zero pressure, matching old Watermasses.py logic."""
    a0 = 999.842594
    a1 = 6.793952e-2
    a2 = -9.095290e-3
    a3 = 1.001685e-4
    a4 = -1.120083e-6
    a5 = 6.536332e-9

    b0 = 8.24493e-1
    b1 = -4.0899e-3
    b2 = 7.6438e-5
    b3 = -8.2467e-7
    b4 = 5.3875e-9

    c0 = -5.72466e-3
    c1 = 1.0227e-4
    c2 = -1.6546e-6

    d0 = 4.8314e-4

    SMOW = a0 + (a1 + (a2 + (a3 + (a4 + a5 * T) * T) * T) * T) * T
    RB = b0 + (b1 + (b2 + (b3 + b4 * T) * T) * T) * T
    RC = c0 + (c1 + c2 * T) * T

    return SMOW + RB * S + RC * (S ** 1.5) + d0 * S * S


def _watermass_mask(temp, salt, sigma0, definition):
    """Create boolean water-mass mask from threshold definitions."""

    mask = xa.ones_like(temp, dtype=bool)

    if "thetao_min" in definition:
        mask = mask & (temp >= definition["thetao_min"])
    if "thetao_max" in definition:
        mask = mask & (temp < definition["thetao_max"])

    if "so_min" in definition:
        mask = mask & (salt >= definition["so_min"])
    if "so_max" in definition:
        mask = mask & (salt <= definition["so_max"])

    if "sigma0_min" in definition:
        sigmin = definition["sigma0_min"]
        if sigmin > 100:
            sigmin -= 1000.0
        mask = mask & (sigma0 >= sigmin)

    if "sigma0_max" in definition:
        sigmax = definition["sigma0_max"]
        if sigmax > 100:
            sigmax -= 1000.0
        mask = mask & (sigma0 <= sigmax)

    return mask

def watermass_transports(
    strait,
    model,
    time_start,
    time_end,
    file_u,
    file_v,
    file_t,
    file_s,
    file_z,
    file_zu=None,
    file_zv=None,
    watermass_definitions=None,
    mesh_dxv=0,
    mesh_dyu=0,
    coords=0,
    set_latlon=False,
    lon_p=0,
    lat_p=0,
    Arakawa="",
    rho=1026.0,
    cp=3996.0,
    Tref=0.0,
    path_save="",
    path_indices="",
    path_mesh="",
    saving=True,
    user_rename_dict=None,
    normalize_time="MS",
    return_diagnostics=False,
):
    """
    Calculate volume, heat and salt transports for user-defined water masses.

    Uses the old StraitFlux/GSR-style UNESCO density at zero pressure,
    matching the Watermasses.py approach more closely than GSW sigma0.

    Example:
        watermass_definitions = {
            "Overflow": {"sigma0_min": 27.8},
            "Polar": {"thetao_max": 4.0, "sigma0_max": 27.7},
            "Atlantic": {"thetao_min": 4.0, "sigma0_max": 27.8},
        }
    """

    if watermass_definitions is None:
        watermass_definitions = {
            "Overflow": {"sigma0_min": 27.8},
            "Polar": {"thetao_max": 4.0, "sigma0_max": 27.7},
            "Atlantic": {"thetao_min": 4.0, "sigma0_max": 27.8},
        }

    partial_func1 = partial(prepro._preprocess1, user_rename_dict=user_rename_dict)

    print("read and load surface fields for indices/grid check")
    ti = xa.open_mfdataset(file_t, preprocess=partial_func1).isel(time=0)
    ui = xa.open_mfdataset(file_u, preprocess=partial_func1).isel(time=0)
    vi = xa.open_mfdataset(file_v, preprocess=partial_func1).isel(time=0)
    ti, ui, vi = func._load_with_progress(ti, ui, vi)

    indices = func._load_or_calculate_indices(
        strait,
        model,
        ti,
        ui,
        vi,
        coords=coords,
        lon_p=lon_p,
        lat_p=lat_p,
        set_latlon=set_latlon,
        path_indices=path_indices,
        path_save=path_save,
        saving=saving,
    )

    grid = func._determine_grid(
        model,
        Arakawa,
        ti,
        ui,
        vi,
        path_mesh=path_mesh,
        saving=saving,
    )

    out_u, out_v, _ = prepare_indices(indices)
    xmin, xmax, ymin, ymax, min_x, min_y = func.calc_section_bounds(
        out_u,
        out_v,
        pad=1,
    )

    partial_func2 = partial(
        prepro._preprocess2,
        lon_bnds=(xmin, xmax),
        lat_bnds=(ymin, ymax),
        user_rename_dict=user_rename_dict,
    )

    mu, mv = func._load_or_prepare_mesh(
        model=model,
        mesh_dxv=mesh_dxv,
        mesh_dyu=mesh_dyu,
        ui=ui,
        vi=vi,
        path_mesh=path_mesh,
        preprocess_func=None,
        saving=saving,
    )

    mu = mu.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
    mv = mv.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

    print("read t, s, u, v and z fields")
    t = xa.open_mfdataset(file_t, preprocess=partial_func2, chunks={"time": 1})
    sdata = xa.open_mfdataset(file_s, preprocess=partial_func2, chunks={"time": 1})
    u = xa.open_mfdataset(file_u, preprocess=partial_func2, chunks={"time": 1})
    v = xa.open_mfdataset(file_v, preprocess=partial_func2, chunks={"time": 1})

    deltaz_ds = xa.open_mfdataset(file_z, preprocess=partial_func2, chunks={"time": 1})
    deltaz = _get_vertical_thickness_dataset(
        deltaz_ds,
        ["thkcello"],
        label="T-cell thickness",
    )

    t = func._select_time_if_present(t, time_start, time_end)
    sdata = func._select_time_if_present(sdata, time_start, time_end)
    u = func._select_time_if_present(u, time_start, time_end)
    v = func._select_time_if_present(v, time_start, time_end)

    t = func._ensure_time_axis(t)
    sdata = func._ensure_time_axis(sdata, template=t)
    u = func._ensure_time_axis(u, template=t)
    v = func._ensure_time_axis(v, template=t)

    t = _normalize_monthly_time(t, normalize_time=normalize_time)
    sdata = _normalize_monthly_time(sdata, normalize_time=normalize_time)
    u = _normalize_monthly_time(u, normalize_time=normalize_time)
    v = _normalize_monthly_time(v, normalize_time=normalize_time)

    deltaz = _prepare_vertical_metric_time(
        deltaz,
        template=t,
        time_start=time_start,
        time_end=time_end,
        normalize_time=normalize_time,
    )

    print("load final section subset")
    t, sdata, u, v, deltaz, mu, mv = func._load_with_progress(
        t,
        sdata,
        u,
        v,
        deltaz,
        mu,
        mv,
    )

    if file_zu and file_zv:
        dzu_ds = xa.open_mfdataset(file_zu, preprocess=partial_func2, chunks={"time": 1})
        dzv_ds = xa.open_mfdataset(file_zv, preprocess=partial_func2, chunks={"time": 1})

        dzu3 = _get_vertical_thickness_var(
            dzu_ds,
            ["thkcello_u", "thkcello"],
            label="U-face thickness",
        )
        dzv3 = _get_vertical_thickness_var(
            dzv_ds,
            ["thkcello_v", "thkcello"],
            label="V-face thickness",
        )

        dzu3 = _prepare_vertical_metric_time(
            dzu3,
            template=u,
            time_start=time_start,
            time_end=time_end,
            normalize_time=normalize_time,
        )
        dzv3 = _prepare_vertical_metric_time(
            dzv3,
            template=v,
            time_start=time_start,
            time_end=time_end,
            normalize_time=normalize_time,
        )

        dzu3, dzv3 = func._load_with_progress(dzu3, dzv3)
    else:
        dzu3, dzv3 = func.calc_dz_faces(deltaz, grid, model, path_mesh)

    if "lev" in deltaz.dims and "lev" in t.dims:
        deltaz = deltaz.assign_coords(lev=t.lev)
    if "lev" in dzu3.dims and "lev" in u.dims:
        dzu3 = dzu3.assign_coords(lev=u.lev)
    if "lev" in dzv3.dims and "lev" in v.dims:
        dzv3 = dzv3.assign_coords(lev=v.lev)

    udata, vdata, dzu3, dzv3, mu2, mv2 = func.transform_Arakawa(
        grid,
        mu,
        mv,
        deltaz,
        dzu3,
        dzv3,
        u.uo,
        v.vo,
    )

    Tu = func.interp_TS(t.thetao, "x")
    Tv = func.interp_TS(t.thetao, "y")
    Su = func.interp_TS(sdata.so, "x")
    Sv = func.interp_TS(sdata.so, "y")

    indices_local = func.shift_indices_to_local(indices, min_x, min_y, pad=1)
    arr = indices_local.indices.values.copy()

    iu = arr[:, 0].astype(int)
    ju = arr[:, 1].astype(int)
    iv = arr[:, 2].astype(int)
    jv = arr[:, 3].astype(int)
    sign_u = arr[:, 4].astype(float)

    u_mask = ~(np.all(np.isnan(arr[:, 0:2]) | (arr[:, 0:2] == 0), axis=1))
    v_mask = ~(np.all(np.isnan(arr[:, 2:4]) | (arr[:, 2:4] == 0), axis=1))

    sign_v = func.calc_sign_v(indices)

    transport_list = []
    temp_list = []
    salt_list = []
    area_list = []
    face_type = []

    for i, j, sg in zip(iu[u_mask], ju[u_mask], sign_u[u_mask]):
        area = func._strip_nondim_coords(mu2.dyu.isel(y=j, x=i) * dzu3.isel(y=j, x=i))
        tr = func._strip_nondim_coords(sg * udata.isel(y=j, x=i) * area)

        transport_list.append(tr)
        temp_list.append(func._strip_nondim_coords(Tu.isel(y=j, x=i)))
        salt_list.append(func._strip_nondim_coords(Su.isel(y=j, x=i)))
        area_list.append(area)
        face_type.append("u")

    for i, j, sg in zip(iv[v_mask], jv[v_mask], sign_v):
        area = func._strip_nondim_coords(mv2.dxv.isel(y=j, x=i) * dzv3.isel(y=j, x=i))
        tr = func._strip_nondim_coords(sg * vdata.isel(y=j, x=i) * area)

        transport_list.append(tr)
        temp_list.append(func._strip_nondim_coords(Tv.isel(y=j, x=i)))
        salt_list.append(func._strip_nondim_coords(Sv.isel(y=j, x=i)))
        area_list.append(area)
        face_type.append("v")

    transport = xa.concat(transport_list, dim="section_element")
    temp_face = xa.concat(temp_list, dim="section_element")
    salt_face = xa.concat(salt_list, dim="section_element")
    area_face = xa.concat(area_list, dim="section_element")

    sec_el = np.arange(transport.sizes["section_element"])
    face_type_arr = np.asarray(face_type)

    for da in [transport, temp_face, salt_face, area_face]:
        da.coords["section_element"] = sec_el
        da.coords["face_type"] = ("section_element", face_type_arr)

    transport = func._transpose_section(transport)
    temp_face = func._transpose_section(temp_face)
    salt_face = func._transpose_section(salt_face)
    area_face = func._transpose_section(area_face)

    if "time" in area_face.dims and area_face.sizes["time"] == 1:
        area_face = area_face.isel(time=0, drop=True)

    wet_bool = np.isfinite(transport) & np.isfinite(area_face) & (area_face > 0)

    transport = transport.where(wet_bool, 0.0)
    temp_face = temp_face.where(wet_bool)
    salt_face = salt_face.where(wet_bool)
    area_face = area_face.where(wet_bool, 0.0)

    # Old paper-style density:
    # full density at zero pressure, then sigma0 = rho0 - 1000.
    rho0_face = _dens0_unesco(salt_face, temp_face)
    sigma0_face = rho0_face - 1000.0

    sum_dims = [d for d in ["lev", "section_element"] if d in transport.dims]

    out_vars = {}

    out_vars["volume_transport_Total_Sv"] = (
        transport.sum(dim=sum_dims) / 1e6
    ).rename("volume_transport_Total_Sv")

    out_vars["heat_transport_Total_PW"] = (
        rho * cp * (transport * (temp_face - Tref)).sum(dim=sum_dims) / 1e15
    ).rename("heat_transport_Total_PW")

    out_vars["salt_transport_Total"] = (
        rho * (transport * salt_face).sum(dim=sum_dims)
    ).rename("salt_transport_Total")

    for wm_name, wm_def in watermass_definitions.items():
        wm_mask = _watermass_mask(temp_face, salt_face, sigma0_face, wm_def) & wet_bool
        safe_name = wm_name.replace(" ", "_")

        out_vars[f"volume_transport_{safe_name}_Sv"] = (
            transport.where(wm_mask, 0.0).sum(dim=sum_dims) / 1e6
        ).rename(f"volume_transport_{safe_name}_Sv")

        out_vars[f"heat_transport_{safe_name}_PW"] = (
            rho
            * cp
            * (transport.where(wm_mask, 0.0) * (temp_face - Tref)).sum(dim=sum_dims)
            / 1e15
        ).rename(f"heat_transport_{safe_name}_PW")

        out_vars[f"salt_transport_{safe_name}"] = (
            rho * (transport.where(wm_mask, 0.0) * salt_face).sum(dim=sum_dims)
        ).rename(f"salt_transport_{safe_name}")

        if return_diagnostics:
            out_vars[f"mask_{safe_name}"] = wm_mask.rename(f"mask_{safe_name}")

    if return_diagnostics:
        out_vars.update(
            dict(
                density0_face=rho0_face.rename("density0_face"),
                sigma0_face=sigma0_face.rename("sigma0_face"),
                temperature_face=temp_face.rename("temperature_face"),
                salinity_face=salt_face.rename("salinity_face"),
                transport_face_m3s=transport.rename("transport_face_m3s"),
                area_face_m2=area_face.rename("area_face_m2"),
            )
        )

    out = xa.Dataset(out_vars)

    for name in out.data_vars:
        if name.startswith("volume_transport"):
            out[name].attrs["units"] = "Sv"
        elif name.startswith("heat_transport"):
            out[name].attrs["units"] = "PW"
        elif name.startswith("salt_transport"):
            out[name].attrs["units"] = "kg s-1"
        elif name == "density0_face":
            out[name].attrs["units"] = "kg m-3"
        elif name == "sigma0_face":
            out[name].attrs["units"] = "kg m-3 - 1000"

    out.attrs["method"] = "StraitFlux water-mass transport decomposition"
    out.attrs["density_method"] = "UNESCO density at zero pressure, matching old Watermasses.py"
    out.attrs["watermass_definitions"] = str(watermass_definitions)

    if saving:
        out.to_netcdf(
            path_save
            + strait
            + "_watermass_transports_"
            + model
            + "_"
            + str(time_start)
            + "-"
            + str(time_end)
            + ".nc"
        )

    return out

def transports_split(
    product,
    strait,
    model,
    time_start,
    time_end,
    file_u,
    file_v,
    file_t,
    file_z,
    split_by,
    split_value,
    file_s="",
    file_zu=None,
    file_zv=None,
    mesh_dxv=0,
    mesh_dyu=0,
    coords=0,
    set_latlon=False,
    lon_p=0,
    lat_p=0,
    Arakawa="",
    rho=1026.0,
    cp=3996.0,
    Tref=0.0,
    path_save="",
    path_indices="",
    path_mesh="",
    saving=True,
    user_rename_dict=None,
    normalize_time="MS",
    salt_is_SA=False,
    temp_is_CT=False,
):
    """
    Calculate total transport plus transports above/below a depth
    or lighter/denser than a density threshold.

    product : {"volume", "heat", "salt"}
    split_by : {"depth", "density"}
    split_value :
        depth in m if split_by="depth";
        sigma0 threshold if split_by="density", e.g. 27.8 or 1027.8.

    For depth splitting, the model layer closest to split_value is used.
    No vertical interpolation is performed.
    """

    if product not in ["volume", "heat", "salt"]:
        raise ValueError("product must be 'volume', 'heat', or 'salt'.")

    if split_by not in ["depth", "density"]:
        raise ValueError("split_by must be 'depth' or 'density'.")

    if split_by == "density" and file_s in ["", None]:
        raise ValueError("Density splitting requires file_s.")

    ov = transports_overturning(
        strait=strait,
        model=model,
        time_start=time_start,
        time_end=time_end,
        file_u=file_u,
        file_v=file_v,
        file_t=file_t,
        file_z=file_z,
        file_s=file_s,
        file_zu=file_zu,
        file_zv=file_zv,
        mesh_dxv=mesh_dxv,
        mesh_dyu=mesh_dyu,
        coords=coords,
        set_latlon=set_latlon,
        lon_p=lon_p,
        lat_p=lat_p,
        Arakawa=Arakawa,
        rho=rho,
        cp=cp,
        Tref=Tref,
        path_save=path_save,
        path_indices=path_indices,
        path_mesh=path_mesh,
        saving=False,
        salt_is_SA=salt_is_SA,
        temp_is_CT=temp_is_CT,
        return_diagnostics=True,
        user_rename_dict=user_rename_dict,
        normalize_time=normalize_time,
    )

    transport = ov["transport_face_m3s"]
    temp = ov["temperature_face"]

    if product == "salt":
        salt = ov["salinity_face"]

    if split_by == "depth":
        lev = transport["lev"]
        ilev = int(np.abs(lev - split_value).argmin())
        actual_split = float(lev.isel(lev=ilev).values)

        mask_upper = lev <= actual_split
        mask_lower = lev > actual_split

        label_upper = "above_split"
        label_lower = "below_split"

    else:
        sigma0 = ov["sigma0_face"]

        split_sigma = split_value
        if split_sigma > 100:
            split_sigma = split_sigma - 1000.0

        actual_split = float(split_sigma)

        mask_upper = sigma0 <= split_sigma
        mask_lower = sigma0 > split_sigma

        label_upper = "lighter_than_split"
        label_lower = "denser_than_split"

    sum_dims = [d for d in ["lev", "section_element"] if d in transport.dims]

    if product == "volume":
        factor = 1e-6
        unit = "Sv"
        total = transport.sum(dim=sum_dims) * factor
        upper = transport.where(mask_upper, 0.0).sum(dim=sum_dims) * factor
        lower = transport.where(mask_lower, 0.0).sum(dim=sum_dims) * factor
        prefix = "volume_transport"

    elif product == "heat":
        factor = rho * cp / 1e15
        unit = "PW"
        total = (transport * temp).sum(dim=sum_dims) * factor
        upper = (transport.where(mask_upper, 0.0) * temp).sum(dim=sum_dims) * factor
        lower = (transport.where(mask_lower, 0.0) * temp).sum(dim=sum_dims) * factor
        prefix = "heat_transport"

    elif product == "salt":
        factor = rho
        unit = "kg s-1"
        total = (transport * salt).sum(dim=sum_dims) * factor
        upper = (transport.where(mask_upper, 0.0) * salt).sum(dim=sum_dims) * factor
        lower = (transport.where(mask_lower, 0.0) * salt).sum(dim=sum_dims) * factor
        prefix = "salt_transport"

    out = xa.Dataset(
        {
            f"{prefix}_total": total.rename(f"{prefix}_total"),
            f"{prefix}_{label_upper}": upper.rename(f"{prefix}_{label_upper}"),
            f"{prefix}_{label_lower}": lower.rename(f"{prefix}_{label_lower}"),
        }
    )

    for var in out.data_vars:
        out[var].attrs["units"] = unit

    out.attrs["split_by"] = split_by
    out.attrs["requested_split_value"] = float(split_value)
    out.attrs["actual_split_value"] = actual_split
    out.attrs["method"] = "StraitFlux line integration split by depth or density"

    if saving:
        out.to_netcdf(
            path_save
            + strait
            + "_"
            + product
            + "_split_"
            + split_by
            + "_"
            + model
            + "_"
            + str(time_start)
            + "-"
            + str(time_end)
            + ".nc"
        )

    return out
