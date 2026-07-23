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




def transports(product,strait,model,time_start,time_end,file_u,file_v,file_t,file_z, file_zu=None, file_zv=None,mesh_dxv=0, mesh_dyu=0,coords=0,set_latlon=False,lon_p=0,lat_p=0,file_s='',file_sic='',file_sit='',file_tracer='',tracer_var='tracer',tracer_units='',Arakawa='',rho=1026,cp=3996, Tref=0, Sref=34.8,path_save='',path_indices='',path_mesh='',saving=True,user_rename_dict=None):

    '''Calculation of Transports using line integration

    INPUT Parameters:
    product (str): volume, heat, salt, freshwater, tracer or ice
    strait (str): desired oceanic strait, either pre-defined from indices file or new
    model (str): desired CMIP6 model or reanalysis
    time_start (str or int): starting year
    time_end (str or int): ending year
    file_u (str OR ): path + filename(s) of u field(s); use ice velocities (ui) for ice transports; (multiple files possible, use *; must be possible to combine files over time coordinate)
    file_v (str): path + filename(s) of v field(s); use ice velocities (vi) for ice transports; (multiple files possible, use *)
    file_t (str): path + filename(s) of temperature field(s); (multiple files possible, use *)
    file_z (str): path + filename(s) of cell thickness field(s); (multiple files possible, use *)

    OPTIONAL:
    file_zu/file_zv (str): Optional U/V-cell thickness file. If omitted, U/V-cell thicknesses are calculated automatically from file_z.
    mesh_dxu/mesh_dyv (array): arrays containing the exact grid cell dimensions at northern and eastern grid cell faces of u and v cells (dxv and dyu); if not supplied will be calculated
    coords (tuple): coordinates for strait, if not pre-defined: (latitude_start,longitude_start,latitude_end,longitude_end). Sections may be defined in either direction (north→south, south→north, west→east or east→west). Positive transports are defined to the left of the specified section direction.
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
    rho (float): default = 1026 kg/m3
    cp (float): default = 3996 J/(kgK)
    Tref (float): default = 0°C
    Sref (float): reference salinity for freshwater transports; default = 34.8
    path_save (str): path to save transport data
    path_indices (str): path to save indices data
    path_mesh (str): path to save mesh data
    user_rename_dict (dict): Optional dictionary containing additional variable renaming rules used during preprocessing.

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

    if product in ['volume','heat','salt','freshwater','tracer']:
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
            if product in ['volume','heat','salt','tracer','freshwater']:
                deltaz=deltaz.load()
    except NameError:
        t=t.load()
        u=u.load()
        v=v.load()
        if product in ['volume','heat','salt','tracer','freshwater']:
            deltaz=deltaz.load()

    if product in ['volume','heat','salt','freshwater','tracer']:
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

    if product in ['salt', 'freshwater']:
        if file_s in ['', None]:
            raise ValueError(f"product='{product}' requires file_s to be supplied.")
        Sdata = xa.open_mfdataset(file_s, preprocess=partial_func, chunks={'time': 1}).sel(time=slice(str(time_start), str(time_end)))
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


    if product in ['volume','heat','salt','freshwater','tracer']:
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

    if product == 'freshwater':
        print('rolling S')
        Sudata = func.interp_TS(Sdata.so, 'x')
        Svdata = func.interp_TS(Sdata.so, 'y')
        print('calc u')
        udata = (udata*mu2.dyu.values*dzu3.values*((Sref-Sudata.values)/Sref))
        print('calc v')
        vdata = (vdata*mv2.dxv.values*dzv3.values*((Sref-Svdata.values)/Sref))
    
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
    if product in ['volume','heat','salt','tracer', 'freshwater']:
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


def _integrate_overturning_streamfunction(
    layer_transport,
    dim,
    direction,
):
    """
    Cumulatively integrate layer/bin transports in a physically defined direction.

    Parameters
    ----------
    layer_transport : xarray.DataArray
        Transport in individual depth layers or density bins.

    dim : str
        Integration dimension, e.g. ``"lev"`` or ``"density_bin"``.

    direction : str
        For depth coordinates:
            - ``"top_down"``
            - ``"bottom_up"``

        For density coordinates:
            - ``"light_to_dense"``
            - ``"dense_to_light"``

    Returns
    -------
    xarray.DataArray
        Cumulative overturning streamfunction, returned in the original
        coordinate order.
    """

    if dim not in layer_transport.dims:
        raise ValueError(
            f"Dimension '{dim}' is not present in layer_transport. "
            f"Available dimensions are {layer_transport.dims}."
        )

    valid_directions = {
        "lev": {"top_down", "bottom_up"},
        "density_bin": {"light_to_dense", "dense_to_light"},
    }

    if dim not in valid_directions:
        raise ValueError(
            f"No integration-direction definitions are available for dim='{dim}'."
        )

    if direction not in valid_directions[dim]:
        allowed = sorted(valid_directions[dim])
        raise ValueError(
            f"Invalid direction '{direction}' for dim='{dim}'. "
            f"Choose one of {allowed}."
        )

    coord = layer_transport[dim]

    if coord.size < 2:
        return layer_transport.cumsum(dim=dim)

    first = float(coord.isel({dim: 0}).values)
    last = float(coord.isel({dim: -1}).values)
    coordinate_increases = last > first

    if dim == "lev":
        # Depth normally increases downward:
        # shallow -> deep means increasing coordinate.
        desired_increasing = direction == "top_down"

    else:
        # Density normally increases from light -> dense.
        desired_increasing = direction == "light_to_dense"

    if coordinate_increases == desired_increasing:
        streamfunction = layer_transport.cumsum(dim=dim)
    else:
        streamfunction = (
            layer_transport
            .isel({dim: slice(None, None, -1)})
            .cumsum(dim=dim)
            .isel({dim: slice(None, None, -1)})
        )

    return streamfunction 

def _normalize_meridional_products(products, has_salinity):
    """Validate and normalize requested native-row transport diagnostics."""

    if products is None:
        requested = ["volume", "heat", "MOC_depth"]
        if has_salinity:
            requested.extend(["salt", "MOC_density"])
    elif isinstance(products, str):
        requested = [products]
    else:
        requested = list(products)

    aliases = {
        "volume": "volume",
        "heat": "heat",
        "salt": "salt",
        "moc_depth": "MOC_depth",
        "depth_moc": "MOC_depth",
        "moc_density": "MOC_density",
        "density_moc": "MOC_density",
    }

    normalized = []
    for product in requested:
        key = str(product).strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown product '{product}'. Choose from "
                "'volume', 'heat', 'salt', 'MOC_depth', and 'MOC_density'."
            )
        canonical = aliases[key]
        if canonical not in normalized:
            normalized.append(canonical)

    if not normalized:
        raise ValueError("products must contain at least one diagnostic.")

    salinity_products = {"salt", "MOC_density"}
    missing_salinity = salinity_products.intersection(normalized) and not has_salinity
    if missing_salinity:
        raise ValueError(
            "Products 'salt' and 'MOC_density' require file_s to be supplied."
        )

    return tuple(normalized)


def meridional_transports(
    model,
    time_start,
    time_end,
    file_v,
    file_t,
    file_z,
    file_s="",
    file_zv=None,
    mesh_dxv=0,
    basin_mask=None,
    basin_vars=None,
    basin_mask_grid="T",
    Arakawa="",
    rho=1026.0,
    cp=3996.0,
    Tref=0.0,
    path_save="",
    path_mesh="",
    saving=True,
    user_rename_dict=None,
    cyclic_columns=0,
    products=None,
    depth_integration_direction="top_down",
    density_integration_direction="light_to_dense",
    sigmin=23.0,
    sigstp=0.02,
    nbins=270,
    salt_is_SA=False,
    temp_is_CT=False,
):
    """Calculate transports along every native V-grid row.

    This intentionally simple implementation assumes that ``v``, ``T``,
    ``e3v`` and ``e1v`` are already supplied on the same complete native grid.
    It performs no grid transformation, mesh calculation, spatial subsetting,
    automatic cyclic-point detection, or coordinate-based horizontal alignment.

    The calculation is exactly

    ``v * tracer_at_v * e3v * e1v``

    followed by summation over ``depth`` and ``x``.  Exactly the final
    ``cyclic_columns`` columns are removed once from every horizontal field.
    """

    has_salinity = file_s not in (None, "")
    products = _normalize_meridional_products(products, has_salinity)
    need_heat = "heat" in products
    need_salt = "salt" in products
    need_depth_moc = "MOC_depth" in products
    need_density_moc = "MOC_density" in products
    need_temperature = need_heat or need_density_moc
    need_salinity = need_salt or need_density_moc

    print("requested products:", ", ".join(products))
    print("read native-grid fields without StraitFlux preprocessing")

    # Only rename dimensions/coordinates/variables to a common convention.
    rename_common = {
        "time_counter": "time",
        "time_counter_bnds": "time_bnds",
        "deptht": "depth",
        "depthu": "depth",
        "depthv": "depth",
        "depthw": "depth",
        "lev": "depth",
        "z": "depth",
        "nav_lon": "lon",
        "nav_lat": "lat",
    }
    if user_rename_dict:
        rename_common.update(user_rename_dict)

    def _rename_existing(obj):
        available = set(obj.dims) | set(obj.coords)
        if isinstance(obj, xa.Dataset):
            available |= set(obj.data_vars)
        mapping = {
            old: new for old, new in rename_common.items()
            if old in available and old != new
        }
        return obj.rename(mapping) if mapping else obj

    def _first_variable(ds, candidates, label):
        if isinstance(ds, xa.DataArray):
            return ds
        for name in candidates:
            if name in ds:
                return ds[name]
        if len(ds.data_vars) == 1:
            return ds[list(ds.data_vars)[0]]
        raise KeyError(
            f"Could not identify {label}. Tried {candidates}; "
            f"available variables are {list(ds.data_vars)}."
        )

    vds = _rename_existing(xa.open_mfdataset(file_v, chunks="auto"))
    tds = _rename_existing(xa.open_mfdataset(file_t, chunks="auto"))
    ztds = _rename_existing(xa.open_mfdataset(file_z, chunks="auto"))

    v = _first_variable(vds, ["vo", "vomecrty", "v"], "V velocity")
    temp = _first_variable(tds, ["thetao", "votemper", "temperature", "temp"], "temperature")
    e3t = _first_variable(
        ztds,
        ["thkcello", "e3t", "e3t_0", "e3t_0_field"],
        "T-cell thickness",
    )

    # Select requested period only when a real time coordinate is present.
    for name in ["v", "temp", "e3t"]:
        obj = locals()[name]
        if "time" in obj.dims and obj.sizes["time"] > 1:
            locals()[name] = obj.sel(time=slice(str(time_start), str(time_end)))
    v = v.sel(time=slice(str(time_start), str(time_end))) if "time" in v.dims and v.sizes["time"] > 1 else v
    temp = temp.sel(time=slice(str(time_start), str(time_end))) if "time" in temp.dims and temp.sizes["time"] > 1 else temp
    e3t = e3t.sel(time=slice(str(time_start), str(time_end))) if "time" in e3t.dims and e3t.sizes["time"] > 1 else e3t

    if "time" not in v.dims:
        v = v.expand_dims(time=temp.time if "time" in temp.dims else [0])
    if "time" not in temp.dims:
        temp = temp.expand_dims(time=v.time)

    # Static metrics may contain a dummy singleton time dimension.
    if "time" in e3t.dims and e3t.sizes["time"] == 1 and v.sizes["time"] != 1:
        e3t = e3t.isel(time=0, drop=True)

    if file_zv not in (None, ""):
        zvds = _rename_existing(xa.open_mfdataset(file_zv, chunks="auto"))
        e3v = _first_variable(
            zvds,
            ["thkcello_v", "thkcello", "e3v", "e3v_0", "e3v_0_field"],
            "V-face thickness",
        )
        if "time" in e3v.dims and e3v.sizes["time"] > 1:
            e3v = e3v.sel(time=slice(str(time_start), str(time_end)))
        if "time" in e3v.dims and e3v.sizes["time"] == 1 and v.sizes["time"] != 1:
            e3v = e3v.isel(time=0, drop=True)
    else:
        print("file_zv not supplied: calculate e3v as the T-cell mean in y")
        e3v = func.interp_TS(e3t, "y")

    if not isinstance(mesh_dxv, (xa.Dataset, xa.DataArray)):
        raise ValueError(
            "This simple meridional function requires mesh_dxv/e1v to be "
            "supplied explicitly as an xarray DataArray or Dataset."
        )
    e1v = _rename_existing(mesh_dxv)
    e1v = _first_variable(e1v, ["dxv", "e1v"], "e1v/dxv")

    # Remove harmless singleton dimensions from static e1v.
    for dim in list(e1v.dims):
        if dim not in ("y", "x") and e1v.sizes[dim] == 1:
            e1v = e1v.isel({dim: 0}, drop=True)

    salt = None
    if need_salinity:
        sds = _rename_existing(xa.open_mfdataset(file_s, chunks="auto"))
        salt = _first_variable(sds, ["so", "vosaline", "salinity", "salt"], "salinity")
        if "time" in salt.dims and salt.sizes["time"] > 1:
            salt = salt.sel(time=slice(str(time_start), str(time_end)))
        if "time" not in salt.dims:
            salt = salt.expand_dims(time=v.time)

    # All arrays are required to describe the same native grid by position.
    fields = {"temperature": temp, "e3v": e3v, "e1v": e1v}
    if salt is not None:
        fields["salinity"] = salt
    for name, da in fields.items():
        for dim in ("y", "x"):
            if dim not in da.dims:
                raise ValueError(f"{name} has dimensions {da.dims}; '{dim}' is required.")
            if da.sizes[dim] != v.sizes[dim]:
                raise ValueError(
                    f"{name} has {dim}={da.sizes[dim]}, while V velocity has "
                    f"{dim}={v.sizes[dim]}. Supply all fields on the same full native grid."
                )
    for name, da in {"v": v, "temperature": temp, "e3v": e3v}.items():
        if "depth" not in da.dims:
            raise ValueError(f"{name} has dimensions {da.dims}; 'depth' is required.")
        if da.sizes["depth"] != v.sizes["depth"]:
            raise ValueError(
                f"{name} has depth={da.sizes['depth']}, while V velocity has "
                f"depth={v.sizes['depth']}."
            )

    # Discard horizontal/depth index labels and align strictly by array position.
    # Time labels are retained.
    native_x = np.arange(v.sizes["x"])
    native_y = np.arange(v.sizes["y"])
    native_depth = v["depth"].values
    v = v.assign_coords(x=native_x, y=native_y, depth=native_depth)
    temp = temp.assign_coords(x=native_x, y=native_y, depth=native_depth)
    e3v = e3v.assign_coords(x=native_x, y=native_y, depth=native_depth)
    e1v = e1v.assign_coords(x=native_x, y=native_y)
    if salt is not None:
        salt = salt.assign_coords(x=native_x, y=native_y, depth=native_depth)

    # Interpolate T/S to V exactly as in the manual calculation.
    temp_v = func.interp_TS(temp, "y") if need_temperature else None
    salt_v = func.interp_TS(salt, "y") if need_salinity else None

    if cyclic_columns in (None, False):
        cyclic_columns = 0
    if not isinstance(cyclic_columns, (int, np.integer)) or cyclic_columns < 0:
        raise ValueError("cyclic_columns must be a non-negative integer.")
    cyclic_columns = int(cyclic_columns)
    if cyclic_columns >= v.sizes["x"]:
        raise ValueError("cyclic_columns cannot remove all x points.")

    # Drop exactly the user-supplied number of final columns, once.
    if cyclic_columns > 0:
        xslice = slice(None, -cyclic_columns)
        v = v.isel(x=xslice)
        e3v = e3v.isel(x=xslice)
        e1v = e1v.isel(x=xslice)
        if temp_v is not None:
            temp_v = temp_v.isel(x=xslice)
        if salt_v is not None:
            salt_v = salt_v.isel(x=xslice)
        print(f"dropped exactly the final {cyclic_columns} x columns")
    else:
        print("no cyclic columns dropped")

    print("final native-grid sizes:", dict(v.sizes))

    # This is the direct manual calculation: v * e3v * e1v.
    area_v = e3v * e1v
    volume_face = (v * area_v).fillna(0.0)

    # Optional masks. Keep this deliberately positional and simple.
    mask_arrays = [xa.ones_like(e1v, dtype=float)]
    basin_names = ["global"]
    if basin_mask is not None:
        if isinstance(basin_mask, str):
            masks_ds = _rename_existing(xa.open_dataset(basin_mask))
        else:
            masks_ds = _rename_existing(basin_mask)

        if isinstance(masks_ds, xa.DataArray):
            mask_items = [(masks_ds.name or "basin", masks_ds)]
        elif isinstance(basin_vars, dict):
            mask_items = [(name, masks_ds[var]) for name, var in basin_vars.items()]
        elif basin_vars is not None:
            names = [basin_vars] if isinstance(basin_vars, str) else list(basin_vars)
            mask_items = [(name, masks_ds[name]) for name in names]
        else:
            mask_items = [(name, masks_ds[name]) for name in masks_ds.data_vars]

        for name, mask in mask_items:
            for dim in list(mask.dims):
                if dim not in ("y", "x") and mask.sizes[dim] == 1:
                    mask = mask.isel({dim: 0}, drop=True)
            if mask.sizes.get("y") != v.sizes["y"] or mask.sizes.get("x") not in (
                v.sizes["x"], v.sizes["x"] + cyclic_columns
            ):
                raise ValueError(f"Basin mask '{name}' does not match the native grid.")
            mask = mask.assign_coords(
                y=np.arange(mask.sizes["y"]), x=np.arange(mask.sizes["x"])
            )
            if cyclic_columns > 0 and mask.sizes["x"] != v.sizes["x"]:
                mask = mask.isel(x=slice(None, -cyclic_columns))
            if str(basin_mask_grid).upper() == "T":
                mask = (mask > 0) & ((mask.shift(y=-1) > 0))
            else:
                mask = mask > 0
            mask_arrays.append(mask.astype(float))
            basin_names.append(str(name))

    masks = xa.concat(mask_arrays, dim=xa.IndexVariable("basin", basin_names))
    volume_basin = volume_face.expand_dims(basin=masks.basin) * masks

    out_vars = {}
    if "volume" in products:
        out_vars["volume_transport_Sv"] = (
            volume_basin.sum(dim=("depth", "x")) / 1e6
        )

    if need_heat:
        out_vars["heat_transport_PW"] = (
            rho * cp * (volume_basin * (temp_v - Tref)).sum(dim=("depth", "x")) / 1e15
        )

    if need_salt:
        out_vars["salt_transport_kg_s"] = (
            rho * (volume_basin * salt_v).sum(dim=("depth", "x"))
        )

    if need_depth_moc:
        layer = (volume_basin.sum(dim="x") / 1e6).rename(
            {"depth": "depth"}
        )
        if depth_integration_direction == "top_down":
            psi = layer.cumsum("depth")
        elif depth_integration_direction == "bottom_up":
            psi = layer.isel(depth=slice(None, None, -1)).cumsum("depth").isel(
                depth=slice(None, None, -1)
            )
        else:
            raise ValueError(
                "depth_integration_direction must be 'top_down' or 'bottom_up'."
            )
        out_vars["overturning_depth_layer_Sv"] = layer
        out_vars["overturning_depth_streamfunction_Sv"] = psi
        out_vars["MOC_depth_Sv"] = psi.max("depth")
        out_vars["MOC_depth_min_Sv"] = psi.min("depth")

    if need_density_moc:
        try:
            import gsw
        except ImportError as exc:
            raise ImportError("Density-space MOC requires gsw.") from exc

        # Use actual depth values when numeric; otherwise fall back to level index.
        depth_values = v["depth"].astype(float)
        lat = vds["lat"] if "lat" in vds else xa.zeros_like(e1v)
        lon = vds["lon"] if "lon" in vds else xa.zeros_like(e1v)
        for coord_name, coord in [("lat", lat), ("lon", lon)]:
            if "time" in coord.dims:
                coord = coord.isel(time=0, drop=True)
            coord = coord.assign_coords(x=np.arange(coord.sizes["x"]), y=np.arange(coord.sizes["y"]))
            if cyclic_columns > 0:
                coord = coord.isel(x=slice(None, -cyclic_columns))
            if coord_name == "lat":
                lat = coord
            else:
                lon = coord

        depth3, lat3 = xa.broadcast(depth_values, lat)
        pressure = xa.apply_ufunc(
            gsw.p_from_z, -depth3, lat3,
            vectorize=True, dask="parallelized", output_dtypes=[float]
        ).broadcast_like(v)
        SA = salt_v if salt_is_SA else xa.apply_ufunc(
            gsw.SA_from_SP, salt_v, pressure,
            lon.broadcast_like(v), lat.broadcast_like(v),
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )
        CT = temp_v if temp_is_CT else xa.apply_ufunc(
            gsw.CT_from_t, SA, temp_v, pressure,
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )
        sigma0 = xa.apply_ufunc(
            gsw.sigma0, SA, CT,
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )

        centers = sigmin + sigstp * (np.arange(nbins) + 0.5)
        vol_np = np.asarray(volume_basin.transpose("time", "basin", "depth", "y", "x"))
        sig_np = np.asarray(sigma0.transpose("time", "depth", "y", "x"))
        nt, nbasin, _, ny, _ = vol_np.shape
        binned = np.zeros((nt, nbasin, nbins, ny), dtype=float)
        for it in range(nt):
            for iy in range(ny):
                sigrow = sig_np[it, :, iy, :].ravel()
                bins = np.floor((sigrow - sigmin) / sigstp).astype("int64")
                valid = np.isfinite(sigrow) & (bins >= 0) & (bins < nbins)
                for ib in range(nbasin):
                    vals = vol_np[it, ib, :, iy, :].ravel()
                    good = valid & np.isfinite(vals)
                    if np.any(good):
                        binned[it, ib, :, iy] = np.bincount(
                            bins[good], weights=vals[good], minlength=nbins
                        )[:nbins]

        layer_density = xa.DataArray(
            binned / 1e6,
            dims=("time", "basin", "density_bin", "y"),
            coords={
                "time": v.time,
                "basin": masks.basin,
                "density_bin": centers,
                "y": v.y,
            },
        )
        if density_integration_direction == "light_to_dense":
            psi_density = layer_density.cumsum("density_bin")
        elif density_integration_direction == "dense_to_light":
            psi_density = layer_density.isel(
                density_bin=slice(None, None, -1)
            ).cumsum("density_bin").isel(density_bin=slice(None, None, -1))
        else:
            raise ValueError(
                "density_integration_direction must be 'light_to_dense' or 'dense_to_light'."
            )
        out_vars["overturning_density_bin_Sv"] = layer_density
        out_vars["overturning_density_streamfunction_Sv"] = psi_density
        out_vars["MOC_density_Sv"] = psi_density.max("density_bin")
        out_vars["MOC_density_min_Sv"] = psi_density.min("density_bin")

    out = xa.Dataset(out_vars)

    # Row-latitude information, when available.
    if "lat" in vds:
        lat = vds["lat"]
        if "time" in lat.dims:
            lat = lat.isel(time=0, drop=True)
        lat = lat.assign_coords(x=np.arange(lat.sizes["x"]), y=np.arange(lat.sizes["y"]))
        if cyclic_columns > 0:
            lat = lat.isel(x=slice(None, -cyclic_columns))
        out = out.assign_coords(
            latitude=("y", lat.mean("x", skipna=True).values),
            latitude_min=("y", lat.min("x", skipna=True).values),
            latitude_max=("y", lat.max("x", skipna=True).values),
        )

    units = {
        "volume_transport_Sv": "Sv",
        "heat_transport_PW": "PW",
        "salt_transport_kg_s": "kg s-1",
        "overturning_depth_layer_Sv": "Sv",
        "overturning_depth_streamfunction_Sv": "Sv",
        "MOC_depth_Sv": "Sv",
        "MOC_depth_min_Sv": "Sv",
        "overturning_density_bin_Sv": "Sv",
        "overturning_density_streamfunction_Sv": "Sv",
        "MOC_density_Sv": "Sv",
        "MOC_density_min_Sv": "Sv",
    }
    for name, unit in units.items():
        if name in out:
            out[name].attrs["units"] = unit

    out.attrs.update(
        method="Direct native V-grid row integration",
        formula="v * e3v * e1v, with optional T/S at V points",
        cyclic_columns_dropped=cyclic_columns,
        requested_products=", ".join(products),
        rho_kg_m3=float(rho),
        cp_J_kgK=float(cp),
        reference_temperature_C=float(Tref),
        depth_integration_direction=depth_integration_direction,
        density_integration_direction=density_integration_direction,
    )
    out = out.load()

    if saving:
        filename = (
            path_save + "meridional_transports_" + model + "_"
            + str(time_start) + "-" + str(time_end) + ".nc"
        )
        out.to_netcdf(filename)

    return out


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
    depth_integration_direction="top_down",
    density_integration_direction="light_to_dense",
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
        _integrate_overturning_streamfunction(
            overturning_depth_layer_Sv,
            dim="lev",
            direction=depth_integration_direction,
        )
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
            _integrate_overturning_streamfunction(
                overturning_density_bin_Sv,
                dim="density_bin",
                direction=density_integration_direction,
            )
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
    out.attrs["depth_integration_direction"] = depth_integration_direction
    out.attrs["density_integration_direction"] = density_integration_direction

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
    depth_integration_direction="top_down",
    density_integration_direction="light_to_dense",
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
    t = xa.open_mfdataset(file_t, preprocess=partial_func2, chunks="auto")
    u = xa.open_mfdataset(file_u, preprocess=partial_func2, chunks="auto")
    v = xa.open_mfdataset(file_v, preprocess=partial_func2, chunks="auto")
    deltaz_ds = xa.open_mfdataset(file_z, preprocess=partial_func2, chunks="auto")
    deltaz = _get_vertical_thickness_dataset(deltaz_ds, ["thkcello"], label="T-cell thickness")

    has_salinity = file_s not in ["", None]
    sdata = xa.open_mfdataset(file_s, preprocess=partial_func2, chunks="auto") if has_salinity else None

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
            dzu_ds = xa.open_mfdataset(file_zu, preprocess=partial_func2, chunks="auto")
            dzv_ds = xa.open_mfdataset(file_zv, preprocess=partial_func2, chunks="auto")
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
        depth_integration_direction=depth_integration_direction,
        density_integration_direction=density_integration_direction,
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
    depth_integration_direction="top_down",
    density_integration_direction="light_to_dense",
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
        depth_integration_direction=depth_integration_direction,
        density_integration_direction=density_integration_direction,
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

def meridional_transports_at_latitudes(
    model,
    time_start,
    time_end,
    target_latitudes,
    file_u,
    file_v,
    file_t,
    file_z,
    file_s="",
    file_zu=None,
    file_zv=None,
    mesh_dxv=0,
    mesh_dyu=0,
    basin_mask=None,
    basin_vars=None,
    rho=1026.0,
    cp=3996.0,
    Tref=0.0,
    path_save="",
    saving=True,
    user_rename_dict=None,
    cyclic_columns=0,
    products=None,
    depth_integration_direction="top_down",
):
    """Calculate transports across exact geographic latitude contours.

    A latitude contour is represented as the boundary between T cells south
    and north of ``target_latitudes``. On a curvilinear Arakawa-C grid this
    boundary crosses both V faces and U faces. The transport is therefore
    calculated conservatively from

    ``v * e1v * e3v * crossing_v + u * e2u * e3u * crossing_u``.

    Positive transport is northward. The routine deliberately performs only
    simple variable/dimension renaming and positional grid matching. Exactly
    ``cyclic_columns`` final x columns are removed once, after interpolation
    and construction of the crossing masks.

    Parameters
    ----------
    target_latitudes : scalar or 1-D sequence
        Geographic latitude(s) in degrees north.
    mesh_dxv : xarray.DataArray or Dataset
        Native V-face zonal width, usually ``e1v``.
    mesh_dyu : xarray.DataArray or Dataset
        Native U-face meridional width, usually ``e2u``.
    basin_mask : optional DataArray, Dataset, or path
        Two-dimensional T-grid mask(s), with 1/True inside and 0/False outside.
        A face is included only when both adjacent T cells belong to the basin.
    products : sequence of str, optional
        Any of ``volume``, ``heat``, ``salt``, and ``MOC_depth``.
    """

    has_salinity = file_s not in (None, "")
    if products is None:
        products = ["volume", "heat", "MOC_depth"]
        if has_salinity:
            products.append("salt")
    elif isinstance(products, str):
        products = [products]
    else:
        products = list(products)

    aliases = {
        "volume": "volume",
        "heat": "heat",
        "salt": "salt",
        "moc_depth": "MOC_depth",
        "depth_moc": "MOC_depth",
    }
    normalized = []
    for product in products:
        key = str(product).strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown product '{product}'. Choose from volume, heat, "
                "salt, and MOC_depth."
            )
        canonical = aliases[key]
        if canonical not in normalized:
            normalized.append(canonical)
    products = tuple(normalized)

    if "salt" in products and not has_salinity:
        raise ValueError("Product 'salt' requires file_s.")

    need_temp = "heat" in products
    need_salt = "salt" in products

    target_latitudes = np.atleast_1d(np.asarray(target_latitudes, dtype=float))
    if target_latitudes.ndim != 1 or target_latitudes.size == 0:
        raise ValueError("target_latitudes must be a scalar or a non-empty 1-D sequence.")

    print("requested products:", ", ".join(products))
    print("calculate transports across exact geographic latitude contours")

    rename_common = {
        "time_counter": "time",
        "time_counter_bnds": "time_bnds",
        "deptht": "depth",
        "depthu": "depth",
        "depthv": "depth",
        "depthw": "depth",
        "lev": "depth",
        "z": "depth",
        "nav_lon": "lon",
        "nav_lat": "lat",
    }
    if user_rename_dict:
        rename_common.update(user_rename_dict)

    def _rename_existing(obj):
        available = set(obj.dims) | set(obj.coords)
        if isinstance(obj, xa.Dataset):
            available |= set(obj.data_vars)
        mapping = {
            old: new for old, new in rename_common.items()
            if old in available and old != new
        }
        return obj.rename(mapping) if mapping else obj

    def _open(obj):
        if isinstance(obj, (xa.Dataset, xa.DataArray)):
            return _rename_existing(obj)
        return _rename_existing(xa.open_mfdataset(obj, chunks="auto"))

    def _first_variable(obj, candidates, label):
        if isinstance(obj, xa.DataArray):
            return obj
        for name in candidates:
            if name in obj:
                return obj[name]
        if len(obj.data_vars) == 1:
            return obj[list(obj.data_vars)[0]]
        raise KeyError(
            f"Could not identify {label}. Tried {candidates}; available "
            f"variables are {list(obj.data_vars)}."
        )

    def _select_time(da):
        if "time" in da.dims and da.sizes["time"] > 1:
            return da.sel(time=slice(str(time_start), str(time_end)))
        return da

    def _drop_static_singletons(da, keep=("depth", "y", "x")):
        for dim in list(da.dims):
            if dim not in keep and da.sizes[dim] == 1:
                da = da.isel({dim: 0}, drop=True)
        return da

    uds = _open(file_u)
    vds = _open(file_v)
    tds = _open(file_t)
    ztds = _open(file_z)

    u = _select_time(_first_variable(uds, ["uo", "vozocrtx", "u"], "U velocity"))
    v = _select_time(_first_variable(vds, ["vo", "vomecrty", "v"], "V velocity"))
    temp = _select_time(_first_variable(
        tds, ["thetao", "votemper", "temperature", "temp"], "temperature"
    ))
    e3t = _select_time(_first_variable(
        ztds, ["thkcello", "e3t", "e3t_0", "e3t_0_field"], "T-cell thickness"
    ))

    if "lat" not in tds:
        raise ValueError(
            "The temperature/T-grid file must contain nav_lat or lat on the T grid."
        )
    lat_t = tds["lat"]
    if "time" in lat_t.dims:
        lat_t = lat_t.isel(time=0, drop=True)
    lat_t = _drop_static_singletons(lat_t, keep=("y", "x"))

    if file_zu not in (None, ""):
        e3u = _select_time(_first_variable(
            _open(file_zu),
            ["thkcello_u", "thkcello", "e3u", "e3u_0", "e3u_0_field"],
            "U-face thickness",
        ))
    else:
        print("file_zu not supplied: calculate e3u as the T-cell mean in x")
        e3u = func.interp_TS(e3t, "x")

    if file_zv not in (None, ""):
        e3v = _select_time(_first_variable(
            _open(file_zv),
            ["thkcello_v", "thkcello", "e3v", "e3v_0", "e3v_0_field"],
            "V-face thickness",
        ))
    else:
        print("file_zv not supplied: calculate e3v as the T-cell mean in y")
        e3v = func.interp_TS(e3t, "y")

    if not isinstance(mesh_dxv, (xa.Dataset, xa.DataArray)):
        raise ValueError("mesh_dxv must explicitly contain native e1v/dxv.")
    if not isinstance(mesh_dyu, (xa.Dataset, xa.DataArray)):
        raise ValueError("mesh_dyu must explicitly contain native e2u/dyu.")

    e1v = _first_variable(_rename_existing(mesh_dxv), ["e1v", "dxv"], "e1v/dxv")
    e2u = _first_variable(_rename_existing(mesh_dyu), ["e2u", "dyu"], "e2u/dyu")
    e1v = _drop_static_singletons(e1v, keep=("y", "x"))
    e2u = _drop_static_singletons(e2u, keep=("y", "x"))

    salt = None
    if need_salt:
        salt = _select_time(_first_variable(
            _open(file_s), ["so", "vosaline", "salinity", "salt"], "salinity"
        ))

    # Ensure a time dimension exists for all time-varying fields.
    reference_time = None
    for da in (u, v, temp, salt):
        if da is not None and "time" in da.dims:
            reference_time = da["time"]
            break
    if reference_time is None:
        reference_time = xa.IndexVariable("time", [0])
    for name in ("u", "v", "temp", "salt"):
        da = locals()[name]
        if da is not None and "time" not in da.dims:
            locals()[name] = da.expand_dims(time=reference_time)
    # locals() assignment is not reliable in all Python implementations.
    if "time" not in u.dims:
        u = u.expand_dims(time=reference_time)
    if "time" not in v.dims:
        v = v.expand_dims(time=reference_time)
    if "time" not in temp.dims:
        temp = temp.expand_dims(time=reference_time)
    if salt is not None and "time" not in salt.dims:
        salt = salt.expand_dims(time=reference_time)

    # Static thickness fields may carry a singleton time dimension.
    for name in ("e3u", "e3v"):
        da = locals()[name]
        if "time" in da.dims and da.sizes["time"] == 1 and u.sizes["time"] != 1:
            da = da.isel(time=0, drop=True)
        if name == "e3u":
            e3u = da
        else:
            e3v = da

    # All native fields must have identical positional horizontal/depth sizes.
    horizontal = {
        "U velocity": u,
        "V velocity": v,
        "temperature": temp,
        "e3u": e3u,
        "e3v": e3v,
        "e1v": e1v,
        "e2u": e2u,
        "T-grid latitude": lat_t,
    }
    if salt is not None:
        horizontal["salinity"] = salt
    for label, da in horizontal.items():
        for dim in ("y", "x"):
            if dim not in da.dims:
                raise ValueError(f"{label} has dimensions {da.dims}; '{dim}' is required.")
            if da.sizes[dim] != u.sizes[dim]:
                raise ValueError(
                    f"{label} has {dim}={da.sizes[dim]}, while U velocity has "
                    f"{dim}={u.sizes[dim]}. Supply fields on the same full native grid."
                )
    for label, da in {
        "U velocity": u, "V velocity": v, "temperature": temp,
        "e3u": e3u, "e3v": e3v,
    }.items():
        if "depth" not in da.dims:
            raise ValueError(f"{label} has dimensions {da.dims}; 'depth' is required.")
        if da.sizes["depth"] != u.sizes["depth"]:
            raise ValueError(
                f"{label} has depth={da.sizes['depth']}, while U velocity has "
                f"depth={u.sizes['depth']}."
            )

    # Discard index labels and match strictly by native array position.
    nx, ny, nz = u.sizes["x"], u.sizes["y"], u.sizes["depth"]
    xcoord = np.arange(nx)
    ycoord = np.arange(ny)
    depthcoord = u["depth"].values
    for name in ("u", "v", "temp", "e3u", "e3v", "salt"):
        da = locals()[name]
        if da is None:
            continue
        coords = {"x": xcoord, "y": ycoord}
        if "depth" in da.dims:
            coords["depth"] = depthcoord
        da = da.assign_coords(coords)
        if name == "u": u = da
        elif name == "v": v = da
        elif name == "temp": temp = da
        elif name == "e3u": e3u = da
        elif name == "e3v": e3v = da
        elif name == "salt": salt = da
    e1v = e1v.assign_coords(x=xcoord, y=ycoord)
    e2u = e2u.assign_coords(x=xcoord, y=ycoord)
    lat_t = lat_t.assign_coords(x=xcoord, y=ycoord)

    # Interpolate tracers before cyclic columns are removed.
    temp_u = func.interp_TS(temp, "x") if need_temp else None
    temp_v = func.interp_TS(temp, "y") if need_temp else None
    salt_u = func.interp_TS(salt, "x") if need_salt else None
    salt_v = func.interp_TS(salt, "y") if need_salt else None

    # Remove exactly the requested trailing cyclic columns once.  Tracer
    # interpolation was done on the complete native grid so that its behaviour
    # remains identical to the existing meridional-row routine.
    if cyclic_columns in (None, False):
        cyclic_columns = 0
    if not isinstance(cyclic_columns, (int, np.integer)) or cyclic_columns < 0:
        raise ValueError("cyclic_columns must be a non-negative integer.")
    cyclic_columns = int(cyclic_columns)
    if cyclic_columns >= nx:
        raise ValueError("cyclic_columns cannot remove all x points.")

    if cyclic_columns > 0:
        xslice = slice(None, -cyclic_columns)
        u = u.isel(x=xslice)
        v = v.isel(x=xslice)
        e3u = e3u.isel(x=xslice)
        e3v = e3v.isel(x=xslice)
        e1v = e1v.isel(x=xslice)
        e2u = e2u.isel(x=xslice)
        lat_t_work = lat_t.isel(x=xslice)
        if temp_u is not None:
            temp_u = temp_u.isel(x=xslice)
            temp_v = temp_v.isel(x=xslice)
        if salt_u is not None:
            salt_u = salt_u.isel(x=xslice)
            salt_v = salt_v.isel(x=xslice)
        print(f"dropped exactly the final {cyclic_columns} x columns")
    else:
        lat_t_work = lat_t
        print("no cyclic columns dropped")

    nx_work = u.sizes["x"]
    ny_work = u.sizes["y"]

    # Basin masks are supplied on T points.  A native face is retained only
    # when both adjacent T cells lie inside the basin.  For U faces, x is
    # periodic after the duplicate cyclic columns have been removed.
    basin_names = ["global"]
    basin_masks_t = [np.ones((ny_work, nx_work), dtype=bool)]

    if basin_mask is not None:
        masks_obj = (
            _open(basin_mask)
            if isinstance(basin_mask, str)
            else _rename_existing(basin_mask)
        )
        if isinstance(masks_obj, xa.DataArray):
            mask_items = [(masks_obj.name or "basin", masks_obj)]
        elif isinstance(basin_vars, dict):
            mask_items = [(name, masks_obj[var]) for name, var in basin_vars.items()]
        elif basin_vars is not None:
            names = [basin_vars] if isinstance(basin_vars, str) else list(basin_vars)
            mask_items = [(name, masks_obj[name]) for name in names]
        else:
            mask_items = [(name, masks_obj[name]) for name in masks_obj.data_vars]

        for name, mask in mask_items:
            mask = _drop_static_singletons(mask, keep=("y", "x"))
            if mask.sizes.get("y") != ny or mask.sizes.get("x") != nx:
                raise ValueError(
                    f"Basin mask '{name}' must have the full native size "
                    f"(y={ny}, x={nx}); got {dict(mask.sizes)}."
                )
            mask = mask.assign_coords(x=xcoord, y=ycoord)
            if cyclic_columns > 0:
                mask = mask.isel(x=slice(None, -cyclic_columns))
            basin_names.append(str(name))
            basin_masks_t.append(np.asarray((mask > 0).values, dtype=bool))

    basin_face_masks_u = []
    basin_face_masks_v = []
    for mask_t_np in basin_masks_t:
        # U face between (y,x) and (y,x+1), including the periodic seam.
        mask_u_np = mask_t_np & np.roll(mask_t_np, -1, axis=1)
        # V face between (y,x) and (y+1,x).  There is no artificial external
        # boundary at the northern or southern edge.
        mask_v_np = np.zeros_like(mask_t_np, dtype=bool)
        mask_v_np[:-1, :] = mask_t_np[:-1, :] & mask_t_np[1:, :]
        basin_face_masks_u.append(mask_u_np)
        basin_face_masks_v.append(mask_v_np)

    # Build each exact latitude contour only once from the static T-grid
    # latitude field.  We store only the crossed native faces and their signs,
    # rather than constructing a huge (latitude,y,x) mask that is repeatedly
    # broadcast over time and depth.
    lat_np = np.asarray(lat_t_work.values)
    contours = []
    n_u_faces = []
    n_v_faces = []
    closure_errors = []

    for latitude in target_latitudes:
        north = lat_np >= latitude

        # Signed boundary of the north-of-latitude T-cell set.
        # Each native face can occur at most once in a contour.
        cross_u_np = np.roll(north, -1, axis=1).astype(np.int8) - north.astype(np.int8)
        cross_v_np = np.zeros_like(north, dtype=np.int8)
        cross_v_np[:-1, :] = (
            north[1:, :].astype(np.int8) - north[:-1, :].astype(np.int8)
        )

        yu, xu = np.nonzero(cross_u_np)
        yv, xv = np.nonzero(cross_v_np)
        su = cross_u_np[yu, xu].astype(float)
        sv = cross_v_np[yv, xv].astype(float)

        # A closed discrete contour has zero net signed crossings in each
        # horizontal direction.  This is a useful topology check and catches
        # contours that touch an unhandled domain edge.
        closure_error = int(abs(su.sum()) + abs(sv.sum()))
        closure_errors.append(closure_error)
        n_u_faces.append(len(su))
        n_v_faces.append(len(sv))
        contours.append((yu, xu, su, yv, xv, sv))

    if any(err != 0 for err in closure_errors):
        bad = [
            f"{lat:g}° (error={err})"
            for lat, err in zip(target_latitudes, closure_errors)
            if err != 0
        ]
        print(
            "warning: some latitude contours may touch an unhandled grid "
            "boundary: " + ", ".join(bad[:10])
        )

    volume_u = (u * e2u * e3u).fillna(0.0)
    volume_v = (v * e1v * e3v).fillna(0.0)

    def _sum_sparse_faces(field_u, field_v, basin_index, contour):
        """Sum one 3-D face flux field along one sparse closed contour."""
        yu, xu, su, yv, xv, sv = contour
        pieces = []

        if len(su):
            keep_u = basin_face_masks_u[basin_index][yu, xu]
            if np.any(keep_u):
                yu_keep = yu[keep_u]
                xu_keep = xu[keep_u]
                su_keep = su[keep_u]
                index_u = xa.DataArray(np.arange(len(yu_keep)), dims="face")
                selected_u = field_u.isel(
                    y=xa.DataArray(yu_keep, dims="face", coords={"face": index_u}),
                    x=xa.DataArray(xu_keep, dims="face", coords={"face": index_u}),
                )
                sign_u = xa.DataArray(su_keep, dims="face", coords={"face": index_u})
                pieces.append((selected_u * sign_u).sum("face"))

        if len(sv):
            keep_v = basin_face_masks_v[basin_index][yv, xv]
            if np.any(keep_v):
                yv_keep = yv[keep_v]
                xv_keep = xv[keep_v]
                sv_keep = sv[keep_v]
                index_v = xa.DataArray(np.arange(len(yv_keep)), dims="face")
                selected_v = field_v.isel(
                    y=xa.DataArray(yv_keep, dims="face", coords={"face": index_v}),
                    x=xa.DataArray(xv_keep, dims="face", coords={"face": index_v}),
                )
                sign_v = xa.DataArray(sv_keep, dims="face", coords={"face": index_v})
                pieces.append((selected_v * sign_v).sum("face"))

        if not pieces:
            template = field_u.isel(y=0, x=0, drop=True) * 0.0
            return template
        result = pieces[0]
        for piece in pieces[1:]:
            result = result + piece
        return result

    # Compute only the crossed faces for each latitude and basin.  The loop is
    # inexpensive because a contour contains O(nx) faces rather than ny*nx.
    layer_volume_by_basin = []
    for basin_index, basin_name in enumerate(basin_names):
        layer_volume_by_lat = [
            _sum_sparse_faces(volume_u, volume_v, basin_index, contour)
            for contour in contours
        ]
        layer_volume_by_basin.append(
            xa.concat(
                layer_volume_by_lat,
                dim=xa.IndexVariable("latitude", target_latitudes),
            )
        )
    layer_volume = xa.concat(
        layer_volume_by_basin,
        dim=xa.IndexVariable("basin", basin_names),
    )

    out_vars = {}
    if "volume" in products:
        out_vars["volume_transport_Sv"] = layer_volume.sum("depth") / 1e6

    if need_temp:
        heat_face_u = rho * cp * volume_u * (temp_u - Tref)
        heat_face_v = rho * cp * volume_v * (temp_v - Tref)
        heat_by_basin = []
        for basin_index, basin_name in enumerate(basin_names):
            heat_by_lat = [
                _sum_sparse_faces(heat_face_u, heat_face_v, basin_index, contour)
                for contour in contours
            ]
            heat_by_basin.append(
                xa.concat(
                    heat_by_lat,
                    dim=xa.IndexVariable("latitude", target_latitudes),
                )
            )
        heat_layer = xa.concat(
            heat_by_basin,
            dim=xa.IndexVariable("basin", basin_names),
        )
        out_vars["heat_transport_PW"] = heat_layer.sum("depth") / 1e15

    if need_salt:
        salt_face_u = rho * volume_u * salt_u
        salt_face_v = rho * volume_v * salt_v
        salt_by_basin = []
        for basin_index, basin_name in enumerate(basin_names):
            salt_by_lat = [
                _sum_sparse_faces(salt_face_u, salt_face_v, basin_index, contour)
                for contour in contours
            ]
            salt_by_basin.append(
                xa.concat(
                    salt_by_lat,
                    dim=xa.IndexVariable("latitude", target_latitudes),
                )
            )
        salt_layer = xa.concat(
            salt_by_basin,
            dim=xa.IndexVariable("basin", basin_names),
        )
        out_vars["salt_transport_kg_s"] = salt_layer.sum("depth")

    if "MOC_depth" in products:
        layer_sv = layer_volume / 1e6
        if depth_integration_direction == "top_down":
            psi = layer_sv.cumsum("depth")
        elif depth_integration_direction == "bottom_up":
            psi = layer_sv.isel(depth=slice(None, None, -1)).cumsum("depth").isel(
                depth=slice(None, None, -1)
            )
        else:
            raise ValueError(
                "depth_integration_direction must be 'top_down' or 'bottom_up'."
            )
        out_vars["overturning_depth_layer_Sv"] = layer_sv
        out_vars["overturning_depth_streamfunction_Sv"] = psi
        out_vars["MOC_depth_Sv"] = psi.max("depth")
        out_vars["MOC_depth_min_Sv"] = psi.min("depth")

    out = xa.Dataset(out_vars)
    out = out.assign_coords(latitude=("latitude", target_latitudes))

    units = {
        "volume_transport_Sv": "Sv",
        "heat_transport_PW": "PW",
        "salt_transport_kg_s": "kg s-1",
        "overturning_depth_layer_Sv": "Sv",
        "overturning_depth_streamfunction_Sv": "Sv",
        "MOC_depth_Sv": "Sv",
        "MOC_depth_min_Sv": "Sv",
    }
    for name, unit in units.items():
        if name in out:
            out[name].attrs["units"] = unit

    out.attrs.update({
        "model": str(model),
        "method": "sparse exact geographic latitude contour on native Arakawa-C faces",
        "positive_direction": "northward",
        "cyclic_columns_dropped": cyclic_columns,
        "rho_kg_m3": float(rho),
        "cp_J_kg_K": float(cp),
        "reference_temperature_degC": float(Tref),
    })

    # Diagnostics: every listed face is unique within its contour.  The closure
    # error should be zero for a fully closed latitude contour.
    out["number_of_U_faces"] = xa.DataArray(
        np.asarray(n_u_faces, dtype=np.int64), dims="latitude",
        coords={"latitude": target_latitudes},
    )
    out["number_of_V_faces"] = xa.DataArray(
        np.asarray(n_v_faces, dtype=np.int64), dims="latitude",
        coords={"latitude": target_latitudes},
    )
    out["contour_closure_error"] = xa.DataArray(
        np.asarray(closure_errors, dtype=np.int64), dims="latitude",
        coords={"latitude": target_latitudes},
    )
    out["number_of_U_faces"].attrs["description"] = "unique native U faces crossed by latitude contour"
    out["number_of_V_faces"].attrs["description"] = "unique native V faces crossed by latitude contour"
    out["contour_closure_error"].attrs["description"] = (
        "sum of absolute signed U- and V-face crossings; zero indicates a closed contour"
    )
    out = out.load()

    if saving:
        filename = (
            path_save + "meridional_exact_latitudes_" + str(model) + "_"
            + str(time_start) + "-" + str(time_end) + ".nc"
        )
        out.to_netcdf(filename)
        print("saved", filename)

    return out
    
