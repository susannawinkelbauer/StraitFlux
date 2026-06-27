import xarray as xa
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('skipping matplotlib')
from tqdm import tqdm
import sys
try:
    from dask.diagnostics import ProgressBar
except ImportError:
    print('skipping dask import')
from StraitFlux.indices import check_availability_indices, prepare_indices
#from indices import check_availability_indices, prepare_indices
import StraitFlux.preprocessing as prepro
#import preprocessing as prepro

def check_Arakawa(u_data,v_data,T_data,model):

    u=u_data
    v=v_data
    t=T_data

    print('checking grid')
    if model in ['MPI-ESM1-2-LR','MPI-ESM1-2-HR']:
        grid='Arakawa-C'
        print(grid)
    elif u.lat[int(len(t.y)/2),0].values == t.lat[int(len(t.y)/2),0].values and v.lat[int(len(t.y)/2),0].values != t.lat[int(len(t.y)/2),0].values:
        if v.lon[int(len(t.y)/2),0].values == t.lon[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values != t.lon[int(len(t.y)/2),0].values:
            grid='Arakawa-C'
            print(grid)
        else:
            raise ValueError(
                f"Grid type for model '{model}' was not recognized automatically. "
                "Please specify Arakawa='Arakawa-A', "
                "'Arakawa-B', or 'Arakawa-C'."
            )
    elif u.lat[int(len(t.y)/2),0].values == v.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values == v.lon[int(len(t.y)/2),0].values:
        if u.lat[int(len(t.y)/2),0].values != t.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values != t.lon[int(len(t.y)/2),0].values:
            grid='Arakawa-B'
            print(grid)
        elif u.lat[int(len(t.y)/2),0].values == t.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values == t.lon[int(len(t.y)/2),0].values:
            grid='Arakawa-A'
            print(grid)
        elif u.lat[int(len(t.y)/2),0].values == t.lat[int(len(t.y)/2),0].values == v.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values != t.lon[int(len(t.y)/2),0].values != v.lon[int(len(ti.y)/2),0].values:
            grid='Arakawa-E'
            print(grid+'?')
        else:
            raise ValueError(
                f"Grid type for model '{model}' was not recognized automatically. "
                "Please specify Arakawa='Arakawa-A', "
                "'Arakawa-B', or 'Arakawa-C'."
            )
    else:
        raise ValueError(
            f"Grid type for model '{model}' was not recognized automatically. "
            "Please specify Arakawa='Arakawa-A', "
            "'Arakawa-B', or 'Arakawa-C'."
        )
    return grid

def transform_Arakawa(grid,mu,mv,deltaz,dzu3,dzv3,udata,vdata):

    deltaz2=deltaz.thkcello###.mean(dim='time')
    dzs=deltaz2.sum(dim='lev').where(deltaz2.sum(dim='lev')!=0) #axis=0

    if grid == 'Arakawa-C':
        mu2=mu
        mv2=mv
    if grid == 'Arakawa-B':
        print('transforming Arakawa-B to Arakawa-C')
        dzuv=deltaz2.rolling(x=2,min_periods=1).mean().rolling(y=2,min_periods=1).mean()
        dzuv2=dzuv.cumsum('lev').where(dzuv.cumsum('lev')<=dzs) #dzuvs
        dzuv2=dzuv2.fillna(dzs)#dzuvs
        dzuv3=dzuv2.copy(data=np.diff(dzuv2, axis=dzuv2.get_axis_num("lev"), prepend=0))
        mu2=mu.rolling(x=2,min_periods=1).mean().rolling(y=2,min_periods=1).mean()
        mv2=mv.rolling(y=2,min_periods=1).mean().rolling(x=2,min_periods=1).mean()
        udata2=(udata*mu2.dyu.values*dzuv3.values).rolling(y=2,min_periods=1).mean()/(mu.dyu.values*dzu3.values)#.fillna(0)
        vdata2=(vdata*mv2.dxv.values*dzuv3.values).rolling(x=2,min_periods=1).mean()/(mv.dxv.values*dzv3.values)
        udata2=udata2.where(udata2>-2).where(udata2<2).fillna(0)
        vdata2=vdata2.where(vdata2>-2).where(vdata2<2).fillna(0)
        udata=udata2*(udata/udata).values
        vdata=vdata2*(udata/udata).values


    if grid == 'Arakawa-A':
        print('transforming Arakawa-A to Arakawa-C')
        mu2=mu.rolling(x=2,min_periods=1).mean()#.rolling(y=2,min_periods=1).mean()
        mv2=mv.rolling(y=2,min_periods=1).mean()#.rolling(x=2,min_periods=1).mean()
        print('equation to get u/v at T faces')
        udata2=(udata*mu.dyu.values*deltaz2.values).rolling(x=2,min_periods=1).mean()/(mu2.dyu.values*dzu3.values)#.fillna(0)
        vdata2=(vdata*mv.dxv.values*deltaz2.values).rolling(y=2,min_periods=1).mean()/(mv2.dxv.values*dzv3.values)
        udata=udata2.where(udata2>-1000).where(udata2<1000).fillna(0)
        vdata=vdata2.where(vdata2>-1000).where(vdata2<1000).fillna(0)

    return udata,vdata,dzu3,dzv3,mu2,mv2

def check_indices(indices,out_u,out_v,t,u,v,strait,model,path_save):
    lp=indices.indices[-1][indices.indices[-1] != 0].values
    slp=indices.indices[-2][indices.indices[-2] != 0].values
    fp=indices.indices[0][indices.indices[0] != 0].values
    sfp=indices.indices[1][indices.indices[1] != 0].values
    tfp=indices.indices[2][indices.indices[2] != 0].values
    #last point:
    if indices.indices[-1][0] == 0 and indices.indices[-1][1] == 0:
        if v.vo[int(lp[1]-1),int(lp[0]-1)].values > 0 or v.vo[int(lp[1]-1),int(lp[0]-1)].values < 0:
            if v.vo[int(slp[1]-1),int(slp[0]-1)].values > 0 or v.vo[int(slp[1]-1),int(slp[0]-1)].values < 0:
                print('!!!ATTENTION!!!: last point water, recheck indices line!')
            else:
                print('dropping last point...')
        else:
            print('line good')
    else:
        if u.uo[int(lp[1]-1),int(lp[0]-1)].values > 0 or u.uo[int(lp[1]-1),int(lp[0]-1)].values < 0:
            if u.uo[int(slp[1]-1),int(slp[0]-1)].values > 0 or u.uo[int(slp[1]-1),int(slp[0]-1)].values < 0:
                print('!!!ATTENTION!!!: last point water, recheck indices line!')
            else:
                print('dropping last point...')
        else:
            print('line good')
            

    #first point:
    if indices.indices[0][2] == 0 and indices.indices[0][3] == 0:
        if u.uo[int(fp[1]-1),int(fp[0]-1)].values > 0 or u.uo[int(fp[1]-1),int(fp[0]-1)].values < 0:
            if u.uo[int(sfp[1]-1),int(sfp[0]-1)].values > 0 or u.uo[int(sfp[1]-1),int(sfp[0]-1)].values < 0:
                print('!!!ATTENTION!!!: first point water, recheck indices line!')
            else:
                print('dropping last point...')
        else:
            print('line good')
    else:
        if v.vo[int(fp[1]-1),int(fp[0]-1)].values > 0 or v.vo[int(fp[1]-1),int(fp[0]-1)].values < 0:
            if v.vo[int(sfp[1]-1),int(sfp[0]-1)].values > 0 or v.vo[int(sfp[1]-1),int(sfp[0]-1)].values < 0:
                print('!!!ATTENTION!!!: first point water, recheck indices line!')
            else:
                print('dropping last point...')
        else:
            print('line good')
       
    
    out_u,out_v,out_u_vz = prepare_indices(indices)

    min_x=np.nanmin((min(out_u[:,0],default=np.nan),min(out_v[:,0],default=np.nan)))
    max_x=np.nanmax((max(out_u[:,0],default=np.nan),max(out_v[:,0],default=np.nan)))
    min_y=np.nanmin((min(out_u[:,1],default=np.nan),min(out_v[:,1],default=np.nan)))
    max_y=np.nanmax((max(out_u[:,1],default=np.nan),max(out_v[:,1],default=np.nan)))

    if min_x == -1:
        min_x = 0
        max_x = max_x + 1
        
    t=t.sel(x=slice(int(min_x)-2,int(max_x)+2),y=slice(int(min_y)-2,int(max_y)+2)).load()
    u=u.sel(x=slice(int(min_x)-1,int(max_x)+1),y=slice(int(min_y)-1,int(max_y)+1)).load()
    v=v.sel(x=slice(int(min_x)-1,int(max_x)+1),y=slice(int(min_y)-1,int(max_y)+1)).load()
    try:
        plt.title(model+'_'+strait,fontsize=14)
        try:
            plt.pcolormesh(t.x,t.y,(t.thetao/t.thetao),cmap='tab20c')
        except AttributeError:
            plt.pcolormesh(t.x,t.y,(t.sithick/t.sithick),cmap='tab20c')
        plt.scatter(out_v[:,0],out_v[:,1]+0.5,marker='_',c='r',s=200)
        plt.scatter(out_u[:,0]+0.5,out_u[:,1],marker='|',c='r',s=200)
        plt.ylabel('y',fontsize=14)
        plt.xlabel('x',fontsize=14)
        plt.savefig(path_save+strait+'_'+model+'_indices_check.png')
        plt.close()
    except:
        print('skipping Plot')

def interp_TS(ds, d):
    return 0.5 * (ds + ds.shift({d: -1}))
    #return ds.rolling({d:2},min_periods=1).mean()

def calc_dz_faces(deltaz,grid,model,path_mesh):

    if model in ['MPI-ESM1-2-LR','MPI-ESM1-2-HR']:
        print('swap')
        deltaz['y']=np.arange(len(deltaz.y)-1,-1,-1)
        deltaz=deltaz.sortby('y')
    try:
        with ProgressBar():
            zv=deltaz.thkcello.values
    except NameError:
        zv=deltaz.thkcello.values
    z0u=np.zeros(np.shape(zv))
    z0v=np.zeros(np.shape(zv))
    print('calc dz at cell faces')
    if 'time' in deltaz.dims:
        deltazu=xa.Dataset({'thkcello':(('time','lev','y','x'),np.zeros(np.shape(zv)))},coords=({'time':('time',deltaz.time.data),'lev':('lev',deltaz.lev.data),'x':('x',deltaz.x.data),'y':('y',deltaz.y.data)}))
        deltazv=xa.Dataset({'thkcello':(('time','lev','y','x'),np.zeros(np.shape(zv)))},coords=({'time':('time',deltaz.time.data),'lev':('lev',deltaz.lev.data),'x':('x',deltaz.x.data),'y':('y',deltaz.y.data)}))
    else:
        deltazu=xa.Dataset({'thkcello':(('lev','y','x'),np.zeros(np.shape(zv)))},coords=({'lev':('lev',deltaz.lev.data),'x':('x',deltaz.x.data),'y':('y',deltaz.y.data)}))
        deltazv=xa.Dataset({'thkcello':(('lev','y','x'),np.zeros(np.shape(zv)))},coords=({'lev':('lev',deltaz.lev.data),'x':('x',deltaz.x.data),'y':('y',deltaz.y.data)}))


    if grid == 'Arakawa-C':
        if 'time' in deltaz.dims:
            for i in tqdm(range(len(deltaz.y)-1)):
                #print(i)
                for j in range(len(deltaz.x)-1):
                    p=np.nansum(zv[:,:,i,j:j+2],axis=1).argmin(axis=1)
                    for k in range(len(p)):
                        if p[k] == 0:
                            z0u[k,:,i,j]=zv[k,:,i,j]
                        if p[k] ==1:
                            l=np.isnan(zv[k,:,i,j+1]).argmax(axis=0)-1
                            z0u[k,:l,i,j]=zv[k,:l,i,j]
                            z0u[k,l,i,j]=zv[k,l,i,j+1]
                            if l >= 0:
                                z0u[k,l+1:,i,j]=np.nan
            deltazu['thkcello'][:,:,:,:]=z0u
            deltazu['thkcello'][:,:,:,-1]=zv[:,:,:,-1]
            deltazu['thkcello'][:,:,-1,:]=zv[:,:,-1,:]

            for j in tqdm(range(len(deltaz.x)-1)):
                #print(j)
                for i in range(len(deltaz.y)-1):
                    p=np.nansum(zv[:,:,i:i+2,j],axis=1).argmin(axis=1)
                    for k in range(len(p)):
                        if p[k] == 0:
                            z0v[k,:,i,j]=zv[k,:,i,j]
                        if p[k] ==1:
                            l=np.isnan(zv[k,:,i+1,j]).argmax(axis=0)-1
                            z0v[k,:l,i,j]=zv[k,:l,i,j]
                            z0v[k,l,i,j]=zv[k,l,i+1,j]
                            if l >= 0:
                                z0v[k,l+1:,i,j]=np.nan

            deltazv['thkcello'][:,:,:,:]=z0v
            deltazv['thkcello'][:,:,:,-1]=zv[:,:,:,-1]
            deltazv['thkcello'][:,:,-1,:]=zv[:,:,-1,:]

        else:
            for i in tqdm(range(len(deltaz.y)-1)):
                #print(i)
                for j in range(len(deltaz.x)-1):
                    p=np.nansum(zv[:,i,j:j+2],axis=0).argmin()
                    if p == 0:
                        z0u[:,i,j]=zv[:,i,j]
                    if p ==1:
                        z0u[:,i,j]=zv[:,i,j+1]
            deltazu['thkcello'][:,:,:]=z0u
            deltazu['thkcello'][:,:,-1]=zv[:,:,-1]
            deltazu['thkcello'][:,-1,:]=zv[:,-1,:]

            for j in tqdm(range(len(deltaz.x)-1)):
                for i in range(len(deltaz.y)-1):
                    p=np.nansum(zv[:,i:i+2,j],axis=0).argmin()
                    if p == 0:
                        z0v[:,i,j]=zv[:,i,j]
                    if p ==1:
                        z0v[:,i,j]=zv[:,i+1,j]
            deltazv['thkcello'][:,:,:]=z0v
            deltazv['thkcello'][:,:,-1]=zv[:,:,-1]
            deltazv['thkcello'][:,-1,:]=zv[:,-1,:]

    elif grid in ['Arakawa-B','Arakawa-A']:
        if 'time' in deltaz.dims:
            for k in tqdm(range(len(deltaz.time))):
                #print(k)
                zv=deltaz.thkcello[k].values
                z0u=np.zeros(np.shape(zv))
                z0v=np.zeros(np.shape(zv))
                for i in range(len(deltaz.y)-1):
                    #print(i)
                    for j in range(len(deltaz.x)-1):
                        p=np.argwhere(np.nansum(zv[:,i:i+2,j:j+2],axis=0) == np.min(np.nansum(zv[:,i:i+2,j:j+2],axis=0)))[0]
                        if p[0] == 0:
                            if p[1] == 0:
                                z0u[:,i,j]=zv[:,i,j]
                            elif p[1] == 1:
                                z0u[:,i,j]=zv[:,i,j+1]
                        elif p[0] == 1:
                            if  p[1] == 0:
                                z0u[:,i,j]=zv[:,i+1,j]
                            elif p[1] == 1:
                                z0u[:,i,j]=zv[:,i+1,j+1]
                deltazu['thkcello'][k,:,:,:]=z0u
                deltazu['thkcello'][k,:,:,-1]=zv[:,:,-1]
                deltazu['thkcello'][k,:,-1,:]=zv[:,-1,:]
                deltazv['thkcello'][k,:,:,:]=z0u
                deltazv['thkcello'][k,:,:,-1]=zv[:,:,-1]
                deltazv['thkcello'][k,:,-1,:]=zv[:,-1,:]


        else:
            for i in tqdm(range(len(deltaz.y)-1)):
                #print(i)
                for j in range(len(deltaz.x)-1):
                    p=np.argwhere(np.nansum(zv[:,i:i+2,j:j+2],axis=0) == np.min(np.nansum(zv[:,i:i+2,j:j+2],axis=0)))[0]
                    if p[0] == 0:
                        if p[1] == 0:
                            z0u[:,i,j]=zv[:,i,j]
                        elif p[1] == 1:
                            z0u[:,i,j]=zv[:,i,j+1]
                    elif p[0] == 1:
                        if  p[1] == 0:
                            z0u[:,i,j]=zv[:,i+1,j]
                        elif p[1] == 1:
                            z0u[:,i,j]=zv[:,i+1,j+1]
            deltazu['thkcello'][:,:,:]=z0u
            deltazu['thkcello'][:,:,-1]=zv[:,:,-1]
            deltazu['thkcello'][:,-1,:]=zv[:,-1,:]
            deltazv['thkcello'][:,:,:]=z0u
            deltazv['thkcello'][:,:,-1]=zv[:,:,-1]
            deltazv['thkcello'][:,-1,:]=zv[:,-1,:]

    if model in ['MPI-ESM1-2-LR','MPI-ESM1-2-HR']:
        print('swap')
        deltazu['y']=np.arange(len(deltazu.y)-1,-1,-1)
        deltazu=deltazu.sortby('y')
        deltazv['y']=np.arange(len(deltazv.y)-1,-1,-1)
        deltazv=deltazv.sortby('y')

    return deltazu.thkcello,deltazv.thkcello


def _strip_nondim_coords(da):
    """Drop non-dimensional auxiliary coordinates that can block xr.concat."""
    return da.reset_coords(drop=True)


def _transpose_section(da, tdim="time", zdim="lev"):
    order = [d for d in [tdim, zdim, "section_element"] if d in da.dims]
    return da.transpose(*order)


def _bincount_timewise(arr, ibin, centers, nbins, tdim="time", name="binned"):
    """Bin a section quantity into density classes for each time step."""
    if tdim not in arr.dims:
        arr = arr.expand_dims({tdim: [0]})
    if tdim not in ibin.dims:
        ibin = ibin.expand_dims({tdim: arr[tdim]})

    data = np.empty((arr.sizes[tdim], nbins), dtype=float)
    for it in range(arr.sizes[tdim]):
        a = arr.isel({tdim: it}).values.ravel()
        b = ibin.isel({tdim: it}).values.ravel()
        valid = np.isfinite(a) & (b >= 0)
        data[it, :] = np.bincount(
            b[valid].astype(int),
            weights=a[valid],
            minlength=nbins,
        )[:nbins]

    return xa.DataArray(
        data,
        dims=(tdim, "density_bin"),
        coords={tdim: arr[tdim], "density_bin": centers},
        name=name,
    )

def calc_section_bounds(out_u, out_v, pad=1):
    min_x = np.nanmin((min(out_u[:, 0], default=np.nan), min(out_v[:, 0], default=np.nan)))
    max_x = np.nanmax((max(out_u[:, 0], default=np.nan), max(out_v[:, 0], default=np.nan)))
    min_y = np.nanmin((min(out_u[:, 1], default=np.nan), min(out_v[:, 1], default=np.nan)))
    max_y = np.nanmax((max(out_u[:, 1], default=np.nan), max(out_v[:, 1], default=np.nan)))

    if min_x == -1:
        min_x = 0
        max_x = max_x + 1

    return (
        int(min_x) - pad,
        int(max_x) + pad,
        int(min_y) - pad,
        int(max_y) + pad,
        int(min_x),
        int(min_y),
    )


def calc_sign_v(indices):
    sign_v = []
    indi = indices.indices[:, 2][indices.indices[:, 3] != 0]

    for ind in range(len(indi) - 1):
        if indi[ind] < indi[ind + 1]:
            sign_v = np.append(sign_v, 1)
        elif indi[ind + 1] in [1, 0, -1]:
            sign_v = np.append(sign_v, 1)
        else:
            sign_v = np.append(sign_v, -1)

    try:
        sign_v = np.append(sign_v, sign_v[-1])
    except IndexError:
        pass

    return sign_v


def shift_indices_to_local(indices, min_x, min_y, pad=1):
    indices_local = indices.copy(deep=True)
    arr = indices_local.indices.values.copy()

    xoff = int(min_x) - pad
    yoff = int(min_y) - pad

    u_mask = ~np.all(np.isnan(arr[:, 0:2]) | (arr[:, 0:2] == 0), axis=1)
    v_mask = ~np.all(np.isnan(arr[:, 2:4]) | (arr[:, 2:4] == 0), axis=1)

    arr[u_mask, 0] = arr[u_mask, 0] - xoff
    arr[u_mask, 1] = arr[u_mask, 1] - yoff
    arr[v_mask, 2] = arr[v_mask, 2] - xoff
    arr[v_mask, 3] = arr[v_mask, 3] - yoff

    indices_local["indices"][:] = arr
    return indices_local

def _sf_preprocess1_ds(ds, preprocess=True, user_rename_dict=None):
    """Preprocess an already-open Dataset for index/grid detection."""
    if preprocess:
        return prepro._preprocess1(ds, user_rename_dict=user_rename_dict)
    if "time" in ds.dims:
        ds = ds.isel(time=0)
    if "lev" in ds.dims:
        ds = ds.isel(lev=0)
    return ds


def _sf_preprocess2_ds(ds, lon_bnds, lat_bnds, preprocess=True, user_rename_dict=None):
    """Preprocess and subset an already-open Dataset for section calculations."""
    if preprocess:
        return prepro._preprocess2(ds, lon_bnds=lon_bnds, lat_bnds=lat_bnds, user_rename_dict=user_rename_dict)
    return ds.sel(x=slice(*lon_bnds), y=slice(*lat_bnds))


def _sf_get_var_dataset(obj, target_name):
    """
    Accept either a Dataset or DataArray and return a Dataset with target_name.
    Useful for supplied mesh metrics or e3u/e3v arrays.
    """
    if obj is None or isinstance(obj, (int, float)):
        return None
    if isinstance(obj, xa.DataArray):
        return obj.to_dataset(name=target_name)
    if isinstance(obj, xa.Dataset):
        if target_name in obj:
            return obj
        original_var_name = list(obj.data_vars)[0]
        return obj.rename({original_var_name: target_name})
    return None


def _ensure_time_axis(ds, template=None):
    """Ensure a Dataset/DataArray has a time dimension."""
    if "time" not in ds.dims:
        if template is not None and "time" in template.coords:
            return ds.expand_dims(dim={"time": template.time})
        return ds.expand_dims(dim={"time": 1})
    return ds


def _select_time_if_present(ds, time_start, time_end):
    """Select requested time range if time is present and not singleton."""
    if "time" in ds.dims and ds.sizes.get("time", 0) > 1:
        return ds.sel(time=slice(str(time_start), str(time_end)))
    return ds


def _load_with_progress(*objects):
    """Load one or more xarray objects, using dask ProgressBar when available."""
    try:
        with ProgressBar():
            loaded = [obj.load() if obj is not None else None for obj in objects]
    except NameError:
        loaded = [obj.load() if obj is not None else None for obj in objects]
    return loaded if len(loaded) > 1 else loaded[0]

def _load_or_calculate_indices(
    strait,
    model,
    ti,
    ui,
    vi,
    coords=0,
    lon_p=0,
    lat_p=0,
    set_latlon=False,
    path_indices="",
    path_save="",
    saving=True,
):
    """Load cached StraitFlux indices or calculate them from surface fields."""
    try:
        return xa.open_dataset(path_indices + model + "_" + strait + "_indices.nc")
    except OSError:
        print("calc indices")
        indices, line = check_availability_indices(ti, strait, model, coords, lon_p, lat_p, set_latlon)
        i2 = indices.indices.where(indices.indices != 0)
        try:
            plt.pcolormesh((ti.thetao / ti.thetao), cmap="tab20c")
            plt.scatter(i2[:, 2], i2[:, 3], color="tab:red", s=0.1, marker="x")
            plt.scatter(i2[:, 0], i2[:, 1], color="tab:red", s=0.1, marker="x")
            plt.title(model + "_" + strait, fontsize=14)
            plt.ylabel("y", fontsize=14)
            plt.xlabel("x", fontsize=14)
            plt.savefig(path_save + strait + "_" + model + "_indices.png")
            plt.close()
        except Exception:
            print("skipping Plot")

        out_u, out_v, _ = prepare_indices(indices)
        check_indices(indices, out_u, out_v, ti, ui, vi, strait, model, path_save)
        if saving and path_indices != "":
            indices.to_netcdf(path_indices + model + "_" + strait + "_indices.nc")
        return indices


def _determine_grid(model, Arakawa, ti, ui, vi, path_mesh="", saving=True):
    """Read cached Arakawa grid type or determine it from u/v/T positions."""
    if Arakawa in ["Arakawa-A", "Arakawa-B", "Arakawa-C"]:
        return Arakawa
    if Arakawa != "":
        raise ValueError("grid not known")

    try:
        with open(path_mesh + model + "grid.txt", "r") as f:
            return f.read()
    except OSError:
        grid = check_Arakawa(ui, vi, ti, model)
        if saving and path_mesh != "":
            with open(path_mesh + model + "grid.txt", "w") as f:
                f.write(grid)
        return grid
    

def _load_or_prepare_mesh(
    model,
    mesh_dxv=0,
    mesh_dyu=0,
    ui=None,
    vi=None,
    path_mesh="",
    preprocess_func=None,
    saving=True,
):

    def _apply_preprocess(ds):
        if preprocess_func is None:
            return ds
        return preprocess_func(ds)

    supplied_dxv = _sf_get_var_dataset(mesh_dxv, "dxv")
    supplied_dyu = _sf_get_var_dataset(mesh_dyu, "dyu")

    if (supplied_dxv is None) != (supplied_dyu is None):
        raise ValueError(
            "Please provide both mesh_dxv and mesh_dyu, or neither of them."
        )

    # 1. User-supplied mesh has highest priority.
    if supplied_dxv is not None and supplied_dyu is not None:
        mv = supplied_dxv
        mu = supplied_dyu

        if saving and path_mesh != "":
            mv.to_netcdf(path_mesh + "mesh_dxv_" + model + ".nc")
            mu.to_netcdf(path_mesh + "mesh_dyu_" + model + ".nc")

        return _apply_preprocess(mu), _apply_preprocess(mv)

    # 2. Otherwise try cached mesh files.
    try:
        mu = xa.open_mfdataset(
            path_mesh + "mesh_dyu_" + model + ".nc",
            preprocess=preprocess_func,
        )
        mv = xa.open_mfdataset(
            path_mesh + "mesh_dxv_" + model + ".nc",
            preprocess=preprocess_func,
        )
        return mu, mv
    except Exception:
        pass

    # 3. Otherwise calculate metrics from u/v grids.
    if ui is None or vi is None:
        raise ValueError(
            "Horizontal meshes were not supplied and cached mesh files were not found. "
            "Please provide ui and vi so meshes can be calculated, or provide "
            "mesh_dxv and mesh_dyu."
        )

    print("meshes not supplied or cached: calc horizontal meshes")
    mu, mv = prepro.calc_dxdy(model, ui, vi, path_mesh)

    return _apply_preprocess(mu), _apply_preprocess(mv)


def calc_vvl_e3_from_ssh(
    model,
    ssh,
    mesh,
    path_save,
    ssh_name="zos",
    time_dim="time",
    e3t0_name="e3t_0",
    e3u0_name="e3u_0",
    e3v0_name="e3v_0",
    tmask_name="tmask",
    umask_name="umask",
    vmask_name="vmask",
    e1t_name="e1t",
    e2t_name="e2t",
    e1u_name="e1u",
    e2u_name="e2u",
    e1v_name="e1v",
    e2v_name="e2v",
    lon_name="nav_lon",
    lat_name="nav_lat",
    lev_t_name="deptht",
    lev_u_name="depthu",
    lev_v_name="depthv",
    save=True,
    dtype="float32",
):
    """
    Calculate time-varying NEMO-style VVL cell thicknesses e3t, e3u and e3v
    from sea-surface height and static mesh variables.

    Parameters
    ----------
    model : str
        Model/product name used in output filenames.

    ssh : xr.Dataset, xr.DataArray, or str
        SSH/zos data or path/pattern to SSH files.

    mesh : xr.Dataset or str
        Mesh-mask dataset or path to mesh file.

    path_save : str
        Folder where monthly e3t/e3u/e3v files are saved.

    Returns
    -------
    xr.Dataset
        Dataset containing e3t, e3u and e3v.
    """

    if isinstance(mesh, str):
        mesh = xa.open_dataset(mesh)

    if isinstance(ssh, str):
        ssh = xa.open_mfdataset(ssh, chunks={time_dim: 1})

    if isinstance(ssh, xa.Dataset):
        ssh = ssh[ssh_name]

    os.makedirs(path_save, exist_ok=True)

    def _drop_first_dim_if_needed(da):
        if da.ndim >= 3 and da.dims[0] not in [lev_t_name, lev_u_name, lev_v_name, "nav_lev", "lev"]:
            return da.isel({da.dims[0]: 0})
        return da

    e3t0 = _drop_first_dim_if_needed(mesh[e3t0_name])
    e3u0 = _drop_first_dim_if_needed(mesh[e3u0_name])
    e3v0 = _drop_first_dim_if_needed(mesh[e3v0_name])

    tmask = _drop_first_dim_if_needed(mesh[tmask_name])
    umask = _drop_first_dim_if_needed(mesh[umask_name])
    vmask = _drop_first_dim_if_needed(mesh[vmask_name])

    e1t = _drop_first_dim_if_needed(mesh[e1t_name])
    e2t = _drop_first_dim_if_needed(mesh[e2t_name])
    e1u = _drop_first_dim_if_needed(mesh[e1u_name])
    e2u = _drop_first_dim_if_needed(mesh[e2u_name])
    e1v = _drop_first_dim_if_needed(mesh[e1v_name])
    e2v = _drop_first_dim_if_needed(mesh[e2v_name])

    e3tm = e3t0 * tmask
    H = e3tm.sum(dim=e3tm.dims[0])

    out_list = []

    for it in range(ssh.sizes[time_dim]):

        ssh_t = ssh.isel({time_dim: it})
        time_value = ssh[time_dim].isel({time_dim: it}).values
        timestamp = pd.Timestamp(time_value)

        print(timestamp)

        lev_t_dim = e3tm.dims[0]
        y_dim = e3tm.dims[-2]
        x_dim = e3tm.dims[-1]

        e3t_vvl = e3tm * (1.0 + ssh_t / H)
        e3t_vvl = e3t_vvl.rename("e3t")

        e3u_anom = np.zeros_like(e3t_vvl.values)
        e3v_anom = np.zeros_like(e3t_vvl.values)

        e3t_v = e3t_vvl.values
        e3t0_v = e3t0.values

        e1t_v = e1t.values
        e2t_v = e2t.values
        e1u_v = e1u.values
        e2u_v = e2u.values
        e1v_v = e1v.values
        e2v_v = e2v.values

        umask_v = umask.values
        vmask_v = vmask.values

        for k in range(e3t_v.shape[0]):
            e3u_anom[k, :, :-1] = (
                0.5
                * umask_v[k, :, :-1]
                * (1.0 / (e1u_v[:, :-1] * e2u_v[:, :-1]))
                * (
                    e1t_v[:, :-1] * e2t_v[:, :-1] * (e3t_v[k, :, :-1] - e3t0_v[k, :, :-1])
                    + e1t_v[:, 1:] * e2t_v[:, 1:] * (e3t_v[k, :, 1:] - e3t0_v[k, :, 1:])
                )
            )

            e3v_anom[k, :-1, :] = (
                0.5
                * vmask_v[k, :-1, :]
                * (1.0 / (e1v_v[:-1, :] * e2v_v[:-1, :]))
                * (
                    e1t_v[:-1, :] * e2t_v[:-1, :] * (e3t_v[k, :-1, :] - e3t0_v[k, :-1, :])
                    + e1t_v[1:, :] * e2t_v[1:, :] * (e3t_v[k, 1:, :] - e3t0_v[k, 1:, :])
                )
            )

        e3u_combined = e3u0.values * umask_v + e3u_anom
        e3v_combined = e3v0.values * vmask_v + e3v_anom

        # cyclic / last-column handling, as in your ORAS6 script
        e3u_combined[:, :, -1] = e3u_combined[:, :, 1]

        depthu = mesh[lev_u_name] if lev_u_name in mesh.coords else e3t_vvl[lev_t_dim]
        depthv = mesh[lev_v_name] if lev_v_name in mesh.coords else e3t_vvl[lev_t_dim]

        coords_3d_u = {
            lev_u_name: depthu.values,
            y_dim: e3t_vvl[y_dim].values,
            x_dim: e3t_vvl[x_dim].values,
        }

        coords_3d_v = {
            lev_v_name: depthv.values,
            y_dim: e3t_vvl[y_dim].values,
            x_dim: e3t_vvl[x_dim].values,
        }

        e3u_da = xa.DataArray(
            e3u_combined,
            dims=(lev_u_name, y_dim, x_dim),
            coords=coords_3d_u,
            name="e3u",
            attrs={"long_name": "U-cell thickness with VVL", "units": "m"},
        )

        e3v_da = xa.DataArray(
            e3v_combined,
            dims=(lev_v_name, y_dim, x_dim),
            coords=coords_3d_v,
            name="e3v",
            attrs={"long_name": "V-cell thickness with VVL", "units": "m"},
        )

        if lon_name in mesh:
            e3t_vvl = e3t_vvl.assign_coords({lon_name: mesh[lon_name], lat_name: mesh[lat_name]})
            e3u_da = e3u_da.assign_coords({lon_name: mesh[lon_name], lat_name: mesh[lat_name]})
            e3v_da = e3v_da.assign_coords({lon_name: mesh[lon_name], lat_name: mesh[lat_name]})

        e3t_month = e3t_vvl.expand_dims({time_dim: [time_value]}).astype(dtype)
        e3u_month = e3u_da.expand_dims({time_dim: [time_value]}).astype(dtype)
        e3v_month = e3v_da.expand_dims({time_dim: [time_value]}).astype(dtype)

        ds_month = xa.Dataset(
            {
                "e3t": e3t_month,
                "e3u": e3u_month,
                "e3v": e3v_month,
            }
        )

        out_list.append(ds_month)

        if save:
            year = timestamp.year
            month = timestamp.month

            e3t_month.to_dataset(name="e3t").to_netcdf(
                f"{path_save}/{model}_e3t_{year}{month:02d}.nc"
            )
            e3u_month.to_dataset(name="e3u").to_netcdf(
                f"{path_save}/{model}_e3u_{year}{month:02d}.nc"
            )
            e3v_month.to_dataset(name="e3v").to_netcdf(
                f"{path_save}/{model}_e3v_{year}{month:02d}.nc"
            )

    return xa.concat(out_list, dim=time_dim)


def normalize_monthly_time(ds, time_dim="time", anchor="MS"):
    """
    Normalize time coordinates to month-start or month-end timestamps.

    anchor:
      "MS" = month start
      "ME" = month end
    """
    if time_dim not in ds.dims:
        return ds

    import pandas as pd

    time = pd.to_datetime(ds[time_dim].values)

    if anchor == "MS":
        new_time = time.to_period("M").to_timestamp(how="start")
    elif anchor == "ME":
        new_time = time.to_period("M").to_timestamp(how="end")
    else:
        raise ValueError("anchor must be 'MS' or 'ME'")

    return ds.assign_coords({time_dim: new_time})