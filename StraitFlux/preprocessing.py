import xarray as xa
import numpy as np
from tqdm import tqdm
from xmip.preprocessing import rename_cmip6,promote_empty_dims, broadcast_lonlat, correct_coordinates

def renaming_dict_exp():
    rename_dict = {
        # dim labels (order represents the priority when checking for the dim labels)
        "x": ["i", "ni", "xh", "nlon","ncl3"],
        "y": ["j", "nj", "yh", "nlat","ncl2"],
        "lev": ["deptht", "olevel", "zlev", "olev", "depth","depthu","depthv","ncl1"],
        "bnds": ["bnds", "axis_nbounds", "d2"],
        "vertex": ["vertex", "nvertex", "vertices"],
        # coordinate labels
        "lon": ["longitude", "nav_lon"],
        "lat": ["latitude", "nav_lat"],
        "lev_bounds": [
            "deptht_bounds",
            "lev_bnds",
            "olevel_bounds",
            "zlev_bnds",
        ],
        "lon_bounds": [
            "bounds_lon",
            "bounds_nav_lon",
            "lon_bnds",
            "x_bnds",
            "vertices_longitude",
        ],
        "lat_bounds": [
            "bounds_lat",
            "bounds_nav_lat",
            "lat_bnds",
            "y_bnds",
            "vertices_latitude",
        ],
        "time_bounds": ["time_bnds"],
        # variables
        "uo": ["vozocrtx"],
        "vo": ["vomecrty"],
        "thetao": ["votemper"],
        "so": ["vosaline"],
        "thkcello": ["e3t_0_field"],
    }
    return rename_dict


def lon_to_180(ds):
    lon = ds["lon"].where(ds["lon"] <= 180, ds["lon"] - 360)
    ds = ds.assign_coords(lon=lon)
    return ds

def corr_updown(ds):
    if np.sign(ds.lat[-1,0].values) == -1:
        ds['y']=np.arange(len(ds.y)-1,-1,-1)
        ds=ds.sortby('y')
    return ds

def corr_indices(ds):
    if np.sign(ds.y.min()) == -1:
        ds['y']=np.arange(1,len(ds.y)+1)
    if np.sign(ds.x.min()) == -1:
        ds['x']=np.arange(1,len(ds.x)+1)
    if ds.x.min() <= 0.9:
        ds['x']=np.arange(1,len(ds.x)+1)
    if ds.y.min() <= 0.9:
        ds['y']=np.arange(1,len(ds.y)+1)
    return ds

def corr_CMCC(ds):
    ds=ds.rename({'x':'y','y':'x'})
    return ds


def corr_xyseries(ds):
    try:
        if [i for i in ds[str([i for i in ds.data_vars][0])].dims].index("x") < [i for i in ds[str([i for i in ds.data_vars][0])].dims].index("y"):
            ds=ds.rename({'x':'y','y':'x'})
    except ValueError:
        pass
    return ds

def corr_dims_latlon(ds):
    if ds.lon.dims[0]=='x' and ds.lon.dims[1]=='y':
        print('correcting lat/lon dimensions')
        ds['lon']=ds.lon.transpose("y", "x")
        ds['lat']=ds.lat.transpose("y", "x")
    return ds

def corr_xy_points(ds):
    ds['x']=np.arange(0,len(ds.x))
    ds['y']=np.arange(0,len(ds.y))
    return ds

def corr_lonlat_dims(ds):
    if len(ds.lon.dims) > 2:
        ds['lon']=ds.lon[0]
        ds['lat']=ds.lat[0]
    return ds

def rename_time_lev(ds):
    rename_dict = {
        "lev": ["deptht", "olevel", "zlev", "olev", "depth","depthu","depthv","ncl1"],
        "time": ["time_counter"],
    }
    for i in ds.coords:
        for target, candidates in rename_dict.items():
            if i in candidates:
                ds = ds.rename({i: target})
    return ds

def wrapper(ds):
    ds = ds.copy()
    ds = rename_time_lev(ds)
    ds = rename_cmip6(ds,rename_dict=renaming_dict_exp())
    ds = promote_empty_dims(ds)
    ds = broadcast_lonlat(ds)
    ds = correct_coordinates(ds)
    ds = lon_to_180(ds)
    ds = corr_xyseries(ds)
    ds = corr_indices(ds)
    ds = corr_dims_latlon(ds)
    ds = corr_lonlat_dims(ds)
    ds = corr_updown(ds)
    ds = corr_xy_points(ds)
    return ds

def distance(lat1,lon1,lat2,lon2):
    '''
    This function calculates the distance between two lat/lon points
    args:
        lat1: latitude of point 1
        lon1: longitude of point 1
        lat2: latitude of point 2
        lon2: longitude of point 2
    '''
    lat11 = np.radians(lat1)
    lon11 = np.radians(lon1)
    lat22 = np.radians(lat2)
    lon22 = np.radians(lon2)
    a = 6378.388    
    dx=np.cos(lat22)*np.cos(lon22)-np.cos(lat11)*np.cos(lon11)
    dy=np.cos(lat22)*np.sin(lon22)-np.cos(lat11)*np.sin(lon11)
    dz=np.sin(lat22)-np.sin(lat11)
    c=np.sqrt(dx*dx+dy*dy+dz*dz)
    distance=a*2*np.arcsin(c/2) 
    return distance

def calc_dxdy(model,u,v,path_mesh):

    dx=xa.DataArray(data=np.zeros(u.lat.shape),coords=u.lat.coords,dims=u.lat.dims)
    dy=xa.DataArray(data=np.zeros(v.lat.shape),coords=v.lat.coords,dims=v.lat.dims)
    for i in tqdm(range(0,len(u.y)-1)):
        #print(i)
        dy[i+1,:]=distance(u.lat[i,:],u.lon[i,:],u.lat[i+1,:],u.lon[i+1,:])
    for i in tqdm(range(0,len(v.x)-1)):
        #print(i)
        dx[:,i+1]=distance(v.lat[:,i],v.lon[:,i],v.lat[:,i+1],v.lon[:,i+1])
            
    mu=(dy*1000).to_dataset(name='dyu')
    mv=(dx*1000).to_dataset(name='dxv')
    mu.to_netcdf(path_mesh+'mesh_dyu_'+model+'.nc')
    mv.to_netcdf(path_mesh+'mesh_dxv_'+model+'.nc')
    return mu,mv

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('',a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape(unique_a.shape[0],a.shape[1])

def select_area(ds,out_u,out_v):
    min_x=np.nanmin((min(out_u[:,0],default=np.nan),min(out_v[:,0],default=np.nan)))
    max_x=np.nanmax((max(out_u[:,0],default=np.nan),max(out_v[:,0],default=np.nan)))
    min_y=np.nanmin((min(out_u[:,1],default=np.nan),min(out_v[:,1],default=np.nan)))
    max_y=np.nanmax((max(out_u[:,1],default=np.nan),max(out_v[:,1],default=np.nan)))
    ds = ds.sel(x=slice(int(min_x)-1,int(max_x)+1),y=slice(int(min_y)-1,int(max_y)+1))
    return ds

## for indices selection
def _preprocess1(x):
    x=wrapper(x)
    return x.isel(lev=0)

## for actuall fields
def _preprocess2(x, lon_bnds, lat_bnds):
    x=wrapper(x)
    return x.sel(x=slice(*lon_bnds), y=slice(*lat_bnds))


def kugel_2_kart(lat,lon):
    '''
    This function transforms from spherical to kartesian coordinates
    args:
        lat and lon
    '''
    radius =  6378388. # m
    x = radius * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = radius * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = radius * np.sin(np.radians(lat))
    return x,y,z

def kart_2_kugel(x,y,z):
    '''
    This function transforms from kartesian to spherical coordinates
    args:
        x,y,z
    '''
    lat = np.degrees(np.arcsin(z/(np.sqrt(x*x+y*y+z*z))))
    lon2 = []
    for i in range(len(x)):
        if x[i] > 0:
            lon = np.degrees(np.arctan(y[i]/x[i]))
        elif y[i] > 0:
            lon = np.degrees(np.arctan(y[i]/x[i])) + 180
        else:
            lon = np.degrees(np.arctan(y[i]/x[i])) - 180
        lon2=np.append(lon2,lon)
    return lat,lon2

