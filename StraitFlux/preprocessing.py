import xarray as xa
import numpy as np
from tqdm import tqdm
from xmip.preprocessing import rename_cmip6,promote_empty_dims, broadcast_lonlat, correct_coordinates

def _merge_rename_dicts(default_dict, user_rename_dict=None):
    """Merge user-supplied rename candidates into the default rename dictionary.

    user_rename_dict should have the same structure as renaming_dict_exp(),
    e.g. {"thetao": ["temp"], "uo": ["uvel"]}. User names are
    prepended so they have priority, while default names remain available.
    """
    if user_rename_dict is None:
        return default_dict

    merged = {key: list(value) for key, value in default_dict.items()}

    for key, values in user_rename_dict.items():
        if isinstance(values, str):
            values = [values]
        else:
            values = list(values)

        if key in merged:
            merged[key] = list(dict.fromkeys(values + merged[key]))
        else:
            merged[key] = list(dict.fromkeys(values))

    return merged


def renaming_dict_exp(user_rename_dict=None):
    rename_dict = {
        # dim labels (order represents the priority when checking for the dim labels)
        "x": ["i", "ni", "xh", "nlon","ncl3"],
        "y": ["j", "nj", "yh", "nlat","ncl2"],
        "lev": ["deptht", "olevel", "zlev", "olev", "depth","depthu","depthv","ncl1","nav_lev", "z"],
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
        "uo": ["vozocrtx","siu","u"],
        "vo": ["vomecrty","siv","v"],
        "thetao": ["votemper"],
        "so": ["vosaline"],
        "thkcello": ["e3t", "e3t_0", "e3t_0_field"],
        "dxv":["e1v"],
        "dyu":["e2u"],
        "sithick": ["sithick", "sit", "SIT", "iicethic"],
        "siconc": ["siconc", "sic", "SIC", "iiceconc"]
    }
    return _merge_rename_dicts(rename_dict, user_rename_dict)


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
        "lev": ["deptht", "olevel", "zlev", "olev", "depth", "depthu", "depthv", "ncl1", "nav_lev", "z"],
        "time": ["time_counter", "t"],
    }
    for i in ds.coords:
        for target, candidates in rename_dict.items():
            if i in candidates:
                ds = ds.rename({i: target})
    return ds

def drop_cyclic_points(ds):
    tt,ind=np.unique(ds.lon,axis=1,return_index=True)
    ds=ds.sel(x=np.sort(ind))
    return ds

def drop_northsouth_duplicate_points(ds):
    tt,ind=np.unique(ds.lat,axis=0,return_index=True)
    ds=ds.sel(y=np.sort(ind))
    return ds

def drop_cyclic_points(ds):
    tt,ind=np.unique(ds.lon,axis=1,return_index=True)
    ds=ds.sel(x=np.sort(ind))
    return ds


def rename_vertical_metrics(ds):
    """Rename U/V vertical thickness metrics without losing T/U/V identity.

    The generic CMIP-style name ``thkcello`` is reserved for T-cell
    thickness. U- and V-face thicknesses are kept separately as
    ``thkcello_u`` and ``thkcello_v``. This allows one mesh file to contain
    e3t_0, e3u_0 and e3v_0 simultaneously.
    """
    rename_map = {}

    t_candidates = ["e3t", "e3t_0", "e3t_0_field"]
    u_candidates = ["e3u", "e3u_0", "e3u_0_field"]
    v_candidates = ["e3v", "e3v_0", "e3v_0_field"]

    for old in t_candidates:
        if old in ds and "thkcello" not in ds:
            rename_map[old] = "thkcello"
            break

    for old in u_candidates:
        if old in ds and "thkcello_u" not in ds:
            rename_map[old] = "thkcello_u"
            break

    for old in v_candidates:
        if old in ds and "thkcello_v" not in ds:
            rename_map[old] = "thkcello_v"
            break

    if rename_map:
        ds = ds.rename(rename_map)

    return ds

def ensure_2d_latlon(ds):
    """
    Convert 1D lat/lon coordinates on regular grids to 2D lat(y, x), lon(y, x).

    StraitFlux internally expects 2D latitude/longitude arrays. Curvilinear grids
    already have this format, but regular lat-lon grids often store lat(y) and
    lon(x). This function converts those to 2D arrays.
    """

    if "lat" not in ds or "lon" not in ds:
        return ds

    lat = ds["lat"]
    lon = ds["lon"]

    if lat.ndim == 2 and lon.ndim == 2:
        return ds

    if lat.ndim != 1 or lon.ndim != 1:
        return ds

    lat_dim = lat.dims[0]
    lon_dim = lon.dims[0]

    lat2d, lon2d = np.meshgrid(lat.values, lon.values, indexing="ij")

    ds = ds.assign_coords(
        {
            "lat": ((lat_dim, lon_dim), lat2d),
            "lon": ((lat_dim, lon_dim), lon2d),
        }
    )

    return ds

def _save_1d_latlon(ds):
    """Save 1D latitude/longitude values before xmip renaming can drop them."""

    lat_candidates = ["lat", "latitude", "nav_lat"]
    lon_candidates = ["lon", "longitude", "nav_lon"]

    lat_name = next((name for name in lat_candidates if name in ds), None)
    lon_name = next((name for name in lon_candidates if name in ds), None)

    if lat_name is None or lon_name is None:
        return ds, None, None

    lat = ds[lat_name]
    lon = ds[lon_name]

    if lat.ndim == 1 and lon.ndim == 1:
        return ds, lat.values.copy(), lon.values.copy()

    return ds, None, None

def _restore_regular_1d_latlon(ds, lat_vals, lon_vals):
    """Restore saved regular-grid 1D lat/lon as 2D StraitFlux coordinates."""

    if lat_vals is None or lon_vals is None:
        return ds, False

    if "y" not in ds.dims or "x" not in ds.dims:
        return ds, False

    if ds.sizes["y"] != len(lat_vals) or ds.sizes["x"] != len(lon_vals):
        return ds, False

    lon_vals = np.asarray(lon_vals)
    lat_vals = np.asarray(lat_vals)

    lon_vals = np.where(lon_vals > 180, lon_vals - 360, lon_vals)

    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    for name in ["lat", "lon", "latitude", "longitude", "nav_lat", "nav_lon"]:
        if name in ds:
            ds = ds.drop_vars(name)

    ds = ds.assign_coords(
        y=np.arange(ds.sizes["y"]),
        x=np.arange(ds.sizes["x"]),
        lat=(("y", "x"), lat2d),
        lon=(("y", "x"), lon2d),
    )

    return ds, True



def wrapper(ds, user_rename_dict=None):
    ds = ds.copy()
    ds, saved_lat, saved_lon = _save_1d_latlon(ds)
    ds = rename_time_lev(ds)
    ds = rename_cmip6(ds, rename_dict=renaming_dict_exp(user_rename_dict))
    ds = rename_vertical_metrics(ds)
    ds, restored_regular_grid = _restore_regular_1d_latlon(ds, saved_lat, saved_lon)
    #if restored_regular_grid:
    #    return ds
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
    ds = drop_cyclic_points(ds)
    ds = drop_northsouth_duplicate_points(ds)

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

def calc_dxdy(model, u, v, path_mesh):

    lat_u = u.lat.values
    lon_u = u.lon.values
    lat_v = v.lat.values
    lon_v = v.lon.values

    Ny_u, Nx_u = lat_u.shape
    Ny_v, Nx_v = lat_v.shape

    dy = np.zeros_like(lat_u, dtype=np.float64)
    dx = np.zeros_like(lat_v, dtype=np.float64)

    for i in tqdm(range(Ny_u - 1), desc="dy (north-south)"):
        dy[i + 1, :] = distance(
            lat_u[i, :], lon_u[i, :],
            lat_u[i + 1, :], lon_u[i + 1, :]
        )

    for j in tqdm(range(Nx_v - 1), desc="dx (east-west)"):
        dx[:, j + 1] = distance(
            lat_v[:, j], lon_v[:, j],
            lat_v[:, j + 1], lon_v[:, j + 1]
        )

    dy_da = xa.DataArray(
        dy * 1000.0,
        coords=u.lat.coords,
        dims=u.lat.dims,
        name="dyu",
    )

    dx_da = xa.DataArray(
        dx * 1000.0,
        coords=v.lat.coords,
        dims=v.lat.dims,
        name="dxv",
    )

    mu = dy_da.to_dataset(name="dyu")
    mv = dx_da.to_dataset(name="dxv")

    mu.to_netcdf(path_mesh + "mesh_dyu_" + model + ".nc")
    mv.to_netcdf(path_mesh + "mesh_dxv_" + model + ".nc")

    return mu, mv


def calc_dxdy_old(model,u,v,path_mesh):

    dy=xa.DataArray(data=np.zeros(u.lat.shape),coords=u.lat.coords,dims=u.lat.dims)
    dx=xa.DataArray(data=np.zeros(v.lat.shape),coords=v.lat.coords,dims=v.lat.dims)
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
def _preprocess1(x, user_rename_dict=None):
    x=wrapper(x, user_rename_dict=user_rename_dict)
    return x.isel(lev=0)

def _preprocess11(x, user_rename_dict=None):
    x=wrapper(x, user_rename_dict=user_rename_dict)
    return x

## for indices selection ice
def _preprocess1i(x, user_rename_dict=None):
    x=wrapper(x, user_rename_dict=user_rename_dict)
    return x

## for actuall fields
def _preprocess2(x, lon_bnds, lat_bnds, user_rename_dict=None):
    x=wrapper(x, user_rename_dict=user_rename_dict)
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

def make_3d_dz_from_depth(
    ds,
    depth_name="depth",
    lat_name="latitude",
    lon_name="longitude",
    varname="thkcello",
):
    """
    Create 3D layer thickness thkcello(depth, latitude, longitude)
    from a 1D depth coordinate.

    No bathymetry or partial bottom cells are applied.
    """

    z = ds[depth_name].values.astype(float)

    # Layer interfaces halfway between depth centers
    zi = np.zeros(len(z) + 1)
    zi[0] = 0.0
    zi[1:-1] = 0.5 * (z[:-1] + z[1:])
    zi[-1] = z[-1] + 0.5 * (z[-1] - z[-2])

    dz = np.diff(zi)

    # Broadcast to 3D
    dz3d = np.broadcast_to(
        dz[:, None, None],
        (len(ds[depth_name]), len(ds[lat_name]), len(ds[lon_name])),
    )

    thkcello = xa.DataArray(
        dz3d,
        dims=(depth_name, lat_name, lon_name),
        coords={
            depth_name: ds[depth_name],
            lat_name: ds[lat_name],
            lon_name: ds[lon_name],
        },
        name=varname,
    )

    thkcello.attrs["units"] = "m"
    thkcello.attrs["long_name"] = "layer thickness derived from depth coordinate"

    return thkcello.to_dataset()


