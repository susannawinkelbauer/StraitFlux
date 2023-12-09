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




def transports(product,strait,model,time_start,time_end,file_u,file_v,file_t,file_z,coords=0,set_latlon=False,lon_p=0,lat_p=0,file_s='',file_sic='',file_sit='',Arakawa='',rho=1026,cp=3996, Tref=0,path_save='',path_indices='',path_mesh='',saving=True):

    '''Calculation of Transports using line integration

    INPUT Parameters:
    product (str): volume, heat, salt or ice
    strait (str): desired oceanic strait, either pre-defined from indices file or new
    model (str): desired CMIP6 model or reanalysis
    time_start (str or int): starting year
    time_end (str or int): ending year
    file_u (str): path + filename(s) of u field(s); use ice velocities (ui) for ice transports; (multiple files possible, use *; must be possible to combine files over time coordinate)
    file_v (str): path + filename(s) of v field(s); use ice velocities (vi) for ice transports; (multiple files possible, use *)
    file_t (str): path + filename(s) of temperature field(s); (multiple files possible, use *)
    file_z (str): path + filename(s) of cell thickness field(s); (multiple files possible, use *)

    OPTIONAL:
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


    partial_func = partial(prepro._preprocess1)


    try:
        indices=xa.open_dataset(path_indices+model+'_'+strait+'_indices.nc')
    except OSError:
        print('calc indices')
        print('read and load files for indices')
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
            plt.pcolormesh((ui.uo))
            plt.scatter(i2[:,2],i2[:,3],color='tab:red',s=0.1,marker='x')
            plt.scatter(i2[:,0],i2[:,1],color='tab:red',s=0.1,marker='x')
            plt.title(model+'_'+strait,fontsize=14)
            plt.ylabel('y',fontsize=14)
            plt.xlabel('x',fontsize=14)
            plt.savefig(path_save+strait+'_'+model+'_indices.png')
            plt.close()
        except NameError:
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
                grid = func.check_Arakawa(ui,vi,ti,product,model)
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
                grid = func.check_Arakawa(ui,vi,ti,product,model)
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

    try:
        mu=xa.open_dataset(path_mesh+'mesh_dyu_'+model+'.nc')
        mv=xa.open_dataset(path_mesh+'mesh_dxv_'+model+'.nc')
    except FileNotFoundError:
        print('calc horizontal meshes')
        try:
            mu,mv = prepro.calc_dxdy(model,ui,vi,path_mesh)
        except NameError:
            print('read and load files for mesh')
            ui = xa.open_mfdataset(file_u, preprocess=partial_func).isel(time=0)
            vi = xa.open_mfdataset(file_v, preprocess=partial_func).isel(time=0)
            try:
                with ProgressBar():
                    ui=ui.load()
                    vi=vi.load()
            except NameError:
                ui=ui.load()
                vi=vi.load()
            mu,mv = prepro.calc_dxdy(model,ui,vi,path_mesh)


    print('read t, u and v fields')
    partial_func = partial(prepro._preprocess2,lon_bnds=(int(min_x)-1,int(max_x)+1),lat_bnds=(int(min_y)-1,int(max_y)+1))
    t = xa.open_mfdataset(file_t, preprocess=partial_func,chunks={'time':1})
    u = xa.open_mfdataset(file_u, preprocess=partial_func,chunks={'time':1})
    v = xa.open_mfdataset(file_v, preprocess=partial_func,chunks={'time':1})
    if 'time' in t.dims and t.dims['time'] > 1:
        t=t.sel(time=slice(str(time_start),str(time_end)))
        u=u.sel(time=slice(str(time_start),str(time_end)))
        v=v.sel(time=slice(str(time_start),str(time_end)))
    elif 'time' not in t.dims:
        t=t.expand_dims(dim={"time": 1})
        u=u.expand_dims(dim={"time": 1})
        v=v.expand_dims(dim={"time": 1})
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
            deltaz=deltaz.load()
    except NameError:
        t=t.load()
        u=u.load()
        v=v.load()
        deltaz=deltaz.load()


    dzu3,dzv3 = func.calc_dz_faces(deltaz,grid,model,path_mesh)
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
        print('calc u')
        udata=udata*mu.dyu.values*sit.sithick.values*sic.siconc.values
        print('calc v')
        vdata=vdata*mv.dxv.values*sit.sithick.values*sic.siconc.values

    udata = udata.fillna(0.)
    vdata = vdata.fillna(0.)
    print('calc line')
    if product in ['volume','heat','salt']:
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
    trans = trans.drop_vars('tot_'+product+'_flux')
    trans.to_netcdf(path_save+strait+'_'+product+'_'+model+'_'+str(time_start)+'-'+str(time_end)+'.nc')

    return trans




