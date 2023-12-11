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
import xesmf as xe
try:
    from dask.diagnostics import ProgressBar
except ImportError:
    print('skipping dask import')

from xmip.preprocessing import rename_cmip6, promote_empty_dims, broadcast_lonlat, correct_coordinates
import StraitFlux.preprocessing as prepro
import StraitFlux.functions as func
import StraitFlux.functions_VP as func2
from StraitFlux.indices import check_availability_indices, prepare_indices

def vel_projection(strait,model,time_start,time_end,file_u,file_v,file_t,file_z,coords=0,set_latlon=False,lat_p=0,lon_p=0,path_save='',path_indices='',path_mesh='',Arakawa='',saving=True):
    '''
    This function calculates the u/v crossection by projecting the vectors onto the strait 
    by multiplying with the amount of the vector going through the strait and
    regridding the values onto the T-proj points on the reference line 

    INPUT Parameters:
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
    Arakawa (str): Arakawa-A, Arakawa-B or Arakawa-C; only needed if automatic check fails
    path_save (str): path to save transport data
    path_indices (str): path to save indices data
    path_mesh (str): path to save mesh data

    RETURNS:
    crosssection of currents at given section

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
        with ProgressBar():
            ti=ti.load()
            ui=ui.load()
            vi=vi.load()
        indices,line = check_availability_indices(ti,strait,model,coords,lon_p,lat_p,set_latlon)
        i2=indices.indices.where(indices.indices!=0) 
        try:
            (ti.thetao/ti.thetao).plot(add_colorbar=False)
            plt.scatter(i2[:,2],i2[:,3],color='tab:red',s=0.1,marker='x')
            plt.scatter(i2[:,0],i2[:,1],color='tab:red',s=0.1,marker='x')
            plt.title(model+'_'+strait,fontsize=14)
            plt.ylabel('y',fontsize=14)
            plt.xlabel('x',fontsize=14)
            plt.savefig(path_save+strait+'_'+model+'_indices.png')
            plt.close()
        except NameError:
            print('skipping Plot')
        indices.to_netcdf(path_indices+model+'_'+strait+'_indices.nc')

    #######
    if Arakawa in ['Arakawa-A','Arakawa-B','Arakawa-C']:
        grid=Arakawa
    elif Arakawa == '':
        try:
            file = open(path_mesh+model+'grid.txt', 'r')
            grid= file.read()
        except OSError:
            partial_func = partial(prepro._preprocess1)
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
            with open(path_mesh+model+'grid.txt', 'w') as f:
                f.write(grid)
    else:
        print('grid not known')
        sys.exit()
    ################     
                 

    try:
        indices = xa.open_dataset(path_save+'indices_long_'+model+strait+'.nc')
        betrag_u = np.loadtxt(path_save+'betrag_u_'+model+strait+'.txt')
        betrag_v = np.loadtxt(path_save+'betrag_v_'+model+strait+'.txt')
        tu = np.loadtxt(path_save+'tu_'+model+strait+'.txt')
        tv = np.loadtxt(path_save+'tv_'+model+strait+'.txt')
        un = np.loadtxt(path_save+'un_'+model+strait+'.txt')
        dist_listT_kurz2 = np.loadtxt(path_save+'dx_'+model+strait+'.txt')
        T_proj_points=xa.open_dataset(path_save+'T_proj_points_'+model+strait+'.nc')
        
    except FileNotFoundError:
        try:
            T_data=ti
            u_data=ui
            v_data=vi
        except NameError:      
            T_data = xa.open_mfdataset(file_t, preprocess=partial_func).isel(time=0)
            u_data = xa.open_mfdataset(file_u, preprocess=partial_func).isel(time=0)
            v_data = xa.open_mfdataset(file_v, preprocess=partial_func).isel(time=0)
            with ProgressBar():
                T_data=T_data.load()
                u_data=u_data.load()
                v_data=v_data.load()
        start = time.time()
        indices,ref_line=check_availability_indices(T_data,strait,model,coords,lon_p,lat_p,set_latlon)
        end = time.time()
        print(end - start)
        start = time.time()
        un,u_line,v_line,T_line,u_line2,v_line2=func2.select_indices(indices,u_data,v_data,T_data)
        proj_u,proj_v,T_proj, u_proj, v_proj=func2.proj_vec(ref_line,u_line,v_line,T_line,u_line2,v_line2)
        T_proj_points,dist_listT,dist_listT_kurz2=func2.calc_interpolation_points(indices,T_data, ref_line)
        betrag_u=func2.calc_betrag(proj_u)
        betrag_v=func2.calc_betrag(proj_v)
        tu,tv=func2.multi_factors(T_line,u_line,v_line,T_proj,u_proj,v_proj)
        end = time.time()
        print(end - start)
        
        if saving == True:
            np.savetxt(path_save+'betrag_u_'+model+strait+'.txt', betrag_u)
            np.savetxt(path_save+'betrag_v_'+model+strait+'.txt', betrag_v)
            np.savetxt(path_save+'tu_'+model+strait+'.txt', tu)
            np.savetxt(path_save+'tv_'+model+strait+'.txt', tv)
            np.savetxt(path_save+'un_'+model+strait+'.txt', un)
            np.savetxt(path_save+'dx_'+model+strait+'.txt', dist_listT_kurz2)
            T_proj_points.to_netcdf(path_save+'T_proj_points_'+model+strait+'.nc')
            indices.to_netcdf(path_save+'indices_long_'+model+strait+'.nc')

    
    out_u,out_v,out_u_vz = prepare_indices(indices)
    min_x=np.nanmin((min(out_u[:,0],default=np.nan),min(out_v[:,0],default=np.nan)))
    max_x=np.nanmax((max(out_u[:,0],default=np.nan),max(out_v[:,0],default=np.nan)))
    min_y=np.nanmin((min(out_u[:,1],default=np.nan),min(out_v[:,1],default=np.nan)))
    max_y=np.nanmax((max(out_u[:,1],default=np.nan),max(out_v[:,1],default=np.nan)))

    if min_x == -1:
        min_x = 0
        max_x = max_x + 1
    #print(min_x,max_x,min_y,max_y)
    #sys.exit()
    print('read t, u and v fields')
    partial_func = partial(prepro._preprocess2,lon_bnds=(int(min_x)-2,int(max_x)+2),lat_bnds=(int(min_y)-2,int(max_y)+2))
    t = xa.open_mfdataset(file_t, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))
    u = xa.open_mfdataset(file_u, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))
    v = xa.open_mfdataset(file_v, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))
    deltaz = xa.open_mfdataset(file_z, preprocess=partial_func,chunks={'time':1})[['thkcello']]
    
    if 'time' in deltaz.dims:
        deltaz=deltaz.sel(time=slice(str(time_start),str(time_end)))

    print('load t, u and v fields')
    with ProgressBar():
        t=t.load()
        u=u.load()
        v=v.load()
        deltaz=deltaz.load()
        
        
    dzu3,dzv3 = func.calc_dz_faces(deltaz,grid,model,path_mesh)
    
    start = time.time()
    print('calculating regridder')
    regridder_u=xe.Regridder(u,T_proj_points,'bilinear',ignore_degenerate=True)
    regridder_v=xe.Regridder(v,T_proj_points,'bilinear',ignore_degenerate=True)
    end = time.time()
    print(end - start)    
    u_beitrag = np.zeros((len(u.time),len(u.lev),len(T_proj_points.lat))) # time,z=75,x=coords of strait
    v_beitrag = np.zeros((len(u.time),len(u.lev),len(T_proj_points.lat)))

    u_trans = np.zeros((len(u.time),len(u.lev),len(u.y),len(u.x)))
    v_trans = np.zeros((len(u.time),len(u.lev),len(u.y),len(u.x)))
    start = time.time()
    for i in range(len(un)):
        if 'time' in deltaz.dims:
            u_trans[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]=u.uo[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]*dzu3[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)].values*betrag_u[i]*tu[i]#
            v_trans[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]=v.vo[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]*dzv3[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)].values*betrag_v[i]*tv[i]        
        else:
            u_trans[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]=u.uo[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]*dzu3[:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)].values*betrag_u[i]*tu[i]#
            v_trans[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]=v.vo[:,:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)]*dzv3[:,int(un[i,1]-min_y+2),int(un[i,0]-min_x+2)].values*betrag_v[i]*tv[i]#

    
    
    u['uo'][:,:,:,:]=u_trans[:,:,:,:]#
    v['vo'][:,:,:,:]=v_trans[:,:,:,:]#
    end = time.time()
    print(end - start) 
    print('regridding')
    start = time.time()
    u_l=regridder_u(u.uo.fillna(0))#
    v_l=regridder_v(v.vo.fillna(0))#
    end = time.time()
    print(end - start) 
    for m in range(len(u_l.lat)):
        for k in range(len(u.lev)):
            u_beitrag[:,k,m] = u_l[:,k].isel(lat=m,lon=m).values
            v_beitrag[:,k,m] = v_l[:,k].isel(lat=m,lon=m).values
            
    gesamt=np.where(np.isnan(u_beitrag),0,u_beitrag)+np.where(np.isnan(v_beitrag),0,v_beitrag)
    dz2=np.gradient(u.lev)
    dz3=np.transpose([dz2]*np.shape(gesamt)[-1])
    uv_tot = xa.Dataset({'uv':(('time','depth','x'),gesamt/dz3*(u_beitrag[0]/u_beitrag[0])),'dx_int':(('x'),dist_listT_kurz2),'dz_int':(('depth'),dz2.data)},coords=dict(time=u.time,depth=u.lev.data,x=np.cumsum(dist_listT_kurz2)))
    uv_tot.to_netcdf(path_save+strait+'_crosssection_uv_'+model+'_'+str(time_start)+'-'+str(time_end)+'.nc')
    return uv_tot


def TS_interp(product,strait,model,time_start,time_end,file_u,file_t,file_s='',coords=0,set_latlon=False,lat_p=0,lon_p=0,path_save='',path_indices='',path_mesh='',saving=True):
    '''
    This function calculates the u/v crossection by projecting the vectors onto the strait 
    by multiplying with the amount of the vector going through the strait and
    regridding the values onto the T-proj points on the reference line 

    INPUT Parameters:
    product (str): T or S
    strait (str): desired oceanic strait, either pre-defined from indices file or new
    model (str): desired CMIP6 model or reanalysis
    time_start (str or int): starting year
    time_end (str or int): ending year
    file_u (str): path + filename(s) of u field(s); use ice velocities (ui) for ice transports; (multiple files possible, use *; must be possible to combine files over time coordinate)
    file_t (str): path + filename(s) of temperature field(s); (multiple files possible, use *)

    OPTIONAL:
    coords (tuple): coordinates for strait, if not pre-defined: (latitude_start,longitude_start,latitude_end,longitude_end)
    set_latlon: set True if you wish to pass arrays of latitudes and longitudes
    lon (array): longitude coordinates for strait, if not pre-defined. (range -180 to 180; same length as lat needed!)
    lat (array): latitude coordinates for strait, if not pre-defined. (range -90 to 90; same length as lon needed!)
    file_s (str): only needed for salinity transports; path + filename(s) of salinity field(s); (multiple files possible, use *)
    Arakawa (str): Arakawa-A, Arakawa-B or Arakawa-C; only needed if automatic check fails
    path_save (str): path to save transport data
    path_indices (str): path to save indices data
    path_mesh (str): path to save mesh data

    RETURNS:
    crosssection of temperature (T) or salinity (S) at given section

    '''
    
    partial_func = partial(prepro._preprocess1)
    try:
        indices=xa.open_dataset(path_indices+model+'_'+strait+'_indices.nc')
    except OSError:
        print('calc indices')
        print('read and load files for indices')
        ti = xa.open_mfdataset(file_t, preprocess=partial_func).isel(time=0)
        with ProgressBar():
            ti=ti.load()
        indices,line = check_availability_indices(ti,strait,model,coords,lon_p,lat_p,set_latlon)
        i2=indices.indices.where(indices.indices!=0) 
        try:
            (ti.thetao/ti.thetao).plot(add_colorbar=False)
            plt.scatter(i2[:,2],i2[:,3],color='tab:red',s=0.1,marker='x')
            plt.scatter(i2[:,0],i2[:,1],color='tab:red',s=0.1,marker='x')
            plt.title(model+'_'+strait,fontsize=14)
            plt.ylabel('y',fontsize=14)
            plt.xlabel('x',fontsize=14)
            plt.savefig(path_save+strait+'_'+model+'_indices.png')
            plt.close()
        except NameError:
            print('skipping Plot')
        indices.to_netcdf(path_indices+model+'_'+strait+'_indices.nc')
        
    out_u,out_v,out_u_vz = prepare_indices(indices)
    min_x=np.nanmin((min(out_u[:,0],default=np.nan),min(out_v[:,0],default=np.nan)))
    max_x=np.nanmax((max(out_u[:,0],default=np.nan),max(out_v[:,0],default=np.nan)))
    min_y=np.nanmin((min(out_u[:,1],default=np.nan),min(out_v[:,1],default=np.nan)))
    max_y=np.nanmax((max(out_u[:,1],default=np.nan),max(out_v[:,1],default=np.nan)))

    if min_x == -1:
        min_x = 0
        max_x = max_x + 1
    #print(min_x,max_x,min_y,max_y)
    
    
    try:
        T_proj_points=xa.open_dataset(path_save+'T_proj_points_'+model+strait+'.nc')
        dist_listT_kurz2 = np.loadtxt(path_save+'dx_'+model+strait+'.txt')
    except FileNotFoundError:
        try:
            T_data=ti
        except NameError:      
            T_data = xa.open_mfdataset(file_t, preprocess=partial_func).isel(time=0)
            with ProgressBar():
                T_data=T_data.load()
        start = time.time()
        indices,ref_line=check_availability_indices(T_data,strait,model,coords,lon_p,lat_p,set_latlon)
        T_proj_points,dist_listT,dist_listT_kurz2=func2.calc_interpolation_points(indices,T_data, ref_line)
        
        if saving == True:
            np.savetxt(path_save+'dx_'+model+strait+'.txt', dist_listT_kurz2)
            T_proj_points.to_netcdf(path_save+'T_proj_points_'+model+strait+'.nc')
    

    print('read t and/or s fields')
    partial_func = partial(prepro._preprocess2,lon_bnds=(int(min_x)-5,int(max_x)+5),lat_bnds=(int(min_y)-5,int(max_y)+5))
    if product == 'T':
        t = xa.open_mfdataset(file_t, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))
    elif product == 'S':
        t = xa.open_mfdataset(file_s, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end)))

    u = xa.open_mfdataset(file_u, preprocess=partial_func,chunks={'time':1}).sel(time=slice(str(time_start),str(time_end))).isel(time=0)
    with ProgressBar():
        t=t.load()
        u=u.load() 
    if product == 'T':
        t['mask'] = t.thetao[0]/t.thetao[0]
        t['mask'] = xa.where(~np.isnan(t['mask']),1,0)
    elif product == 'S':
        t['mask'] = t.so[0]/t.so[0]
        t['mask'] = xa.where(~np.isnan(t['mask']),1,0)
    
    regridder=[]
    print('calculating regridder')
    for s in tqdm(range(len(t.lev))):
        regridder_T=xe.Regridder(t.isel(lev=s),T_proj_points,'bilinear',ignore_degenerate=True,extrap_method='nearest_s2d')
        regridder=np.append(regridder,regridder_T)
    
    T_beitrag2 = np.zeros((len(t.time),len(t.lev),len(T_proj_points.lat),len(T_proj_points.lat)))
    
    print('regridding')
    if product == 'T':
        for s in range(len(t.lev)):
            T_beitrag2[:,s,:,:] = regridder[s](t['thetao'].isel(lev=s))
    elif product == 'S':
        for s in range(len(t.lev)):
            T_beitrag2[:,s,:,:] = regridder[s](t['so'].isel(lev=s))
        
    T_beitrag = np.zeros((len(t.time),len(t.lev),len(T_proj_points.lat)))
    
    for j in range(len(T_proj_points.lat)):
        for k in range(len(t.lev)): ##75
            T_beitrag[:,k,j] = T_beitrag2[:,k,j,j]

            
    ## Mask same as for uv profiles:
    M_beitrag = np.zeros((len(t.lev),len(T_proj_points.lat)))
    regridder_M=xe.Regridder(u,T_proj_points,'bilinear',ignore_degenerate=True)
    M=regridder_M(u.uo.fillna(0))
    for m in range(len(M.lat)):
        for k in range(len(M.lev)):
            M_beitrag[k,m] = M[k].isel(lat=m,lon=m).values
    if product == 'T':            
        T_tot = xa.Dataset({'T':(('time','depth','x'),T_beitrag*(M_beitrag/M_beitrag))},coords=dict(time=t.time,depth=t.lev.data,x=np.cumsum(dist_listT_kurz2)))
        T_tot.to_netcdf(path_save+strait+'_crosssection_T_'+model+'_'+str(time_start)+'-'+str(time_end)+'.nc')
    elif product == 'S':            
        T_tot = xa.Dataset({'S':(('time','depth','x'),T_beitrag*(M_beitrag/M_beitrag))},coords=dict(time=t.time,depth=t.lev.data,x=np.cumsum(dist_listT_kurz2)))
        T_tot.to_netcdf(path_save+strait+'_crosssection_S_'+model+'_'+str(time_start)+'-'+str(time_end)+'.nc')
    return T_tot
