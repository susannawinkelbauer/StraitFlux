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

def check_Arakawa(u_data,v_data,T_data,product,model):

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
            print('grid not recognized, check manually')
            sys.exit()
    elif u.lat[int(len(t.y)/2),0].values == v.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values == v.lon[int(len(t.y)/2),0].values:
        if u.lat[int(len(t.y)/2),0].values != t.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values != t.lon[int(len(t.y)/2),0].values:
            grid='Arakawa-B'
            print(grid)
        elif u.lat[int(len(t.y)/2),0].values == t.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values == t.lon[int(len(t.y)/2),0].values:
            grid='Arakawa-A'
            print(grid)
        elif u.lat[int(len(t.y)/2),0].values == t.lat[int(len(t.y)/2),0].values == v.lat[int(len(t.y)/2),0].values and u.lon[int(len(t.y)/2),0].values != t.lon[int(len(ti.y)/2),0].values != v.lon[int(len(ti.y)/2),0].values:
            grid='Arakawa-E'
            print(grid+'?')
        else:
            print('grid not recognized, check manually')
            sys.exit()
    else:
        print('grid not recognized, check manually')
        sys.exit()
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
    #print(t)
    try:
        plt.title(model+'_'+strait,fontsize=14)
        u.uo.plot()
        plt.scatter(out_v[:,0],out_v[:,1]+0.5,marker='_',c='r',s=200)
        plt.scatter(out_u[:,0]+0.5,out_u[:,1],marker='|',c='r',s=200)
        plt.ylabel('y',fontsize=14)
        plt.xlabel('x',fontsize=14)
        plt.savefig(path_save+strait+'_'+model+'_indices_check.png')
        plt.close()
    except NameError:
        print('skipping Plot')

def interp_TS(ds,d):
    return ds.rolling({d:2},min_periods=1).mean()

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
