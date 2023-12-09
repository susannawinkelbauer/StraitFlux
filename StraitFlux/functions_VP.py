import xarray as xa
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('skipping matplotlib')
from tqdm import tqdm
import sys
from dask.diagnostics import ProgressBar
from StraitFlux.indices import check_availability_indices, prepare_indices
import StraitFlux.preprocessing as prepro

def select_indices(indices,u_data,v_data,T_data):
    '''
    This function selects indices on T,u and v grid
    args:
        path_indices: path with indices files
        strait: Fram,Barents,GSR,etc.
        path_raw_data: path to raw fata files
        reana: ORAS5,FOAM,CGLORS,GLORYS2V4
    '''

    selection = indices.indices.values
    sels = np.zeros((len(selection),2))
    i=0
    for i in range(len(selection)):
        if selection[i,0] == 0 and selection[i,1] == 0:
            sels[i,0] = selection[i,2]
            sels[i,1] = selection[i,3]
        else:
            sels[i,0] = selection[i,0]
            sels[i,1] = selection[i,1]

    ### take points north, east, west and south of indices
    uniques=np.unique(sels,axis=0)
    uniques_n=uniques.copy()
    uniques_n[:,0]=uniques[:,0]+1
    uniques_e=uniques.copy()
    uniques_e[:,1]=uniques[:,1]+1
    uniques_s=uniques.copy()
    uniques_s[:,0]=uniques[:,0]-1
    uniques_w=uniques.copy()
    uniques_w[:,1]=uniques[:,1]-1
    uniques_ne=uniques.copy()
    uniques_ne[:,:]=uniques+1
    uniques_se=uniques.copy()
    uniques_se[:,1]=uniques[:,1]+1
    uniques_se[:,0]=uniques[:,0]-1
    uniques_nw=uniques.copy()
    uniques_nw[:,1]=uniques[:,1]-1
    uniques_nw[:,0]=uniques[:,0]+1
    uniques_sw=uniques.copy()
    uniques_sw[:,:]=uniques-1
    
    

    uniques2=np.concatenate((uniques,uniques_n,uniques_e,uniques_s,uniques_w,uniques_ne,uniques_se,uniques_nw,uniques_sw),axis=0) 
    uniques_new=np.unique(uniques2,axis=0)


    u_points_lons, u_points_lats = [], []
    v_points_lons, v_points_lats = [], []
    T_points_lons, T_points_lats = [], []
    u_points_lons2, u_points_lats2 = [], []
    v_points_lons2, v_points_lats2 = [], []
    T_points_lons2, T_points_lats2 = [], []
    
    
    # coordinates of u_points:
    i=0
    for i in range(len(uniques_new)):
        # selected points:
        u_points_lon = u_data.isel(x=int(uniques_new[i,0]),y=int(uniques_new[i,1])).lon.values
        u_points_lons = np.append(u_points_lons,u_points_lon)
        u_points_lat = u_data.isel(x=int(uniques_new[i,0]),y=int(uniques_new[i,1])).lat.values
        u_points_lats = np.append(u_points_lats,u_points_lat)
    # coordinates of v_points:
    j=0
    for j in range(len(uniques_new)):
        v_points_lon = v_data.isel(x=int(uniques_new[j,0]),y=int(uniques_new[j,1])).lon.values
        v_points_lons = np.append(v_points_lons,v_points_lon)
        v_points_lat = v_data.isel(x=int(uniques_new[j,0]),y=int(uniques_new[j,1])).lat.values
        v_points_lats = np.append(v_points_lats,v_points_lat)
    # coordinates of T_points:
    k=0
    for k in range(len(uniques_new)):
        T_points_lon = T_data.isel(x=int(uniques_new[k,0]),y=int(uniques_new[k,1])).lon.values
        T_points_lons = np.append(T_points_lons,T_points_lon)
        T_points_lat = T_data.isel(x=int(uniques_new[k,0]),y=int(uniques_new[k,1])).lat.values
        T_points_lats = np.append(T_points_lats,T_points_lat)
    k=0
    for k in range(len(sels)):
        T_points_lon2 = T_data.isel(x=int(sels[k,0]),y=int(sels[k,1])).lon.values
        T_points_lons2 = np.append(T_points_lons2,T_points_lon2)
        T_points_lat2 = T_data.isel(x=int(sels[k,0]),y=int(sels[k,1])).lat.values
        T_points_lats2 = np.append(T_points_lats2,T_points_lat2)

    i=0
    for i in range(len(uniques_new)):
        # selected points:
        try:
            u_points_lon2 = u_data.isel(x=int(uniques_new[i,0]+1),y=int(uniques_new[i,1])).lon.values
        except IndexError:
            u_points_lon2 = u_data.isel(x=0,y=int(uniques_new[i,1])).lon.values
        u_points_lons2 = np.append(u_points_lons2,u_points_lon2)
        try:
            u_points_lat2 = u_data.isel(x=int(uniques_new[i,0]+1),y=int(uniques_new[i,1])).lat.values
        except IndexError:
            u_points_lat2 = u_data.isel(x=0,y=int(uniques_new[i,1])).lat.values
        u_points_lats2 = np.append(u_points_lats2,u_points_lat2)
    # coordinates of v_points:
    j=0
    for j in range(len(uniques_new)):
        try:
            v_points_lon2 = v_data.isel(x=int(uniques_new[j,0]),y=int(uniques_new[j,1])+1).lon.values
        except IndexError:
            v_points_lon2 = v_data.isel(x=int(uniques_new[j,0]),y=int(uniques_new[j,1])).lon.values
        v_points_lons2 = np.append(v_points_lons2,v_points_lon2)
        try:
            v_points_lat2 = v_data.isel(x=int(uniques_new[j,0]),y=int(uniques_new[j,1])+1).lat.values
        except IndexError:
            v_points_lat2 = v_data.isel(x=int(uniques_new[j,0]),y=int(uniques_new[j,1])).lat.values
        v_points_lats2 = np.append(v_points_lats2,v_points_lat2)
    # coordinates of T_points:


    # provide line
    u_line = xa.DataArray(np.full((len(uniques_new), len(uniques_new)),1),dims=['lat','lon'],coords=[u_points_lats,u_points_lons])
    v_line = xa.DataArray(np.full((len(uniques_new), len(uniques_new)),1),dims=['lat','lon'],coords=[v_points_lats,v_points_lons])
    T_line = xa.DataArray(np.full((len(uniques_new), len(uniques_new)),1),dims=['lat','lon'],coords=[T_points_lats,T_points_lons])
    T_line2 = xa.DataArray(np.full((len(sels), len(sels)),1),dims=['lat','lon'],coords=[T_points_lats2,T_points_lons2])
    u_line2 = xa.DataArray(np.full((len(uniques_new), len(uniques_new)),1),dims=['lat','lon'],coords=[u_points_lats2,u_points_lons2])
    v_line2 = xa.DataArray(np.full((len(uniques_new), len(uniques_new)),1),dims=['lat','lon'],coords=[v_points_lats2,v_points_lons2])

    return uniques_new,u_line,v_line,T_line,u_line2,v_line2

def get_nearest_r(ref_line,u_line,v_line,T_line):
    '''
    This function calculates the nearest point on the reference line and it's neighbour for each T-point on the native grid
    '''

    mini = np.zeros((len(T_line),2)) # two columns, first for nearest point and second for neighbour
    i=0
    
    for i in tqdm(range(len(T_line))):
    #calc distance for every point of line to every point of ref line
        dist_list_single_point = []
        j=0
        for j in range(len(ref_line)):
            dist = prepro.distance(ref_line.lat[j].values,ref_line.lon[j].values,T_line.lat[i].values,T_line.lon[i].values)
            dist_list_single_point = np.append(dist_list_single_point,dist)

        minimum = dist_list_single_point.tolist().index(min(dist_list_single_point))
        #print(len(ref_line))
        #sys.exit()
        mini[i,0] = minimum     # nearest point
        if mini[i,0] == len(ref_line)-1:
            mini[i,1] = minimum-1  # left neighbouring point
        else:
            mini[i,1] = minimum+1  # right neighbouring point

    r_1_lat=[]
    r_1_lon=[]
    r_2_lat=[]
    r_2_lon=[]
    ##### select points on T-line
    for i in range(len(T_line)):
        r1lat=ref_line[int(mini[i,0]),int(mini[i,0])].lat
        r1lon=ref_line[int(mini[i,0]),int(mini[i,0])].lon
        r2lat=ref_line[int(mini[i,1]),int(mini[i,1])].lat
        r2lon=ref_line[int(mini[i,1]),int(mini[i,1])].lon
        r_1_lat=np.append(r_1_lat,r1lat)
        r_1_lon=np.append(r_1_lon,r1lon)
        r_2_lat=np.append(r_2_lat,r2lat)
        r_2_lon=np.append(r_2_lon,r2lon)
        
    return r_1_lat,r_1_lon,r_2_lat,r_2_lon

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def calc_normvec(ref_line,u_line,v_line,T_line):
    '''
    This function calculates the vectors normal to the reference line at each u,v and T point using 3 cross products
    '''
    a = 6378388
    normu_x, normu_y, normu_z = [], [], []
    normv_x, normv_y, normv_z = [], [], []
    normT_x, normT_y, normT_z = [], [], []
    
    r_1_lat,r_1_lon,r_2_lat,r_2_lon=get_nearest_r(ref_line,u_line,v_line,T_line)
    ref_x,ref_y,ref_z=prepro.kugel_2_kart(ref_line.lat.values,ref_line.lon.values)
    u_x,u_y,u_z=prepro.kugel_2_kart(u_line.lat.values,u_line.lon.values)
    v_x,v_y,v_z=prepro.kugel_2_kart(v_line.lat.values,v_line.lon.values)
    T_x,T_y,T_z=prepro.kugel_2_kart(T_line.lat.values,T_line.lon.values)
    r1_x,r1_y,r1_z=prepro.kugel_2_kart(r_1_lat,r_1_lon)
    r2_x,r2_y,r2_z=prepro.kugel_2_kart(r_2_lat,r_2_lon)
    
    Tprojx, Tprojy, Tprojz= [], [], []
    uprojx, uprojy, uprojz= [], [], []
    vprojx, vprojy, vprojz= [], [], []
    
    for i in range(len(T_x)):
        r_1=[r1_x[i],r1_y[i],r1_z[i]]
        r_2=[r2_x[i],r2_y[i],r2_z[i]]
        n=np.cross(np.array(r_1),np.array(r_2))  ## cross product 1
        #calc normvec T
        T=[T_x[i],T_y[i],T_z[i]]
        pT=np.cross(np.array(T),n)  ## cross product 2
        T_proj=np.cross(n,pT)  ## cross product 3
        T_proj=normalize(T_proj)*a  ## normalize and multiply by Earth radius to obtain point on ref line
        Tprojx=np.append(T_proj[0],Tprojx)
        Tprojy=np.append(T_proj[1],Tprojy)
        Tprojz=np.append(T_proj[2],Tprojz)
        T_t_x=T_proj[0]-T[0]   ### difference between T point and projected point on ref-line = normal vector
        T_t_y=T_proj[1]-T[1]
        T_t_z=T_proj[2]-T[2]
        normT_x=np.append(normT_x,T_t_x)
        normT_y=np.append(normT_y,T_t_y)
        normT_z=np.append(normT_z,T_t_z)
        #calc normvec u
        u=[u_x[i],u_y[i],u_z[i]]
        pu=np.cross(np.array(u),n)
        u_proj=np.cross(n,pu)
        u_proj=normalize(u_proj)*a
        uprojx=np.append(u_proj[0],uprojx)
        uprojy=np.append(u_proj[1],uprojy)
        uprojz=np.append(u_proj[2],uprojz)
        u_t_x=u_proj[0]-u[0]
        u_t_y=u_proj[1]-u[1]
        u_t_z=u_proj[2]-u[2]
        normu_x=np.append(normu_x,u_t_x)
        normu_y=np.append(normu_y,u_t_y)
        normu_z=np.append(normu_z,u_t_z)
        #calc normvec v
        v=[v_x[i],v_y[i],v_z[i]]
        pv=np.cross(np.array(v),n)
        v_proj=np.cross(n,pv)
        v_proj=normalize(v_proj)*a
        vprojx=np.append(v_proj[0],vprojx)
        vprojy=np.append(v_proj[1],vprojy)
        vprojz=np.append(v_proj[2],vprojz)
        v_t_x=v_proj[0]-v[0]
        v_t_y=v_proj[1]-v[1]
        v_t_z=v_proj[2]-v[2]
        normv_x=np.append(normv_x,v_t_x)
        normv_y=np.append(normv_y,v_t_y)
        normv_z=np.append(normv_z,v_t_z)
        
    Tproj_lat,Tproj_lon=prepro.kart_2_kugel(Tprojx,Tprojy,Tprojz)
    T_proj = xa.DataArray(np.full((len(Tproj_lat), len(Tproj_lon)),1),dims=['lat','lon'],coords=[np.flip(Tproj_lat),np.flip(Tproj_lon)])
    T_proj = T_proj.to_dataset(name='T')
    uproj_lat,uproj_lon=prepro.kart_2_kugel(uprojx,uprojy,uprojz)
    u_proj = xa.DataArray(np.full((len(uproj_lat), len(uproj_lon)),1),dims=['lat','lon'],coords=[np.flip(uproj_lat),np.flip(uproj_lon)])
    u_proj = u_proj.to_dataset(name='u')
    vproj_lat,vproj_lon=prepro.kart_2_kugel(vprojx,vprojy,vprojz)
    v_proj = xa.DataArray(np.full((len(vproj_lat), len(vproj_lon)),1),dims=['lat','lon'],coords=[np.flip(vproj_lat),np.flip(vproj_lon)])
    v_proj = v_proj.to_dataset(name='v')
    return normT_x,normT_y,normT_z, T_proj, u_proj, v_proj

def calc_dir_vector(u_line,v_line,u_line2,v_line2):
    '''
    This function calculates the direct u and v vectors
    '''

    ux,uy,uz = prepro.kugel_2_kart(u_line.lat.values,u_line.lon.values) #koordinaten uPunkt -Koordinaten TPunkt
    vx,vy,vz = prepro.kugel_2_kart(v_line.lat.values,v_line.lon.values) #koordinaten vPunkt -Koordinaten T PUnkt
    ux2,uy2,uz2 = prepro.kugel_2_kart(u_line2.lat.values,u_line2.lon.values) #koordinaten uPunkt -Koordinaten TPunkt
    vx2,vy2,vz2 = prepro.kugel_2_kart(v_line2.lat.values,v_line2.lon.values) #koordinaten vPunkt -Koordinaten T PUnkt

    rv_u = np.zeros((len(u_line),3))
    rv_v = np.zeros((len(v_line),3))
    i=0
    for i in tqdm(range(len(u_line))):
        vec_ux = ux2[i]-ux[i]#ux[i] - tx[i]#
        vec_uy = uy2[i]-uy[i]#uy[i] - ty[i]#
        vec_uz = uz2[i]-uz[i]#uz[i] - tz[i]#
        rv_u[i,0] = vec_ux
        rv_u[i,1] = vec_uy
        rv_u[i,2] = vec_uz
        
        vec_vx = vx2[i]-vx[i]#vx[j] - tx[j]#
        vec_vy = vy2[i]-vy[i]#vy[j] - ty[j]#
        vec_vz = vz2[i]-vz[i]#vz[j] - tz[j]#
        rv_v[i,0] = vec_vx
        rv_v[i,1] = vec_vy
        rv_v[i,2] = vec_vz
        
    return rv_u, rv_v

def proj_vec(ref_line,u_line,v_line,T_line,u_line2,v_line2):
    '''
    This function calculates the projection vectors usind the normal and direct vectors
    '''
    
    print('.. calculating normal vectors')
    normT_x,normT_y,normT_z,T_proj, u_proj, v_proj = calc_normvec(ref_line,u_line,v_line,T_line) #normu_x,normu_y,normu_z,normv_x,normv_y,normv_z
    print('.. calculating direct vectors')
    u_dirvec, v_dirvec = calc_dir_vector(u_line,v_line,u_line2,v_line2)
    u_dir = np.zeros((len(u_dirvec),3))
    v_dir = np.zeros((len(v_dirvec),3))
    k=0
    for k in range(len(u_dirvec)):
        u_dir[k,0] = (1/np.sqrt((u_dirvec[k,0]**2 + u_dirvec[k,1]**2 + u_dirvec[k,2]**2))) * u_dirvec[k,0]
        u_dir[k,1] = (1/np.sqrt((u_dirvec[k,0]**2 + u_dirvec[k,1]**2 + u_dirvec[k,2]**2))) * u_dirvec[k,1]
        u_dir[k,2] = (1/np.sqrt((u_dirvec[k,0]**2 + u_dirvec[k,1]**2 + u_dirvec[k,2]**2))) * u_dirvec[k,2]
        v_dir[k,0] = (1/np.sqrt((v_dirvec[k,0]**2 + v_dirvec[k,1]**2 + v_dirvec[k,2]**2))) * v_dirvec[k,0]
        v_dir[k,1] = (1/np.sqrt((v_dirvec[k,0]**2 + v_dirvec[k,1]**2 + v_dirvec[k,2]**2))) * v_dirvec[k,1]
        v_dir[k,2] = (1/np.sqrt((v_dirvec[k,0]**2 + v_dirvec[k,1]**2 + v_dirvec[k,2]**2))) * v_dirvec[k,2]

    norm_betragT = np.sqrt((normT_x**2 + normT_y**2 +normT_z**2))
    zaehler_u = (u_dir[:,0] * normT_x) + (u_dir[:,1] * normT_y) + (u_dir[:,2] * normT_z)
    zaehler_v = (v_dir[:,0] * normT_x) + (v_dir[:,1] * normT_y) + (v_dir[:,2] * normT_z)
    term1_u = zaehler_u/norm_betragT**2 
    term1_v = zaehler_v/norm_betragT**2
    proj_u = np.zeros((len(normT_x),3))
    proj_v = np.zeros((len(normT_x),3))
    for i in range(len(normT_x)):
        proj_u[i,0] = term1_u[i] *normT_x[i]
        proj_u[i,1] = term1_u[i] *normT_y[i]
        proj_u[i,2] = term1_u[i] *normT_z[i]
        proj_v[i,0] = term1_v[i] *normT_x[i]
        proj_v[i,1] = term1_v[i] *normT_y[i]
        proj_v[i,2] = term1_v[i] *normT_z[i]

    return proj_u,proj_v,T_proj, u_proj, v_proj

def calc_dx(T_proj_points):
    
    dist_listT = []
    for k in range(len(T_line)-1):
        distsT = prepro.distance(T_proj_points.lat[k],T_proj_points.lon[k],T_proj_points.lat[k+1],T_proj_points.lon[k+1])
        dist_listT = np.append(dist_listT,distsT)
    dist_lastT = prepro.distance(T_proj_points.lat[-2],T_proj_points.lon[-2],T_proj_points.lat[-1],T_proj_points.lon[-1])
    dist_listT = np.append(dist_listT,dist_lastT)

    T_proj_lat=T_proj_points.lat*dist_listT/dist_listT
    T_proj_lat=T_proj_lat[~np.isnan(T_proj_lat)]
    T_proj_lon=T_proj_points.lon*dist_listT/dist_listT
    T_proj_lon=T_proj_lon[~np.isnan(T_proj_lon)]
    
    dist_listT_kurz2 = []
    dist_firstT_kurz2 = prepro.distance(T_proj_lat[0],T_proj_lon[0],T_proj_lat[1],T_proj_lon[1])
    dist_listT_kurz2 = np.append(dist_listT_kurz2,dist_firstT_kurz2)
    for k in range(1,len(T_proj_lon)-1):      
        distsT_kurz2 = prepro.distance(T_proj_lat[k-1],T_proj_lon[k-1],T_proj_lat[k],T_proj_lon[k])/2 + prepro.distance(T_proj_lat[k],T_proj_lon[k],T_proj_lat[k+1],T_proj_lon[k+1])/2
        dist_listT_kurz2 = np.append(dist_listT_kurz2,distsT_kurz2)
    dist_lastT_kurz2 = prepro.distance(T_proj_lat[-2],T_proj_lon[-2],T_proj_lat[-1],T_proj_lon[-1])
    dist_listT_kurz2 = np.append(dist_listT_kurz2,dist_lastT_kurz2)

    return dist_listT,dist_listT_kurz2,T_proj_lat, T_proj_lon

def calc_betrag(projvar):
    betrag = np.sqrt((projvar[:,0]**2 + projvar[:,1]**2 + projvar[:,2]**2))
    return betrag

########### Hier ev. noch verbessern!!!!

def multi_factors(T_line,u_line,v_line,T_proj,u_proj,v_proj):
    T_dist=prepro.distance(T_line.lat.values+1,T_line.lon.values+1,T_proj.lat.values,T_proj.lon.values)
    u_dist=prepro.distance(u_line.lat.values+1,u_line.lon.values+1,u_proj.lat.values,u_proj.lon.values)
    v_dist=prepro.distance(v_line.lat.values+1,v_line.lon.values+1,v_proj.lat.values,v_proj.lon.values)
    tu=u_dist-T_dist
    tv=v_dist-T_dist
    tu[tu<0]=-1
    tu[tu>0]=1
    tv[tv<0]=-1
    tv[tv>0]=1
    
    return tu,tv

def calc_interpolation_points(indices,T_data, ref_line):
    '''
    This function selects indices on T,u and v grid
    args:
        path_indices: path with indices files
        strait: Fram,Barents,GSR,etc.
        path_raw_data: path to raw fata files
        reana: ORAS5,FOAM,CGLORS,GLORYS2V4
    '''
    selection = indices.indices.values
    sels = np.zeros((len(selection),2))
    i=0
    for i in range(len(selection)):
        if selection[i,0] == 0:
            sels[i,0] = selection[i,2]
            sels[i,1] = selection[i,3]
        else:
            sels[i,0] = selection[i,0]
            sels[i,1] = selection[i,1]


    # prepate for coordinates
    T_points_lons = []
    T_points_lats = []
    # coordinates of T_points:
    k=0
    for k in range(len(sels)):
        T_points_lon = T_data.isel(x=int(sels[k,0]),y=int(sels[k,1])).lon.values
        T_points_lons = np.append(T_points_lons,T_points_lon)
        T_points_lat = T_data.isel(x=int(sels[k,0]),y=int(sels[k,1])).lat.values
        T_points_lats = np.append(T_points_lats,T_points_lat)
    T_line = xa.DataArray(np.full((len(sels), len(sels)),1),dims=['lat','lon'],coords=[T_points_lats,T_points_lons])

    mini = np.zeros((len(T_line),2)) # two columns, first for nearest point and second for neighbour
    i=0
    for i in range(len(T_line)):
    #calc distance for every point of line to every point of ref line
        dist_list_single_point = []
        j=0
        for j in range(len(ref_line)):
            dist = prepro.distance(ref_line.lat[j].values,ref_line.lon[j].values,T_line.lat[i].values,T_line.lon[i].values)
            dist_list_single_point = np.append(dist_list_single_point,dist)
        minimum = dist_list_single_point.tolist().index(min(dist_list_single_point))

        mini[i,0] = minimum     # nearest point
        if mini[i,0] == len(ref_line)-1:
            mini[i,1] = minimum-1  # left neighbouring point
        else:
            mini[i,1] = minimum+1  # right neighbouring point


    r_1_lat=[]
    r_1_lon=[]
    r_2_lat=[]
    r_2_lon=[]
    ##### select points on T-line
    for i in range(len(T_line)):
        r1lat=ref_line[int(mini[i,0]),int(mini[i,0])].lat
        r1lon=ref_line[int(mini[i,0]),int(mini[i,0])].lon
        r2lat=ref_line[int(mini[i,1]),int(mini[i,1])].lat
        r2lon=ref_line[int(mini[i,1]),int(mini[i,1])].lon
        r_1_lat=np.append(r_1_lat,r1lat)
        r_1_lon=np.append(r_1_lon,r1lon)
        r_2_lat=np.append(r_2_lat,r2lat)
        r_2_lon=np.append(r_2_lon,r2lon)

    a = 6378388
    normT_x=[]
    normT_y=[]
    normT_z=[]
    ref_x,ref_y,ref_z=prepro.kugel_2_kart(ref_line.lat.values,ref_line.lon.values)
    T_x,T_y,T_z=prepro.kugel_2_kart(T_line.lat.values,T_line.lon.values)
    r1_x,r1_y,r1_z=prepro.kugel_2_kart(r_1_lat,r_1_lon)
    r2_x,r2_y,r2_z=prepro.kugel_2_kart(r_2_lat,r_2_lon)
    Tprojx=[]
    Tprojy=[]
    Tprojz=[]
    for i in range(len(T_x)):
        r_1=[r1_x[i],r1_y[i],r1_z[i]]
        r_2=[r2_x[i],r2_y[i],r2_z[i]]
        n=np.cross(np.array(r_1),np.array(r_2))  ## cross product 1
        #calc normvec T
        T=[T_x[i],T_y[i],T_z[i]]
        pT=np.cross(np.array(T),n)  ## cross product 2
        T_proj=np.cross(n,pT)  ## cross product 3
        T_proj=normalize(T_proj)*a  ## normalize and multiply by Earth radius to obtain point on ref line
        Tprojx=np.append(T_proj[0],Tprojx)
        Tprojy=np.append(T_proj[1],Tprojy)
        Tprojz=np.append(T_proj[2],Tprojz)

    res,ind = np.unique(Tprojx, return_index=True)
    Tprojx = res[np.argsort(ind)]
    res,ind = np.unique(Tprojy, return_index=True)
    Tprojy = res[np.argsort(ind)]
    res,ind = np.unique(Tprojz, return_index=True)
    Tprojz = res[np.argsort(ind)]
    
    
    Tproj_lat,Tproj_lon=prepro.kart_2_kugel(Tprojx,Tprojy,Tprojz)
    T_proj = xa.DataArray(np.full((len(Tproj_lat), len(Tproj_lon)),1),dims=['lat','lon'],coords=[np.flip(Tproj_lat),np.flip(Tproj_lon)])
    T_proj = T_proj.to_dataset(name='T')

    dist_listT = []
    for k in range(len(Tprojx)-1):
        distsT = prepro.distance(T_proj.lat[k],T_proj.lon[k],T_proj.lat[k+1],T_proj.lon[k+1])
        dist_listT = np.append(dist_listT,distsT)
    dist_lastT = prepro.distance(T_proj.lat[-2],T_proj.lon[-2],T_proj.lat[-1],T_proj.lon[-1])
    dist_listT = np.append(dist_listT,dist_lastT)

    T_proj_lat=T_proj.lat*dist_listT/dist_listT
    T_proj_lat=T_proj_lat[~np.isnan(T_proj_lat)]
    T_proj_lon=T_proj.lon*dist_listT/dist_listT
    T_proj_lon=T_proj_lon[~np.isnan(T_proj_lon)]
    
    dist_listT_kurz2 = []
    dist_firstT_kurz2 = prepro.distance(T_proj_lat[0],T_proj_lon[0],T_proj_lat[1],T_proj_lon[1])
    dist_listT_kurz2 = np.append(dist_listT_kurz2,dist_firstT_kurz2)
    for k in range(1,len(T_proj_lon)-1):      
        distsT_kurz2 = prepro.distance(T_proj_lat[k-1],T_proj_lon[k-1],T_proj_lat[k],T_proj_lon[k])/2 + prepro.distance(T_proj_lat[k],T_proj_lon[k],T_proj_lat[k+1],T_proj_lon[k+1])/2
        dist_listT_kurz2 = np.append(dist_listT_kurz2,distsT_kurz2)
    dist_lastT_kurz2 = prepro.distance(T_proj_lat[-2],T_proj_lon[-2],T_proj_lat[-1],T_proj_lon[-1])
    dist_listT_kurz2 = np.append(dist_listT_kurz2,dist_lastT_kurz2)

    return T_proj,dist_listT*1000,dist_listT_kurz2*1000

