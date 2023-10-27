import os
import xarray as xa
import numpy as np
import sys


def def_indices(strait,coords,lon_p,lat_p,set_latlon):
    if set_latlon == True:
        lon = lon_p
        lat = lat_p
    elif coords!=0:
        if np.abs(coords[2]-coords[0]) <= np.abs(coords[3]-coords[1]):
            lon = np.array(np.arange(coords[1],coords[3],0.1))
            lat = np.array(np.linspace(coords[0],coords[2],len(lon)))
        else:
            lat = np.array(np.arange(coords[0],coords[2],0.1))
            lon = np.array(np.linspace(coords[1],coords[3],len(lat)))
    elif strait == 'Bering':
        lon = np.array(np.arange(-170.5,-166,0.1))
        lat = np.array(np.linspace(65.99,65.75,len(lon)))
    elif strait == 'Fram':
        lon = np.array(np.arange(-20.7,12,0.1))
        lat = np.array(np.linspace(78.82,78.836,len(lon)))
    elif strait == 'Davis':
        lon = np.array(np.arange(-61.8,-52.5,0.1))
        lat = np.array(np.linspace(66.65,67.31,len(lon)))
    elif strait == 'Barents':
        lat = np.array(np.arange(78,69.2,-0.1))
        lon = np.array(np.linspace(18,19.8,len(lat)))
    elif strait == 'RAPID':
        lon = np.array(np.arange(-80.5,-13.5,0.1))
        lat = np.array(np.linspace(26.0,26.0,len(lon)))
    elif strait == 'OSNAP':
        lon1 = np.array(np.arange(-57,-44.6,0.1))
        lat1 = np.array(np.linspace(52,60.1888,len(lon1)))
        lon2 = np.array(np.arange(-44.6,-5.5,0.1))
        lat2 = np.array(np.linspace(60.1888,56.5,len(lon2)))
        lat = np.append(lat1,lat2)
        lon = np.append(lon1,lon2)
    elif strait == 'GSR':
        lon1 = np.array(np.arange(-30.8212,-23.2447,0.1))
        lon2 = np.array(np.arange(-23.2447,-15.0679,0.1))
        lon3 = np.array(np.arange(-15.0679,-6.8700,0.1))
        lon4 = np.array(np.arange(-6.8700,-1.1695,0.1))
        lon5 = np.array(np.arange(-1.1695,6.10547,0.1))
        lon = []
        lon = np.append(lon,lon1)
        lon = np.append(lon,lon2)
        lon = np.append(lon,lon3)
        lon = np.append(lon,lon4)
        lon = np.append(lon,lon5)
        lat1 = np.array(np.linspace(68.5318,66.0050,len(lon1)))
        lat2 = np.array(np.linspace(66.0050,64.4121,len(lon2)))
        lat3 = np.array(np.linspace(64.4121,62.0669,len(lon3)))
        lat4 = np.array(np.linspace(62.0669,60.2777,len(lon4)))
        lat5 = np.array(np.linspace(60.2777,59.4703,len(lon5)))
        lat = []
        lat = np.append(lat,lat1)
        lat = np.append(lat,lat2)
        lat = np.append(lat,lat3)
        lat = np.append(lat,lat4)
        lat = np.append(lat,lat5)
    elif strait == 'Makassar':
        lon = np.array(np.arange(116,119,0.1))
        lat = np.array(np.linspace(-2.87,-2.87,len(lon)))
    elif strait == 'Hudson':
        lon = np.array(np.arange(-82.832771,-82.59107,0.1))
        lat = np.array(np.linspace(69.66432,69.89028,len(lon)))
    elif strait == 'Hudson_NW':
        lon = np.array(np.arange(-84.500731,-84.317820,0.1))
        lat = np.array(np.linspace(69.85032,70.001132,len(lon)))
    elif strait == 'Färöer':
        lat = np.array(np.arange(62.5,63.875,0.1))
        lon = np.array(np.linspace(-6.125,-6.125,len(lat)))
    elif strait == 'Gibraltar':
        lat = np.array(np.arange(34.5,37.5,0.1))
        lon = np.array(np.linspace(-5.59,-5.59,len(lat)))
    elif strait == 'NIIC':
        lon = np.array(np.arange(-30.8212,-23.2447,0.1))
        lat = np.array(np.linspace(68.5318,66.0050,len(lon)))
    elif strait == 'IF':
        lon = np.array(np.arange(-15.0679,-6.8700,0.1))
        lat = np.array(np.linspace(64.4121,62.0669,len(lon)))
    elif strait == 'FS':
        lon = np.array(np.arange(-6.8700,-1.1695,0.1))
        lat = np.array(np.linspace(62.0669,60.2777,len(lon)))
    else:
        print('strait not defined, please provide coordinates: coords=(lat_start,lon_start,lat_end,lon_end)')
        sys.exit()

    return lat,lon


def distance(lat1,lon1,lat2,lon2):
    '''
    This funtion provides the distance calculation on the sphere surface (source: https://www.kompf.de/gps/distcalc.html)
    args:
        lat1,lon1 = pair of values on reference line
        lat2,lat2 = pair of values for which the distance to the reference line should be tested
    returns: 
        distance of (lat1,lon1) and (lat2,lon2) in kilometers
    '''
    lat11 = np.radians(lat1)
    lon11 = np.radians(lon1)
    lat22 = np.radians(lat2)
    lon22 = np.radians(lon2)
    a = 6378.388
    part1 = np.sin(lat11)*np.sin(lat22) 
    part2 = np.cos(lat11)*np.cos(lat22)*np.cos(lon22-lon11)
    term = part1+part2
    # following if/else statement is necessary due to numerical issues
    if type(term) != float:   
        term = np.where(term>1,1,term)
    else:
        if term >1:
            term=1
    distance = a * np.arccos(term)
    return distance

def selection_window(lat1,lon1,lat2,lon2,Tdataset):
    '''
    This funtion selects the nearest point on Tgrid within a selection window with a size of +- 0.5 degrees around the first point on the reference curve.
    args:
         Tdataset: xa.Dataset of Tdata on Tgrid 
         lat1,lon1 = pair of values on reference line
         lat2,lat2 = pair of values for which the distance to the reference line should be tested
    returns:
         xloc: index of nearest x on Tgrid
         yloc: index of nearest y on Tgrid
    '''
    delta=2
    x1_x2 = np.where((lat2>(lat1-delta))&(lat2<(lat1+delta)),1,0)
    y1_y2 = np.where((lon2>(lon1-delta))&(lon2<(lon1+delta)),1,0)
    fenster = x1_x2 + y1_y2
    fenster = np.where(fenster==2,1,0)
    lat2 = (lat2 * fenster)
    lon2 = (lon2 * fenster)
    dist1 = distance(lat1,lon1,lat2,lon2)

    if Tdataset.lat.dims[0] == 'x':
        xloc,yloc = np.where(dist1==np.nanmin(dist1))
    else:
        yloc,xloc = np.where(dist1==np.nanmin(dist1))

    yloc=yloc[0]
    xloc=xloc[0]
    return xloc, yloc

def select_points(Tdataset,xstart,ystart,lat_line_point,lon_line_point):
    '''
    This function test the neighboring grid boxes which index is the next one.
    args:
        Tdataset: xa.Dataset of Tdata on Tgrid
        xstart: x index of point from where the neighboring boxes have to be checked
        ystart: y index of point from where the neighhoring boxes have to be checked
        lat_line_point: latitude of point at reference curve
        lon_line_point: longitude of point at reference curve
    returns:
        xstart: x index of new next point
        ystart: y index of new next point
    '''
    # select adjoining points/gridboxes
    mid = Tdataset.isel(x=xstart,y=ystart)
    if ystart >= len(Tdataset.y)-1:
        above = Tdataset.isel(x=xstart,y=ystart)
    else:
        above = Tdataset.isel(x=xstart,y=ystart+1)
    under = Tdataset.isel(x=xstart,y=ystart-1)
    left = Tdataset.isel(x=xstart-1,y=ystart)
    right = Tdataset.isel(x=xstart+1,y=ystart)
    # calculate distance:
    mid_dist = distance(lat_line_point,lon_line_point,mid.lat.values,mid.lon.values)
    abo_dist =  distance(lat_line_point,lon_line_point,above.lat.values,above.lon.values)
    und_dist =  distance(lat_line_point,lon_line_point,under.lat.values,under.lon.values)
    lef_dist =  distance(lat_line_point,lon_line_point,left.lat.values,left.lon.values)
    rig_dist =  distance(lat_line_point,lon_line_point,right.lat.values,right.lon.values)
    # select point with minmal distance:
    testlist = list([mid_dist,abo_dist,und_dist,lef_dist,rig_dist])

    min_pos = testlist.index(min(testlist))
    
    if min_pos == 0:
        xstart,ystart = xstart,ystart
    elif min_pos ==1:
        xstart,ystart = xstart,ystart+1
    elif min_pos ==2:
        xstart,ystart = xstart,ystart-1
    elif min_pos ==3:
        xstart,ystart = xstart-1,ystart
    elif min_pos ==4:
        xstart,ystart = xstart+1,ystart
    return xstart,ystart


def check_availability_indices(Tdataset,strait,model,coords,lon_p,lat_p,set_latlon):
    print('calculating indices...')

    lat,lon = def_indices(strait,coords,lon_p,lat_p,set_latlon)


    line = xa.DataArray(data=1,dims=['lat','lon'],coords=[lat,lon])


    # first point of line:
    start_lat, start_lon = line.isel(lat=0,lon=0).lat.values, line.isel(lat=0,lon=0).lon.values
    indices = np.zeros((len(line.lon.values),3))######
    for i in range(len(line.lon.values)):#####
        if i == 0:
            xstart,ystart =selection_window(start_lat,start_lon,Tdataset.lat.values,Tdataset.lon.values,Tdataset)
            indices[i,0] = xstart
            indices[i,1] = ystart
        else:
            xstart,ystart = select_points(Tdataset,xstart,ystart,line.lat[i].values,line.lon[i].values)
            #print(line.lon[i].values)
            #print(Tdataset.lon[ystart,xstart].values)
            # to provide circle:
            if xstart == len(Tdataset.x)-1:
                xstart = 1
            elif ystart == len(Tdataset.y)-1:
                ystart = len(Tdataset.y)-3
                xstart = len(Tdataset.x)-1-xstart
            indices[i,0] = xstart
            indices[i,1] = ystart

    # remove duplicates:
    row_mask = np.append([True],np.any(np.diff(indices,axis=0),1))
    out = indices[row_mask]

    # if first and last point of out are the same:
    if out[0,0] == out[-1,0] and out[0,1] == out[-1,1]:
        out = np.delete(out,-1,0)

    # allocting if a u point or a v point are taken:
    selection = np.zeros((len(out),5))
    for i in range(1,len(out)): # starting at 1 because point at i-1 is checked to see how point at i is reached
        # came from left:
        if out[i-1,0]+1 == out[i,0] and out[i-1,1] == out[i,1]:
            if i == 1:
                selection[i-1,2] = out[i-1,0]
                selection[i-1,3] = out[i-1,1]
                selection[i-i,4] = 1
            selection[i,2] = out[i,0]
            selection[i,3] = out[i,1]
            selection[i,4] = 1

        # came from right:
        elif out[i-1,0]-1 == out[i,0] and out[i-1,1] == out[i,1]:
            if i ==1:
                selection[i-1,2] = out[i-1,0] +1
                selection[i-1,3] = out[i-1,1]
                selection[i-1,4] =1
            selection[i,2] = out[i,0] +1
            selection[i,3] = out[i,1]
            selection[i,4] = 1

        # came from above:
        elif out[i-1,0] == out[i,0] and out[i-1,1]-1 == out[i,1]:
            if i == 1:
                selection[i-1,0] = out[i-1,0]
                selection[i-1,1] = out[i-1,1]
                selection[i-1,4] =1
            selection[i,0] = out[i-1,0]
            selection[i,1] = out[i-1,1]
            selection[i,4] =1
    
        # came from below:
        elif out[i-1,0] == out[i,0] and out[i-1,1]+1 == out[i,1]:
            if i == 1:
                selection[i-1,0] = out[i-1,0]
                selection[i-1,1] = out[i-1,1]
                selection[i-1,4] = -1
            selection[i,0] = out[i,0]
            selection[i,1] = out[i,1]
            selection[i,4] = -1

        # y is getting smaller:
        elif out[i-1,1] == out[i,1]+1:
            selection[i,0] = out[i,0]
            selection[i,1] = out[i,1]
            selection[i,4] = -1
    
        # x index is changing:
        elif out[i-1,0] > 1000 and out[i,0] < 1000:
            selection[i,2] = out[i,0]
            selection[i,3] = out[i,1]
            selection[i,4] = 1
        else:
            ('hmmm...')


       
    row_mask = np.append([True],np.any(np.diff(selection,axis=0),1))
    selection = selection[row_mask]
    res,ind = np.unique(selection, return_index=True, axis=0)
    selection = res[np.argsort(ind)]

    # saving choosen indices
    selected = xa.Dataset({'indices':(('dim0','dim1'),selection)})
    selected.indices.attrs['dim0'] = 'points'
    selected.indices.attrs['dim1'] = 'columns with points on ugrid,vgrid and sign'
    selected.indices.attrs['dim1_general'] = 'if values are 0 then on the other grid'
    selected.indices.attrs['dim1=0'] = 'selected x index on ugrid'
    selected.indices.attrs['dim1=1'] = 'selected y index on ugrid'
    selected.indices.attrs['dim1=2'] = 'selected x index on vgrid'
    selected.indices.attrs['dim1=3'] = 'selected y index on vgrid'
    selected.indices.attrs['dim1=4'] = 'sign concerning ugrid (when it has to be counted negative)'

    return selected,line



def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('',a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape(unique_a.shape[0],a.shape[1])


def prepare_indices(indices):
    '''
    This function prepares the points when ugrid or vgrid have to be choosen.
    args:
        indices: xa.Dataset where the indices where saved
    returns:
        out_u: np.array with indices where ugrid needs to be selected
        out_v: np.array with indices where vgrid needs to be selected
        out_u_vz: np.array with signs for the indices on ugrid
    '''
    select = indices.indices.values
    selection = unique_rows(select)
    out_u = selection[:,0:2]
    masku = np.all(np.isnan(out_u)|np.equal(out_u,0),axis=1)
    out_u = out_u[~masku]
    out_v = selection[:,2:4]
    maskv = np.all(np.isnan(out_v)|np.equal(out_v,0),axis=1)
    out_v = out_v[~maskv]
    out_u_vz = np.delete(selection,2,1)
    out_u_vz = np.delete(out_u_vz,2,1)
    out_u_vz = out_u_vz[~masku]

    return out_u,out_v,out_u_vz

