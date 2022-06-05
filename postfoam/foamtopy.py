from array import array
from matplotlib.patches import Polygon
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os import listdir
import os.path  
from scipy.interpolate import griddata
import re
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class FoamCase:
    def __init__(self, filepath, time = None):
        self.filepath = filepath
        self.parameters = {}

        self.get_parameters(time)


    def read_vectorField(self, path):
        '''
        Private
        '''
        file = open(self.filepath+path, 'r')
        Lines = file.readlines()
        index = Lines.index("(\n")
        c1 = np.loadtxt(self.filepath+path, dtype=str,usecols=0, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c2 = np.loadtxt(self.filepath+path, dtype=str,usecols=1, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c3 = np.loadtxt(self.filepath+path, dtype=str,usecols=2, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        for i in range(0,len(c1)):
            c1[i] = c1[i][1:]
            c3[i] = c3[i][:-1]
        c1 = c1.astype(float)
        c2 = c2.astype(float)
        c3 = c3.astype(float)
        return np.vstack((c1,c2,c3)).T

    def read_faceList(self):
        '''
        Private
        '''
        '''faceList of type 4()'''
        path = '/constant/polyMesh/faces'
        file = open(self.filepath+path, 'r')
        Lines = file.readlines()
        index = Lines.index("(\n")
        c1 = np.loadtxt(self.filepath+path, dtype=str,usecols=0, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c2 = np.loadtxt(self.filepath+path, dtype=str,usecols=1, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c3 = np.loadtxt(self.filepath+path, dtype=str,usecols=2, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c4 = np.loadtxt(self.filepath+path, dtype=str,usecols=3, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        for i in range(0,len(c1)):
            c1[i] = c1[i][2:]
            c4[i] = c4[i][:-1]
        c1 = c1.astype(float)
        c2 = c2.astype(float)
        c3 = c3.astype(float)
        c4 = c4.astype(float)
        return np.vstack((c1,c2,c3,c4)).T

    def read_boundary(self):
        '''Private'''
        path = '/constant/polyMesh/boundary'
        file = open(self.filepath+path, 'r')
        Lines = file.readlines()
        index = Lines.index("(\n")
        boundary_index = np.array([])
        for i in range(index,len(Lines)):
            if "{" in Lines[i]:
                nFaces = ''; startFace = ''; empty = False
                while '}' not in Lines[i]:
                    i+=1
                    for p in [r'\w+']:
                        match= re.findall(p, Lines[i])
                        try:
                            if match[0] == 'type' and match[1] == 'empty':
                                empty = True
                            elif match[0] == 'nFaces':
                                nFaces = match[1]
                            elif match[0] == 'startFace':
                                startFace = match[1]
                        except:
                            continue
                    if empty: break
                if not empty: boundary_index=np.hstack((boundary_index,np.arange(int(startFace),int(startFace)+int(nFaces),1)))
        boundary_index = boundary_index.astype(int)
        boundary_cells = self.get_cellCentres(boundary_index)
        return boundary_cells

    def get_cellCentres(self, face_index):
        '''Private'''
        point_index = np.array(self.read_faceList()[face_index])
        point1 = self.get_points()[point_index[:,0].astype(int)]
        point2 = self.get_points()[point_index[:,1].astype(int)]
        point3 = self.get_points()[point_index[:,2].astype(int)]
        point4 = self.get_points()[point_index[:,3].astype(int)]
        cell_center = np.vstack(((point1[:,0]+point2[:,0]+point3[:,0]+point4[:,0])/4, ((point1[:,1]+point2[:,1]+point3[:,1]+point4[:,1])/4))).T
        return cell_center
           
    def get_points(self):
        return self.read_vectorField('/constant/polyMesh/points')

    def read_scalarField(self,path):
        '''
        Private
        '''
        try:
            file = open(self.filepath+path, 'r')
            Lines = file.readlines()
            index = Lines.index("(\n")
            c = np.loadtxt(self.filepath+path, dtype=float,usecols=0, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
            return c
        except:
            pass

    def get_parameters(self, time = None):
        '''
        Private
        '''
        if time == None:
            numbers=[]
            for i in listdir(self.filepath):
                try:
                    if '.' in i:
                        numbers.append(float(i))
                    else:
                        numbers.append(int(i))
                except:
                    pass
            time = max(numbers)
        else:
            pass
        for i in listdir(self.filepath+'/'+str(time)):
            if os.path.isfile(self.filepath+'/'+str(time)+'/'+i):
                file = open(self.filepath+'/'+str(time)+'/'+i, 'r')
                Lines = file.readlines()
                for j in Lines:
                    if 'volVectorField' in j:
                        param = self.read_vectorField('/'+str(time)+'/'+i)
                        self.parameters[i]=param
                    elif 'volScalarField' in j:
                        param = self.read_scalarField('/'+str(time)+'/'+i)
                        self.parameters[i]=param
                    else:
                        pass

    def show_parameters(self):
        print('Parameters found: ', end = ' ')
        print(*list(self.parameters.keys()), sep = ', ')

    def get_data(self, parameter: str):
        return self.parameters.get(parameter)

    def file_path(self):
        return self.filepath

    def get_lineFace(self, start: array, end: array, field: str, index: int, nPoints: int):
        coords = np.vstack((np.linspace(start[0],end[0],nPoints),np.linspace(start[1],end[1],nPoints))).T
        xi,yi,zi = self.interp_data(field, index)
        lines = np.zeros((nPoints,1))
        xi = xi.ravel(); yi = yi.ravel(); zi = zi.ravel()
        for j in range(0,nPoints):
            point = np.argmin(np.sum(np.vstack((abs(xi-coords[j,0])**2, abs(yi-coords[j,1])**2)).T,1))
            lines[j] = zi[point]
        return np.hstack((coords,lines))

    def interp_data(self, field, index):
        '''Private'''
        x = self.parameters.get('C')[:,0]
        y = self.parameters.get('C')[:,1]
        z = self.parameters.get(field)
        if index != None:
            z = z[:,index]
        xi = np.linspace(min(x),max(x),3000)
        yi = np.linspace(min(y),max(y),3000)
        xi, yi = np.meshgrid(xi,yi)
        zi = griddata((x,y),z,(xi,yi),method='linear')
        return xi,yi,zi

    def get_mask_line(self):
        xy_bounds = self.read_boundary()
        inv_index = np.array([0],dtype = int)
        for i in range(2,len(xy_bounds)):
            if abs(xy_bounds[i,0]-xy_bounds[i-1,0])>(max(xy_bounds[:,0])-min(xy_bounds[:,0]))/10 or abs(xy_bounds[i,1]-xy_bounds[i-1,1])>(max(xy_bounds[:,1])-min(xy_bounds[:,1]))/10: 
                inv_index = np.append(inv_index,i)
        inv_index = np.append(inv_index,len(xy_bounds))
        boundary_types = np.array([])
        for i in range(1,len(inv_index)):
            if max(xy_bounds[inv_index[i-1]:inv_index[i],1]) <= min(np.delete(xy_bounds,np.arange(inv_index[i-1],inv_index[i],1), axis=0)[:,1]):
                boundary_types = np.append(boundary_types, 'bottom')
            elif max(xy_bounds[inv_index[i-1]:inv_index[i],1]) >= max(np.delete(xy_bounds,np.arange(inv_index[i-1],inv_index[i],1), axis=0)[:,1]):
                boundary_types = np.append(boundary_types, 'top')
            elif max(xy_bounds[inv_index[i-1]:inv_index[i],0]) <= min(np.delete(xy_bounds,np.arange(inv_index[i-1],inv_index[i],1), axis=0)[:,0]):
                boundary_types = np.append(boundary_types, 'left')
            elif min(xy_bounds[inv_index[i-1]:inv_index[i],0]) >= max(np.delete(xy_bounds,np.arange(inv_index[i-1],inv_index[i],1), axis=0)[:,0]):
                boundary_types = np.append(boundary_types, 'right')
            else:
                print('Error boundary not found')

        line_completed = False
        line = np.empty([1,2])
        while not line_completed:
            if 'bottom' in boundary_types:
                i = np.where(boundary_types=='bottom')[0][0] + 1
                line = np.vstack((line,xy_bounds[inv_index[i-1]:inv_index[i],:]))
                bottom = xy_bounds[inv_index[i-1]:inv_index[i],:]
                boundary_types[np.where(boundary_types=='bottom')] = 'skip'
            elif 'right' in boundary_types:
                i = np.where(boundary_types=='right')[0][0] + 1
                line = np.vstack((line,xy_bounds[inv_index[i-1]:inv_index[i],:]))
                right = xy_bounds[inv_index[i-1]:inv_index[i],:]
                boundary_types[np.where(boundary_types=='right')] = 'skip'
            elif 'top' in boundary_types:
                i = np.where(boundary_types=='top')[0][0] + 1
                line = np.vstack((line,xy_bounds[inv_index[i-1]:inv_index[i],:]))
                boundary_types[np.where(boundary_types=='top')] = 'skip'
                top = xy_bounds[inv_index[i-1]:inv_index[i],:]
            elif 'left' in boundary_types:
                i = np.where(boundary_types=='left')[0][0] + 1
                line = np.vstack((line,xy_bounds[inv_index[i-1]:inv_index[i],:]))
                left = xy_bounds[inv_index[i-1]:inv_index[i],:]
                boundary_types[np.where(boundary_types=='left')] = 'skip'
                line_completed = True
        line = line[1:,:]
            
        inv_index = np.array([0],dtype = int)
        for i in range(2,len(line)):
            if abs(line[i,0]-line[i-1,0])>(max(line[:,0])-min(line[:,0]))/10 or abs(line[i,1]-line[i-1,1])>(max(line[:,1])-min(line[:,1]))/10: 
                inv_index = np.append(inv_index,i)
        inv_index = np.append(inv_index,len(line))
        cont_line = np.empty([1,2])
        for i in range(1,len(inv_index)):
            if i % 2 != 0:
                cont_line = np.vstack((cont_line,line[inv_index[i-1]:inv_index[i],:]))
            else:
                cont_line = np.vstack((cont_line,line[inv_index[i-1]:inv_index[i],:][::-1]))
        cont_line = cont_line[1:,:]
        return cont_line, bottom, top, left, right

    def plotSurface(self, field: str, index: int = None):
        '''Cell coordinates have to be present in file C'''
        x = self.parameters.get('C')[:,0]
        y = self.parameters.get('C')[:,1]
        xi,yi,zi = self.interp_data(field, index)
        plt.imshow(zi, extent=[min(x),max(x),min(y),max(y)], origin='lower', cmap='viridis')
        #plt.contourf(xi, yi, zi, cmap='viridis')
        self.plot_boundary()
        plt.savefig('out.png', dpi = 300, bbox_inches='tight')
        plt.show()

    def plot_boundary(self):
        boundary, bottom, top, left, right = self.get_mask_line()
        plt.fill_between(bottom[:,0],bottom[:,1], min(boundary[:,1]), color = 'w')
        plt.fill_between(top[:,0],top[:,1], max(boundary[:,1]), color = 'w')
        plt.fill_betweenx(left[:,1],left[:,0], min(boundary[:,0]), color = 'w')
        plt.fill_betweenx(right[:,1],right[:,0], max(boundary[:,0]), color = 'w')