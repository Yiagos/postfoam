import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os import listdir
import os.path  
from scipy.interpolate import griddata
import re
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import pandas as pd



class FoamCase:
    def __init__(self, filepath, time = None, readfromfile = False, read_parameters = True):
        self.filepath = filepath
        self.parameters = {}
        self.postParameters = {}
        if not readfromfile:
            if read_parameters == True:
                self.get_parameters(time)
            else:
                pass
        else:
            self.load_pffile()


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
        try:
            c1 = c1.astype(float)
        except:
            c_float = np.zerps(len(c1))
            for d in range(0,len(c1)):
                try:
                    c_float[d] = float(c1[d])
                except:
                    c_float[d] = 0
            c1 = c_float
        try:
            c2 = c2.astype(float)
        except:
            c_float = np.zeros(len(c2))
            for d in range(0,len(c2)):
                try:
                    c_float[d] = float(c2[d])
                except:
                    c_float[d] = 0
            c2 = c_float
        try:
            c3 = c3.astype(float)
        except:
            c_float = np.zerps(len(c3))
            for d in range(0,len(c3)):
                try:
                    c_float[d] = float(c3[d])
                except:
                    c_float[d] = 0
            c3 = c_float
        return np.vstack((c1,c2,c3)).T

    def read_tensorField(self, path):
        '''
        Private
        '''
        file = open(self.filepath+path, 'r')
        Lines = file.readlines()
        index = Lines.index("(\n")
        c1 = np.loadtxt(self.filepath+path, dtype=str,usecols=0, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c2 = np.loadtxt(self.filepath+path, dtype=str,usecols=1, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c3 = np.loadtxt(self.filepath+path, dtype=str,usecols=2, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c4 = np.loadtxt(self.filepath+path, dtype=str,usecols=3, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c5 = np.loadtxt(self.filepath+path, dtype=str,usecols=4, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c6 = np.loadtxt(self.filepath+path, dtype=str,usecols=5, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c7 = np.loadtxt(self.filepath+path, dtype=str,usecols=6, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c8 = np.loadtxt(self.filepath+path, dtype=str,usecols=7, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c9 = np.loadtxt(self.filepath+path, dtype=str,usecols=8, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        for i in range(0,len(c1)):
            c1[i] = c1[i][1:]
            c9[i] = c9[i][:-1]
        c1 = c1.astype(float)
        c2 = c2.astype(float)
        c3 = c3.astype(float)
        c4 = c1.astype(float)
        c5 = c2.astype(float)
        c6 = c3.astype(float)
        c7 = c1.astype(float)
        c8 = c2.astype(float)
        c9 = c3.astype(float)
        return np.vstack((c1,c2,c3,c4,c5,c6,c7,c8,c9)).T

    def read_symmTensorField(self, path):
        '''
        Private
        '''
        file = open(self.filepath+path, 'r')
        Lines = file.readlines()
        index = Lines.index("(\n")
        c1 = np.loadtxt(self.filepath+path, dtype=str,usecols=0, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c2 = np.loadtxt(self.filepath+path, dtype=str,usecols=1, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c3 = np.loadtxt(self.filepath+path, dtype=str,usecols=2, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c4 = np.loadtxt(self.filepath+path, dtype=str,usecols=3, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c5 = np.loadtxt(self.filepath+path, dtype=str,usecols=4, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c6 = np.loadtxt(self.filepath+path, dtype=str,usecols=5, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        for i in range(0,len(c1)):
            c1[i] = c1[i][1:]
            c6[i] = c6[i][:-1]
        c1 = c1.astype(float)
        c2 = c2.astype(float)
        c3 = c3.astype(float)
        c4 = c1.astype(float)
        c5 = c2.astype(float)
        c6 = c3.astype(float)
        return np.vstack((c1,c2,c3,c4,c5,c6)).T

    def read_faceList(self):
        '''
        Private
        '''
        '''faceList of type 4()'''
        path = '/constant/polyMesh/faces'
        file = open(self.filepath+path, 'r')
        Lines = file.readlines()
        index = Lines.index("(\n")
        c1 = np.loadtxt(self.filepath+path, dtype='<U9',usecols=0, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c2 = np.loadtxt(self.filepath+path, dtype='<U9',usecols=1, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c3 = np.loadtxt(self.filepath+path, dtype='<U9',usecols=2, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        c4 = np.loadtxt(self.filepath+path, dtype='<U9',usecols=-1, skiprows=index+1, max_rows=int(Lines[index-1]), delimiter=" ")
        for i in range(0,len(c1)):
            c1[i] = c1[i][2:]
            if c4[i][-1]==')':
                c4[i] = c4[i][:-1]
            if c3[i][-1]==')':
                c3[i] = c3[i][:-1]
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
                                if int(nFaces)>1500:
                                    empty=True
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
                    elif 'volTensorField' in j:
                        param = self.read_tensorField('/'+str(time)+'/'+i)
                        self.parameters[i]=param
                    elif 'volSymmTensorField' in j:
                        param = self.read_symmTensorField('/'+str(time)+'/'+i)
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

    def get_lineFace(self, start: np.array, end: np.array, field: str, index: int, nPoints: int):
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
        xi = np.linspace(min(x),max(x),1000)
        yi = np.linspace(min(y),max(y),1000)
        xi, yi = np.meshgrid(xi,yi)
        zi = griddata((x,y),z,(xi,yi),method='linear')

        bounds = self.get_mask_line()
        if len(bounds)>10000:
            bounds = bounds[::50]
        points = np.vstack((xi.flatten(),yi.flatten())).T
        path = Path(bounds)
        grid = path.contains_points(points)
        grid = grid.reshape((len(yi),len(xi)))
        zi[grid == False] = np.nan

        return xi,yi,zi

    def plotSurface(self, field: str, index: int = None, out = "out.png", colorbar: bool = False, show: bool = True, save: bool = True):
        '''Cell coordinates have to be present in file C'''
        plt.figure()
        ax = plt.subplot()
        x = self.parameters.get('C')[:,0]
        y = self.parameters.get('C')[:,1]
        xi,yi,zi = self.interp_data(field, index)
        im = plt.imshow(zi, extent=[min(x),max(x),min(y),max(y)], origin='lower', cmap='viridis')
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()
        if save:
            plt.savefig(out, dpi = 300, bbox_inches='tight')
        if show:
            plt.show()

    def get_mask_line(self):
        xy_bounds = self.read_boundary()
        inv_index = np.array([0],dtype = int)
        'Update'
        diff_x = abs(xy_bounds[1:,0]-xy_bounds[:-1,0])
        diff_y = abs(xy_bounds[1:,1]-xy_bounds[:-1,1])
        cond_x = (max(xy_bounds[:,0])-min(xy_bounds[:,0]))/10
        cond_y = (max(xy_bounds[:,1])-min(xy_bounds[:,1]))/10
        for i in range(0,len(diff_x)):
            if diff_x[i]>cond_x or diff_y[i]>cond_y: 
                inv_index = np.append(inv_index,i+1)
        #for i in range(2,len(xy_bounds)):
        #    if abs(xy_bounds[i,0]-xy_bounds[i-1,0])>(max(xy_bounds[:,0])-min(xy_bounds[:,0]))/10 or abs(xy_bounds[i,1]-xy_bounds[i-1,1])>(max(xy_bounds[:,1])-min(xy_bounds[:,1]))/10: 
        #        inv_index = np.append(inv_index,i)
        inv_index = np.append(inv_index,len(xy_bounds))
        lines = {}
        for i in range(1,len(inv_index)):
            lines[i-1] = xy_bounds[inv_index[i-1]:inv_index[i],:]

        line_completed = False
        line = lines.get(0)
        while not line_completed:
            dx = np.inf
            dx_1=np.empty(len(lines)-1)
            dx_2=np.empty(len(lines)-1)
            for i in range(1,len(inv_index)-1):
                dx_1[i-1] = np.sqrt((lines.get(i)[0,0]-line[-1,0])**2+(lines.get(i)[0,1]-line[-1,1])**2)
                dx_2[i-1] = np.sqrt((lines.get(i)[-1,0]-line[-1,0])**2+(lines.get(i)[-1,1]-line[-1,1])**2)
                if min(dx_1[i-1],dx_2[i-1])<dx:
                    dx = min(dx_1[i-1],dx_2[i-1])
                    next_line = i
            if dx_2[next_line-1]<dx_1[next_line-1]:
                line = np.vstack((line,lines.get(next_line)[::-1]))
            else:
                line = np.vstack((line,lines.get(next_line)))
            lines[next_line]=np.full((lines[next_line].shape[0], lines[next_line].shape[1]), np.inf)
            #plt.plot(line[:,0],line[:,1])
            #plt.savefig('test.png', dpi = 300, bbox_inches='tight')
            #plt.show()
            if dx == np.inf:
                line_completed = True 
        return line

    def create_field(self, field_name: str, equation: str):
        'Space is required between the equation variables'
        s = equation.split(" ")
        for i in range(0,len(s)):
            if s[i] in list(self.parameters.keys()) or s[i][:-3] in list(self.parameters.keys()):
                if "[" in s[i]:
                    s[i] = "self.parameters.get('" + s[i][:-3] +  "')" + '[' + ':,' + s[i][-2:]
                else:
                    s[i] = "self.parameters.get('" + s[i] + "')"
        s = ''.join(s)
        self.parameters[field_name] = eval(s)

    def add_constant(self, constant_name: str, value):
        self.parameters[constant_name]=value

    def plotGeometry(self, out = "out.png"):
        self.parameters['Surface']=np.ones(len(self.parameters.get('C')[:,0]))
        self.plotSurface('Surface')

    def read_postData(self, folder_name = 'all', time = None):
        '''Can read csv files'''
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
        try:
            if folder_name != 'all':
                for i in listdir(self.filepath+'/postProcessing/'+folder_name+'/'+str(time)):
                    try:
                        if os.path.isfile(self.filepath+'/postProcessing/'+folder_name+'/'+str(time)+'/'+i):
                            self.postParameters[i[:-4]] = np.loadtxt(self.filepath+'/postProcessing/'+folder_name+'/'+str(time)+'/'+i, skiprows=1, delimiter=',')
                    except:
                        print('Files could not be read')
                        break
                print('Post data found: ', end = ' ')
                print(*list(self.postParameters.keys()), sep = ', ')
            else:
                for folder_name in listdir(self.filepath+'/postProcessing/'):
                    for i in listdir(self.filepath+'/postProcessing/'+folder_name+'/'+str(time)):
                        try:
                            if os.path.isfile(self.filepath+'/postProcessing/'+folder_name+'/'+str(time)+'/'+i):
                                self.postParameters[i[:-4]] = np.loadtxt(self.filepath+'/postProcessing/'+folder_name+'/'+str(time)+'/'+i, skiprows=1, delimiter=',')
                        except:
                            print('Files could not be read')
                            break
                print('Post data found: ', end = ' ')
                print(*list(self.postParameters.keys()), sep = ', ')
        except:
            print('Folder not found')

    def load_pffile(self):
        'Not working currently'
        df = pd.read_pickle(self.filepath)
        for col in df.columns:
            if "post" not in col:
                if "#" not in col:
                    self.parameters[col]=df.loc[:,col].to_numpy().T
                else:
                    if col[:-2] not in self.parameters.keys():
                        self.parameters[col[:-2]]=df.loc[:,col].to_numpy().T
                    else:
                        self.parameters[col[:-2]]=np.vstack((self.parameters.get(col[:-2]).T, df.loc[:,col].to_numpy())).T
            else:
                pass
        print(self.parameters.get('C'))


def savepf(Case: FoamCase, filename: str = "FoamCase.pf"):
    df = pd.DataFrame()
    index = 0
    for key in list(Case.parameters.keys()):
        try:
            if Case.parameters.get(key).ndim == 1:
                df.insert(index, key, Case.parameters.get(key))
                index+=1
            else:
                for i in range(0,Case.parameters.get(key).shape[1]):
                    df.insert(index, key+'#'+str(i), Case.parameters.get(key)[:,i])
                    index+=1
        except AttributeError:
            pass
    Case.read_postData()
    for key in list(Case.postParameters.keys()):
        try:
            if Case.postParameters.get(key).ndim == 1:
                df.insert(index, key+'_post', Case.postParameters.get(key))
                index+=1
            else:
                for i in range(0,Case.postParameters.get(key).shape[1]):
                    data = np.full(len(df), np.NaN)
                    data[0:len(Case.postParameters.get(key)[:,i])] = Case.postParameters.get(key)[:,i]
                    df.insert(index, key+'#'+str(i)+'_post', data)
                    index+=1
        except AttributeError:
            pass
    df.to_pickle(filename)
    
def loadpf(path: str):
    'Not working currently'
    return FoamCase(path, readfromfile=True)