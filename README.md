# postfoam
A python library for postprocessing openfoam simulations. (Currently limited to 2D geometries)
## Installing
```
pip install wheel
pip install postfoam
```
## Available functions
- Case.show_parameters() # shows fields found in directory
- Case.plotGeometry() # Plot case geometry
- Case.get_data() # Extracts data from specified field
- Case.plotSurface() # Plots contour of given field
- Case.get_lineFace() # Interpolates cell data into specified line
- Case.add_constant() # Adds a constant into the case parameters for further numerical operations
- Case.create_field() # creates a new field based on a mathematical operation of the already existing ones

## Usage
### Important Dependecies
- C file with cell coordinates must be present in directory
- Geometry must be 2D

### Importing an OpenFOAM case
```python
from postfoam import foamtopy as fp

pe_hills = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/periodic_hills")
pe_hills.show_parameters()
```
```
Output:
Parameters found:  epsilon, Cz, Cy, k, U, p, nut, C, Cx
```
```python
# Plots case geometry
pe_hills.plotGeometry()
```
### Surface plot
```python
pe_hills.plotSurface('k', out = "surface_k.png", colorbar=True)

pe_hills.plotSurface('U', 0 , show = False, colorbar=True)
plt.savefig('surface_U.png', dpi = 300, bbox_inches='tight')
plt.show()

'''Create your own plot'''
plt.figure()
xi,yi,zi = pe_hills.interp_data('U', 0)
plt.contourf(xi, yi, zi, 20, cmap='viridis')
#or plt.imshow(zi, extent=[min(x),max(x),min(y),max(y)], origin='lower', cmap='viridis')
plt.savefig('contour_U.png', dpi = 300, bbox_inches='tight')
plt.show()
```
### Data Interpolation and Line Plots
```python
line = pe_hills.get_lineFace(start=[1, 3.035, 0], end=[1, 0, 0], field="U", index=0, nPoints=100)
plt.figure()
plt.plot(line[:,2], line[:,1])
plt.savefig('line_U.png', dpi = 300, bbox_inches='tight')
plt.show()
```
### Data manipulation
```python
'''Manipulate fields'''
pe_hills.add_constant('U_ref', 0.2)
pe_hills.create_field("Ux/U_ref", "U[0] / U_ref")
pe_hills.show_parameters()
```
### Reading OpenFOAM Postprocessing
```python
'''Get postprocessing data'''
pe_hills.read_postData()
U_line = pe_hills.postParameters.get('lineuh_1_U')[:,0]
```
