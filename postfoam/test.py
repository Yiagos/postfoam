import foamtopy as fp
import matplotlib.pyplot as plt

pe_hills = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/periodic_hills")
pe_hills.show_parameters()
pe_hills.plotGeometry()
U = pe_hills.get_data("U")

pe_hills.plotSurface('k', out = "surface_k.png", colorbar=True)

pe_hills.plotSurface('U', 0 , show = False, colorbar=True)
plt.savefig('surface_U.png', dpi = 300, bbox_inches='tight')
plt.show()

line = pe_hills.get_lineFace(start=[1, 3.035, 0], end=[1, 0, 0], field="U", index=0, nPoints=100)
plt.figure()
plt.plot(line[:,2], line[:,1])
plt.savefig('line_U.png', dpi = 300, bbox_inches='tight')
plt.show()

'''Create your own plot'''
plt.figure()
xi,yi,zi = pe_hills.interp_data('U', 0)
plt.contourf(xi, yi, zi, 20, cmap='viridis')
#plt.imshow(zi, extent=[min(x),max(x),min(y),max(y)], origin='lower', cmap='viridis')
plt.savefig('contour_U.png', dpi = 300, bbox_inches='tight')
plt.show()

'''Manipulate fields'''
pe_hills.add_constant('U_ref', 0.2)
pe_hills.create_field("Ux/U_ref", "U[0] / U_ref")
pe_hills.show_parameters()

'''Get postprocessing data'''
pe_hills.read_postData()
U_line = pe_hills.postParameters.get('lineuh_1_U')[:,0]

'''Testing for different case'''
pitzDaily = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/pitzDaily", 200)
pitzDaily.show_parameters()
pitzDaily.plotGeometry("pitzDaily")
pitzDaily.plotSurface('p')
line = pitzDaily.get_lineFace(start=[0, 0, 0], end=[0, 1, 0], field="U", index=0, nPoints=100)
pitzDaily.create_field('k_norm', 'k / U[0] **2')