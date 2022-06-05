from readline import get_line_buffer
import foamtopy as fp
import matplotlib.pyplot as plt
import numpy as np

'''RUN CODE'''
'''
pitzDaily = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/pitzDaily")
print(pitzDaily.file_path())
a = pitzDaily.get_points()
b = pitzDaily.read_scalarField("/100/k")
pitzDaily.show_parameters()
c = pitzDaily.get_data("k")
print(c)
print(type(pitzDaily))
print(pitzDaily.parameters.get("C")[:,0])
#z = fp.plotSurface(pitzDaily, "k")
#print(z)
'''

'''
line = pe_hills.get_lineFace(start=[1, 3.035, 0], end=[1, 0, 0], field="U", index=0, nPoints=100)

u=np.loadtxt('/home/yiagoskyrits//OpenFOAM/yiagoskyrits-9/run/case_1p0/postProcessing/sampleDict2/3000/lineuh_1_U.csv', dtype=float,skiprows=1,usecols=1, delimiter=",")
y = np.loadtxt('/home/yiagoskyrits//OpenFOAM/yiagoskyrits-9/run/case_1p0/postProcessing/sampleDict2/3000/lineuh_1_U.csv', dtype=float,skiprows=1,usecols=0, delimiter=",")
plt.plot(line[:,2], line[:,1])
plt.plot(u,y)
plt.savefig('out.png', dpi = 300, bbox_inches='tight')
plt.show()
'''
pe_hills = fp.FoamCase("/home/yiagoskyrits//OpenFOAM/yiagoskyrits-9/run/case_1p0", 3000)
pe_hills.show_parameters()
pe_hills.plotSurface('k')



