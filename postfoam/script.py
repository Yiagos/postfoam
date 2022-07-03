from readline import get_line_buffer
import foamtopy as fp
import matplotlib.pyplot as plt
import numpy as np


#pe_hills = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/periodic_hills")
#fp.savepf(pe_hills, "peCase.pf")
#pe_hills2 = fp.loadpf("peCase.pf")
#pe_hills2.plotSurface('k', out = "surface_k.png", colorbar=True)

#cbfs = fp.FoamCase('/home/yiagoskyrits/OpenFOAM/yiagoskyrits-9/run/convdiv20580')
#cbfs.plotSurface('k', out='test.png')

pe_hill = fp.FoamCase("/home/yiagoskyrits/OpenFOAM/yiagoskyrits-9/run/case_1p0")
gu = pe_hill.parameters.get('UPrime2Mean')
print(gu)