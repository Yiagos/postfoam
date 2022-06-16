from readline import get_line_buffer
import foamtopy as fp
import matplotlib.pyplot as plt
import numpy as np


pe_hills = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/periodic_hills")
pe_hills.plotSurface('k', out = "surface_k.png", colorbar=True)



