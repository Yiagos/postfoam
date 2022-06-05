from readline import get_line_buffer
import foamtopy as fp
import matplotlib.pyplot as plt
import numpy as np


'''Test for different case'''
pitzDaily = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/pitzDaily", 200)
pitzDaily.show_parameters()
pitzDaily.plotSurface('p')
line = pitzDaily.get_lineFace(start=[0, 0, 0], end=[0, 1, 0], field="U", index=0, nPoints=100)


