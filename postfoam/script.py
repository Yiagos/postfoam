from readline import get_line_buffer
import foamtopy as fp
import matplotlib.pyplot as plt
import numpy as np


pe_hills = fp.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/periodic_hills")
pe_hills.show_parameters()
pe_hills.add_constant('U_ref', 0.2)
pe_hills.create_field("Ux/U_ref", "k / U[0] **2")
pe_hills.show_parameters()



