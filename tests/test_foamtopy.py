from cmath import exp
from postfoam import foamtopy

def test_foamtopy():
    expected = "/home/yiagoskyrits/postfoam/tests/test_case/pitzDaily"
    actual = foamtopy.FoamCase("/home/yiagoskyrits/postfoam/tests/test_case/pitzDaily")
    assert actual==expected, "test failed"