import numpy as np

def write_coeff1(path):
    filename = path + '/system/controlDict'
    file = open(filename, 'w')
    writeString = """
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.1.1                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         3000;

deltaT          1;

writeControl    runTime;

writeInterval   3000;

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      fixed;

timePrecision   0;

runTimeModifiable true;

//libs ("libmyMomentumTransportModels.so");

functions
    {
    //    #includeFunc  sampleDict;
        #includeFunc    sampleDict2;
    }
// ************************************************************************* //"""
    file.write(writeString)
    file.close()

directory = "/home/yiagoskyrits/OpenFOAM/yiagoskyrits-9/run/case_1p0"
write_coeff1(directory)
