import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt


# Functions
def write_array(res):
    res = res.reshape(-1,1)
    array_init = np.array(["""
/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  9
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "3000";
    object      multbetaStarField;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<scalar>
14751
(
"""])

    array_str = res.astype(str)
    array_end = np.array(["""
)
;

boundaryField
{
    bottomWall
    {
        type            calculated;
        value           uniform 1;
    }
    defaultFaces
    {
        type            empty;
    }
    inlet
    {
        type            cyclic;
    }
    outlet
    {
        type            cyclic;
    }
    topWall
    {
        type            calculated;
        value           uniform 1;
    }
}
}


// ************************************************************************* //
"""])
    array_final = np.vstack((array_init,array_str,array_end))
    return array_final


def get_error_all(path, v, points, u005, u05, u1, u2, u3, u4, u5, u6, u7, u8):
    file_u_0_05 = path + "/postProcessing/sampleDict2/3000/lineuh_0.05_U.csv"
    file_u_0_5 = path + "/postProcessing/sampleDict2/3000/lineuh_0.5_U.csv"
    file_u_1 = path + "/postProcessing/sampleDict2/3000/lineuh_1_U.csv"
    file_u_2 = path + "/postProcessing/sampleDict2/3000/lineuh_2_U.csv"
    file_u_3 = path + "/postProcessing/sampleDict2/3000/lineuh_3_U.csv"
    file_u_4 = path + "/postProcessing/sampleDict2/3000/lineuh_4_U.csv"
    file_u_5 = path + "/postProcessing/sampleDict2/3000/lineuh_5_U.csv"
    file_u_6 = path + "/postProcessing/sampleDict2/3000/lineuh_6_U.csv"
    file_u_7 = path + "/postProcessing/sampleDict2/3000/lineuh_7_U.csv"
    file_u_8 = path + "/postProcessing/sampleDict2/3000/lineuh_8_U.csv"
    err005 = 0
    err05 = 0
    err1 = 0
    err2 = 0
    err3 = 0
    err4 = 0
    err5 = 0
    err6 = 0
    err7 = 0
    err8 = 0

    u_005 = np.loadtxt(file_u_0_05, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_005_points = np.loadtxt(file_u_0_05, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_05 = np.loadtxt(file_u_0_5, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_05_points = np.loadtxt(file_u_0_5, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_1 = np.loadtxt(file_u_1, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_1_points = np.loadtxt(file_u_1, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_2 = np.loadtxt(file_u_2, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_2_points = np.loadtxt(file_u_2, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_3 = np.loadtxt(file_u_3, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_3_points = np.loadtxt(file_u_3, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_4 = np.loadtxt(file_u_4, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_4_points = np.loadtxt(file_u_4, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_5 = np.loadtxt(file_u_5, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_5_points = np.loadtxt(file_u_5, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_6 = np.loadtxt(file_u_6, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_6_points = np.loadtxt(file_u_6, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_7 = np.loadtxt(file_u_7, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_7_points = np.loadtxt(file_u_7, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    u_8 = np.loadtxt(file_u_8, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_8_points = np.loadtxt(file_u_8, usecols=[0], dtype=float, delimiter=",", skiprows=1)


    for j in range(0, len(points)):
        diff005 = np.abs(u_005_points - points[j])
        index005 = diff005.argmin()
        err005 = err005 + (u005[j] - u_005[index005]) ** 2

        diff05 = np.abs(u_05_points - points[j])
        index05 = diff05.argmin()
        err05 = err05 + (u05[j] - u_05[index05]) ** 2

        diff1 = np.abs(u_1_points - points[j])
        index1 = diff1.argmin()
        err1 = err1 + (u1[j] - u_1[index1]) ** 2

        diff2 = np.abs(u_2_points - points[j])
        index2 = diff2.argmin()
        err2 = err2 + (u2[j] - u_2[index2]) ** 2

        diff3 = np.abs(u_3_points - points[j])
        index3 = diff3.argmin()
        err3 = err3 + (u3[j] - u_3[index3]) ** 2

        diff4 = np.abs(u_4_points - points[j])
        index4 = diff4.argmin()
        err4 = err4 + (u4[j] - u_4[index4]) ** 2

        diff5 = np.abs(u_5_points - points[j])
        index5 = diff5.argmin()
        err5 = err5 + (u5[j] - u_5[index5]) ** 2

        diff6 = np.abs(u_6_points - points[j])
        index6 = diff6.argmin()
        err6 = err6 + (u6[j] - u_6[index6]) ** 2

        diff7 = np.abs(u_7_points - points[j])
        index7 = diff7.argmin()
        err7 = err7 + (u7[j] - u_7[index7]) ** 2

        diff8 = np.abs(u_8_points - points[j])
        index8 = diff8.argmin()
        err8 = err8 + (u8[j] - u_8[index8]) ** 2


    mse005 = err005/len(points)
    mse05 = err05/len(points)
    mse1 = err1/len(points)
    mse2 = err2/len(points)
    mse3 = err3/len(points)
    mse4 = err4/len(points)
    mse5 = err5/len(points)
    mse6 = err6/len(points)
    mse7 = err7/len(points)
    mse8 = err8/len(points)

    mse = (mse005+mse05+mse1+mse2+mse3+mse4+mse5+mse6+mse7+mse8)/10
    print(mse)
    return mse


start = time.time()
# Constants
directory = "/home/yiagoskyrits/OpenFOAM/yiagoskyrits-9/run/case_1p0"
v = 0.028          # Velocity
rho = 1.2     # Density
dist_num = 120    # Number of distributions
loop_num = 5
sd = 0.01
field = np.loadtxt(directory+'/0/cellFields', usecols=[0], dtype=float, skiprows=21, max_rows=14751)
betaStar = np.full((len(field),1), 0.09)


exp_points_u005 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp0.05.dat", usecols=[0], dtype=float, delimiter=",")
exp_u005 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp0.05.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u005 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp0.05.dat", usecols=[0], dtype=float, delimiter=",")
exp_u05 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp0.5.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u05 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp0.5.dat", usecols=[0], dtype=float, delimiter=",")
exp_u1 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp1.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u1 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp1.dat", usecols=[0], dtype=float, delimiter=",")
exp_u2 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp2.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u2 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp2.dat", usecols=[0], dtype=float, delimiter=",")
exp_u3 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp3.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u3 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp3.dat", usecols=[0], dtype=float, delimiter=",")
exp_u4 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp4.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u4 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp4.dat", usecols=[0], dtype=float, delimiter=",")
exp_u5 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp5.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u5 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp5.dat", usecols=[0], dtype=float, delimiter=",")
exp_u6 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp6.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u6 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp6.dat", usecols=[0], dtype=float, delimiter=",")
exp_u7 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp7.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u7 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp7.dat", usecols=[0], dtype=float, delimiter=",")
exp_u8 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp8.dat", usecols=[1], dtype=float, delimiter=",")
exp_points_u8 = np.loadtxt("/home/yiagoskyrits/DataFiles/data_files_ph/exp8.dat", usecols=[0], dtype=float, delimiter=",")



for loop in range(loop_num):

    # Clean
    subprocess.check_call(directory + '/Allclean', shell=True)

    if loop != 0:                                   # Update distribution parameters
        ind = np.argmin(error)
        print(error[ind])
        dist_num = int(dist_num/2)
        betaStar = mult[:,ind]
        sd = sd/2

    # Create initial distributions
    mult = np.ones((len(field),dist_num))
    for i in range(0,dist_num):
        for j in range(0,len(field)):
            if field[j] != 0:
                mult[j,i] = abs(np.round(np.random.normal(betaStar[i], sd, 1), 3))/0.09

    error = np.ones(dist_num)       # Array of errors

    for i in range(0,dist_num):

        print("Writing to file")
        array_final = write_array(mult[:,i])
        np.savetxt(directory+'/0/multbetaStarField', array_final, fmt='%s')


        # simpleFoam
        subprocess.check_call(directory + '/runSim', shell=True)
        error[i] = get_error_all(directory, v, exp_points_u1, exp_u005, exp_u05, exp_u1, exp_u2, exp_u3, exp_u4, exp_u5, exp_u6, exp_u7, exp_u8)

        if i != dist_num-1:
            # clean
            subprocess.check_call(directory + '/Allclean', shell=True)
            print("Case cleaned")

    print()

end = time.time()
print("Time: {}s".format(end-start))

