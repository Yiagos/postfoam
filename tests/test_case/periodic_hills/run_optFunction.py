import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt


# Functions
def write_coeff(path, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8):     # Replaces model coefficients
    filename = path + '/constant/turbulenceProperties'
    print('Writing coefficients' + ' to file ' + filename)
    file = open(filename, 'w')
    writeString = """/*--------------------------------*- C++ -*----------------------------------*\
    | =========                 |                                                 |
    | \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\    /   O peration     | Version:  2.3.0                                 |
    |   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
    |    \\/     M anipulation  |                                                 |
    \*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        location    "constant";
        object      turbulenceProperties;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    simulationType RAS;

    RAS
    {
    RASModel        kOmegaSSTFS;

        turbulence      on;

        printCoeffs     on;

        kOmegaSSTFSCoeffs
        {
            a2          %s;
            b2          %s;
            betaStar2   %s;
	    coeff1	%s;
	    coeff2      %s;
	    coeff3      %s;
            coeff4      %s;
            coeff5      %s;
        }
    }

    // ************************************************************************* //""" % (coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8)
    file.write(writeString)
    file.close()


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


def get_error(path, v, points, u):
    file_u_4 = path + "/postProcessing/sampleDict2/3000/lineuh_4_U.csv"
    err = 0

    u_4 = np.loadtxt(file_u_4, usecols=[1], dtype=float, delimiter=",", skiprows=1) / v
    u_4_points = np.loadtxt(file_u_4, usecols=[0], dtype=float, delimiter=",", skiprows=1)

    for j in range(0, len(points)):
        diff = np.abs(u_4_points - points[j])
        index = diff.argmin()
        err = err + (u[j] - u_4[index]) ** 2

    mse = err / len(points)
    print(mse)
    return mse


start = time.time()
# Constants
directory = "/home/yiagoskyrits/OpenFOAM/yiagoskyrits-9/run/case_1p0"
v = 0.028          # Velocity
rho = 1.2     # Density
a1 = 0.09
b1 = 0.09
betaStar = 0.09
c1 = 0.09
c2 = 0.09
c3 = 0.09
c4 = 0.09
c5 = 0.09
dist_num = 120    # Number of distributions
h = 1           # Step height
loop_num = 5
sd1 = 0.01       # Standard deviation for each distribution
sd2 = 0.01      # Make sure sd gives values within the required limits
sd3 = 0.01

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


out = np.zeros((1, 9))

for loop in range(loop_num):

    # Clean
    subprocess.check_call(directory + '/Allclean', shell=True)

    if loop != 0:                                   # Update distribution parameters
        dist_num = int(dist_num/2)
        a1 = result[0][0]
        b1 = result[0][1]
        betaStar = result[0][2]
        c1 = result[0][3]
        c2 = result[0][4]
        c3 = result[0][5]
        c4 = result[0][6]
        c5 = result[0][7]
        sd1 = sd1/2
        sd2 = sd2/2
        sd3 = sd3/2

    # Create initial distributions
    a1_dist = abs(np.round(np.random.normal(a1, sd1, dist_num), 3))
    b1_dist = abs(np.round(np.random.normal(b1, sd2, dist_num), 3))
    betaStar_dist = abs(np.round(np.random.normal(betaStar, sd3, dist_num), 3))
    c1_dist = abs(np.round(np.random.normal(c1, sd1, dist_num), 3))
    c2_dist = abs(np.round(np.random.normal(c2, sd1, dist_num), 3))
    c3_dist = abs(np.round(np.random.normal(c3, sd1, dist_num), 3))
    c4_dist = abs(np.round(np.random.normal(c4, sd1, dist_num), 3))
    c5_dist = abs(np.round(np.random.normal(c5, sd1, dist_num), 3))
    comb = np.vstack((a1_dist, b1_dist, betaStar_dist, c1_dist, c2_dist, c3_dist, c4_dist, c5_dist)).T     # Matrix of coefficients

    error = np.ones(dist_num)       # Array of errors

    for i in range(dist_num):
        write_coeff(directory, comb[i][0], comb[i][1], comb[i][2], comb[i][3], comb[i][4] ,comb[i][5] ,comb[i][6] ,comb[i][7])      # Change coefficients
        # simpleFoam
        subprocess.check_call(directory + '/runSim', shell=True)
        #error[i] = get_error(directory, v, exp_points_u4, exp_u4)
        error[i] = get_error_all(directory, v, exp_points_u1, exp_u005, exp_u05, exp_u1, exp_u2, exp_u3, exp_u4, exp_u5, exp_u6, exp_u7, exp_u8)

        if i != dist_num-1:
            # clean
            subprocess.check_call(directory + '/Allclean', shell=True)
            print("Case cleaned")

    if loop == 0:
        result = np.column_stack((comb, error))
    else:
        result = np.column_stack((comb, error))
        result = np.row_stack((result, prev_result))
    result = result[np.argsort(result[:, 8])]
    prev_result = result[0]
    print(result)
    print()
    out = np.append(out, result, axis=0)

out = out[1:, :]
end = time.time()
print("Time: {}s".format(end-start))

np.savetxt("opt_output.txt", out, delimiter=" ")
