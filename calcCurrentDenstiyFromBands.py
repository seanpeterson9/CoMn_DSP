from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import numpy as np
import os
import csv

def fermi_deriv(E,T):
    return 1/(2.*np.cosh(E/(2.*T)))**2.

def deltaGaussHerm(Ek,Ek_prime,f):
    hermiteZero = 1
    hermiteTwo = 4.*(((Ek - Ek_prime)**2)/(2*f*f)) - 2.
    coeffZero = 1./np.sqrt(np.pi)
    coeffTwo = -1./(4.*np.sqrt(np.pi))
    #coeffTwo = 0

    gauss = np.exp(-((Ek - Ek_prime)**2)/(2*f*f))
    return (1./(f*np.sqrt(2.)))*gauss*(coeffZero*hermiteZero + coeffTwo*hermiteTwo)

"""
E = np.linspace(-1.5,1.5,101)
dE = E[1] - E[0]
E2 = np.linspace(-.48,.48,101)
dE2 = E2[1] - E2[0]
print(np.sum(dE*deltaGaussHerm(E,0,.03)),dE2*np.sum(deltaGaussHerm(E2,0,.025*.48)))
plt.scatter(E,deltaGaussHerm(E,0,.03),c = 'blue')
plt.scatter(E2,deltaGaussHerm(E2,0,.025*.48),c = 'red')
plt.show()
#"""

def cosFitTest(x,fitParams,n):
    ret = np.zeros(x.shape)
    for i in range(0,n):
        ret = ret + fitParams[i]*np.cos(2*i*np.pi*x)
    return ret

def cosFitTestWrapper(x,A0,A1,A2,A3,A4,A5,A6):
    return cosFitTest(x,[A0,A1,A2,A3,A4,A5,A6],7)

def cosFit(x,A,B,C,D,E,F,G):#,H,I):
    #H = (B - 3*D + 5*F)/7.
    #return A + B*np.cos(np.pi*x) + C*np.cos(2*np.pi*x) + D*np.cos(3*np.pi*x) + E*np.cos(4*np.pi*x) + F*np.cos(5*np.pi*x) + G*np.cos(6*np.pi*x) + H*np.cos(7*np.pi*x) #+ I*np.cos(8*np.pi*x)
    return A + B*np.cos(2*np.pi*x) + C*np.cos(4*np.pi*x) + D*np.cos(6*np.pi*x) + E*np.cos(8*np.pi*x) + F*np.cos(10*np.pi*x) + G*np.cos(12*np.pi*x)# + H*np.cos(7*np.pi*x) #+ I*np.cos(8*np.pi*x)

def cosFitTest_2D(x,y,fitParams,n):
    ret = np.zeros(x.shape)
    idx = 0
    for i in range(0,n):
        for j in range(0,i+1):
            ret = ret + fitParams[idx]*(np.cos(2*i*np.pi*x)*np.cos(2*j*np.pi*y) + np.cos(2*j*np.pi*x)*np.cos(2*i*np.pi*y))
            idx = int(idx + 1)
    return ret

def cosFitTestWrapper_2D(xy,A00,A10,A11,A20,A21,A22,A30,A31,A32,A33,A40,A41,A42,A43,A44,A50,A51,A52,A53,A54,A55,A60,A61,A62,A63,A64,A65,A66,A70,A71,A72,A73,A74,A75,A76,A77):
    fitParams = [A00,A10,A11,A20,A21,A22,A30,A31,A32,A33,A40,A41,A42,A43,A44,A50,A51,A52,A53,A54,A55,A60,A61,A62,A63,A64,A65,A66,A70,A71,A72,A73,A74,A75,A76,A77]
    x,y = xy
    return cosFitTest_2D(x,y,fitParams,8)

def cosFitTest_3D(x,y,z,fitParams,n):
    ret = np.zeros(x.shape)
    idx = 0
    for i in range(0,n):
        for j in range(0,i+1):
            for k in range(0,n):
                ret = ret + fitParams[idx]*(np.cos(2*i*np.pi*x)*np.cos(2*j*np.pi*y) + np.cos(2*j*np.pi*x)*np.cos(2*i*np.pi*y))*np.cos(2*k*np.pi*z)
                idx = int(idx + 1)
    return ret

def cosFitTest_3D_grad(x,y,z,fitParams,n):
    vx = np.zeros(x.shape)
    vy = np.zeros(x.shape)
    vz = np.zeros(x.shape)
    idx = 0
    for i in range(0,n):
        for j in range(0,i+1):
            for k in range(0,n):
                #if i > 0:
                vx = vx - 2*np.pi*fitParams[idx]*(i*np.sin(2*i*np.pi*x)*np.cos(2*j*np.pi*y) + j*np.sin(2*j*np.pi*x)*np.cos(2*i*np.pi*y))*np.cos(2*k*np.pi*z)
                #if j > 0:
                vy = vy - 2*np.pi*fitParams[idx]*(j*np.cos(2*i*np.pi*x)*np.sin(2*j*np.pi*y) + i*np.cos(2*j*np.pi*x)*np.sin(2*i*np.pi*y))*np.cos(2*k*np.pi*z)
                #if k > 0:
                vz = vz - 2*k*np.pi*fitParams[idx]*(np.cos(2*i*np.pi*x)*np.cos(2*j*np.pi*y) + np.cos(2*j*np.pi*x)*np.cos(2*i*np.pi*y))*np.sin(2*k*np.pi*z)
                idx = int(idx + 1)
    return vx, vy, vz
"""
def cosFitTestWrapper_3D(xyz,A000,A001,A002,A003,A004,A005,A006,A007,A100,A101,A102,A103,A104,A105,A106,A107,A110,A111,A112,A113,A114,A115,A116,A117,A200,A201,A202,A203,A204,A205,A206,A207,A210,A211,A212,A213,A214,A215,A216,A217,A220,A221,A222,A223,A224,A225,A226,A227,A300,A301,A302,A303,A304,A305,A306,A307,A310,A311,A312,A313,A314,A315,A316,A317,A320,A321,A322,A323,A324,A325,A326,A327,A330,A331,A332,A333,A334,A335,A336,A337,A400,A401,A402,A403,A404,A405,A406,A407,A410,A411,A412,A413,A414,A415,A416,A417,A420,A421,A422,A423,A424,A425,A426,A427,A430,A431,A432,A433,A434,A435,A436,A437,A440,A441,A442,A443,A444,A445,A446,A447,A500,A501,A502,A503,A504,A505,A506,A507,A510,A511,A512,A513,A514,A515,A516,A517,A520,A521,A522,A523,A524,A525,A526,A527,A530,A531,A532,A533,A534,A535,A536,A537,A540,A541,A542,A543,A544,A545,A546,A547,A550,A551,A552,A553,A554,A555,A556,A557,A600,A601,A602,A603,A604,A605,A606,A607,A610,A611,A612,A613,A614,A615,A616,A617,A620,A621,A622,A623,A624,A625,A626,A627,A630,A631,A632,A633,A634,A635,A636,A637,A640,A641,A642,A643,A644,A645,A646,A647,A650,A651,A652,A653,A654,A655,A656,A657,A660,A661,A662,A663,A664,A665,A666,A667,A700,A701,A702,A703,A704,A705,A706,A707,A710,A711,A712,A713,A714,A715,A716,A717,A720,A721,A722,A723,A724,A725,A726,A727,A730,A731,A732,A733,A734,A735,A736,A737,A740,A741,A742,A743,A744,A745,A746,A747,A750,A751,A752,A753,A754,A755,A756,A757,A760,A761,A762,A763,A764,A765,A766,A767,A770,A771,A772,A773,A774,A775,A776,A777):
    x,y,z = xyz
    fitParams = [A000,A001,A002,A003,A004,A005,A006,A007,A100,A101,A102,A103,A104,A105,A106,A107,A110,A111,A112,A113,A114,A115,A116,A117,A200,A201,A202,A203,A204,A205,A206,A207,A210,A211,A212,A213,A214,A215,A216,A217,A220,A221,A222,A223,A224,A225,A226,A227,A300,A301,A302,A303,A304,A305,A306,A307,A310,A311,A312,A313,A314,A315,A316,A317,A320,A321,A322,A323,A324,A325,A326,A327,A330,A331,A332,A333,A334,A335,A336,A337,A400,A401,A402,A403,A404,A405,A406,A407,A410,A411,A412,A413,A414,A415,A416,A417,A420,A421,A422,A423,A424,A425,A426,A427,A430,A431,A432,A433,A434,A435,A436,A437,A440,A441,A442,A443,A444,A445,A446,A447,A500,A501,A502,A503,A504,A505,A506,A507,A510,A511,A512,A513,A514,A515,A516,A517,A520,A521,A522,A523,A524,A525,A526,A527,A530,A531,A532,A533,A534,A535,A536,A537,A540,A541,A542,A543,A544,A545,A546,A547,A550,A551,A552,A553,A554,A555,A556,A557,A600,A601,A602,A603,A604,A605,A606,A607,A610,A611,A612,A613,A614,A615,A616,A617,A620,A621,A622,A623,A624,A625,A626,A627,A630,A631,A632,A633,A634,A635,A636,A637,A640,A641,A642,A643,A644,A645,A646,A647,A650,A651,A652,A653,A654,A655,A656,A657,A660,A661,A662,A663,A664,A665,A666,A667,A700,A701,A702,A703,A704,A705,A706,A707,A710,A711,A712,A713,A714,A715,A716,A717,A720,A721,A722,A723,A724,A725,A726,A727,A730,A731,A732,A733,A734,A735,A736,A737,A740,A741,A742,A743,A744,A745,A746,A747,A750,A751,A752,A753,A754,A755,A756,A757,A760,A761,A762,A763,A764,A765,A766,A767,A770,A771,A772,A773,A774,A775,A776,A777]
    return cosFitTest_3D(x,y,z,fitParams,8)
#"""
"""
def cosFitTestWrapper_3D(xyz,A000,A001,A002,A003,A004,A005,A100,A101,A102,A103,A104,A105,A110,A111,A112,A113,A114,A115,A200,A201,A202,A203,A204,A205,A210,A211,A212,A213,A214,A215,A220,A221,A222,A223,A224,A225,A300,A301,A302,A303,A304,A305,A310,A311,A312,A313,A314,A315,A320,A321,A322,A323,A324,A325,A330,A331,A332,A333,A334,A335,A400,A401,A402,A403,A404,A405,A410,A411,A412,A413,A414,A415,A420,A421,A422,A423,A424,A425,A430,A431,A432,A433,A434,A435,A440,A441,A442,A443,A444,A445,A500,A501,A502,A503,A504,A505,A510,A511,A512,A513,A514,A515,A520,A521,A522,A523,A524,A525,A530,A531,A532,A533,A534,A535,A540,A541,A542,A543,A544,A545,A550,A551,A552,A553,A554,A555):
    x,y,z = xyz
    fitParams = [A000,A001,A002,A003,A004,A005,A100,A101,A102,A103,A104,A105,A110,A111,A112,A113,A114,A115,A200,A201,A202,A203,A204,A205,A210,A211,A212,A213,A214,A215,A220,A221,A222,A223,A224,A225,A300,A301,A302,A303,A304,A305,A310,A311,A312,A313,A314,A315,A320,A321,A322,A323,A324,A325,A330,A331,A332,A333,A334,A335,A400,A401,A402,A403,A404,A405,A410,A411,A412,A413,A414,A415,A420,A421,A422,A423,A424,A425,A430,A431,A432,A433,A434,A435,A440,A441,A442,A443,A444,A445,A500,A501,A502,A503,A504,A505,A510,A511,A512,A513,A514,A515,A520,A521,A522,A523,A524,A525,A530,A531,A532,A533,A534,A535,A540,A541,A542,A543,A544,A545,A550,A551,A552,A553,A554,A555]
    return cosFitTest_3D(x,y,z,fitParams,6)
#"""
#"""
def cosFitTestWrapper_3D(xyz,A000,A001,A002,A003,A004,A100,A101,A102,A103,A104,A110,A111,A112,A113,A114,A200,A201,A202,A203,A204,A210,A211,A212,A213,A214,A220,A221,A222,A223,A224,A300,A301,A302,A303,A304,A310,A311,A312,A313,A314,A320,A321,A322,A323,A324,A330,A331,A332,A333,A334,A400,A401,A402,A403,A404,A410,A411,A412,A413,A414,A420,A421,A422,A423,A424,A430,A431,A432,A433,A434,A440,A441,A442,A443,A444):
    x,y,z = xyz
    fitParams = [A000,A001,A002,A003,A004,A100,A101,A102,A103,A104,A110,A111,A112,A113,A114,A200,A201,A202,A203,A204,A210,A211,A212,A213,A214,A220,A221,A222,A223,A224,A300,A301,A302,A303,A304,A310,A311,A312,A313,A314,A320,A321,A322,A323,A324,A330,A331,A332,A333,A334,A400,A401,A402,A403,A404,A410,A411,A412,A413,A414,A420,A421,A422,A423,A424,A430,A431,A432,A433,A434,A440,A441,A442,A443,A444]
    return cosFitTest_3D(x,y,z,fitParams,5)
#"""

def cosFitTestConstrained_3D(fitParams,constStrParam,gradStrParam,xyz,E):
    kx,ky,kz = xyz
    constStr = 1+4./(1.+np.exp(-.02*constStrParam))
    gradStr = 2./(1.+np.exp(-.02*gradStrParam))

    dkx = np.gradient(kx)
    dky = np.gradient(ky)
    dkz = np.gradient(kz)
    dk = np.sqrt(dkx**2 + dky**2 + dkz**2)
    dE = np.gradient(E)
    
    ux = dkx/dk
    uy = dky/dk
    uz = dkz/dk
    
    vx,vy,vz = cosFitTest_3D_grad(kx,ky,kz,fitParams,5)
    #n_fit = 5 has 75 fit parameters I believe
    #mask = np.abs(E) < 0.2
    ret1 = np.sign(E - cosFitTest_3D(kx,ky,kz,fitParams,5))*np.abs(E - cosFitTest_3D(kx,ky,kz,fitParams,5))**(2./3.)
    ret2 = np.sign(dE/dk - vx*ux - vy*uy - vz*uz)*np.abs(dE/dk - vx*ux - vy*uy - vz*uz)**(2./3.)
    #ret[mask] = ret[mask] + (E[mask] - cosFitTest_3D(x[mask],y[mask],z[mask],fitParams,5))/(np.abs(E[mask])+.005)
    ret1 = ret1 + constStr*deltaGaussHerm(0,E,.09)*np.sign(E - cosFitTest_3D(kx,ky,kz,fitParams,5))*np.abs(E - cosFitTest_3D(kx,ky,kz,fitParams,5))**(2./3.)
    ret2 = ret2 + constStr*deltaGaussHerm(0,E,.09)*np.sign(dE/dk - vx*ux - vy*uy - vz*uz)*np.abs(dE/dk - vx*ux - vy*uy - vz*uz)**(2./3.)
    #print(np.sum(ret1),np.sum(ret2),np.sum(ret1 + np.sign(ret1)*np.abs(ret2)))
    #return ret1 + np.sign(ret1)*np.abs(ret2)
    ret = []
    for i in range(0,len(kx)):
        #ret.append((ret1[i],gradStr*ret2[i]))
        ret.append(ret1[i])
        ret.append(gradStr*ret2[i])
    #return np.abs(ret1) + gradStr*np.abs(ret2)
    return np.abs(ret)

def cosFitTestConstrainedWrapper_3D(strFitParams,xyz,E):
    kx,ky,kz = xyz
    dkx = np.gradient(kx)
    dky = np.gradient(ky)
    dkz = np.gradient(kz)
    dk = np.sqrt(dkx**2 + dky**2 + dkz**2)
    dE = np.gradient(E)
    
    ux = dkx/dk
    uy = dky/dk
    uz = dkz/dk
    
    popt = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(strFitParams[0],strFitParams[1],(kx,ky,kz),E))[0]
    fitParams = popt
    vx,vy,vz = cosFitTest_3D_grad(kx,ky,kz,fitParams,5)
    ret1 = deltaGaussHerm(0,E,.12)*(E - cosFitTest_3D(kx,ky,kz,fitParams,5))
    mask = (np.abs(E) > .16) & (np.abs(cosFitTest_3D(kx,ky,kz,fitParams,5)) < .16)
    ret1[mask] = ret1[mask] + deltaGaussHerm(0,cosFitTest_3D(kx[mask],ky[mask],kz[mask],fitParams,5),.12)*(E[mask] - cosFitTest_3D(kx[mask],ky[mask],kz[mask],fitParams,5))

    ret2 = deltaGaussHerm(0,E,.12)*(dE/dk - vx*ux - vy*uy - vz*uz)
    #N = np.linspace(0,1,len(kx))
    #plt.scatter(N,dE/dk,c = 'red')
    #plt.scatter(N,vx*ux+vy*uy+vz*uz,c = 'blue')
    #plt.scatter(N,ux**2+uy**2+uz**2,c = 'k')
    #plt.show()
    #print(strFitParams,np.sum(np.abs(ret1[mask])),np.sqrt(3.)*np.sum(np.abs(ret2[mask]))/max(np.sqrt(vx**2+vy**2+vz**2).flatten()))
    print(1+4./(1.+np.exp(-.02*strFitParams[0])),2/(1.+np.exp(-.02*strFitParams[1])),np.sum(np.abs(ret1)),np.sum(np.abs(ret2))/max(dE/dk))
    #return [np.sum(np.abs(ret1[mask])),np.sqrt(3.)*np.sum(np.abs(ret2[mask]))/max(np.sqrt(vx**2+vy**2+vz**2).flatten())]
    return [np.sum(np.abs(ret1)**(2./3.)),np.sum(np.abs(ret2)**(2./3.))/max(np.abs(dE/dk)**(2./3.))]

def cosFit_deriv(x,A,B,C,D,E,F,G):
    return 2*B*np.pi*np.sin(2*np.pi*x) + 4*C*np.pi*np.sin(4*np.pi*x) + 6*D*np.pi*np.sin(6*np.pi*x) + 8*E*np.pi*np.sin(8*np.pi*x) + 10*F*np.pi*np.sin(10*np.pi*x) + 12*G*np.pi*np.sin(12*np.pi*x)


def cosFit_2D(xy,C00,C20,C22,C40,C42,C44,C60,C62,C64,C66,C80,C82,C84,C86,C88,C90,C92,C94,C96,C98,C99):
    x,y = xy
    return C00 + C20*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)) + C22*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + C40*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y)) + C42*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y)) + C44*np.cos(4*np.pi*x)*np.cos(4*np.pi*y) + C60*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y)) + C62*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y)) + C64*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y)) + C66*np.cos(6*np.pi*x)*np.cos(6*np.pi*y) + C80*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y)) + C82*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y)) + C84*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y)) + C86*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y)) + C88*np.cos(8*np.pi*x)*np.cos(8*np.pi*y) + C90*(np.cos(10*np.pi*x)+np.cos(10*np.pi*y)) + C92*(np.cos(10*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(10*np.pi*y)) + C94*(np.cos(10*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(10*np.pi*y)) + C96*(np.cos(10*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(10*np.pi*y)) + C98*(np.cos(10*np.pi*x)*np.cos(8*np.pi*y)+np.cos(8*np.pi*x)*np.cos(10*np.pi*y)) + C99*np.cos(10*np.pi*x)*np.cos(10*np.pi*y)

def cosFit_3D_wrapper(x,y,z,fitParamsList):
    return cosFit_3D((x,y,z),fitParamsList[0],fitParamsList[1],fitParamsList[2],fitParamsList[3],fitParamsList[4],fitParamsList[5],fitParamsList[6],fitParamsList[7],fitParamsList[8],fitParamsList[9],fitParamsList[10],fitParamsList[11],fitParamsList[12],fitParamsList[13],fitParamsList[14],fitParamsList[15],fitParamsList[16],fitParamsList[17],fitParamsList[18],fitParamsList[19],fitParamsList[20],fitParamsList[21],fitParamsList[22],fitParamsList[23],fitParamsList[24],fitParamsList[25],fitParamsList[26],fitParamsList[27],fitParamsList[28],fitParamsList[29],fitParamsList[30],fitParamsList[31],fitParamsList[32],fitParamsList[33],fitParamsList[34],fitParamsList[35],fitParamsList[36],fitParamsList[37],fitParamsList[38],fitParamsList[39],fitParamsList[40],fitParamsList[41],fitParamsList[42],fitParamsList[43],fitParamsList[44],fitParamsList[45],fitParamsList[46],fitParamsList[47],fitParamsList[48],fitParamsList[49],fitParamsList[50],fitParamsList[51],fitParamsList[52],fitParamsList[53],fitParamsList[54],fitParamsList[55],fitParamsList[56],fitParamsList[57],fitParamsList[58],fitParamsList[59],fitParamsList[60],fitParamsList[61],fitParamsList[62],fitParamsList[63],fitParamsList[64],fitParamsList[65],fitParamsList[66],fitParamsList[67],fitParamsList[68],fitParamsList[69],fitParamsList[70],fitParamsList[71],fitParamsList[72],fitParamsList[73])

def cosFit_3D(xyz,C000,C200,C220,C002,C202,C222,C400,C420,C440,C004,C204,C402,C404,C224,C422,C424,C442,C444,C600,C620,C640,C660,C006,C206,C602,C226,C622,C406,C604,C426,C642,C624,C446,C664,C606,C626,C662,C646,C666,C800,C820,C840,C860,C880,C008,C208,C802,C228,C822,C408,C804,C428,C842,C824,C448,C844,C608,C806,C628,C862,C826,C648,C864,C846,C668,C866,C808,C828,C882,C848,C884,C868,C886,C888):
    #function has like 73 fit params! Probably an overfit, but doesn't matter since the fit parameters are meaningless and we just 
    # are using this instead of an interpolation
    x,y,z = xyz
    secondOrder = C200*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)) + C220*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + C002*np.cos(2*np.pi*z) + C202*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(2*np.pi*z) + C222*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(2*np.pi*z)

    fourthOrder = C400*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y)) + C420*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y)) + C440*np.cos(4*np.pi*x)*np.cos(4*np.pi*y) + C004*np.cos(4*np.pi*z) + C204*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(4*np.pi*z) + C402*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.cos(2*np.pi*z) + C404*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.cos(4*np.pi*z) + C224*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(4*np.pi*z) + C424*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(4*np.pi*z) + C442*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(2*np.pi*z) + C444*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(4*np.pi*z)

    sixthOrder = C600*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y)) + C620*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y)) + C640*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y)) + C660*np.cos(6*np.pi*x)*np.cos(6*np.pi*y) + C006*np.cos(6*np.pi*z) + C206*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(6*np.pi*z) + C602*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(2*np.pi*z) + C226*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(6*np.pi*z) + C622*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(2*np.pi*z) + C406*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.cos(6*np.pi*z) + C604*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(4*np.pi*z) + C426*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(6*np.pi*z) + C642*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y))*np.cos(2*np.pi*z) + C624*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(4*np.pi*z) + C446*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(6*np.pi*z) + C664*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(4*np.pi*z) + C606*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(6*np.pi*z) + C626*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(6*np.pi*z) + C662*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(2*np.pi*z) + C646*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y))*np.cos(6*np.pi*z) + C666*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(6*np.pi*z)

    eighthOrder = C800*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y)) + C820*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y)) + C840*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y)) + C860*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y)) + C880*np.cos(8*np.pi*x)*np.cos(8*np.pi*y) + C008*np.cos(8*np.pi*z) + C208*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(8*np.pi*z) + C802*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C228*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(8*np.pi*z) + C822*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C408*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.cos(8*np.pi*z) + C804*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C428*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(8*np.pi*z) + C842*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C824*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C448*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(8*np.pi*z) + C844*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C608*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(8*np.pi*z) + C806*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C628*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(8*np.pi*z) + C862*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C826*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C648*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*x))*np.cos(8*np.pi*z) + C864*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C846*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C668*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(8*np.pi*z) + C866*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C808*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C828*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C882*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(2*np.pi*z) + C848*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C884*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(4*np.pi*z) + C868*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C886*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(6*np.pi*z) + C888*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(8*np.pi*z)
    
    return C000 + secondOrder + fourthOrder + sixthOrder + eighthOrder

def cosFit_3D_grad_wrapper(x,y,z,fitParamsList):
    return cosFit_3D_grad((x,y,z),fitParamsList[0],fitParamsList[1],fitParamsList[2],fitParamsList[3],fitParamsList[4],fitParamsList[5],fitParamsList[6],fitParamsList[7],fitParamsList[8],fitParamsList[9],fitParamsList[10],fitParamsList[11],fitParamsList[12],fitParamsList[13],fitParamsList[14],fitParamsList[15],fitParamsList[16],fitParamsList[17],fitParamsList[18],fitParamsList[19],fitParamsList[20],fitParamsList[21],fitParamsList[22],fitParamsList[23],fitParamsList[24],fitParamsList[25],fitParamsList[26],fitParamsList[27],fitParamsList[28],fitParamsList[29],fitParamsList[30],fitParamsList[31],fitParamsList[32],fitParamsList[33],fitParamsList[34],fitParamsList[35],fitParamsList[36],fitParamsList[37],fitParamsList[38],fitParamsList[39],fitParamsList[40],fitParamsList[41],fitParamsList[42],fitParamsList[43],fitParamsList[44],fitParamsList[45],fitParamsList[46],fitParamsList[47],fitParamsList[48],fitParamsList[49],fitParamsList[50],fitParamsList[51],fitParamsList[52],fitParamsList[53],fitParamsList[54],fitParamsList[55],fitParamsList[56],fitParamsList[57],fitParamsList[58],fitParamsList[59],fitParamsList[60],fitParamsList[61],fitParamsList[62],fitParamsList[63],fitParamsList[64],fitParamsList[65],fitParamsList[66],fitParamsList[67],fitParamsList[68],fitParamsList[69],fitParamsList[70],fitParamsList[71],fitParamsList[72],fitParamsList[73])

def cosFit_3D_grad(xyz,C000,C200,C220,C002,C202,C222,C400,C420,C440,C004,C204,C402,C404,C224,C422,C424,C442,C444,C600,C620,C640,C660,C006,C206,C602,C226,C622,C406,C604,C426,C642,C624,C446,C664,C606,C626,C662,C646,C666,C800,C820,C840,C860,C880,C008,C208,C802,C228,C822,C408,C804,C428,C842,C824,C448,C844,C608,C806,C628,C862,C826,C648,C864,C846,C668,C866,C808,C828,C882,C848,C884,C868,C886,C888):
    #function has like 73 fit params! Probably an overfit, but doesn't matter since the fit parameters are meaningless and we just 
    # are using this instead of an interpolation
    x,y,z = xyz
    secondOrder_x = -2*np.pi*C200*np.sin(2*np.pi*x) - 2*np.pi*C220*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) - 2*np.pi*C202*np.sin(2*np.pi*x)*np.cos(2*np.pi*z) - 2*np.pi*C222*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(2*np.pi*z)
    secondOrder_y = -2*np.pi*C200*np.sin(2*np.pi*y) - 2*np.pi*C220*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) - 2*np.pi*C202*np.sin(2*np.pi*y)*np.cos(2*np.pi*z) - 2*np.pi*C222*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)*np.cos(2*np.pi*z)
    secondOrder_z = -2*np.pi*C002*np.sin(2*np.pi*z) - 2*np.pi*C202*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.sin(2*np.pi*z) - 2*np.pi*C222*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.sin(2*np.pi*z)

    fourthOrder_x = -4*np.pi*C400*np.sin(4*np.pi*x) - np.pi*C420*(4*np.sin(4*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(4*np.pi*y)) - 4*np.pi*C440*np.sin(4*np.pi*x)*np.cos(4*np.pi*y) - 2*np.pi*C204*np.sin(2*np.pi*x)*np.cos(4*np.pi*z) - 4*np.pi*C402*np.sin(4*np.pi*x)*np.cos(2*np.pi*z) - 4*np.pi*C404*np.sin(4*np.pi*x)*np.cos(4*np.pi*z) - 2*np.pi*C224*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(4*np.pi*z) - np.pi*C424*(4*np.sin(4*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(4*np.pi*z) - 4*np.pi*C442*np.sin(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(2*np.pi*z) - 4*np.pi*C444*np.sin(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(4*np.pi*z)
    
    fourthOrder_y = -4*np.pi*C400*np.sin(4*np.pi*y) - np.pi*C420*(2*np.cos(4*np.pi*x)*np.sin(2*np.pi*y)+4*np.cos(2*np.pi*x)*np.sin(4*np.pi*y)) - 4*np.pi*C440*np.cos(4*np.pi*x)*np.sin(4*np.pi*y) - 2*np.pi*C204*np.sin(2*np.pi*y)*np.cos(4*np.pi*z) - 4*np.pi*C402*np.sin(4*np.pi*y)*np.cos(2*np.pi*z) - 4*np.pi*C404*np.sin(4*np.pi*y)*np.cos(4*np.pi*z) - 2*np.pi*C224*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)*np.cos(4*np.pi*z) - np.pi*C424*(2*np.cos(4*np.pi*x)*np.sin(2*np.pi*y)+4*np.cos(2*np.pi*x)*np.sin(4*np.pi*y))*np.cos(4*np.pi*z) - 4*np.pi*C442*np.cos(4*np.pi*x)*np.sin(4*np.pi*y)*np.cos(2*np.pi*z) - 4*np.pi*C444*np.cos(4*np.pi*x)*np.sin(4*np.pi*y)*np.cos(4*np.pi*z)
    
    fourthOrder_z = -4*np.pi*C004*np.sin(4*np.pi*z) - 4*np.pi*C204*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.sin(4*np.pi*z) - 2*np.pi*C402*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.sin(2*np.pi*z) - 4*np.pi*C404*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.sin(4*np.pi*z) - 4*np.pi*C224*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.sin(4*np.pi*z) - 4*np.pi*C424*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.sin(4*np.pi*z) - 2*np.pi*C442*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.sin(2*np.pi*z) - 4*np.pi*C444*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.sin(4*np.pi*z)

    sixthOrder_x = -6*np.pi*C600*np.sin(6*np.pi*x) - np.pi*C620*(6*np.sin(6*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(6*np.pi*y)) - np.pi*C640*(6*np.sin(6*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(6*np.pi*y)) - 6*np.pi*C660*np.sin(6*np.pi*x)*np.cos(6*np.pi*y)  - 2*np.pi*C206*np.sin(2*np.pi*x)*np.cos(6*np.pi*z) - 6*np.pi*C602*np.sin(6*np.pi*x)*np.cos(2*np.pi*z) - 2*np.pi*C226*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(6*np.pi*z) - np.pi*C622*(6*np.sin(6*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(2*np.pi*z) - 4*np.pi*C406*np.sin(4*np.pi*x)*np.cos(6*np.pi*z) - 6*np.pi*C604*np.sin(6*np.pi*x)*np.cos(4*np.pi*z) - np.pi*C426*(4*np.sin(4*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(6*np.pi*z) - np.pi*C642*(6*np.sin(6*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(6*np.pi*y))*np.cos(2*np.pi*z) - np.pi*C624*(6*np.sin(6*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(4*np.pi*z) - 4*np.pi*C446*np.sin(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(6*np.pi*z) - 6*np.pi*C664*np.sin(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(4*np.pi*z) - 6*np.pi*C606*np.sin(6*np.pi*x)*np.cos(6*np.pi*z) - np.pi*C626*(6*np.sin(6*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(6*np.pi*z) - 6*np.pi*C662*np.sin(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(2*np.pi*z) - np.pi*C646*(6*np.sin(6*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(6*np.pi*y))*np.cos(6*np.pi*z) - 6*np.pi*C666*np.sin(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(6*np.pi*z)
    
    sixthOrder_y = -6*np.pi*C600*np.cos(6*np.pi*y) - np.pi*C620*(2*np.cos(6*np.pi*x)*np.sin(2*np.pi*y)+6*np.cos(2*np.pi*x)*np.sin(6*np.pi*y)) - np.pi*C640*(4*np.cos(6*np.pi*x)*np.sin(4*np.pi*y)+6*np.cos(4*np.pi*x)*np.sin(6*np.pi*y)) - 6*np.pi*C660*np.cos(6*np.pi*x)*np.sin(6*np.pi*y) - 2*np.pi*C206*np.sin(2*np.pi*y)*np.cos(6*np.pi*z) - 6*np.pi*C602*np.sin(6*np.pi*y)*np.cos(2*np.pi*z) - 2*np.pi*C226*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)*np.cos(6*np.pi*z) - np.pi*C622*(2*np.cos(6*np.pi*x)*np.sin(2*np.pi*y)+6*np.cos(2*np.pi*x)*np.sin(6*np.pi*y))*np.cos(2*np.pi*z) - 4*np.pi*C406*np.sin(4*np.pi*y)*np.cos(6*np.pi*z) - 6*np.pi*C604*np.sin(6*np.pi*y)*np.cos(4*np.pi*z) - np.pi*C426*(2*np.cos(4*np.pi*x)*np.sin(2*np.pi*y)+4*np.cos(2*np.pi*x)*np.sin(4*np.pi*y))*np.cos(6*np.pi*z) - np.pi*C642*(4*np.cos(6*np.pi*x)*np.sin(4*np.pi*y)+6*np.cos(4*np.pi*x)*np.sin(6*np.pi*y))*np.cos(2*np.pi*z) - np.pi*C624*(2*np.cos(6*np.pi*x)*np.sin(2*np.pi*y)+6*np.cos(2*np.pi*x)*np.sin(6*np.pi*y))*np.cos(4*np.pi*z) - 4*np.pi*C446*np.cos(4*np.pi*x)*np.sin(4*np.pi*y)*np.cos(6*np.pi*z) - 6*np.pi*C664*np.cos(6*np.pi*x)*np.sin(6*np.pi*y)*np.cos(4*np.pi*z) - 6*np.pi*C606*np.cos(6*np.pi*y)*np.cos(6*np.pi*z) - np.pi*C626*(2*np.cos(6*np.pi*x)*np.sin(2*np.pi*y)+6*np.cos(2*np.pi*x)*np.sin(6*np.pi*y))*np.cos(6*np.pi*z) - 6*np.pi*C662*np.cos(6*np.pi*x)*np.sin(6*np.pi*y)*np.cos(2*np.pi*z) - np.pi*C646*(4*np.cos(6*np.pi*x)*np.sin(4*np.pi*y)+6*np.cos(4*np.pi*x)*np.sin(6*np.pi*y))*np.cos(6*np.pi*z) - 6*np.pi*C666*np.cos(6*np.pi*x)*np.sin(6*np.pi*y)*np.cos(6*np.pi*z)
    
    sixthOrder_z = -6*np.pi*C006*np.sin(6*np.pi*z) - 6*np.pi*C206*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.sin(6*np.pi*z) - 2*np.pi*C602*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.sin(2*np.pi*z) - 6*np.pi*C226*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.sin(6*np.pi*z) - 2*np.pi*C622*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.sin(2*np.pi*z) - 6*np.pi*C406*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.sin(6*np.pi*z) - 4*np.pi*C604*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.sin(4*np.pi*z) - 6*np.pi*C426*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.sin(6*np.pi*z) - 2*np.pi*C642*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y))*np.sin(2*np.pi*z) - 4*np.pi*C624*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.sin(4*np.pi*z) - 6*np.pi*C446*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.sin(6*np.pi*z) - 4*np.pi*C664*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.sin(4*np.pi*z) - 6*np.pi*C606*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.sin(6*np.pi*z) - 6*np.pi*C626*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.sin(6*np.pi*z) - 2*np.pi*C662*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.sin(2*np.pi*z) - 6*np.pi*C646*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y))*np.sin(6*np.pi*z) - 6*np.pi*C666*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.sin(6*np.pi*z)

    eighthOrder_x = -8*np.pi*C800*np.sin(8*np.pi*x) - np.pi*C820*(8*np.sin(8*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(8*np.pi*y)) - np.pi*C840*(8*np.sin(8*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(8*np.pi*y)) - np.pi*C860*(8*np.sin(8*np.pi*x)*np.cos(6*np.pi*y)+6*np.sin(6*np.pi*x)*np.cos(8*np.pi*y)) - 8*np.pi*C880*np.sin(8*np.pi*x)*np.cos(8*np.pi*y) - 2*np.pi*C208*np.sin(2*np.pi*x)*np.cos(8*np.pi*z) - 8*np.pi*C802*np.sin(8*np.pi*x)*np.cos(2*np.pi*z) - 2*np.pi*C228*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(8*np.pi*z) - np.pi*C822*(8*np.sin(8*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) - 4*np.pi*C408*np.sin(4*np.pi*x)*np.cos(8*np.pi*z) - 8*np.pi*C804*np.sin(8*np.pi*x)*np.cos(4*np.pi*z) - np.pi*C428*(4*np.sin(4*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(8*np.pi*z) - np.pi*C842*(8*np.sin(8*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) - np.pi*C824*(8*np.sin(8*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) - 4*np.pi*C448*np.sin(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(8*np.pi*z) - np.pi*C844*(8*np.sin(8*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) - 6*np.pi*C608*np.sin(6*np.pi*x)*np.cos(8*np.pi*z) - 8*np.pi*C806*np.sin(8*np.pi*x)*np.cos(6*np.pi*z) - np.pi*C628*(6*np.sin(6*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(8*np.pi*z) - np.pi*C862*(8*np.sin(8*np.pi*x)*np.cos(6*np.pi*y)+6*np.sin(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) - np.pi*C826*(8*np.sin(8*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) - np.pi*C648*(6*np.sin(6*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(6*np.pi*x))*np.cos(8*np.pi*z) - np.pi*C864*(8*np.sin(8*np.pi*x)*np.cos(6*np.pi*y)+6*np.sin(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) - np.pi*C846*(8*np.sin(8*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) - 6*np.pi*C668*np.sin(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(8*np.pi*z) - np.pi*C866*(8*np.sin(8*np.pi*x)*np.cos(6*np.pi*y)+6*np.sin(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) - 8*np.pi*C808*np.sin(8*np.pi*x)*np.cos(8*np.pi*z) - np.pi*C828*(8*np.sin(8*np.pi*x)*np.cos(2*np.pi*y)+2*np.sin(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) - 8*np.pi*C882*np.sin(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(2*np.pi*z) - np.pi*C848*(8*np.sin(8*np.pi*x)*np.cos(4*np.pi*y)+4*np.sin(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) - 8*np.pi*C884*np.sin(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(4*np.pi*z) - np.pi*C868*(8*np.sin(8*np.pi*x)*np.cos(6*np.pi*y)+6*np.sin(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) - 8*np.pi*C886*np.sin(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(6*np.pi*z) - 8*np.pi*C888*np.sin(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(8*np.pi*z)
    
    eighthOrder_y = -8*np.pi*C800*np.sin(8*np.pi*y) - np.pi*C820*(2*np.cos(8*np.pi*x)*np.sin(2*np.pi*y)+8*np.cos(2*np.pi*x)*np.sin(8*np.pi*y)) - np.pi*C840*(4*np.cos(8*np.pi*x)*np.sin(4*np.pi*y)+8*np.cos(4*np.pi*x)*np.sin(8*np.pi*y)) - np.pi*C860*(6*np.cos(8*np.pi*x)*np.sin(6*np.pi*y)+8*np.cos(6*np.pi*x)*np.sin(8*np.pi*y)) - 8*np.pi*C880*np.cos(8*np.pi*x)*np.sin(8*np.pi*y) - 2*np.pi*C208*np.sin(2*np.pi*y)*np.cos(8*np.pi*z) - 8*np.pi*C802*np.sin(8*np.pi*y)*np.cos(2*np.pi*z) - 2*np.pi*C228*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)*np.cos(8*np.pi*z) - np.pi*C822*(2*np.cos(8*np.pi*x)*np.sin(2*np.pi*y)+8*np.cos(2*np.pi*x)*np.sin(8*np.pi*y))*np.cos(2*np.pi*z) - 4*np.pi*C408*np.sin(4*np.pi*y)*np.cos(8*np.pi*z) - 8*np.pi*C804*np.sin(8*np.pi*y)*np.cos(4*np.pi*z) - np.pi*C428*(2*np.cos(4*np.pi*x)*np.sin(2*np.pi*y)+4*np.cos(2*np.pi*x)*np.sin(4*np.pi*y))*np.cos(8*np.pi*z) - np.pi*C842*(4*np.cos(8*np.pi*x)*np.sin(4*np.pi*y)+8*np.cos(4*np.pi*x)*np.sin(8*np.pi*y))*np.cos(2*np.pi*z) - np.pi*C824*(2*np.cos(8*np.pi*x)*np.sin(2*np.pi*y)+8*np.cos(2*np.pi*x)*np.sin(8*np.pi*y))*np.cos(4*np.pi*z) - 4*np.pi*C448*np.cos(4*np.pi*x)*np.sin(4*np.pi*y)*np.cos(8*np.pi*z) - np.pi*C844*(4*np.cos(8*np.pi*x)*np.sin(4*np.pi*y)+8*np.cos(4*np.pi*x)*np.sin(8*np.pi*y))*np.cos(4*np.pi*z) - 6*np.pi*C608*np.sin(6*np.pi*y)*np.cos(8*np.pi*z) - 8*np.pi*C806*np.sin(8*np.pi*y)*np.cos(6*np.pi*z) - np.pi*C628*(2*np.cos(6*np.pi*x)*np.sin(2*np.pi*y)+6*np.cos(2*np.pi*x)*np.sin(6*np.pi*y))*np.cos(8*np.pi*z) - np.pi*C862*(6*np.cos(8*np.pi*x)*np.sin(6*np.pi*y)+8*np.cos(6*np.pi*x)*np.sin(8*np.pi*y))*np.cos(2*np.pi*z) - np.pi*C826*(2*np.cos(8*np.pi*x)*np.sin(2*np.pi*y)+8*np.cos(2*np.pi*x)*np.sin(8*np.pi*y))*np.cos(6*np.pi*z) - np.pi*C648*(4*np.cos(6*np.pi*x)*np.sin(4*np.pi*y)+6*np.cos(4*np.pi*x)*np.sin(6*np.pi*x))*np.cos(8*np.pi*z) - np.pi*C864*(6*np.cos(8*np.pi*x)*np.sin(6*np.pi*y)+8*np.cos(6*np.pi*x)*np.sin(8*np.pi*y))*np.cos(4*np.pi*z) - np.pi*C846*(4*np.cos(8*np.pi*x)*np.sin(4*np.pi*y)+8*np.cos(4*np.pi*x)*np.sin(8*np.pi*y))*np.cos(6*np.pi*z) - 6*np.pi*C668*np.cos(6*np.pi*x)*np.sin(6*np.pi*y)*np.cos(8*np.pi*z) - np.pi*C866*(6*np.cos(8*np.pi*x)*np.sin(6*np.pi*y)+8*np.cos(6*np.pi*x)*np.sin(8*np.pi*y))*np.cos(6*np.pi*z) - 8*np.pi*C808*np.sin(8*np.pi*y)*np.cos(8*np.pi*z) - np.pi*C828*(2*np.cos(8*np.pi*x)*np.sin(2*np.pi*y)+8*np.cos(2*np.pi*x)*np.sin(8*np.pi*y))*np.cos(8*np.pi*z) - 8*np.pi*C882*np.cos(8*np.pi*x)*np.sin(8*np.pi*y)*np.cos(2*np.pi*z) - np.pi*C848*(4*np.cos(8*np.pi*x)*np.sin(4*np.pi*y)+8*np.cos(4*np.pi*x)*np.sin(8*np.pi*y))*np.cos(8*np.pi*z) - 8*np.pi*C884*np.cos(8*np.pi*x)*np.sin(8*np.pi*y)*np.cos(4*np.pi*z) - np.pi*C868*(6*np.cos(8*np.pi*x)*np.sin(6*np.pi*y)+8*np.cos(6*np.pi*x)*np.sin(8*np.pi*y))*np.cos(8*np.pi*z) - 8*np.pi*C886*np.cos(8*np.pi*x)*np.sin(8*np.pi*y)*np.cos(6*np.pi*z) - 8*np.pi*C888*np.cos(8*np.pi*x)*np.sin(8*np.pi*y)*np.cos(8*np.pi*z)
    
    eighthOrder_z = -8*np.pi*C008*np.sin(8*np.pi*z) - 8*np.pi*C208*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.sin(8*np.pi*z) - 2*np.pi*C802*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.sin(2*np.pi*z) - 8*np.pi*C228*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.sin(8*np.pi*z) - 2*np.pi*C822*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.sin(2*np.pi*z) - 8*np.pi*C408*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.sin(8*np.pi*z) - 4*np.pi*C804*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.sin(4*np.pi*z) - 8*np.pi*C428*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.sin(8*np.pi*z) - 2*np.pi*C842*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.sin(2*np.pi*z) - 4*np.pi*C824*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.sin(4*np.pi*z) - 8*np.pi*C448*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.sin(8*np.pi*z) - 4*np.pi*C844*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.sin(4*np.pi*z) - 8*np.pi*C608*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.sin(8*np.pi*z) - 6*np.pi*C806*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.sin(6*np.pi*z) - 8*np.pi*C628*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.sin(8*np.pi*z) - 2*np.pi*C862*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.sin(2*np.pi*z) - 6*np.pi*C826*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.sin(6*np.pi*z) - 8*np.pi*C648*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*x))*np.sin(8*np.pi*z) - 4*np.pi*C864*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.sin(4*np.pi*z) - 6*np.pi*C846*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.sin(6*np.pi*z) - 8*np.pi*C668*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.sin(8*np.pi*z) - 6*np.pi*C866*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.sin(6*np.pi*z) - 8*np.pi*C808*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.sin(8*np.pi*z) - 8*np.pi*C828*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.sin(8*np.pi*z) - 2*np.pi*C882*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.sin(2*np.pi*z) - 8*np.pi*C848*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.sin(8*np.pi*z) - 4*np.pi*C884*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.sin(4*np.pi*z) - 8*np.pi*C868*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.sin(8*np.pi*z) - 6*np.pi*C886*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.sin(6*np.pi*z) - 8*np.pi*C888*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.sin(8*np.pi*z)
    
    return secondOrder_x+fourthOrder_x+sixthOrder_x+eighthOrder_x,secondOrder_y+fourthOrder_y+sixthOrder_y+eighthOrder_y,secondOrder_z+fourthOrder_z+sixthOrder_z+eighthOrder_z

def readBandFile(EBANDSFilename,nbands):
    os.system('./abinitEBANDSReader.sh ' + EBANDSFilename + ' kpts_tmp bands_tmp')
    with open('kpts_tmp','r') as myFile:
        read = csv.reader(myFile,delimiter = ',')
        kx = []
        ky = []
        kz = []
        for row in read:
            kx.append(float(row[0]))
            ky.append(float(row[1]))
            kz.append(float(row[2]))
    
    kx = np.array(kx)
    ky = np.array(ky)
    kz = np.array(kz)
    
    E = []
    for i in range(0,2*nbands):
        E.append([])
    
    with open('bands_tmp','r') as myFile:
        read = csv.reader(myFile,delimiter = ' ')
        for row in read:
            for i in range(0,2*nbands):
                E[i].append(float(row[i]))
    E = np.array(E)
    
    return kx,ky,kz,E

def genFitParams(kx,ky,kz,E,nbands,E_cutoff):
    bandID = []
    fitParamsList = []
    strFitParamsList = []
    for i in range(0,2*nbands):
        if (len(E[i][np.abs(E[i]) < E_cutoff]) > 1):# and (i == 34):
            """
            E_c = E_cutoff
            mask = np.abs(E[i]) < E_c
            while len(E[i][mask]) < 75:
                E_c = 1.2 * E_c
                mask = np.abs(E[i]) < E_c
            """
            #sort = np.argsort(np.abs(E[i]))
            print(i,len(E[i][np.abs(E[i]) < E_cutoff]))
            bandID.append(i)
            #poptInit = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(0,(kx,ky,kz),E[i]))[0]
            popt = leastsq(cosFitTestConstrainedWrapper_3D,x0 = [-30.23,0],args=((kx,ky,kz),E[i]),epsfcn=.05,maxfev=20)[0]
            strFitParamsList.append(popt)
            popt = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(strFitParamsList[-1][0],strFitParamsList[-1][1],(kx,ky,kz),E[i]))[0]
            #popt = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(7.9,.396,(kx,ky,kz),E[i]))[0]
            #popt,pcurve = curve_fit(cosFitTestWrapper_3D,(kx[sort],ky[sort],kz[sort]),E[i][sort])#,p0 = guess)
            fitParamsList.append(popt)
    fitParamsList = np.array(fitParamsList)
    return fitParamsList,bandID

def genEquipotentialCyl(fitParamsList,E,N_theta,N_z):
    theta = np.linspace(0.,np.pi/2.,N_theta)
    z = np.linspace(0.,.5-1e-6,N_z)
    Theta,Z = np.meshgrid(theta,z)
    
    n_fit = 5
    
    dkz = z[1] - z[0]
    
    kr1 = np.zeros(Theta.shape)
    kx1 = np.zeros(Theta.shape)
    ky1 = np.zeros(Theta.shape)
    kz1 = np.zeros(Theta.shape)
    dkr1 = np.zeros(Theta.shape)
    
    kr2 = np.zeros(Theta.shape)
    kx2 = np.zeros(Theta.shape)
    ky2 = np.zeros(Theta.shape)
    kz2 = np.zeros(Theta.shape)
    dkr2 = np.zeros(Theta.shape)
    
    kr3 = np.zeros(Theta.shape)
    kx3 = np.zeros(Theta.shape)
    ky3 = np.zeros(Theta.shape)
    kz3 = np.zeros(Theta.shape)
    dkr3 = np.zeros(Theta.shape)
    
    kr4 = np.zeros(Theta.shape)
    kx4 = np.zeros(Theta.shape)
    ky4 = np.zeros(Theta.shape)
    kz4 = np.zeros(Theta.shape)
    dkr4 = np.zeros(Theta.shape)
    
    for i in range(0,N_z):
        for j in range(0,N_theta):
            
            kr1[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(np.abs(k)*np.cos(Theta[i][j]),np.abs(k)*np.sin(Theta[i][j]),Z[i][j],fitParamsList,n_fit) - E)**2,.09))
            kx1[i][j] = np.abs(kr1[i][j])*np.cos(Theta[i][j])
            ky1[i][j] = np.abs(kr1[i][j])*np.sin(Theta[i][j])
            kz1[i][j] = Z[i][j]


            #print(Theta[i][j],Phi[i][j],kr1[i][j],kx1[i][j],ky1[i][j],kz1[i][j])
            
            kr2[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5 - np.abs(k)*np.cos(Theta[i][j]),.5 - np.abs(k)*np.sin(Theta[i][j]),Z[i][j],fitParamsList,n_fit) - E)**2,.09))
            kx2[i][j] = .5 - np.abs(kr2[i][j])*np.cos(Theta[i][j])
            ky2[i][j] = .5 - np.abs(kr2[i][j])*np.sin(Theta[i][j])
            kz2[i][j] = Z[i][j]

            kr3[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5 - np.abs(k)*np.cos(Theta[i][j]),np.abs(k)*np.sin(Theta[i][j]),Z[i][j],fitParamsList,n_fit) - E)**2,.09))
            kx3[i][j] = .5 - np.abs(kr3[i][j])*np.cos(Theta[i][j])
            ky3[i][j] = np.abs(kr3[i][j])*np.sin(Theta[i][j])
            kz3[i][j] = Z[i][j]
            
            kr4[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(np.abs(k)*np.cos(Theta[i][j]),.5 - np.abs(k)*np.sin(Theta[i][j]),Z[i][j],fitParamsList,n_fit) - E)**2,.09))
            kx4[i][j] = np.abs(kr4[i][j])*np.cos(Theta[i][j])
            ky4[i][j] = .5 - np.abs(kr4[i][j])*np.sin(Theta[i][j])
            kz4[i][j] = Z[i][j]
            #print(Theta[i][j],Z[i][j],kr1[i][j],kr2[i][j],kr3[i][j],kr4[i][j])
            """
            kr3[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),np.abs(k)*np.cos(Phi[i][j]),fitParamsList,8) - E)**2,.12))
            kx3[i][j] = .5 - np.abs(kr3[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky3[i][j] = .5 - np.abs(kr3[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz3[i][j] = np.abs(kr3[i][j])*np.cos(Phi[i][j])
            """
            if ((cosFitTest_3D(kr1[i][j]*np.cos(Theta[i][j]),kr1[i][j]*np.sin(Theta[i][j]),kz1[i][j],fitParamsList,n_fit) - E)**2 > 1e-5) or (kx1[i][j] < 0.) or (kx1[i][j] > .5) or (ky1[i][j] < 0.) or (ky1[i][j] > .5)  or (kz1[i][j] < 0.) or (kz1[i][j] > .5):
                kr1[i][j] = -.5
            if ((cosFitTest_3D(.5-kr2[i][j]*np.cos(Theta[i][j]),.5-kr2[i][j]*np.sin(Theta[i][j]),kz2[i][j],fitParamsList,n_fit) - E)**2 > 1e-5) or (kx2[i][j] < 0.) or (kx2[i][j] > .5) or (ky2[i][j] < 0.) or (ky2[i][j] > .5) or (kz2[i][j] < 0.) or (kz2[i][j] > .5):
                kr2[i][j] = -.5
            if ((cosFitTest_3D(.5-kr3[i][j]*np.cos(Theta[i][j]),kr3[i][j]*np.sin(Theta[i][j]),kz3[i][j],fitParamsList,n_fit) - E)**2 > 1e-5) or (kx3[i][j] < 0.) or (kx3[i][j] > .5) or (ky3[i][j] < 0.) or (ky3[i][j] > .5) or (kz3[i][j] < 0.) or (kz3[i][j] > .5):
                kr3[i][j] = -.5
            if ((cosFitTest_3D(kr4[i][j]*np.cos(Theta[i][j]),.5-kr4[i][j]*np.sin(Theta[i][j]),kz4[i][j],fitParamsList,n_fit) - E)**2 > 1e-5) or (kx4[i][j] < 0.) or (kx4[i][j] > .5) or (ky4[i][j] < 0.) or (ky4[i][j] > .5) or (kz4[i][j] < 0.) or (kz4[i][j] > .5):
                kr4[i][j] = -.5

        dkr1[i] = np.sqrt(np.gradient(kx1[i])**2 + np.gradient(ky1[i])**2)
        dkr2[i] = np.sqrt(np.gradient(kx2[i])**2 + np.gradient(ky2[i])**2)
        dkr3[i] = np.sqrt(np.gradient(kx3[i])**2 + np.gradient(ky3[i])**2)
        dkr4[i] = np.sqrt(np.gradient(kx4[i])**2 + np.gradient(ky4[i])**2)
    """
    dk1_dtheta = np.zeros(kr1.shape)
    dk1_dphi = np.zeros(kr1.shape)
    dk2_dtheta = np.zeros(kr2.shape)
    dk2_dphi = np.zeros(kr2.shape)
    dk3_dtheta = np.zeros(kr3.shape)
    dk3_dphi = np.zeros(kr3.shape)
    for i in range(0,N_theta):
        dk1_dtheta[i,] = np.sqrt(np.gradient(kx1[i,])**2 + np.gradient(ky1[i,])**2 + np.gradient(kz1[i,])**2)
        dk2_dtheta[i,] = np.sqrt(np.gradient(kx2[i,])**2 + np.gradient(ky2[i,])**2 + np.gradient(kz2[i,])**2)
        dk3_dtheta[i,] = np.sqrt(np.gradient(kx3[i,])**2 + np.gradient(ky3[i,])**2 + np.gradient(kz3[i,])**2)
    for i in range(0,N_phi):
        dk1_dphi[:,i] = np.sqrt(np.gradient(kx1[:,i])**2 + np.gradient(ky1[:,i])**2 + np.gradient(kz1[:,i])**2)
        dk2_dphi[:,i] = np.sqrt(np.gradient(kx2[:,i])**2 + np.gradient(ky2[:,i])**2 + np.gradient(kz2[:,i])**2)
        dk3_dphi[:,i] = np.sqrt(np.gradient(kx3[:,i])**2 + np.gradient(ky3[:,i])**2 + np.gradient(kz3[:,i])**2)

    dS1 = dk1_dtheta * dk1_dphi
    dS2 = dk2_dtheta * dk2_dphi
    dS3 = dk3_dtheta * dk3_dphi
    """
    
    dS1 = dkr1*dkz
    dS2 = dkr2*dkz
    dS3 = dkr3*dkz
    dS4 = dkr4*dkz
    
    kx = []
    ky = []
    kz = []
    dS = []
    
    for i in range(0,len(kx1[kr1 != -.5])):
        kx.append(kx1[kr1 != -.5][i])
        ky.append(ky1[kr1 != -.5][i])
        kz.append(kz1[kr1 != -.5][i])
        dS.append(dS1[kr1 != -.5][i])
    
    for i in range(0,len(kx2[kr2 != -.5])):
        mask = (kx1[kr1 != -.5] == kx2[kr2 != -.5][i]) & (ky1[kr1 != -.5] == ky2[kr2 != -.5][i]) & (kz1[kr1 != -.5] == kz2[kr2 != -.5][i])
        if len(kx1[kr1 != -.5][mask]) == 0:
            kx.append(kx2[kr2 != -.5][i])
            ky.append(ky2[kr2 != -.5][i])
            kz.append(kz2[kr2 != -.5][i])
            dS.append(dS2[kr2 != -.5][i])
    
    for i in range(0,len(kx3[kr3 != -.5])):
        mask1 = (kx1[kr1 != -.5] == kx3[kr3 != -.5][i]) & (ky1[kr1 != -.5] == ky3[kr3 != -.5][i]) & (kz1[kr1 != -.5] == kz3[kr3 != -.5][i])
        mask2 = (kx2[kr2 != -.5] == kx3[kr3 != -.5][i]) & (ky2[kr2 != -.5] == ky3[kr3 != -.5][i]) & (kz2[kr2 != -.5] == kz3[kr3 != -.5][i])
        if len(kx1[kr1 != -.5][mask1]) == 0 and len(kx2[kr2 != -.5][mask2]) == 0:
            kx.append(kx3[kr3 != -.5][i])
            ky.append(ky3[kr3 != -.5][i])
            kz.append(kz3[kr3 != -.5][i])
            dS.append(dS3[kr3 != -.5][i])
    
    for i in range(0,len(kx4[kr4 != -.5])):
        mask1 = (kx1[kr1 != -.5] == kx4[kr4 != -.5][i]) & (ky1[kr1 != -.5] == ky4[kr4 != -.5][i]) & (kz1[kr1 != -.5] == kz4[kr4 != -.5][i])
        mask2 = (kx2[kr2 != -.5] == kx4[kr4 != -.5][i]) & (ky2[kr2 != -.5] == ky4[kr4 != -.5][i]) & (kz2[kr2 != -.5] == kz4[kr4 != -.5][i])
        mask3 = (kx3[kr3 != -.5] == kx4[kr4 != -.5][i]) & (ky3[kr3 != -.5] == ky4[kr4 != -.5][i]) & (kz3[kr3 != -.5] == kz4[kr4 != -.5][i])
        if len(kx1[kr1 != -.5][mask1]) == 0 and len(kx2[kr2 != -.5][mask2]) == 0 and len(kx3[kr3 != -.5][mask3]) == 0:
            kx.append(kx4[kr4 != -.5][i])
            ky.append(ky4[kr4 != -.5][i])
            kz.append(kz4[kr4 != -.5][i])
            dS.append(dS4[kr4 != -.5][i])
   
    kx = np.array(kx)
    ky = np.array(ky)
    kz = np.array(kz)
    dS = np.array(dS)
    
    mask = (dS < 3*np.sum(dS)/len(dS)) & (~np.isnan(dS))
    kx = kx[mask]
    ky = ky[mask]
    kz = kz[mask]
    dS = dS[mask]
    
    #vx,vy,vz = cosFit_3D_grad_wrapper(kx,ky,kz,fitParamsList)
    vx,vy,vz = cosFitTest_3D_grad(kx,ky,kz,fitParamsList,n_fit)
    
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    mask = (~np.isnan(v))
    kx = kx[mask]
    ky = ky[mask]
    kz = kz[mask]
    dS = dS[mask]
    vz = vz[mask]
    v = v[mask]
    
    #print(' ')
    #print(kz)
    #"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(kx,ky,kz,c = np.abs(v),cmap = 'inferno')
    plt.show()
    #"""
    return kx,ky,kz,dS,vz,v

def genEquipotentialFsolve(fitParams,E,dk,dkz_guess):
    n_fit = 5
    
    kx = np.arange(0.,0.5+dk,dk)
    ky = np.arange(0.,0.5+dk,dk)
    kz_guess = np.arange(0.,0.5+dkz_guess,dkz_guess)
    kz_guess[0] = kz_guess[0] + 1e-4
    kz_guess[-1] = kz_guess[-1] - 1e-4
    
    Kx, Ky = np.meshgrid(kx,ky)
    
    Kx_grid = []
    Ky_grid = []
    Kz_guess = []
    for i in range(0,len(kz_guess)):
        Kx_grid.append(Kx)
        Ky_grid.append(Ky)
        Kz_guess.append(kz_guess[i]*np.ones(Kx.shape))
    Kx_grid = np.array(Kx_grid)
    Ky_grid = np.array(Ky_grid)
    Kz_guess = np.array(Kz_guess)
    """
    kx_guess = np.arange(0.,0.5+dk_guess,dk_guess)
    ky_guess = np.arange(0.,0.5+dk_guess,dk_guess)
    kz_guess = np.arange(0.,0.5+dk_guess,dk_guess)
    
    Kx_guess,Ky_guess,Kz_guess = np.meshgrid(kx_guess,ky_guess,kz_guess)
    Kx_guess = Kx_guess.flatten()
    Ky_guess = Ky_guess.flatten()
    Kz_guess = Kz_guess.flatten()
    
    Kx = np.zeros(len(Kx_guess))
    Ky = np.zeros(len(Ky_guess))
    Kz = np.zeros(len(Kz_guess))
    """
    Kz_grid = np.zeros(Kx_grid.shape)
    for i in range(0,len(Kx_grid)):
        for j in range(0,len(Kx_grid[i])):
            #for l in range(0,len(Kx_grid[i][j])):
            for l in range(0,j+1):
                """
                K = fsolve(lambda k: [(E-cosFitTest_3D(k[0],k[1],k[2],fitParams,n_fit))**2,0,0],x0=[[Kx_guess[i],Ky_guess[i],Kz_guess[i]]])
                Kx[i] = K[0]
                Ky[i] = K[1]
                Kz[i] = K[2]
                """
                #if j == 0 and l == 0:
                Kz_grid[i][j][l] = .5/(1+np.exp(-.01*fsolve(lambda k: (E-cosFitTest_3D(Kx_grid[i][j][l],Ky_grid[i][j][l],.5/(1+np.exp(-.01*k)),fitParams,n_fit))**2,x0=[100*np.log(2*Kz_guess[i][j][l]/(1-2*Kz_guess[i][j][l]))],factor=0.1)))
                #elif j == 0:
                #    Kz_grid[i][j][l] = fsolve(lambda k: (E-cosFitTest_3D(Kx_grid[i][j][l],Ky_grid[i][j][l],k,fitParams,n_fit))**2,x0=[Kz_grid[i][j][l-1]],factor=0.1)
                #elif l == 0:
                #    Kz_grid[i][j][l] = fsolve(lambda k: (E-cosFitTest_3D(Kx_grid[i][j][l],Ky_grid[i][j][l],k,fitParams,n_fit))**2,x0=[Kz_grid[i][j-1][-1]],factor=0.1)
                
                Kz_grid[i][l][j] = Kz_grid[i][j][l]
                #print(E,i,j,l,Kx_grid[i][j][l],Ky_grid[i][j][l],Kz_grid[i][j][l],cosFitTest_3D(Kx_grid[i][j][l],Ky_grid[i][j][l],Kz_grid[i][j][l],fitParams,n_fit))
    #dS = []
    #dS_new = np.zeros(Kx_grid.shape)
    for i in range(0,len(Kx_grid[0])):
        for j in range(0,len(Kx_grid[0][i])):
            sort = np.argsort(Kz_grid[:,i,j])
            Kz_grid[:,i,j] = Kz_grid[:,i,j][sort]
            """
            for k in range(0,len(Kx_grid)):
                if i != len(Kx_grid[0])-1:
                    dkz_x_tmp = np.abs(Kz_grid[k][i+1][j] - Kz_grid[k][i][j])
                else:
                    dkz_x_tmp = np.abs(Kz_grid[k][-1][j] - Kz_grid[k][-2][j])
                if j != len(Kx_grid[0][i])-1:
                    dkz_y_tmp = np.abs(Kz_grid[k][i][j+1] - Kz_grid[k][i][j])
                else:
                    dkz_y_tmp = np.abs(Kz_grid[k][i][-1] - Kz_grid[k][i][-2])
                for l in range(0,len(Kx_grid)):
                    if i != len(Kx_grid[0])-1:
                        dkz_x_tmp_new = np.abs(Kz_grid[l][i+1][j] - Kz_grid[k][i][j])
                    else:
                        dkz_x_tmp_new = np.abs(Kz_grid[k][-1][j] - Kz_grid[l][-2][j])
                    if j != len(Kx_grid[0][i])-1:
                        dkz_y_tmp_new = np.abs(Kz_grid[l][i][j+1] - Kz_grid[k][i][j])
                    else:
                        dkz_y_tmp_new = np.abs(Kz_grid[k][i][-1] - Kz_grid[l][i][-2])

                    if dkz_x_tmp > dkz_x_tmp_new:
                        dkz_x_tmp = dkz_x_tmp_new
                    if dkz_y_tmp > dkz_y_tmp_new:
                        dkz_y_tmp = dkz_y_tmp_new
                dS_new[k][i][j] = np.sqrt(dk**4 + (dkz_x_tmp*dk)**2 + (dkz_y_tmp*dk)**2)
                #dS_new[k][i][j] = np.sqrt(dk**2 + dkz_x_tmp**2) * np.sqrt(dk**2 + dkz_y_tmp**2)
            """
    """
    for i in range(0,len(kz_guess)):
        dkz_y,dkz_x = np.gradient(Kz_grid[i])
        dS.append(np.sqrt(dk**2 + dkz_x**2)*np.sqrt(dk**2 + dkz_y**2))
    #"""
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(Kx_grid[i],Ky_grid[i],Kz_grid[i],c = dS[i],cmap = 'inferno')
        #plt.show()
        #dKz_x.append(dkz_x)
        #dKz_y.append(dkz_y)
    #dKz_x = np.array(dKz_x)
    #dKz_y = np.array(dKz_y)
    #dS = np.array(dS)
    kx = []
    ky = []
    kz = []
    #dSk = []
    for i in range(0,len(kz_guess)):
        if i == 0:
            kx.append(Kx_grid[i].flatten())
            ky.append(Ky_grid[i].flatten())
            kz.append(Kz_grid[i].flatten())
            #dSk.append(dS[i].flatten())
            #dSk.append(dS_new[i].flatten())
        else:
            mask = (np.round(Kz_grid[0],3) != np.round(Kz_grid[i],3))
            #print(np.round(Kz_grid[0],8)[~mask])
            #print(' ')
            #print(np.round(Kz_grid[1],8)[~mask])
            for j in range(1,i):
                if i != j:
                    mask = mask & (np.round(Kz_grid[j],3) != np.round(Kz_grid[i],3))
            kx.append(Kx_grid[i][mask])
            ky.append(Ky_grid[i][mask])
            kz.append(Kz_grid[i][mask])
            #dSk.append(dS[i][mask])
            #dSk.append(dS_new[i][mask])
    kx_1d = []
    ky_1d = []
    kz_1d = []
    #dS_1d = []
    for i in range(0,len(kx)):
        for j in range(0,len(kx[i])):
            kx_1d.append(kx[i][j])
            ky_1d.append(ky[i][j])
            kz_1d.append(kz[i][j])
            #dS_1d.append(dSk[i][j])
    kx = np.array(kx_1d)
    ky = np.array(ky_1d)
    kz = np.array(kz_1d)
    #dSk = np.array(dS_1d)
    
    kx_midpoint = kx + dk/2
    ky_midpoint = ky + dk/2
    kz_midpoint = np.zeros(len(kx))

    for i in range(0,len(kx)):
        kz_midpoint[i] = .5/(1+np.exp(-.01*fsolve(lambda k: (E-cosFitTest_3D(kx_midpoint[i],ky_midpoint[i],.5/(1+np.exp(-.01*k)),fitParams,n_fit))**2,x0=[100*np.log(2*kz[i]/(1-2*kz[i]))],factor=0.1)))

    vx,vy,vz = cosFitTest_3D_grad(kx,ky,kz,fitParams,n_fit)
    dSk = dk*dk*np.sqrt(1. + (vx/vz)**2 + (vy/vz)**2)
    vx,vy,vz = cosFitTest_3D_grad(kx_midpoint,ky_midpoint,kz_midpoint,fitParams,n_fit)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    #mask = (kz >= 0) & (kz <= .5) & (~np.isnan(v)) & (np.abs(v) > 1e-3) & ((E-cosFitTest_3D(kx,ky,kz,fitParams,n_fit))**2 < 1e-3) & (dSk < 20*dk*dk)
    mask = (kz_midpoint >= 0) & (kz_midpoint <= .5) & (kx_midpoint <= ky_midpoint) & (~np.isnan(v)) & (np.abs(v) > 1e-3) & ((E-cosFitTest_3D(kx_midpoint,ky_midpoint,kz_midpoint,fitParams,n_fit))**2 < 1e-3) & (dSk < 10*dk*dk)
    
    kx_midpoint_symm = np.concatenate((kx_midpoint[mask],ky_midpoint[mask]))
    ky_midpoint_symm = np.concatenate((ky_midpoint[mask],kx_midpoint[mask]))
    kz_midpoint_symm = np.concatenate((kz_midpoint[mask],kz_midpoint[mask]))
    vx_symm = np.concatenate((vx[mask],vy[mask]))
    vy_symm = np.concatenate((vy[mask],vx[mask]))
    vz_symm = np.concatenate((vz[mask],vz[mask]))
    v_symm = np.concatenate((v[mask],v[mask]))
    dSk_symm = np.concatenate((dSk[mask],dSk[mask]))
    """
    for i in range(0,len(kx[mask])):
        print(kx[mask][i],ky[mask][i],kz[mask][i],dSk[mask][i],v[mask][i])
    """
    #print(len(Kz_grid.flatten()),len(kz_midpoint[mask]),np.sum(dSk[mask]/v[mask]))
    #print(len(Kz_grid.flatten()),len(kz_midpoint_symm),np.sum(dSk_symm/v_symm))
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(kx_midpoint[mask],ky_midpoint[mask],kz_midpoint[mask],c = dSk[mask],cmap = 'inferno')
    #ax.scatter(kx_midpoint_symm,ky_midpoint_symm,kz_midpoint_symm,c = dSk_symm,cmap = 'inferno')
    #plt.show()
    #mask = (Kz_grid >= 0) & (Kz_grid <= .5)# & (~np.isnan(v))
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(Kx_grid[mask],Ky_grid[mask],Kz_grid[mask])
    #plt.show()
    
    #return kx_midpoint[mask],ky_midpoint[mask],kz_midpoint[mask],dSk[mask],vz[mask],v[mask]
    return kx_midpoint_symm,ky_midpoint_symm,kz_midpoint_symm,dSk_symm,vz_symm,v_symm
"""
kx,ky,kz,E = readBandFile('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr',20)
print('interpolating bands')
fitParamsList,bandID = genFitParams(kx,ky,kz,E,20,.6)
genEquipotentialFsolve(fitParamsList[0],0,.02,.0625)
#"""
def genEquipotentialAdaptiveCartesian(fitParams,E_cut_final,dk_init,n):
    kx = np.arange(0.,0.5+dk_init,dk_init)
    ky = np.arange(0.,0.5+dk_init,dk_init)
    kz = np.arange(0.,0.5+dk_init,dk_init)
    Kx,Ky,Kz = np.meshgrid(kx,ky,kz)
    dKx = dk_init*np.ones(Kx.shape)
    dKy = dk_init*np.ones(Ky.shape)
    dKz = dk_init*np.ones(Kz.shape)

    E_cut = 48
    n_fit = 5
    
    #for i in range(0,n_refine):
    while E_cut > E_cut_final:
        #dkx = np.arange(-.5*dk,.5*dk,dk/2)
        #dky = np.arange(-.5*dk,.5*dk,dk/2)
        #dkz = np.arange(-.5*dk,.5*dk,dk/2)
        #dKx,dKy,dKz = np.meshgrid(dkx,dky,dkz)
        #dKx = dKx.flatten()
        #dKy = dKy.flatten()
        #dKz = dKz.flatten()
        #dk = dk/2
        E = cosFitTest_3D(Kx,Ky,Kz,fitParams,n_fit)
        mask = (np.abs(E) < E_cut) & (Ky < Kx)
        #print(len(E[mask]))
        E_cut = E_cut/1.2
        kx = Kx[mask]
        ky = Ky[mask]
        kz = Kz[mask]
        vx,vy,vz = cosFitTest_3D_grad(kx,ky,kz,fitParams,n_fit)
        vx_max = max(np.abs(vx))
        vy_max = max(np.abs(vy))
        vz_max = max(np.abs(vz))
        #vx_max = 1
        #vy_max = 1
        #vz_max = 1
        E = E[mask]
        dKx_old = dKx[mask]
        dKy_old = dKy[mask]
        dKz_old = dKz[mask]
        Kx = []
        Ky = []
        Kz = []
        dKx = []
        dKy = []
        dKz = []
        if len(E) < int(200000*np.sqrt(n)):
            for j in range(0,len(E)):
                vx,vy,vz = cosFitTest_3D_grad(kx[j],ky[j],kz[j],fitParams,n_fit)
                #print(np.abs(vx_max - np.abs(vx))**4/vx_max)
                #vx = 0
                #vy = 0
                #vz = 0
                #dkx = dKx_old[j]/np.round(2*vx_max/np.abs(1.2*vx_max - np.abs(vx)),0)
                #dky = dKy_old[j]/np.round(2*vy_max/np.abs(1.2*vy_max - np.abs(vy)),0)
                #dkz = dKz_old[j]/np.round(2*vz_max/np.abs(1.2*vz_max - np.abs(vz)),0)
                #dkx = dKx_old[j]/np.round(4*np.exp(3*np.abs(vx)/vx_max)-3,0)
                #dky = dKy_old[j]/np.round(4*np.exp(3*np.abs(vy)/vy_max)-3,0)
                #dkz = dKz_old[j]/np.round(4*np.exp(3*np.abs(vz)/vz_max)-3,0)
                dkx = dKx_old[j] / np.round(1+((2+2*(E_cut_final/E_cut))/(np.exp((.25-np.abs(vx/vx_max))/.1)+1)),0)
                dky = dKy_old[j] / np.round(1+((2+2*(E_cut_final/E_cut))/(np.exp((.25-np.abs(vy/vy_max))/.1)+1)),0)
                dkz = dKz_old[j] / np.round(1+((2+2*(E_cut_final/E_cut))/(np.exp((.25-np.abs(vz/vz_max))/.1)+1)),0)
                if dkx < 10**(np.log10(E_cut/E_cut_final)-7):
                    dkx = dKx_old[j]
                if dky < 10**(np.log10(E_cut/E_cut_final)-7):
                    dky = dKy_old[j]
                if dkz < 10**(np.log10(E_cut/E_cut_final)-7):
                    dkz = dKz_old[j]
                dkx_array = np.arange(0,dKx_old[j],dkx)
                dky_array = np.arange(0,dKy_old[j],dky)
                dkz_array = np.arange(0,dKz_old[j],dkz)
                dkx_grid,dky_grid,dkz_grid = np.meshgrid(dkx_array,dky_array,dkz_array)
                dkx_grid = dkx_grid.flatten()
                dky_grid = dky_grid.flatten()
                dkz_grid = dkz_grid.flatten()
                
                for k in range(0,len(dkx_grid)):
                    Kx.append(kx[j] + dkx_grid[k])
                    Ky.append(ky[j] + dky_grid[k])
                    Kz.append(kz[j] + dkz_grid[k])
                    dKx.append(dkx)
                    dKy.append(dky)
                    dKz.append(dkz)
        else:
            Kx = kx
            Ky = ky
            Kz = kz
            dKx = dKx_old
            dKy = dKy_old
            dKz = dKz_old
        Kx = np.array(Kx)
        Ky = np.array(Ky)
        Kz = np.array(Kz)
        dKx = np.array(dKx)
        dKy = np.array(dKy)
        dKz = np.array(dKz)
        E = cosFitTest_3D(Kx,Ky,Kz,fitParams,n_fit)
        mask = (np.abs(E) < E_cut) & (Ky < Kx)
        E = E[mask]
        print(dKx[mask],min(dKx[mask]),E_cut,Kx[mask].shape,int(200000*np.sqrt(n)),n)
        """
        sort = np.argsort(E)
        dE = np.gradient(E[sort])
        print(np.sum(dE*deltaGaussHerm(0,E[sort],.01*E_cut)))
        plt.scatter(E,deltaGaussHerm(0,E,.01*E_cut))
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Kx[mask],Ky[mask],Kz[mask],c = E,cmap = 'inferno')
        plt.show()
        """
    return Kx[mask],Ky[mask],Kz[mask],dKx[mask],dKy[mask],dKz[mask],E
"""
kx,ky,kz,E = readBandFile('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr',20)
print('interpolating bands')
fitParamsList,bandID = genFitParams(kx,ky,kz,E,20,.6)
Kx,Ky,Kz,dKx,dKy,dKz,E = genEquipotentialAdaptiveCartesian(fitParamsList[3],.75,.005)
#kx = np.linspace(0,0.5,400)
#ky = np.linspace(0,0.5,400)
#kz = np.linspace(0,0.5,400)
#Kx,Ky,Kz = np.meshgrid(kx,ky,kz)
#E = cosFitTest_3D(Kx,Ky,Kz,fitParamsList[4],5)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Kx,Ky,Kz,c = E)
plt.show()
#"""
def genEquipotential(fitParamsList,E,N_theta,N_phi):
    theta = np.linspace(0.,np.pi/2.,N_theta)
    phi = np.linspace(0.,np.pi/2.,N_phi)
    Theta,Phi = np.meshgrid(theta,phi)
    
    n_fit = 5
    
    kr1 = np.zeros(Theta.shape)
    kx1 = np.zeros(Theta.shape)
    ky1 = np.zeros(Theta.shape)
    kz1 = np.zeros(Theta.shape)
    
    kr2 = np.zeros(Theta.shape)
    kx2 = np.zeros(Theta.shape)
    ky2 = np.zeros(Theta.shape)
    kz2 = np.zeros(Theta.shape)
    
    kr3 = np.zeros(Theta.shape)
    kx3 = np.zeros(Theta.shape)
    ky3 = np.zeros(Theta.shape)
    kz3 = np.zeros(Theta.shape)
    
    kr4 = np.zeros(Theta.shape)
    kx4 = np.zeros(Theta.shape)
    ky4 = np.zeros(Theta.shape)
    kz4 = np.zeros(Theta.shape)
    
    kr5 = np.zeros(Theta.shape)
    kx5 = np.zeros(Theta.shape)
    ky5 = np.zeros(Theta.shape)
    kz5 = np.zeros(Theta.shape)
    
    kr6 = np.zeros(Theta.shape)
    kx6 = np.zeros(Theta.shape)
    ky6 = np.zeros(Theta.shape)
    kz6 = np.zeros(Theta.shape)
    
    kr7 = np.zeros(Theta.shape)
    kx7 = np.zeros(Theta.shape)
    ky7 = np.zeros(Theta.shape)
    kz7 = np.zeros(Theta.shape)
    
    kr8 = np.zeros(Theta.shape)
    kx8 = np.zeros(Theta.shape)
    ky8 = np.zeros(Theta.shape)
    kz8 = np.zeros(Theta.shape)

    kr9 = np.zeros(Theta.shape)
    kx9 = np.zeros(Theta.shape)
    ky9 = np.zeros(Theta.shape)
    kz9 = np.zeros(Theta.shape)

    kr10 = np.zeros(Theta.shape)
    kx10 = np.zeros(Theta.shape)
    ky10 = np.zeros(Theta.shape)
    kz10 = np.zeros(Theta.shape)

    kr11 = np.zeros(Theta.shape)
    kx11 = np.zeros(Theta.shape)
    ky11 = np.zeros(Theta.shape)
    kz11 = np.zeros(Theta.shape)

    kr12 = np.zeros(Theta.shape)
    kx12 = np.zeros(Theta.shape)
    ky12 = np.zeros(Theta.shape)
    kz12 = np.zeros(Theta.shape)
    for i in range(0,N_phi):
        for j in range(0,N_theta):
            
            kr1[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.12))
            kx1[i][j] = np.abs(kr1[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky1[i][j] = np.abs(kr1[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz1[i][j] = np.abs(kr1[i][j])*np.cos(Phi[i][j])

            #print(Theta[i][j],Phi[i][j],kr1[i][j],kx1[i][j],ky1[i][j],kz1[i][j])
            
            kr2[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.12))
            kx2[i][j] = np.abs(kr2[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky2[i][j] = np.abs(kr2[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz2[i][j] = .5 - np.abs(kr2[i][j])*np.cos(Phi[i][j])

            kr3[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.12))
            kx3[i][j] = .5 - np.abs(kr3[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky3[i][j] = .5 - np.abs(kr3[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz3[i][j] = np.abs(kr3[i][j])*np.cos(Phi[i][j])
            
            kr4[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.12))
            kx4[i][j] = .5 - np.abs(kr4[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky4[i][j] = .5 - np.abs(kr4[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz4[i][j] = .5 - np.abs(kr4[i][j])*np.cos(Phi[i][j])
            
            kr5[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.12))
            kx5[i][j] = .5 - np.abs(kr5[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky5[i][j] = np.abs(kr5[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz5[i][j] = np.abs(kr5[i][j])*np.cos(Phi[i][j])
            
            kr6[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.12))
            kx6[i][j] = .5 - np.abs(kr6[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky6[i][j] = np.abs(kr6[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz6[i][j] = .5 - np.abs(kr6[i][j])*np.cos(Phi[i][j])
            #"""
            kr7[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.165+np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.09))
            kx7[i][j] = np.abs(kr7[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky7[i][j] = np.abs(kr7[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz7[i][j] = .165 + np.abs(kr7[i][j])*np.cos(Phi[i][j])
            #"""
            kr8[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.165+np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.09))
            kx8[i][j] = .5 - np.abs(kr8[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky8[i][j] = .5 - np.abs(kr8[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz8[i][j] = .165 + np.abs(kr8[i][j])*np.cos(Phi[i][j])
            #"""
            kr9[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.165+np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.09))
            kx9[i][j] = .5 - np.abs(kr9[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky9[i][j] = np.abs(kr9[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz9[i][j] = .165 + np.abs(kr9[i][j])*np.cos(Phi[i][j])
            
            kr10[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.33+np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.07))
            kx10[i][j] = np.abs(kr10[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky10[i][j] = np.abs(kr10[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz10[i][j] = .33 + np.abs(kr10[i][j])*np.cos(Phi[i][j])
            #"""
            kr11[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.33+np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.07))
            kx11[i][j] = .5 - np.abs(kr11[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky11[i][j] = .5 - np.abs(kr11[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz11[i][j] = .33 + np.abs(kr11[i][j])*np.cos(Phi[i][j])
            #"""
            kr12[i][j] = np.abs(fsolve(lambda k: (cosFitTest_3D(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.33+np.abs(k)*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2,.07))
            kx12[i][j] = .5 - np.abs(kr12[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky12[i][j] = np.abs(kr12[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz12[i][j] = .33 + np.abs(kr12[i][j])*np.cos(Phi[i][j])
            #"""
            if ((cosFitTest_3D(kr1[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr1[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),kr1[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx1[i][j] < 0.) or (kx1[i][j] > .5) or (ky1[i][j] < 0.) or (ky1[i][j] > .5)  or (kz1[i][j] < 0.) or (kz1[i][j] > .5):
                kr1[i][j] = -.5
            if ((cosFitTest_3D(kr2[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr2[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-kr2[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx2[i][j] < 0.) or (kx2[i][j] > .5) or (ky2[i][j] < 0.) or (ky2[i][j] > .5) or (kz2[i][j] < 0.) or (kz2[i][j] > .5):
                kr2[i][j] = -.5
            if ((cosFitTest_3D(.5-kr3[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-kr3[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),kr3[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx3[i][j] < 0.) or (kx3[i][j] > .5) or (ky3[i][j] < 0.) or (ky3[i][j] > .5) or (kz3[i][j] < 0.) or (kz3[i][j] > .5):
                kr3[i][j] = -.5
            if ((cosFitTest_3D(.5-kr4[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-kr4[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-kr4[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx4[i][j] < 0.) or (kx4[i][j] > .5) or (ky4[i][j] < 0.) or (ky4[i][j] > .5) or (kz4[i][j] < 0.) or (kz4[i][j] > .5):
                kr4[i][j] = -.5
            if ((cosFitTest_3D(.5-kr5[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr5[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),kr5[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx5[i][j] < 0.) or (kx5[i][j] > .5) or (ky5[i][j] < 0.) or (ky5[i][j] > .5) or (kz5[i][j] < 0.) or (kz5[i][j] > .5):
                kr5[i][j] = -.5
            if ((cosFitTest_3D(.5-kr6[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr6[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-kr6[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx6[i][j] < 0.) or (kx6[i][j] > .5) or (ky6[i][j] < 0.) or (ky6[i][j] > .5) or (kz6[i][j] < 0.) or (kz6[i][j] > .5):
                kr6[i][j] = -.5
            #"""
            if ((cosFitTest_3D(kr7[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr7[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.165+kr7[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx7[i][j] < 0.) or (kx7[i][j] > .5) or (ky7[i][j] < 0.) or (ky7[i][j] > .5) or (kz7[i][j] < 0.) or (kz7[i][j] > .5):
                kr7[i][j] = -.5
            #"""
            if ((cosFitTest_3D(.5-kr8[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-kr8[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.165+kr8[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx8[i][j] < 0.) or (kx8[i][j] > .5) or (ky8[i][j] < 0.) or (ky8[i][j] > .5) or (kz8[i][j] < 0.) or (kz8[i][j] > .5):
                kr8[i][j] = -.5
            #"""
            if ((cosFitTest_3D(.5-kr9[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr9[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.165+kr9[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx9[i][j] < 0.) or (kx9[i][j] > .5) or (ky9[i][j] < 0.) or (ky9[i][j] > .5) or (kz9[i][j] < 0.) or (kz9[i][j] > .5):
                kr9[i][j] = -.5
            if ((cosFitTest_3D(kr10[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr10[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.33+kr10[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx10[i][j] < 0.) or (kx10[i][j] > .5) or (ky10[i][j] < 0.) or (ky10[i][j] > .5) or (kz10[i][j] < 0.) or (kz10[i][j] > .5):
                kr10[i][j] = -.5
            #"""
            if ((cosFitTest_3D(.5-kr11[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-kr11[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.33+kr11[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx11[i][j] < 0.) or (kx11[i][j] > .5) or (ky11[i][j] < 0.) or (ky11[i][j] > .5) or (kz11[i][j] < 0.) or (kz11[i][j] > .5):
                kr11[i][j] = -.5
            #"""
            if ((cosFitTest_3D(.5-kr12[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr12[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.33+kr12[i][j]*np.cos(Phi[i][j]),fitParamsList,n_fit) - E)**2 > 1e-5) or (kx12[i][j] < 0.) or (kx12[i][j] > .5) or (ky12[i][j] < 0.) or (ky12[i][j] > .5) or (kz12[i][j] < 0.) or (kz12[i][j] > .5):
                kr12[i][j] = -.5
            #"""
    
    dk1_dtheta = np.zeros(kr1.shape)
    dk1_dphi = np.zeros(kr1.shape)
    dk2_dtheta = np.zeros(kr2.shape)
    dk2_dphi = np.zeros(kr2.shape)
    dk3_dtheta = np.zeros(kr3.shape)
    dk3_dphi = np.zeros(kr3.shape)
    dk4_dtheta = np.zeros(kr4.shape)
    dk4_dphi = np.zeros(kr4.shape)
    dk5_dtheta = np.zeros(kr5.shape)
    dk5_dphi = np.zeros(kr5.shape)
    dk6_dtheta = np.zeros(kr6.shape)
    dk6_dphi = np.zeros(kr6.shape)
    dk7_dtheta = np.zeros(kr7.shape)
    dk7_dphi = np.zeros(kr7.shape)
    dk8_dtheta = np.zeros(kr8.shape)
    dk8_dphi = np.zeros(kr8.shape)
    dk9_dtheta = np.zeros(kr9.shape)
    dk9_dphi = np.zeros(kr9.shape)
    dk10_dtheta = np.zeros(kr10.shape)
    dk10_dphi = np.zeros(kr10.shape)
    dk11_dtheta = np.zeros(kr11.shape)
    dk11_dphi = np.zeros(kr11.shape)
    dk12_dtheta = np.zeros(kr12.shape)
    dk12_dphi = np.zeros(kr12.shape)
    for i in range(0,N_theta):
        dk1_dtheta[i,] = np.sqrt(np.gradient(kx1[i,])**2 + np.gradient(ky1[i,])**2 + np.gradient(kz1[i,])**2)
        dk2_dtheta[i,] = np.sqrt(np.gradient(kx2[i,])**2 + np.gradient(ky2[i,])**2 + np.gradient(kz2[i,])**2)
        dk3_dtheta[i,] = np.sqrt(np.gradient(kx3[i,])**2 + np.gradient(ky3[i,])**2 + np.gradient(kz3[i,])**2)
        dk4_dtheta[i,] = np.sqrt(np.gradient(kx4[i,])**2 + np.gradient(ky4[i,])**2 + np.gradient(kz4[i,])**2)
        dk5_dtheta[i,] = np.sqrt(np.gradient(kx5[i,])**2 + np.gradient(ky5[i,])**2 + np.gradient(kz5[i,])**2)
        dk6_dtheta[i,] = np.sqrt(np.gradient(kx6[i,])**2 + np.gradient(ky6[i,])**2 + np.gradient(kz6[i,])**2)
        dk7_dtheta[i,] = np.sqrt(np.gradient(kx7[i,])**2 + np.gradient(ky7[i,])**2 + np.gradient(kz7[i,])**2)
        dk8_dtheta[i,] = np.sqrt(np.gradient(kx8[i,])**2 + np.gradient(ky8[i,])**2 + np.gradient(kz8[i,])**2)
        dk9_dtheta[i,] = np.sqrt(np.gradient(kx9[i,])**2 + np.gradient(ky9[i,])**2 + np.gradient(kz9[i,])**2)
        dk10_dtheta[i,] = np.sqrt(np.gradient(kx10[i,])**2 + np.gradient(ky10[i,])**2 + np.gradient(kz10[i,])**2)
        dk11_dtheta[i,] = np.sqrt(np.gradient(kx11[i,])**2 + np.gradient(ky11[i,])**2 + np.gradient(kz11[i,])**2)
        dk12_dtheta[i,] = np.sqrt(np.gradient(kx12[i,])**2 + np.gradient(ky12[i,])**2 + np.gradient(kz12[i,])**2)
    for i in range(0,N_phi):
        dk1_dphi[:,i] = np.sqrt(np.gradient(kx1[:,i])**2 + np.gradient(ky1[:,i])**2 + np.gradient(kz1[:,i])**2)
        dk2_dphi[:,i] = np.sqrt(np.gradient(kx2[:,i])**2 + np.gradient(ky2[:,i])**2 + np.gradient(kz2[:,i])**2)
        dk3_dphi[:,i] = np.sqrt(np.gradient(kx3[:,i])**2 + np.gradient(ky3[:,i])**2 + np.gradient(kz3[:,i])**2)
        dk4_dphi[:,i] = np.sqrt(np.gradient(kx4[:,i])**2 + np.gradient(ky4[:,i])**2 + np.gradient(kz4[:,i])**2)
        dk5_dphi[:,i] = np.sqrt(np.gradient(kx5[:,i])**2 + np.gradient(ky5[:,i])**2 + np.gradient(kz5[:,i])**2)
        dk6_dphi[:,i] = np.sqrt(np.gradient(kx6[:,i])**2 + np.gradient(ky6[:,i])**2 + np.gradient(kz6[:,i])**2)
        dk7_dphi[:,i] = np.sqrt(np.gradient(kx7[:,i])**2 + np.gradient(ky7[:,i])**2 + np.gradient(kz7[:,i])**2)
        dk8_dphi[:,i] = np.sqrt(np.gradient(kx8[:,i])**2 + np.gradient(ky8[:,i])**2 + np.gradient(kz8[:,i])**2)
        dk9_dphi[:,i] = np.sqrt(np.gradient(kx9[:,i])**2 + np.gradient(ky9[:,i])**2 + np.gradient(kz9[:,i])**2)
        dk10_dphi[:,i] = np.sqrt(np.gradient(kx10[:,i])**2 + np.gradient(ky10[:,i])**2 + np.gradient(kz10[:,i])**2)
        dk11_dphi[:,i] = np.sqrt(np.gradient(kx11[:,i])**2 + np.gradient(ky11[:,i])**2 + np.gradient(kz11[:,i])**2)
        dk12_dphi[:,i] = np.sqrt(np.gradient(kx12[:,i])**2 + np.gradient(ky12[:,i])**2 + np.gradient(kz12[:,i])**2)
    
    dS1 = dk1_dtheta * dk1_dphi
    dS2 = dk2_dtheta * dk2_dphi
    dS3 = dk3_dtheta * dk3_dphi
    dS4 = dk4_dtheta * dk4_dphi
    dS5 = dk5_dtheta * dk5_dphi
    dS6 = dk6_dtheta * dk6_dphi
    dS7 = dk7_dtheta * dk7_dphi
    dS8 = dk8_dtheta * dk8_dphi
    dS9 = dk9_dtheta * dk9_dphi
    dS10 = dk10_dtheta * dk10_dphi
    dS11 = dk11_dtheta * dk11_dphi
    dS12 = dk12_dtheta * dk12_dphi
    
    kx = []
    ky = []
    kz = []
    dS = []

    for i in range(0,len(kx1[kr1 != -.5])):
        kx.append(kx1[kr1 != -.5][i])
        ky.append(ky1[kr1 != -.5][i])
        kz.append(kz1[kr1 != -.5][i])
        dS.append(dS1[kr1 != -.5][i])
    for i in range(0,len(kx2[kr2 != -.5])):
        mask = (kx1[kr1 != -.5] == kx2[kr2 != -.5][i]) & (ky1[kr1 != -.5] == ky2[kr2 != -.5][i]) & (kz1[kr1 != -.5] == kz2[kr2 != -.5][i])
        if len(kx1[kr1 != -.5][mask]) == 0:
            kx.append(kx2[kr2 != -.5][i])
            ky.append(ky2[kr2 != -.5][i])
            kz.append(kz2[kr2 != -.5][i])
            dS.append(dS2[kr2 != -.5][i])
    
    for i in range(0,len(kx3[kr3 != -.5])):
        kx.append(kx3[kr3 != -.5][i])
        ky.append(ky3[kr3 != -.5][i])
        kz.append(kz3[kr3 != -.5][i])
        dS.append(dS3[kr3 != -.5][i])
    for i in range(0,len(kx4[kr4 != -.5])):
        mask3 = (kx3[kr3 != -.5] == kx4[kr4 != -.5][i]) & (ky3[kr3 != -.5] == ky4[kr4 != -.5][i]) & (kz3[kr3 != -.5] == kz4[kr4 != -.5][i])
        if len(kx3[kr3 != -.5][mask3]) == 0:
            kx.append(kx4[kr4 != -.5][i])
            ky.append(ky4[kr4 != -.5][i])
            kz.append(kz4[kr4 != -.5][i])
            dS.append(dS4[kr4 != -.5][i])
    for i in range(0,len(kx5[kr5 != -.5])):
        kx.append(kx5[kr5 != -.5][i])
        ky.append(ky5[kr5 != -.5][i])
        kz.append(kz5[kr5 != -.5][i])
        dS.append(dS5[kr5 != -.5][i])

        kx.append(ky5[kr5 != -.5][i])
        ky.append(kx5[kr5 != -.5][i])
        kz.append(kz5[kr5 != -.5][i])
        dS.append(dS5[kr5 != -.5][i])
    for i in range(0,len(kx6[kr6 != -.5])):
        mask5 = (kx5[kr5 != -.5] == kx6[kr6 != -.5][i]) & (ky5[kr5 != -.5] == ky6[kr6 != -.5][i]) & (kz5[kr5 != -.5] == kz6[kr6 != -.5][i])
        if len(kx5[kr5 != -.5][mask5]) == 0:
            kx.append(kx6[kr6 != -.5][i])
            ky.append(ky6[kr6 != -.5][i])
            kz.append(kz6[kr6 != -.5][i])
            dS.append(dS6[kr6 != -.5][i])
            
            kx.append(ky6[kr6 != -.5][i])
            ky.append(kx6[kr6 != -.5][i])
            kz.append(kz6[kr6 != -.5][i])
            dS.append(dS6[kr6 != -.5][i])
    #"""
    for i in range(0,len(kx7[kr7 != -.5])):
        mask1 = (kx1[kr1 != -.5] == kx7[kr7 != -.5][i]) & (ky1[kr1 != -.5] == ky7[kr7 != -.5][i]) & (kz1[kr1 != -.5] == kz7[kr7 != -.5][i])
        mask2 = (kx2[kr2 != -.5] == kx7[kr7 != -.5][i]) & (ky2[kr2 != -.5] == ky7[kr7 != -.5][i]) & (kz2[kr2 != -.5] == kz7[kr7 != -.5][i])
        if len(kx1[kr1 != -.5][mask1]) == 0 and len(kx2[kr2 != -.5][mask2]) == 0:
            kx.append(kx7[kr7 != -.5][i])
            ky.append(ky7[kr7 != -.5][i])
            kz.append(kz7[kr7 != -.5][i])
            dS.append(dS7[kr7 != -.5][i])
    #"""
    for i in range(0,len(kx8[kr8 != -.5])):
        mask3 = (kx3[kr3 != -.5] == kx8[kr8 != -.5][i]) & (ky3[kr3 != -.5] == ky8[kr8 != -.5][i]) & (kz3[kr3 != -.5] == kz8[kr8 != -.5][i])
        mask4 = (kx4[kr4 != -.5] == kx8[kr8 != -.5][i]) & (ky4[kr4 != -.5] == ky8[kr8 != -.5][i]) & (kz4[kr4 != -.5] == kz8[kr8 != -.5][i])
        if len(kx3[kr3 != -.5][mask3]) == 0 and len(kx4[kr4 != -.5][mask4]) == 0:
            kx.append(kx8[kr8 != -.5][i])
            ky.append(ky8[kr8 != -.5][i])
            kz.append(kz8[kr8 != -.5][i])
            dS.append(dS8[kr8 != -.5][i])
    #"""
    for i in range(0,len(kx9[kr9 != -.5])):
        mask5 = (kx5[kr5 != -.5] == kx9[kr9 != -.5][i]) & (ky5[kr5 != -.5] == ky9[kr9 != -.5][i]) & (kz5[kr5 != -.5] == kz9[kr9 != -.5][i])
        mask6 = (kx6[kr6 != -.5] == kx9[kr9 != -.5][i]) & (ky6[kr6 != -.5] == ky9[kr9 != -.5][i]) & (kz6[kr6 != -.5] == kz9[kr9 != -.5][i])
        if len(kx5[kr5 != -.5][mask5]) == 0 and len(kx6[kr6 != -.5][mask6]) == 0:
            kx.append(kx9[kr9 != -.5][i])
            ky.append(ky9[kr9 != -.5][i])
            kz.append(kz9[kr9 != -.5][i])
            dS.append(dS9[kr9 != -.5][i])
            
            kx.append(ky9[kr9 != -.5][i])
            ky.append(kx9[kr9 != -.5][i])
            kz.append(kz9[kr9 != -.5][i])
            dS.append(dS9[kr9 != -.5][i])
    for i in range(0,len(kx10[kr10 != -.5])):
        mask1 = (kx1[kr1 != -.5] == kx10[kr10 != -.5][i]) & (ky1[kr1 != -.5] == ky10[kr10 != -.5][i]) & (kz1[kr1 != -.5] == kz10[kr10 != -.5][i])
        mask2 = (kx2[kr2 != -.5] == kx10[kr10 != -.5][i]) & (ky2[kr2 != -.5] == ky10[kr10 != -.5][i]) & (kz2[kr2 != -.5] == kz10[kr10 != -.5][i])
        mask7 = (kx7[kr7 != -.5] == kx10[kr10 != -.5][i]) & (ky7[kr7 != -.5] == ky10[kr10 != -.5][i]) & (kz7[kr7 != -.5] == kz10[kr10 != -.5][i])
        if len(kx1[kr1 != -.5][mask1]) == 0 and len(kx2[kr2 != -.5][mask2]) == 0 and len(kx7[kr7 != -.5][mask7]) == 0:
            kx.append(kx10[kr10 != -.5][i])
            ky.append(ky10[kr10 != -.5][i])
            kz.append(kz10[kr10 != -.5][i])
            dS.append(dS10[kr10 != -.5][i])
    #"""
    for i in range(0,len(kx11[kr11 != -.5])):
        mask3 = (kx3[kr3 != -.5] == kx11[kr11 != -.5][i]) & (ky3[kr3 != -.5] == ky11[kr11 != -.5][i]) & (kz3[kr3 != -.5] == kz11[kr11 != -.5][i])
        mask4 = (kx4[kr4 != -.5] == kx11[kr11 != -.5][i]) & (ky4[kr4 != -.5] == ky11[kr11 != -.5][i]) & (kz4[kr4 != -.5] == kz11[kr11 != -.5][i])
        mask8 = (kx8[kr8 != -.5] == kx11[kr11 != -.5][i]) & (ky8[kr8 != -.5] == ky11[kr11 != -.5][i]) & (kz8[kr8 != -.5] == kz11[kr11 != -.5][i])
        if len(kx3[kr3 != -.5][mask3]) == 0 and len(kx4[kr4 != -.5][mask4]) == 0 and len(kx8[kr8 != -.5][mask8]) == 0:
            kx.append(kx11[kr11 != -.5][i])
            ky.append(ky11[kr11 != -.5][i])
            kz.append(kz11[kr11 != -.5][i])
            dS.append(dS11[kr11 != -.5][i])
    #"""
    for i in range(0,len(kx12[kr12 != -.5])):
        mask5 = (kx5[kr5 != -.5] == kx12[kr12 != -.5][i]) & (ky5[kr5 != -.5] == ky12[kr12 != -.5][i]) & (kz5[kr5 != -.5] == kz12[kr12 != -.5][i])
        mask6 = (kx6[kr6 != -.5] == kx12[kr12 != -.5][i]) & (ky6[kr6 != -.5] == ky12[kr12 != -.5][i]) & (kz6[kr6 != -.5] == kz12[kr12 != -.5][i])
        mask9 = (kx9[kr9 != -.5] == kx12[kr12 != -.5][i]) & (ky9[kr9 != -.5] == ky12[kr12 != -.5][i]) & (kz9[kr9 != -.5] == kz12[kr12 != -.5][i])
        if len(kx5[kr5 != -.5][mask5]) == 0 and len(kx6[kr6 != -.5][mask6]) == 0 and len(kx9[kr9 != -.5][mask9]) == 0:
            kx.append(kx12[kr12 != -.5][i])
            ky.append(ky12[kr12 != -.5][i])
            kz.append(kz12[kr12 != -.5][i])
            dS.append(dS12[kr12 != -.5][i])

            kx.append(ky12[kr12 != -.5][i])
            ky.append(kx12[kr12 != -.5][i])
            kz.append(kz12[kr12 != -.5][i])
            dS.append(dS12[kr12 != -.5][i])
    #"""
    kx = np.array(kx)
    ky = np.array(ky)
    kz = np.array(kz)
    dS = np.array(dS)
    
    mask = (dS < 3*np.sum(dS)/len(dS)) & (~np.isnan(dS))
    kx = kx[mask]
    ky = ky[mask]
    kz = kz[mask]
    dS = dS[mask]
    
    #vx,vy,vz = cosFit_3D_grad_wrapper(kx,ky,kz,fitParamsList)
    vx,vy,vz = cosFitTest_3D_grad(kx,ky,kz,fitParamsList,n_fit)
    
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    mask = (~np.isnan(v))
    kx = kx[mask]
    ky = ky[mask]
    kz = kz[mask]
    dS = dS[mask]
    vz = vz[mask]
    v = v[mask]
    
    #print(' ')
    #print(kz)
    #"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(kx,ky,kz,c = np.abs(v),cmap = 'inferno')
    #ax.scatter(kx,-ky,kz,c = np.abs(v),cmap = 'inferno')
    #ax.scatter(-kx,ky,kz,c = np.abs(v),cmap = 'inferno')
    #ax.scatter(-kx,-ky,kz,c = np.abs(v),cmap = 'inferno')
    plt.show()
    #"""
    return kx,ky,kz,dS,vz,v

"""
kx,ky,kz,E = readBandFile('abinitOutputFiles/Co_FM_DOS/bct_a_277_c_277o_EBANDS.agr',20)
Kx = kx[(ky == 0.) & (kz == 0.)]
E_k = E[35][(ky == 0.) & (kz == 0.)]
popt,pcurve = curve_fit(cosFitTestWrapper,Kx,E_k)
x = np.linspace(-.5,.5,100)
plt.plot(x,cosFitTest(x,popt,7),c = 'blue')
plt.scatter(Kx,E_k,c = 'red')
plt.scatter(-Kx,E_k,c = 'red')
plt.show()

Kx = kx[(kz == 0)]
Ky = ky[(kz == 0)]
E_k = E[35][(kz == 0)]
popt,pcurve = curve_fit(cosFitTestWrapper_2D,(Kx,Ky),E_k)

x = np.linspace(-.5,.5,100)
y = np.linspace(-.5,.5,100)
X,Y = np.meshgrid(x,y)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X,Y,cosFitTest_2D(X,Y,popt,7),color = 'blue',cstride = 5,rstride = 5)
ax.scatter(Kx,Ky,E_k,c = 'red')
plt.show()
"""
"""
kx,ky,kz,E = readBandFile('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr',20)
print('interpolating bands')
fitParamsList,bandID = genFitParams(kx,ky,kz,E,20,.6)
#kx = np.linspace(0,0.5,400)
#ky = np.linspace(0,0.5,400)
#kz = np.linspace(0,0.5,400)
#Kx,Ky,Kz = np.meshgrid(kx,ky,kz)
#E = cosFitTest_3D(Kx,Ky,Kz,fitParamsList[4],5)
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(Kx[np.abs(E) < .02],Ky[np.abs(E) < .02],Kz[np.abs(E) < .02])
#plt.show()
genEquipotential(fitParamsList[4],0,21,21)
#"""

def genEquipotentialAdaptiveCartesianGrid(fitParamsList,E_cut,n):
    n_fit = 5
    kx = []
    ky = []
    kz = []
    dkx = []
    dky = []
    dkz = []
    E = []
    vz = []
    
    for i in range(0,len(fitParamsList)):
        kx_tmp,ky_tmp,kz_tmp,dkx_tmp,dky_tmp,dkz_tmp,E_tmp = genEquipotentialAdaptiveCartesian(fitParamsList[i],E_cut,.0025,n[i])
        kx.append(kx_tmp)
        ky.append(ky_tmp)
        kz.append(kz_tmp)
        dkx.append(dkx_tmp)
        dky.append(dky_tmp)
        dkz.append(dkz_tmp)
        E.append(E_tmp)
        vx_tmp,vy_tmp,vz_tmp = cosFitTest_3D_grad(kx[i],ky[i],kz[i],fitParamsList[i],n_fit)
        vz.append(vz_tmp)
        print(i,len(kx_tmp))
    return kx,ky,kz,dkx,dky,dkz,E,vz

def genEquipotentialGrid(fitParamsList,E_cutoff,N_E,N_theta,N_phi):
    E_grid = np.linspace(-E_cutoff,E_cutoff,N_E)

    kx = []
    ky = []
    kz = []
    dS = []
    vz = []
    v = []
    
    for i in range(0,len(fitParamsList)):
        kx.append([])
        ky.append([])
        kz.append([])
        dS.append([])
        vz.append([])
        v.append([])
        for j in range(0,N_E):
            kx[i].append([])
            ky[i].append([])
            kz[i].append([])
            dS[i].append([])
            vz[i].append([])
            v[i].append([])
            kx_tmp,ky_tmp,kz_tmp,dS_tmp,vz_tmp,v_tmp = genEquipotentialFsolve(fitParamsList[i],E_grid[j],.02,.1)
            #kx_tmp,ky_tmp,kz_tmp,dS_tmp,vz_tmp,v_tmp = genEquipotential(fitParamsList[i],E_grid[j],N_theta,N_phi)
            print(i,E_grid[j],len(kx_tmp))
            kx[i][j] = kx_tmp
            ky[i][j] = ky_tmp
            kz[i][j] = kz_tmp
            dS[i][j] = dS_tmp
            vz[i][j] = vz_tmp
            v[i][j] = v_tmp
            
    return E_grid,dS,vz,v

def genDOSAdaptiveCartesian(bandID,nbands,dkx,dky,dkz,E_k,E_cut,N_E):
    E = np.linspace(-E_cut,E_cut,N_E)
    N_up = np.zeros(E.shape)
    N_down = np.zeros(E.shape)
    for n in range(0,N_E):
        for i in range(0,len(bandID)):
            NF = np.sum(dkx[i]*dky[i]*dkz[i] * deltaGaussHerm(E_k[i],E[n],.01*E_cut))
            if bandID[i] < nbands:
                N_up[n] = N_up[n] + NF
            else:
                N_down[n] = N_down[n] + NF
        print(E[n],N_up[n],N_down[n])

    return E,N_up,N_down

def genDOS(bandID,nbands,dS_k,v_k,E_k,E_cut,N_E):
    dE = np.abs(E_k[1] - E_k[0])
    E = np.linspace(-E_cut,E_cut,N_E)
    N_up = np.zeros(E.shape)
    N_down = np.zeros(E.shape)
    for n in range(0,N_E):
        for i in range(0,len(bandID)):
            NF = 0.
            for j in range(0,len(E_k)):
                NF = NF + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*deltaGaussHerm(E_k[j],E[n],.025*E_k[-1])/v_k[i][j][v_k[i][j] != 0.])
            if bandID[i] < nbands:
                N_up[n] = N_up[n] + NF
            else:
                N_down[n] = N_down[n] + NF
        print(E[n],N_up[n],N_down[n])

    return E,N_up,N_down

def genBands(a,c,N):
    
    if a == c:
        # bcc
        d0 = np.sqrt(.5**2)
        N0_array = np.linspace(0,1,N)
        N_list = [0]
        bandpath_x = []
        bandpath_y = []
        bandpath_z = []
        labels = [[0],[r'$\Gamma$']]
        for i in range(0,N-1):
            # Gamma - H 
            bandpath_x.append(0.)
            bandpath_y.append(.5*N0_array[i])
            bandpath_z.append(0.)
            if i > 0:
                N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$H$')
        d = np.sqrt(.25**2 + .25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # H - N
            bandpath_x.append(.25*N_array[i])
            bandpath_y.append(.5-.25*N_array[i])
            bandpath_z.append(0.)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$N$')
        for i in range(0,len(N_array)-1):
            # N - Gamma
            bandpath_x.append(.25-.25*N_array[i])
            bandpath_y.append(.25-.25*N_array[i])
            bandpath_z.append(0.)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$\Gamma$')
        d = np.sqrt(3*(.25**2))
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # Gamma - P
            bandpath_x.append(.25*N_array[i])
            bandpath_y.append(.25*N_array[i])
            bandpath_z.append(.25*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$P$')
        for i in range(0,len(N_array)-1):
            # P - H
            bandpath_x.append(.25 - .25*N_array[i])
            bandpath_y.append(.25 + .25*N_array[i])
            bandpath_z.append(.25 - .25*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$H$')
        
        for i in range(0,int(N/10)):
            #Adding in a gap for my band structure plot before the 
            #P - N path of bcc
            bandpath_x.append(-1)
            bandpath_y.append(-1)
            bandpath_z.append(-1)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1]+1)
        labels[1].append(r'$P$')
        d = np.sqrt(.25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # P - N
            bandpath_x.append(.25)
            bandpath_y.append(.25)
            bandpath_z.append(.25 - .25*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$N$')

    elif c < a:
        #bct1
        #lattice vectors were normalized by 4*pi/a in x/y-components
        #and 4*pi/c in z-components
        b1 = np.array([0,.5,.5])
        b2 = np.array([.5,0,.5])
        b3 = np.array([.5,.5,0])
        
        eta = .25*(1+(c/a)**2)
        
        d0 = np.sqrt(.25**2 + .25**2)
        
        N_plot = []
        N0_array = np.linspace(0,1,N)
        N_list = [0]
        bandpath_x = []
        bandpath_y = []
        bandpath_z = []
        labels = [[0],[r'$\Gamma$']]
        for i in range(0,N-1):
            # Gamma - X
            bandpath_x.append(.25*N0_array[i])
            bandpath_y.append(.25*N0_array[i])
            bandpath_z.append(0.)
            if i > 0:
                N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$X$')
        for i in range(0,N-1):
            #X - M
            bandpath_x.append(.25 + .25*N0_array[i])
            bandpath_y.append(.25 - .25*N0_array[i])
            bandpath_z.append(0.)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$M$')
        d = np.sqrt(.5**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            #M - Gamma
            bandpath_x.append(.5 - .5*N_array[i])
            bandpath_y.append(0.)
            bandpath_z.append(0.)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$\Gamma$')
        d = np.sqrt(eta*eta)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            #Gamma - Z
            bandpath_x.append(0)
            bandpath_y.append(0)
            bandpath_z.append(eta*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$Z$')
        d = np.sqrt(2*(.25)**2 + (eta-.25)**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            #Z - P
            bandpath_x.append(.25*N_array[i])
            bandpath_y.append(.25*N_array[i])
            bandpath_z.append(eta + (.25-eta)*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$P$')
        d = np.sqrt(.25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            #P - N
            bandpath_x.append(.25)
            bandpath_y.append(.25 - .25*N_array[i])
            bandpath_z.append(.25)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$N$')
        d = np.sqrt(.25**2 + (.25-eta)**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            #N - Z1
            bandpath_x.append(.25 + .25*N_array[i])
            bandpath_y.append(0.)
            bandpath_z.append(.25 + (.25-eta)*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$Z_1$')
        d = np.sqrt((.5-eta)**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)):
            #Z1 - M
            bandpath_x.append(.5)
            bandpath_y.append(0.)
            bandpath_z.append(.5-eta - (.5-eta)*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$M$')
        
        for i in range(0,int(N/10)):
            #Adding in a gap for my band structure plot before the 
            #X - P path of bct1
            bandpath_x.append(-1)
            bandpath_y.append(-1)
            bandpath_z.append(-1)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1]+1)
        labels[1].append(r'$X$')
        d = np.sqrt(.25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)):
            #X - P
            bandpath_x.append(.25)
            bandpath_y.append(.25)
            bandpath_z.append(.25*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$P$')

    elif c > a:
        #bct2
        b1 = np.array([0,.5,.5])
        b2 = np.array([.5,0,.5])
        b3 = np.array([.5,.5,0])
        
        eta = .25*(1+(a/c)**2)
        zeta = a*a/(2*c*c)
        
        N_plot = []
        N0_array = np.linspace(0,1,N)
        d0 = np.sqrt(.25**2 + .25**2)
        N_list = [0]
        bandpath_x = []
        bandpath_y = []
        bandpath_z = []
        labels = [[0],[r'$\Gamma$']]
        for i in range(0,N-1):
            # Gamma - X
            bandpath_x.append(.25*N0_array[i])
            bandpath_y.append(.25*N0_array[i])
            bandpath_z.append(0.)
            if i > 0:
                N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$X$')
        d = np.sqrt(2*(.5*zeta)**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            #X - Y
            bandpath_x.append(.25 + .5*zeta*N_array[i])
            bandpath_y.append(.25 - .5*zeta*N_array[i])
            bandpath_z.append(0.)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$Y$')
        d = np.sqrt((.25 + .5*zeta - eta)**2 + (.25 - .5*zeta)**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # Y - Sigma
            bandpath_x.append(.25 + .5*zeta - (.25 + .5*zeta - eta)*N_array[i])
            bandpath_y.append(.25 - .5*zeta - (.25 - .5*zeta)*N_array[i])
            bandpath_z.append(0.)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$\Sigma$')
        d = np.sqrt((eta**2))
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # Sigma - Gamma
            bandpath_x.append(eta - eta*N_array[i])
            bandpath_y.append(0.)
            bandpath_z.append(0.)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$\Gamma$')
        d = np.sqrt(.5**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # Gamma - Z
            bandpath_x.append(0.)
            bandpath_y.append(0.)
            bandpath_z.append(.5*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$Z$')
        d = np.sqrt((.5 - eta)**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # Z - Sigma_1
            bandpath_x.append((.5 - eta)*N_array[i])
            bandpath_y.append(0.)
            bandpath_z.append(.5)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$\Sigma_1$')
        d = np.sqrt((.5 - eta - .25)**2 + .25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # Sigma_1 - N
            bandpath_x.append(.5 - eta - (.5 - eta - .25)*N_array[i])
            bandpath_y.append(0.)
            bandpath_z.append(.5 - .25*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$N$')
        d = np.sqrt(.25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # N - P
            bandpath_x.append(.25)
            bandpath_y.append(.25*N_array[i])
            bandpath_z.append(.25)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$P$')
        d = np.sqrt(2*(.5*zeta**2) + .25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)-1):
            # P - Y1
            bandpath_x.append(.25 - .5*zeta*N_array[i])
            bandpath_y.append(.25 - .5*zeta*N_array[i])
            bandpath_z.append(.25 + .25*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$Y_1$')
        d = np.sqrt((.25-.5*zeta)**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)):
            # Y1 - Z
            bandpath_x.append(.25 - .5*zeta - (.25 - .5*zeta)*N_array[i])
            bandpath_y.append(.25 - .5*zeta - (.25 - .5*zeta)*N_array[i])
            bandpath_z.append(.5)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$Z$')
        
        for i in range(0,int(N/10)):
            #Adding in a gap for my band structure plot before the 
            #X - P path of bct2
            bandpath_x.append(-1)
            bandpath_y.append(-1)
            bandpath_z.append(-1)
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1]+1)
        labels[1].append(r'$X$')
        d = np.sqrt(.25**2)
        N_array = np.linspace(0,1,int(N*d/d0))
        for i in range(0,len(N_array)):
            #X - P
            bandpath_x.append(.25)
            bandpath_y.append(.25)
            bandpath_z.append(.25*N_array[i])
            N_list.append(N_list[-1]+1)
        labels[0].append(N_list[-1])
        labels[1].append(r'$P$')
        
    bandpath_x = np.array(bandpath_x)
    bandpath_y = np.array(bandpath_y)
    bandpath_z = np.array(bandpath_z)
    N_list = np.array(N_list)
    
    return bandpath_x[bandpath_x != -1], bandpath_y[bandpath_x != -1], bandpath_z[bandpath_x != -1], N_list[bandpath_x != -1], labels

def integrateBands_Better(E_k,dS_k,vz_k,v_k,bandID,nbands,T):
    NF_up = 0.
    NF_down = 0.
    sigmaz_up = 0.
    sigmaz_down = 0.

    dE = np.abs(E_k[1] - E_k[0])
    
    tau = []
    print('calculating better electron lifetimes')
    for i in range(0,len(bandID)):
        tau.append([])
        for n in range(0,len(E_k)):
            tau[i].append([])
            tau_inv_tmp = 0.
            for j in range(0,len(bandID)):
                if ((bandID[i] < nbands) and (bandID[j] < nbands)) or ((bandID[i] >= nbands) and (bandID[j] >= nbands)):
                    for m in range(0,len(E_k)):
                        tau_inv_tmp = tau_inv_tmp + dE*np.sum(dS_k[j][m][v_k[j][m] != 0.]*deltaGaussHerm(E_k[m],E_k[n],.02*E_k[-1])/v_k[j][m][v_k[j][m] != 0.])
            if tau_inv_tmp > 1e-8:
                tau[i][n] = np.ones(len(dS_k[i][n]))/tau_inv_tmp
            else:
                tau[i][n] = np.zeros(len(dS_k[i][n]))
    
    print('calculating better conductivity')
    for i in range(0,len(bandID)):
        NF = 0.
        sigma = 0.
        for j in range(0,len(E_k)):
            NF = NF + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*deltaGaussHerm(E_k[j],0.,.02*E_k[-1])/v_k[i][j][v_k[i][j] != 0.])
            sigma = sigma + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*tau[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/v_k[i][j][v_k[i][j] != 0.])
            #print(i,j,NF,sigma)
        if bandID[i] < nbands:
            NF_up = NF_up + NF
            sigmaz_up = sigmaz_up + sigma
        else:
            NF_down = NF_down + NF
            sigmaz_down = sigmaz_down + sigma

    return NF_up,NF_down,(NF_up-NF_down)/(NF_up+NF_down),sigmaz_up,sigmaz_down,(sigmaz_up-sigmaz_down)/(sigmaz_up+sigmaz_down)

def integrateBands(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID,nbands,T,a,b,c,E_cutoff,N_k):
    print(nbands)
    x = np.linspace(-.5+1e-4,.5-1e-4,N_k)
    y = np.linspace(-.5+1e-4,.5-1e-4,N_k)
    z = np.linspace(-.5+1e-4,.5-1e-4,N_k)
    dx = (x[1] - x[0])/a
    dy = (y[1] - y[0])/b
    dz = (z[1] - z[0])/c
    X,Y,Z = np.meshgrid(x,y,z)
    
    NF_up = 0.
    NF_down = 0.
    sigmax_up = 0.
    sigmax_down = 0.
    sigmay_up = 0.
    sigmay_down = 0.
    sigmaz_up = 0.
    sigmaz_down = 0.
    
    E = []
    #vz = []
    tau_inv = []
    tau = []
    for i in range(0,len(bandID)):
        E.append(interp_band_list[i](X,Y,Z))
        #vz.append(interp_vz_list[i](X,Y,Z))
    print('calculating electron lifetimes')
    for i in range(0,len(bandID)):
        mask_i = np.abs(E[i]) < E_cutoff
        tau_inv.append([])
        tau.append([])
        for j in range(0,len(bandID)):
            mask_j = np.abs(E[j]) < E_cutoff
            tau_inv[i].append([])
            for n in range(0,len(E[i][mask_i])):
                if ((bandID[i] < nbands) and (bandID[j] < nbands)) or ((bandID[i] >= nbands) and (bandID[j] >= nbands)):
                    tau_inv[i][j].append(dx*dy*dz*np.sum(deltaGaussHerm(E[i][mask_i][n],E[j][mask_j],.25*T)))
                else:
                    tau_inv[i][j].append(0.)
                #print(bandID[i],bandID[j],n,tau_inv[i][j][n])
        for n in range(0,len(E[i][mask_i])):
            tau_i_inv = 0.
            for j in range(0,len(bandID)):
                tau_i_inv = tau_i_inv + tau_inv[i][j][n]
            tau[i].append(1/tau_i_inv)
        tau[i] = np.array(tau[i])
            
    
    print('calculating conductivity from bands')
    for i in range(0,len(bandID)):
        #E = interp_band_list[i](X,Y,Z)
        mask = np.abs(E[i]) < E_cutoff
        #vx = interp_vx_list[i](X,Y,Z)
        #vy = interp_vy_list[i](X,Y,Z)
        vz = interp_vz_list[i](X,Y,Z)
        #tau = []
        #for j in range(0,len(E[mask])):
        #    tau.append(1/np.sum(deltaGaussHerm(E[mask][j],E[mask],.25*T)))
        #tau = np.array(tau)
        #sigma1 = np.sum(fermi_deriv(E[mask][~np.isnan(vz[mask])],T))
        NF = dx*dy*dz*np.sum(deltaGaussHerm(E[i][mask][~np.isnan(vz[mask])],0.,.25*T))
        sigmaz = dx*dy*dz*np.sum(tau[i][~np.isnan(vz[mask])]*fermi_deriv(E[i][mask][~np.isnan(vz[mask])],T)*vz[mask][~np.isnan(vz[mask])]*vz[mask][~np.isnan(vz[mask])]) #+ np.sum(tau[~np.isnan(vz[mask])]*fermi_deriv(E[mask][~np.isnan(vz[mask])],T)*vz[mask][~np.isnan(vz[mask])]*vx[mask][~np.isnan(vz[mask])]*Y[mask][~np.isnan(vz[mask])]) - np.sum(tau[~np.isnan(vz[mask])]*fermi_deriv(E[mask][~np.isnan(vz[mask])],T)*vz[mask][~np.isnan(vz[mask])]*vy[mask][~np.isnan(vz[mask])]*X[mask][~np.isnan(vz[mask])])
        #print(bandID[i],NF,sigmaz)#,E[mask][~np.isnan(vz[mask])])
        if bandID[i] < nbands:
            NF_up = NF_up + NF
            sigmaz_up = sigmaz_up + sigmaz
        else:
            NF_down = NF_down + NF
            sigmaz_down = sigmaz_down + sigmaz
    print(' ')
    print(NF_up,NF_down,(NF_up-NF_down)/(NF_up+NF_down))
    print(sigmaz_up,sigmaz_down,(sigmaz_up-sigmaz_down)/(sigmaz_up+sigmaz_down))
    return NF_up,NF_down,(NF_up-NF_down)/(NF_up+NF_down),sigmaz_up,sigmaz_down,(sigmaz_up-sigmaz_down)/(sigmaz_up+sigmaz_down)

def main_plotBands(EBANDSFilename,nbands,a,c,E_cutoff,N_theta,N_phi,N_E,N_k = 100,checkFit = False):
    n_fit = 5
    print('importing bands from file')
    kx,ky,kz,E = readBandFile(EBANDSFilename,nbands)
    print('interpolating bands')
    fitParamsList,bandID = genFitParams(kx,ky,kz,E,nbands,E_cutoff)
    # NOTE: should add an option for interp function to not interp
    #       velocities, thats what takes forever
    #bandpath_x,bandpath_y,bandpath_z,N,labels = genBands(interp_band_list,bandID,nbands,UnitCell,a,b,c,N_k)
    num = np.zeros(len(bandID))
    for i in range(0,len(bandID)):
        num[i] = len(E[bandID[i]][np.abs(E[bandID[i]]) < E_cutoff])
    NUM = num/max(num)
    
    if checkFit:
        residuals = np.zeros(len(bandID))
        N_abinit = np.linspace(0,1,len(kx))
        for i in range(0,len(bandID)):
            mask = np.abs(E[bandID[i]]) < E_cutoff
            residuals[i] = np.sum(np.abs(cosFitTest_3D(kx[mask],ky[mask],kz[mask],fitParamsList[i],n_fit)-E[bandID[i]][mask]))/len(kx[mask])
            print(bandID[i],residuals[i])
            if bandID[i] < nbands:
                plt.plot(N_abinit,cosFitTest_3D(kx,ky,kz,fitParamsList[i],n_fit),c = 'blue')
                plt.scatter(N_abinit,E[bandID[i]],c = 'blue')
            else:
                plt.plot(N_abinit,cosFitTest_3D(kx,ky,kz,fitParamsList[i],n_fit),c = 'red')
                plt.scatter(N_abinit,E[bandID[i]],c = 'red')
            plt.plot(N_abinit,np.zeros(len(N_abinit)),c = 'k',ls = ':')
            plt.show()
        print(np.sum(residuals)/len(bandID))

    bandpath_x,bandpath_y,bandpath_z,N,labels = genBands(a,c,N_k)
    
    f, (a1,a2) = plt.subplots(1,2,gridspec_kw = {'width_ratios': [3,1]})
    a1.set_xticks(labels[0],labels = labels[1])
    a1.set_yticks([-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5],labels = ['-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3','0.4','0.5'])
    #a1.set_yticks([-1.25,-1,-.75,-.5,.25,0,.25,.5,.75,1,1.25],labels = ['-1.25','-1','-0.75','-0.5','-0.25','0','0.25','0.5','0.75','1','1.25'])
    a1.plot(N,np.zeros(len(N)),ls = '--',c = 'black')
    a1.set_ylim([-.5,.5])
    #a1.set_ylim([-1.4,1.4])
    a1.set_xlim(N[0],N[-1])
    for i in range(0,len(bandID)):
        if bandID[i] < nbands:
            #a1.plot(N,cosFit_3D_wrapper(bandpath_x,bandpath_y,bandpath_z,fitParamsList[i]),c = 'blue')
            a1.plot(N,cosFitTest_3D(bandpath_x,bandpath_y,bandpath_z,fitParamsList[i],n_fit),c = 'blue')
        else:
            #a1.plot(N,cosFit_3D_wrapper(bandpath_x,bandpath_y,bandpath_z,fitParamsList[i]),c = 'red')
            a1.plot(N,cosFitTest_3D(bandpath_x,bandpath_y,bandpath_z,fitParamsList[i],n_fit),c = 'red')
    
    print('generating integration grid')
    #kx,ky,kz,dkx,dky,dkz,E_k,vz = genEquipotentialAdaptiveCartesianGrid(fitParamsList,E_cutoff,NUM)
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(fitParamsList,E_cutoff,N_E,N_theta,N_phi)
    #print('calculating DOS')
    #E,N_up,N_down = genDOSAdaptiveCartesian(bandID,nbands,dkx,dky,dkz,E_k,E_cutoff,2*N_E)
    E,N_up,N_down = genDOS(bandID,nbands,dS_k,v_k,E_k,E_cutoff,2*N_E)
    
    print('DOS Results: ')
    print(' ')
    for i in range(0,len(E)):
        print(E[i],N_up[i],N_down[i])
    
    plt.subplot(1,2,2)
    
    a2.plot(-N_up,E,c = 'blue')
    a2.plot(N_down,E,c = 'red')
    a2.plot([0,0],[-2,2],ls = '--',c = 'black')
    a2.plot([-10,10],[0,0],ls = '--', c = 'black')
    #a2.set_xticks([-.2,-.1,0,.1,.2],labels = ['0.2','0.1','0','0.1','0.2'])
    a2.set_yticks([-1.25,-1,-.75,-.5,-.25,0,.25,.5,.75,1,1.25],labels = [])
    a2.set_xlabel(r'$N^\uparrow(E)$' + '\t\t' + r'$N^\downarrow(E)$')
    a2.set_ylim([-.5,.5])
    #a2.set_ylim([-1.4,1.4])
    #a2.set_xlim([-.2,.2])
    
    f.tight_layout()
    plt.show()
#main_plotBands('abinitOutputFiles/Co_FM_DOS/bct_a_291_c_249o_EBANDS.agr','bcc',20,2.91,2.91,2.49,2.0,21,21,41)
#main_plotBands('abinitOutputFiles/Co_FM_DOS/bct_a_277_c_277o_EBANDS.agr','bcc',20,2.77,2.77,2.77,2.0,21,21,41)
main_plotBands('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr',20,2.91,2.49,.48,15,15,101,checkFit=False)

def DSP(EBANDSFilename,nbands,a,c,E_cutoff,T,N_theta,N_phi,N_E,N_k = 31):
    print('importing bands from file')
    kx,ky,kz,E = readBandFile(EBANDSFilename,nbands)
    print('interpolating bands')
    fitParamsList,bandID = genFitParams(kx,ky,kz,E,nbands,E_cutoff)
    #plotBands(interp_band_list,bandID,nbands,'bct1',a,b,c,30)
    #genEquipotential(interp_band_list[1],interp_vx_list[1],interp_vy_list[1],interp_vz_list[1],0.,21,21)
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(fitParamsList,E_cutoff,N_E,N_theta,N_phi)
    #plotDOS(bandID,nbands,dS_k,v_k,E_k,E_cutoff,2*N_E)
    NF_up,NF_down,PN,sigmaz_up,sigmaz_down,PNv2 = integrateBands_Better(E_k,dS_k,vz_k,v_k,bandID,nbands,T)
    #NF_up, NF_down, PN, sigmaz_up, sigmaz_down, PNv2 = integrateBands(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID,nbands,T,a,b,c,E_cutoff,N_k)
    #print([NF_up, NF_down, PN, sigmaz_up, sigmaz_down, PNv2])
    #print(' ')
    #print(NF_up_better,NF_down_better,PN_better,sigmaz_up_better,sigmaz_down_better,PNv2_better)
    return [NF_up, NF_down, PN, sigmaz_up, sigmaz_down, PNv2]

#E = np.linspace(-.6,.6,51)
#delta = deltaGaussHerm(E,0,.06)
#plt.scatter(E,delta,c = 'blue')
#plt.show()

#print(DSP('abinitOutputFiles/Co_FM_DOS/bct_a_294_c_246o_EBANDS.agr',20,2.94,2.94,2.46,0.0016,8.617e-5*1,21,21,31))
#print(DSP('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr',20,2.91,2.49,0.4,8.617e-5*300,21,21,81))
#print(DSP('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr',20,2.91,2.49,0.4,8.617e-5*300,15,15,101))

def main(EBANDSDirectory,T,nbands,a,b,c,N_theta,N_phi,N_E):
    EBANDSFilenameList = []
    os.system('ls ' + EBANDSDirectory + '*_DS2_EBANDS.agr > tmpFileList')
    with open('tmpFileList','r') as myFile:
        read = csv.reader(myFile)
        for row in read:
            EBANDSFilenameList.append(row[0])
    os.system('rm tmpFileList')
    print(EBANDSFilenameList)
    #NOTE: should find a way to extract the a- & c-values from the 
    # filenames, this won't really matter though since I'm not 
    # plotting bands
    #T = 8.617e-5*np.linspace(10,300,30)
    tol = 1e-8
    E_cut = T*np.arccosh(1/np.sqrt(4*T*tol))
    NF_up = np.zeros(len(EBANDSFilenameList))
    NF_down = np.zeros(len(EBANDSFilenameList))
    sigmaz_up = np.zeros(len(EBANDSFilenameList))
    sigmaz_down = np.zeros(len(EBANDSFilenameList))
    PN = np.zeros(len(EBANDSFilenameList))
    PNv2 = np.zeros(len(EBANDSFilenameList))
    
    for i in range(0,len(EBANDSFilenameList)):
        results = DSP(EBANDSFilenameList[i],nbands,a,c,E_cut,T,N_theta,N_phi,N_E)
        NF_up[i] = results[0]
        NF_down[i] = results[1]
        PN[i] = results[2]
        sigmaz_up[i] = results[3]
        sigmaz_down[i] = results[4]
        PNv2[i] = results[5]
        print(EBANDSFilenameList[i], NF_up[i], NF_down[i], PN[i], sigmaz_up[i], sigmaz_down[i], PNv2[i])
        
    print('final results')
    for i in range(0,len(EBANDSFilenameList)):
        print(EBANDSFilenameList[i], NF_up[i], NF_down[i], PN[i], sigmaz_up[i], sigmaz_down[i], PNv2[i])
    
#main('abinitOutputFiles/Co_FM_bandStruct/',8.617e-5*300,20,2.77,2.77,2.77,15,15,101)
