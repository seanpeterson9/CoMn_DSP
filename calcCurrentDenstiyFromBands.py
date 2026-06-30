from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.fft import rfft
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

    gauss = np.exp(-((Ek - Ek_prime)**2)/(2*f*f))
    return (1./(f*np.sqrt(2.)))*gauss*(coeffZero*hermiteZero + coeffTwo*hermiteTwo)

def cosFourierTransform_1D(x,f,n):
    FourierCoeffs = np.zeros(int(n)+1)
    FourierFreqs = np.zeros(int(n)+1)
    dx = np.abs(np.gradient(x))
    for i in range(0,n+1):
        FourierFreqs[i] = 2*np.pi*i
        FourierCoeffs[i] = np.sum(dx*f*np.cos(FourierFreqs[i]*x))#/np.abs(x[0]-x[-1])
        #print(i,FourierFreqs[i],FourierCoeffs[i])
        #plt.plot(x,f)
        #plt.plot(x,np.cos(FourierFreqs[i]*x))
        #plt.show()
        f = f - FourierCoeffs[i]*np.cos(FourierFreqs[i]*x)
    return FourierFreqs,FourierCoeffs

def cosFit_wrapper(x,C0,C2,C4,C6,C8,C10,C12):
    return cosFit(x,[C0,C2,C4,C6,C8,C10,C12])

def cosFit(x,coeffs):
#def cosFit(x,A,B,C,D,E,F,G):#,H,I):
    #return A + B*np.cos(2*np.pi*x) + C*np.cos(4*np.pi*x) + D*np.cos(6*np.pi*x) + E*np.cos(8*np.pi*x) + F*np.cos(10*np.pi*x) + G*np.cos(12*np.pi*x)# + H*np.cos(7*np.pi*x) #+ I*np.cos(8*np.pi*x)
    n = 6
    fit = 0.
    for i in range(0,n+1):
        fit = fit + coeffs[i]*np.cos(2*np.pi*i*x)
    return fit

def cosFit_deriv(x,A,B,C,D,E,F,G):
    return 2*B*np.pi*np.sin(2*np.pi*x) + 4*C*np.pi*np.sin(4*np.pi*x) + 6*D*np.pi*np.sin(6*np.pi*x) + 8*E*np.pi*np.sin(8*np.pi*x) + 10*F*np.pi*np.sin(10*np.pi*x) + 12*G*np.pi*np.sin(12*np.pi*x)

def cosFit_2D(xy,C00,C20,C22,C40,C42,C44,C60,C62,C64,C66,C80,C82,C84,C86,C88,C90,C92,C94,C96,C98,C99):
    x,y = xy
    return C00 + C20*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)) + C22*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + C40*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y)) + C42*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y)) + C44*np.cos(4*np.pi*x)*np.cos(4*np.pi*y) + C60*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y)) + C62*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y)) + C64*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y)) + C66*np.cos(6*np.pi*x)*np.cos(6*np.pi*y) + C80*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y)) + C82*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y)) + C84*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y)) + C86*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y)) + C88*np.cos(8*np.pi*x)*np.cos(8*np.pi*y) + C90*(np.cos(10*np.pi*x)+np.cos(10*np.pi*y)) + C92*(np.cos(10*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(10*np.pi*y)) + C94*(np.cos(10*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(10*np.pi*y)) + C96*(np.cos(10*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(10*np.pi*y)) + C98*(np.cos(10*np.pi*x)*np.cos(8*np.pi*y)+np.cos(8*np.pi*x)*np.cos(10*np.pi*y)) + C99*np.cos(10*np.pi*x)*np.cos(10*np.pi*y)

def cosFit_3D(xyz,C000,C200,C220,C002,C202,C222,C400,C420,C440,C004,C204,C402,C404,C224,C422,C424,C442,C444,C600,C620,C640,C660,C006,C206,C602,C226,C622,C406,C604,C426,C642,C624,C446,C664,C606,C626,C662,C646,C666,C800,C820,C840,C860,C880,C008,C208,C802,C228,C822,C408,C804,C428,C842,C824,C448,C844,C608,C806,C628,C862,C826,C648,C864,C846,C668,C866,C808,C828,C882,C848,C884,C868,C886,C888):
    #function has like 73 fit params! Probably an overfit, but doesn't matter since the fit parameters are meaningless and we just 
    # are using this instead of an interpolation
    x,y,z = xyz
    secondOrder = C200*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)) + C220*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + C002*np.cos(2*np.pi*z) + C202*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(2*np.pi*z) + C222*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(2*np.pi*z)

    fourthOrder = C400*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y)) + C420*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y)) + C440*np.cos(4*np.pi*x)*np.cos(4*np.pi*y) + C004*np.cos(4*np.pi*z) + C204*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(4*np.pi*z) + C402*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.cos(2*np.pi*z) + C404*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.cos(4*np.pi*z) + C224*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(4*np.pi*z) + C424*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(4*np.pi*z) + C442*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(2*np.pi*z) + C444*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(4*np.pi*z)

    sixthOrder = C600*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y)) + C620*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y)) + C640*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y)) + C660*np.cos(6*np.pi*x)*np.cos(6*np.pi*y) + C006*np.cos(6*np.pi*z) + C206*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(6*np.pi*z) + C602*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(2*np.pi*z) + C226*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(6*np.pi*z) + C622*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(2*np.pi*z) + C406*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.cos(6*np.pi*z) + C604*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(4*np.pi*z) + C426*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(6*np.pi*z) + C642*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y))*np.cos(2*np.pi*z) + C624*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(4*np.pi*z) + C446*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(6*np.pi*z) + C664*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(4*np.pi*z) + C606*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(6*np.pi*z) + C626*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(6*np.pi*z) + C662*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(2*np.pi*z) + C646*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*y))*np.cos(6*np.pi*z) + C666*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(6*np.pi*z)

    eighthOrder = C800*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y)) + C820*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y)) + C840*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y)) + C860*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y)) + C880*np.cos(8*np.pi*x)*np.cos(8*np.pi*y) + C008*np.cos(8*np.pi*z) + C208*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))*np.cos(8*np.pi*z) + C802*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C228*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(8*np.pi*z) + C822*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C408*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(8*np.pi*z) + C804*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C428*(np.cos(4*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(4*np.pi*y))*np.cos(8*np.pi*z) + C842*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C824*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C448*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*np.cos(8*np.pi*z) + C844*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C608*(np.cos(6*np.pi*x)+np.cos(6*np.pi*y))*np.cos(8*np.pi*z) + C806*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C628*(np.cos(6*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(6*np.pi*y))*np.cos(8*np.pi*z) + C862*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(2*np.pi*z) + C826*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C648*(np.cos(6*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(6*np.pi*x))*np.cos(8*np.pi*z) + C864*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(4*np.pi*z) + C846*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C668*np.cos(6*np.pi*x)*np.cos(6*np.pi*y)*np.cos(8*np.pi*z) + C866*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(6*np.pi*z) + C808*(np.cos(8*np.pi*x)+np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C828*(np.cos(8*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C882*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(2*np.pi*z) + C848*(np.cos(8*np.pi*x)*np.cos(4*np.pi*y)+np.cos(4*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C884*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(4*np.pi*z) + C868*(np.cos(8*np.pi*x)*np.cos(6*np.pi*y)+np.cos(6*np.pi*x)*np.cos(8*np.pi*y))*np.cos(8*np.pi*z) + C886*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(6*np.pi*z) + C888*np.cos(8*np.pi*x)*np.cos(8*np.pi*y)*np.cos(8*np.pi*z)
    
    return C000 + secondOrder + fourthOrder + sixthOrder + eighthOrder

def readBandFile(EBANDSFilename,nbands):
    os.system('./abinitEBANDSReader.sh ' + EBANDSFilename + ' kpts_tmp bands_tmp')
    with open('kpts_tmp','r') as myFile:
        read = csv.reader(myFile,delimiter = ',')
        kx = []
        ky = []
        kz = []
        for row in read:
            #print(row[0],row[1],row[2])
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

#kx,ky,kz,E = readBandFile('abinitOutputFiles/Co_FM_DOS/bct_a_277_c_277o_EBANDS.agr',20)
#Kx_double = np.concatenate((-kx[(ky == 0.) & (kz == 0.)],kx[(ky == 0.) & (kz == 0.)]))
#E_double = np.concatenate((E[14][(ky == 0.) & (kz == 0.)],E[14][(ky == 0.) & (kz == 0.)]))
#sort = np.argsort(Kx_double)
#E_double = E_double[sort]
#Kx_double = Kx_double[sort]
#print(Kx_double)
#print(np.fft.fft(E_double))
#FourierFreqs,FourierCoeffs = cosFourierTransform_1D(kx[(ky == 0.) & (kz == 0.)],E[14][(ky == 0.) & (kz == 0.)],4)
#FourierFreqs,FourierCoeffs = cosFourierTransform_1D(Kx_double,E_double,40)
#dk = np.gradient(Kx_double)
#w = np.fft.fftfreq(len(Kx_double),dk)
#A = np.abs(np.fft.fft(E_double))/len(Kx_double)
#print(w)
#x = np.linspace(-.5,.5,200)
#fit = np.zeros(x.shape)
#for i in range(0,len(FourierFreqs)):
#    fit = fit + FourierCoeffs[i]*np.cos(FourierFreqs[i]*x)

#plt.plot(x,fit,c = 'red')
#plt.scatter(Kx_double,E_double,c = 'blue')
#plt.show()
"""
popt,pcurve = curve_fit(cosFit_wrapper,kx[(ky == 0.) & (kz == 0.)],E[14][(ky == 0.) & (kz == 0.)])
plt.scatter(kx[(ky == 0.) & (kz == 0.)],E[14][(ky == 0.) & (kz == 0.)],c = 'blue')
plt.scatter(-kx[(ky == 0.) & (kz == 0.)],E[14][(ky == 0.) & (kz == 0.)],c = 'blue')
x = np.linspace(-.5,.5,100)
plt.plot(x,cosFit(x,[popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]]),c = 'red')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
popt,pcurve = curve_fit(cosFit_2D,(kx[kz == 0.],ky[kz == 0.]),E[14][kz == 0.])
x = np.linspace(-.5,.5,50)
y = np.linspace(-.5,.5,50)
X,Y = np.meshgrid(x,y)
ax.plot_wireframe(X,Y,cosFit_2D((X,Y),popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9],popt[10],popt[11],popt[12],popt[13],popt[14],popt[15],popt[16],popt[17],popt[18],popt[19],popt[20]),color = 'blue',rstride = 5,cstride = 5)
ax.scatter(kx[kz == 0.],ky[kz == 0.],E[14][kz == 0.],c = 'red')
#ax.scatter(kx[kz == 0.],-ky[kz == 0.],E[14][kz == 0.],c = 'red')
#ax.scatter(-kx[kz == 0.],ky[kz == 0.],E[14][kz == 0.],c = 'red')
#ax.scatter(-kx[kz == 0.],-ky[kz == 0.],E[14][kz == 0.],c = 'red')
#ax.scatter(ky[kz == 0.],kx[kz == 0.],E[14][kz == 0.],c = 'red')
#ax.scatter(ky[kz == 0.],-kx[kz == 0.],E[14][kz == 0.],c = 'red')
#ax.scatter(-ky[kz == 0.],kx[kz == 0.],E[14][kz == 0.],c = 'red')
#ax.scatter(-ky[kz == 0.],-kx[kz == 0.],E[14][kz == 0.],c = 'red')
plt.show()
#"""
def genFitParams(kx,ky,kz,E,nbands,E_cutoff):
    bandID = []
    fitParamsList = []
    for i in range(0,2*nbands):
        if len(E[i][np.abs(E[i]) < E_cutoff]) > 1:
            print(i,len(E[i][np.abs(E[i]) < E_cutoff]))
            bandID.append(i)
            popt,pcurve = curve_fit(cosFit_3D,(kx,ky,kz),E[i])
            fitParamsList.append(popt)
    fitParamsList = np.array(fitParamsList)
    print(fitParamsList.shape)
    return fitParamsList,bandID

def genInterpFuncs(kx,ky,kz,E,nbands,a,b,c,E_cutoff,N_k):
    Kx = np.concatenate((kx,ky,-kx,-ky,kx,ky,-kx,-ky,kx,ky,-kx,-ky,kx,ky,-kx,-ky))
    Ky = np.concatenate((ky,kx,ky,kx,-ky,-kx,-ky,-kx,ky,kx,ky,kx,-ky,-kx,-ky,-kx))
    Kz = np.concatenate((kz,kz,kz,kz,kz,kz,kz,kz,-kz,-kz,-kz,-kz,-kz,-kz,-kz,-kz))
    E_k = []
    for i in range(0,2*nbands):
        E_k.append(np.concatenate((E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i])))
    E_k = np.array(E_k)
    
    sort = np.argsort(Kx)
    bandID = []
    interp_band_list = []
    for i in range(0,2*nbands):
        if len(E_k[i][np.abs(E_k[i]) < E_cutoff]) > 1:
            print(i,len(E_k[i][np.abs(E_k[i]) < E_cutoff]))
            bandID.append(i)
            interp_band_list.append(LinearNDInterpolator((Kx[sort],Ky[sort],Kz[sort]),E_k[i][sort]))
    """
    j = 19
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Kx[(np.abs(E_k[j]) < E_cutoff) & (Kx > 0) & (Ky > 0) & (Kz > 0)],Ky[(np.abs(E_k[j]) < E_cutoff) & (Kx > 0) & (Ky > 0) & (Kz > 0)],Kz[(np.abs(E_k[j]) < E_cutoff) & (Kx > 0) & (Ky > 0) & (Kz > 0)],c = E_k[j][(np.abs(E_k[j]) < E_cutoff) & (Kx > 0) & (Ky > 0) & (Kz > 0)], cmap = 'inferno')
    plt.show()
    """
    x = np.linspace(-.5,.5,N_k)
    y = np.linspace(-.5,.5,N_k)
    z = np.linspace(-.5,.5,N_k)
    dx = (x[1] - x[0])/a
    dy = (y[1] - y[0])/b
    dz = (z[1] - z[0])/c
    X,Y,Z = np.meshgrid(x,y,z)
    
    Vx = []
    Vy = []
    Vz = []

    interp_vx_list = []
    interp_vy_list = []
    interp_vz_list = []
    
    for i in range(0,len(interp_band_list)):
        print(i)
        vy,vx,vz = np.gradient(interp_band_list[i](X,Y,Z))
        Vx.append(vx/dx)
        Vy.append(vy/dy)
        Vz.append(vz/dz)
        interp_vx_list.append(LinearNDInterpolator((X.flatten(),Y.flatten(),Z.flatten()),Vx[i].flatten()))
        interp_vy_list.append(LinearNDInterpolator((X.flatten(),Y.flatten(),Z.flatten()),Vy[i].flatten()))
        interp_vz_list.append(LinearNDInterpolator((X.flatten(),Y.flatten(),Z.flatten()),Vz[i].flatten()))

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(X,Y,Z,c = interp_band_list[3](X,Y,Z),cmap = 'inferno')
    #plt.show()
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(X,Y,Z,c = np.abs(Vz[3]),cmap = 'inferno')
    #plt.show()

    return interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID

def genEquipotential(interp_band_func,interp_vx_func,interp_vy_func,interp_vz_func,E,N_theta,N_phi):
    theta = np.linspace(0.,np.pi/2.,N_theta)
    phi = np.linspace(0.,np.pi/2.,N_phi)
    Theta,Phi = np.meshgrid(theta,phi)

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

    for i in range(0,N_phi):
        for j in range(0,N_theta):
            kr1[i][j] = np.abs(fsolve(lambda k: (interp_band_func(np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),np.abs(k)*np.cos(Phi[i][j])) - E)**2,.15))
            kx1[i][j] = np.abs(kr1[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky1[i][j] = np.abs(kr1[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz1[i][j] = np.abs(kr1[i][j])*np.cos(Phi[i][j])
            
            kr2[i][j] = np.abs(fsolve(lambda k: (interp_band_func(np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-np.abs(k)*np.cos(Phi[i][j])) - E)**2,.15))
            kx2[i][j] = np.abs(kr2[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky2[i][j] = np.abs(kr2[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz2[i][j] = .5 - np.abs(kr2[i][j])*np.cos(Phi[i][j])

            kr3[i][j] = np.abs(fsolve(lambda k: (interp_band_func(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),np.abs(k)*np.cos(Phi[i][j])) - E)**2,.15))
            kx3[i][j] = .5 - np.abs(kr3[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky3[i][j] = .5 - np.abs(kr3[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz3[i][j] = np.abs(kr3[i][j])*np.cos(Phi[i][j])
            
            """
            kr4[i][j] = np.abs(fsolve(lambda k: (interp_band_func(.5-np.abs(k)*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-np.abs(k)*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-np.abs(k)*np.cos(Phi[i][j])) - E)**2,.15))
            kx4[i][j] = .5 - np.abs(kr4[i][j])*np.sin(Phi[i][j])*np.cos(Theta[i][j])
            ky4[i][j] = .5 - np.abs(kr4[i][j])*np.sin(Phi[i][j])*np.sin(Theta[i][j])
            kz4[i][j] = .5 - np.abs(kr4[i][j])*np.cos(Phi[i][j])
            """
            if ((interp_band_func(kr1[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr1[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),kr1[i][j]*np.cos(Phi[i][j])) - E)**2 > 1e-5) or (kx1[i][j] < 0.) or (ky1[i][j] < 0.) or (kz1[i][j] < 0.):
                kr1[i][j] = -.5
            if ((interp_band_func(kr2[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr2[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-kr2[i][j]*np.cos(Phi[i][j])) - E)**2 > 1e-5) or (kx2[i][j] < 0.) or (ky2[i][j] < 0.) or (kz2[i][j] < 0.):
                kr2[i][j] = -.5
            if ((interp_band_func(.5-kr3[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-kr3[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),kr3[i][j]*np.cos(Phi[i][j])) - E)**2 > 1e-5) or (kx3[i][j] < 0.) or (ky3[i][j] < 0.) or (kz3[i][j] < 0.):
                kr3[i][j] = -.5
            """
            if ((interp_band_func(.5-kr4[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),.5-kr4[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),.5-kr4[i][j]*np.cos(Phi[i][j])) - E)**2 > 1e-5) or (kx4[i][j] < 0.) or (ky4[i][j] < 0.) or (kz4[i][j] < 0.):
                kr4[i][j] = -.5
            print(kr1[i][j]*np.sin(Phi[i][j])*np.cos(Theta[i][j]),kr1[i][j]*np.sin(Phi[i][j])*np.sin(Theta[i][j]),kr1[i][j]*np.cos(Phi[i][j]))
            #print(Theta[i][j], Phi[i][j], kr[i][j])
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
    
    #"""
    X1 = kr1*np.sin(Phi)*np.cos(Theta)
    Y1 = kr1*np.sin(Phi)*np.sin(Theta)
    Z1 = kr1*np.cos(Phi)
    X2 = kr2*np.sin(Phi)*np.cos(Theta)
    Y2 = kr2*np.sin(Phi)*np.sin(Theta)
    Z2 = .5 - kr2*np.cos(Phi)
    X3 = .5 - kr3*np.sin(Phi)*np.cos(Theta)
    Y3 = .5 - kr3*np.sin(Phi)*np.sin(Theta)
    Z3 = kr3*np.cos(Phi)
    #X4 = .5 - kr4*np.sin(Phi)*np.cos(Theta)
    #Y4 = .5 - kr4*np.sin(Phi)*np.sin(Theta)
    #Z4 = .5 - kr4*np.cos(Phi)
    
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
    
    kx = np.array(kx)
    ky = np.array(ky)
    kz = np.array(kz)
    dS = np.array(dS)
    
    mask = (dS < 3*np.sum(dS)/len(dS)) & (~np.isnan(dS))
    kx = kx[mask]
    ky = ky[mask]
    kz = kz[mask]
    dS = dS[mask]
    
    vx = interp_vx_func(kx,ky,kz)
    vy = interp_vy_func(kx,ky,kz)
    vz = interp_vz_func(kx,ky,kz)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    mask = (~np.isnan(v))
    kx = kx[mask]
    ky = ky[mask]
    kz = kz[mask]
    dS = dS[mask]
    vz = vz[mask]
    v = v[mask]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(kx,ky,kz,c = np.abs(vz),cmap = 'inferno')
    plt.show()
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X1[kr1 != -.5],Y1[kr1 != -.5],Z1[kr1 != -.5],c = dS1[kr1 != -.5])#,c = interp_band_func(X1[kr1 != -.5],Y1[kr1 != -.5],Z1[kr1 != -.5]),cmap = 'inferno')
    #ax.colorbar()
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X2[kr2 != -.5],Y2[kr2 != -.5],Z2[kr2 != -.5],c = dS2[kr2 != -.5])#,c = interp_band_func(X2[kr2 != -.5],Y2[kr2 != -.5],Z2[kr2 != -.5]),cmap = 'inferno')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X3[kr3 != -.5],Y3[kr3 != -.5],Z3[kr3 != -.5],c = dS3[kr3 != -.5])#,c = interp_band_func(X3[kr3 != -.5],Y3[kr3 != -.5],Z3[kr3 != -.5]),cmap = 'inferno')
    plt.show()
    #"""
    return kx,ky,kz,dS,vz,v

kx,ky,kz,E = readBandFile('abinitOutputFiles/Co_FM_DOS/bct_a_277_c_277o_EBANDS.agr',20)
interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID = genInterpFuncs(kx,ky,kz,E,20,2.77,2.77,2.77,.6,41)
genEquipotential(interp_band_list[6],interp_vx_list[6],interp_vy_list[6],interp_vz_list[6],0,21,21)

def genEquipotentialGrid(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,E_cutoff,N_E,N_theta,N_phi):
    E_grid = np.linspace(-E_cutoff,E_cutoff,N_E)

    kx = []
    ky = []
    kz = []
    dS = []
    vz = []
    v = []
    
    #plt.show()
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    
    #color = ['red','green','blue']
    
    for i in range(0,len(interp_band_list)):
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
            kx_tmp,ky_tmp,kz_tmp,dS_tmp,vz_tmp,v_tmp = genEquipotential(interp_band_list[i],interp_vx_list[i],interp_vy_list[i],interp_vz_list[i],E_grid[j],N_theta,N_phi)
            kx[i][j] = kx_tmp
            ky[i][j] = ky_tmp
            kz[i][j] = kz_tmp
            dS[i][j] = dS_tmp
            vz[i][j] = vz_tmp
            v[i][j] = v_tmp
            
            #ax.scatter(kx[i][j],ky[i][j],kz[i][j],c = color[j])
    #plt.show()
    return E_grid,dS,vz,v

def genDOS(bandID,nbands,dS_k,v_k,E_k,E_cut,N_E):
    dE = np.abs(E_k[1] - E_k[0])
    E = np.linspace(-E_cut,E_cut,N_E)
    N_up = np.zeros(E.shape)
    N_down = np.zeros(E.shape)
    for n in range(0,N_E):
        for i in range(0,len(bandID)):
            NF = 0.
            for j in range(0,len(E_k)):
                NF = NF + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*deltaGaussHerm(E_k[j],E[n],.1*E_k[-1])/v_k[i][j][v_k[i][j] != 0.])
                print(i,j,NF)
            if bandID[i] < nbands:
                N_up[n] = N_up[n] + NF
            else:
                N_down[n] = N_down[n] + NF
        print(E[n],N_up[n],N_down[n])
    #plt.plot(N_up,E,c = 'blue')
    #plt.plot(N_down,E,c = 'red')
    #plt.show()

    return E,N_up,N_down

#def genBands(interp_band_list,bandID,nbands,UnitCell,a,b,c,N):
def genBands(UnitCell,a,b,c,N):
    #NOTE: I don't have the time to fix this rn, but instead of 
    # writing the bandpaths in terms of b1, b2, b3 it'll be easier to
    # do it properly if I write it in terms of kx,ky,kz
    # because rn I am currently calculating the paths improperly
    # none of my paths should go outside (1/2,1/2,1/2)
    
    if UnitCell == 'bcc':
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

    elif UnitCell == 'bct1':
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

        #NOTE: Still need to add the | X-P path which is discontinuous
        # from the rest of the bandpath for some reason
        # also need to scale how long each section of the path looks
        # on the plot by how long it is in k-space
        # and need to come up with a way to label symmetry points
        # on the x-axis of the plot
    
    elif UnitCell == 'bct2':
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
        
        #bandpath1 = .5*b3*N_array
        #bandpath2 = .5*b3 + (-.5*b1 + .5*b2 + .5*b3)*N_array
        #bandpath3 = (-.5*b1 + .5*b2 + .5*b3)*N_array[::-1]
        """
        #bandpath = np.concatenate((bandpath1,bandpath2,bandpath3))
        print(bandpath_x)
        print(bandpath_y)
        print(bandpath_z)
        plt.plot(np.zeros(len(bandpath_x)),c = 'black',ls = '--')
        for i in range(0,len(bandID)):
            if bandID[i] < nbands:
                plt.plot(interp_band_list[i](bandpath_x,bandpath_y,bandpath_z),c = 'blue')
            else:
                plt.plot(interp_band_list[i](bandpath_x,bandpath_y,bandpath_z),c = 'red')
        plt.show()
        #"""

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
                        tau_inv_tmp = tau_inv_tmp + dE*np.sum(dS_k[j][m][v_k[j][m] != 0.]*deltaGaussHerm(E_k[m],E_k[n],.15*E_k[-1])/v_k[j][m][v_k[j][m] != 0.])
            if tau_inv_tmp > 1e-8:
                tau[i][n] = np.ones(len(dS_k[i][n]))/tau_inv_tmp
            else:
                tau[i][n] = np.zeros(len(dS_k[i][n]))
    
    print('calculating better conductivity')
    for i in range(0,len(bandID)):
        NF = 0.
        sigma = 0.
        for j in range(0,len(E_k)):
            NF = NF + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*deltaGaussHerm(E_k[j],0.,.15*E_k[-1])/v_k[i][j][v_k[i][j] != 0.])
            sigma = sigma + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*tau[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/v_k[i][j][v_k[i][j] != 0.])
            print(i,j,NF,sigma)
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
        print(bandID[i],NF,sigmaz)#,E[mask][~np.isnan(vz[mask])])
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

def main_plotBandsNew(EBANDSFilename,UnitCell,nbands,a,b,c,E_cutoff,N_theta,N_phi,N_E,N_k = 100):
    print('importing bands from file')
    kx,ky,kz,E = readBandFile(EBANDSFilename,nbands)
    print('interpolating bands')
    fitParamsList,bandID = genFitParams(kx,ky,kz,E,nbands,E_cutoff)
    #interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID = genInterpFuncs(kx,ky,kz,E,nbands,a,b,c,E_cutoff,30) 
    # NOTE: should add an option for interp function to not interp
    #       velocities, thats what takes forever
    #bandpath_x,bandpath_y,bandpath_z,N,labels = genBands(interp_band_list,bandID,nbands,UnitCell,a,b,c,N_k)
    bandpath_x,bandpath_y,bandpath_z,N,labels = genBands(UnitCell,a,b,c,N_k)
    
    f, (a1,a2) = plt.subplots(1,2,gridspec_kw = {'width_ratios': [3,1]})
    a1.set_xticks(labels[0],labels = labels[1])
    #a1.set_yticks([-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5],labels = ['-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3','0.4','0.5'])
    a1.set_yticks([-1.5,-1,-.5,0,.5,1,1.5],labels = ['-1.5','-1','-0.5','0','0.5','1','1.5'])
    a1.plot(N,np.zeros(len(N)),ls = '--',c = 'black')
    a1.set_ylim([-1.9,1.9])
    a1.set_xlim(N[0],N[-1])
    for i in range(0,len(bandID)):
        if bandID[i] < nbands:
            #a1.plot(N,interp_band_list[i](bandpath_x,bandpath_y,bandpath_z),c = 'blue')
            a1.plot(N,cosFit_3D((bandpath_x,bandpath_y,bandpath_z),fitParamsList[i][0],fitParamsList[i][1],fitParamsList[i][2],fitParamsList[i][3],fitParamsList[i][4],fitParamsList[i][5],fitParamsList[i][6],fitParamsList[i][7],fitParamsList[i][8],fitParamsList[i][9],fitParamsList[i][10],fitParamsList[i][11],fitParamsList[i][12],fitParamsList[i][13],fitParamsList[i][14],fitParamsList[i][15],fitParamsList[i][16],fitParamsList[i][17],fitParamsList[i][18],fitParamsList[i][19],fitParamsList[i][20],fitParamsList[i][21],fitParamsList[i][22],fitParamsList[i][23],fitParamsList[i][24],fitParamsList[i][25],fitParamsList[i][26],fitParamsList[i][27],fitParamsList[i][28],fitParamsList[i][29],fitParamsList[i][30],fitParamsList[i][31],fitParamsList[i][32],fitParamsList[i][33],fitParamsList[i][34],fitParamsList[i][35],fitParamsList[i][36],fitParamsList[i][37],fitParamsList[i][38],fitParamsList[i][39],fitParamsList[i][40],fitParamsList[i][41],fitParamsList[i][42],fitParamsList[i][43],fitParamsList[i][44],fitParamsList[i][45],fitParamsList[i][46],fitParamsList[i][47],fitParamsList[i][48],fitParamsList[i][49],fitParamsList[i][50],fitParamsList[i][51],fitParamsList[i][52],fitParamsList[i][53],fitParamsList[i][54],fitParamsList[i][55],fitParamsList[i][56],fitParamsList[i][57],fitParamsList[i][58],fitParamsList[i][59],fitParamsList[i][60],fitParamsList[i][61],fitParamsList[i][62],fitParamsList[i][63],fitParamsList[i][64],fitParamsList[i][65],fitParamsList[i][66],fitParamsList[i][67],fitParamsList[i][68],fitParamsList[i][69],fitParamsList[i][70],fitParamsList[i][71],fitParamsList[i][72],fitParamsList[i][73]),c = 'blue')
        else:
            #a1.plot(N,interp_band_list[i](bandpath_x,bandpath_y,bandpath_z),c = 'red')
            a1.plot(N,cosFit_3D((bandpath_x,bandpath_y,bandpath_z),fitParamsList[i][0],fitParamsList[i][1],fitParamsList[i][2],fitParamsList[i][3],fitParamsList[i][4],fitParamsList[i][5],fitParamsList[i][6],fitParamsList[i][7],fitParamsList[i][8],fitParamsList[i][9],fitParamsList[i][10],fitParamsList[i][11],fitParamsList[i][12],fitParamsList[i][13],fitParamsList[i][14],fitParamsList[i][15],fitParamsList[i][16],fitParamsList[i][17],fitParamsList[i][18],fitParamsList[i][19],fitParamsList[i][20],fitParamsList[i][21],fitParamsList[i][22],fitParamsList[i][23],fitParamsList[i][24],fitParamsList[i][25],fitParamsList[i][26],fitParamsList[i][27],fitParamsList[i][28],fitParamsList[i][29],fitParamsList[i][30],fitParamsList[i][31],fitParamsList[i][32],fitParamsList[i][33],fitParamsList[i][34],fitParamsList[i][35],fitParamsList[i][36],fitParamsList[i][37],fitParamsList[i][38],fitParamsList[i][39],fitParamsList[i][40],fitParamsList[i][41],fitParamsList[i][42],fitParamsList[i][43],fitParamsList[i][44],fitParamsList[i][45],fitParamsList[i][46],fitParamsList[i][47],fitParamsList[i][48],fitParamsList[i][49],fitParamsList[i][50],fitParamsList[i][51],fitParamsList[i][52],fitParamsList[i][53],fitParamsList[i][54],fitParamsList[i][55],fitParamsList[i][56],fitParamsList[i][57],fitParamsList[i][58],fitParamsList[i][59],fitParamsList[i][60],fitParamsList[i][61],fitParamsList[i][62],fitParamsList[i][63],fitParamsList[i][64],fitParamsList[i][65],fitParamsList[i][66],fitParamsList[i][67],fitParamsList[i][68],fitParamsList[i][69],fitParamsList[i][70],fitParamsList[i][71],fitParamsList[i][72],fitParamsList[i][73]),c = 'red')
    
    #E_k,dS_k,vz_k,v_k = genEquipotentialGrid(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,E_cutoff,N_E,N_theta,N_phi)
    #E,N_up,N_down = genDOS(bandID,nbands,dS_k,v_k,E_k,E_cutoff,2*N_E)
    
    #plt.subplot(1,2,2)
    
    #a2.plot(-N_up,E,c = 'blue')
    #a2.plot(N_down,E,c = 'red')
    #a2.plot([0,0],[-2,2],ls = '--',c = 'black')
    #a2.plot([-.2,.2],[0,0],ls = '--', c = 'black')
    #a2.set_xticks([-.2,-.1,0,.1,.2],labels = ['0.2','0.1','0','0.1','0.2'])
    #a2.set_yticks([-1.5,-1,-.5,0,.5,1,1.5],labels = [])
    #a2.set_xlabel(r'$N^\uparrow(E)$' + '\t\t' + r'$N^\downarrow(E)$')
    #a2.set_ylim([-1.9,1.9])
    #a2.set_xlim([-.2,.2])
    
    f.tight_layout()
    plt.show()
#main_plotBandsNew('abinitOutputFiles/Co_FM_DOS/bct_a_277_c_277o_EBANDS.agr','bcc',20,2.77,2.77,2.77,2.0,21,21,41)

def main_plotBandsDOS(EBANDSFilename,UnitCell,nbands,a,b,c,E_cutoff,N_theta,N_phi,N_E,N_k = 100):
    print('importing bands from file')
    kx,ky,kz,E = readBandFile(EBANDSFilename,nbands)
    print('interpolating bands')
    interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID = genInterpFuncs(kx,ky,kz,E,nbands,a,b,c,E_cutoff,30) 
    # NOTE: should add an option for interp function to not interp
    #       velocities, thats what takes forever
    #bandpath_x,bandpath_y,bandpath_z,N,labels = genBands(interp_band_list,bandID,nbands,UnitCell,a,b,c,N_k)
    bandpath_x,bandpath_y,bandpath_z,N,labels = genBands(UnitCell,a,b,c,N_k)
    
    f, (a1,a2) = plt.subplots(1,2,gridspec_kw = {'width_ratios': [3,1]})
    a1.set_xticks(labels[0],labels = labels[1])
    #a1.set_yticks([-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5],labels = ['-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3','0.4','0.5'])
    a1.set_yticks([-1.5,-1,-.5,0,.5,1,1.5],labels = ['-1.5','-1','-0.5','0','0.5','1','1.5'])
    a1.plot(N,np.zeros(len(N)),ls = '--',c = 'black')
    a1.set_ylim([-1.9,1.9])
    a1.set_xlim(N[0],N[-1])
    for i in range(0,len(bandID)):
        if bandID[i] < nbands:
            a1.plot(N,interp_band_list[i](bandpath_x,bandpath_y,bandpath_z),c = 'blue')
        else:
            a1.plot(N,interp_band_list[i](bandpath_x,bandpath_y,bandpath_z),c = 'red')
    
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,E_cutoff,N_E,N_theta,N_phi)
    E,N_up,N_down = genDOS(bandID,nbands,dS_k,v_k,E_k,E_cutoff,2*N_E)
    
    #plt.subplot(1,2,2)
    a2.plot(-N_up,E,c = 'blue')
    a2.plot(N_down,E,c = 'red')
    a2.plot([0,0],[-2,2],ls = '--',c = 'black')
    a2.plot([-.2,.2],[0,0],ls = '--', c = 'black')
    a2.set_xticks([-.2,-.1,0,.1,.2],labels = ['0.2','0.1','0','0.1','0.2'])
    #a2.set_yticks([-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5],labels = [])
    a2.set_yticks([-1.5,-1,-.5,0,.5,1,1.5],labels = [])
    a2.set_xlabel(r'$N^\uparrow(E)$' + '\t\t' + r'$N^\downarrow(E)$')
    a2.set_ylim([-1.9,1.9])
    a2.set_xlim([-.2,.2])
    f.tight_layout()
    plt.show()

#main_plotBandsDOS('abinitOutputFiles/Co_FM_DOS/bct_a_270_c_306o_EBANDS.agr','bct2',20,2.70,2.70,3.06,2.0,21,21,41)
#main_plotBandsDOS('abinitOutputFiles/Co_FM_DOS/bct_a_294_c_246o_EBANDS.agr','bct1',20,2.94,2.94,2.46,2.0,21,21,41)
#main_plotBandsDOS('abinitOutputFiles/Co_FM_DOS_28/bct_a_277_c_277o_EBANDS.agr','bcc',20,2.77,2.77,2.77,2.0,21,21,41)

"""
bandpath_x,bandpath_y,bandpath_z,N,labels = genBands('bct2',2.46,2.46,3.44,100)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(bandpath_x,bandpath_y,bandpath_z)
plt.show()
"""
def DSP(EBANDSFilename,nbands,a,b,c,E_cutoff,T,N_theta,N_phi,N_E,N_k = 31):
    print('importing bands from file')
    kx,ky,kz,E = readBandFile(EBANDSFilename,nbands)
    print('interpolating bands')
    interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID = genInterpFuncs(kx,ky,kz,E,nbands,a,b,c,E_cutoff,N_k)
    #plotBands(interp_band_list,bandID,nbands,'bct1',a,b,c,30)
    #genEquipotential(interp_band_list[1],interp_vx_list[1],interp_vy_list[1],interp_vz_list[1],0.,21,21)
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,E_cutoff,N_E,N_theta,N_phi)
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

def main(EBANDSFilenameList,T,nbands,a,b,c,N_theta,N_phi,N_E):
    #T = 8.617e-5*np.linspace(10,300,30)
    tol = 1e-8
    E_cut = .67*2*T*np.arccosh(1/np.sqrt(4*T*tol))
    NF_up = np.zeros(T.shape)
    NF_down = np.zeros(T.shape)
    sigmaz_up = np.zeros(T.shape)
    sigmaz_down = np.zeros(T.shape)
    PN = np.zeros(T.shape)
    PNv2 = np.zeros(T.shape)
    
    for i in range(0,len(T)):
        results = DSP(EBANDSFilenameList[i],nbands,a,b,c,E_cut[i],T[i],N_theta,N_phi,N_E)
        NF_up[i] = results[0]
        NF_down[i] = results[1]
        PN[i] = results[2]
        sigmaz_up[i] = results[3]
        sigmaz_down[i] = results[4]
        PNv2[i] = results[5]
        print(T[i], NF_up[i], NF_down[i], PN[i], sigmaz_up[i], sigmaz_down[i], PNv2[i])
        
    print('final results')
    for i in range(0,len(T)):
        print(T[i], NF_up[i], NF_down[i], PN[i], sigmaz_up[i], sigmaz_down[i], PNv2[i])
    
#main(['abinitOutputFiles/Co_FM_DOS_1e-3/bct_a_271_c_303o_EBANDS.agr','abinitOutputFiles/Co_FM_DOS_8e-4/bct_a_271_c_303o_EBANDS.agr','abinitOutputFiles/Co_FM_DOS_6e-4/bct_a_271_c_303o_EBANDS.agr','abinitOutputFiles/Co_FM_DOS_4e-4/bct_a_271_c_303o_EBANDS.agr','abinitOutputFiles/Co_FM_DOS_2e-4/bct_a_271_c_303o_EBANDS.agr','abinitOutputFiles/Co_FM_DOS_1e-4/bct_a_271_c_303o_EBANDS.agr'],8.617*315773*np.array([1e-3,8e-4,6e-4,4e-4,2e-4,1e-4]),20,2.94,2.94,2.46,21,21,31)
