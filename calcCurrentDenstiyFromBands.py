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
    #coeffTwo = -1./(4.*np.sqrt(np.pi))
    coeffTwo = 0

    gauss = np.exp(-((Ek - Ek_prime)**2)/(2*f*f))
    return (1./(f*np.sqrt(2.)))*gauss*(coeffZero*hermiteZero + coeffTwo*hermiteTwo)
"""
E = np.linspace(-.24,.24,101)
plt.scatter(E,fermi_deriv(E,8.617e-5*100),c='blue')
plt.scatter(E,fermi_deriv(E,8.617e-5*200),c='green')
plt.scatter(E,fermi_deriv(E,8.617e-5*300),c='red')
plt.show()
plt.scatter(E,deltaGaussHerm(0,E,.02*.24),c='red')
plt.scatter(E,deltaGaussHerm(0,E,.05*.24),c='blue')
plt.show()
#"""
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

def genInterpFuncs(kx,ky,kz,E,nbands,a,b,c,E_cutoff,N_k):
    Kx = np.concatenate((kx,ky,-kx,-ky,kx,ky,-kx,-ky,kx,ky,-kx,-ky,kx,ky,-kx,-ky))
    Ky = np.concatenate((ky,kx,ky,kx,-ky,-kx,-ky,-kx,ky,kx,ky,kx,-ky,-kx,-ky,-kx))
    Kz = np.concatenate((kz,kz,kz,kz,kz,kz,kz,kz,-kz,-kz,-kz,-kz,-kz,-kz,-kz,-kz))
    mask = (Kx > -.05) & (Ky > -.05) & (Kz > -.05)
    Kx = Kx[mask]
    Ky = Ky[mask]
    Kz = Kz[mask]
    E_k = []
    for i in range(0,2*nbands):
        E_k.append(np.concatenate((E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i],E[i]))[mask])
    E_k = np.array(E_k)
    
    sort = np.argsort(Kx)
    bandID = []
    interp_band_list = []
    for i in range(0,2*nbands):
        if len(E_k[i][np.abs(E_k[i]) < E_cutoff]) > 1:
            print(i,len(E_k[i][np.abs(E_k[i]) < E_cutoff]))
            bandID.append(i)
            interp_band_list.append(LinearNDInterpolator((Kx[sort],Ky[sort],Kz[sort]),E_k[i][sort]))
    
    x = np.linspace(0,.5,N_k)
    y = np.linspace(0,.5,N_k)
    z = np.linspace(0,.5,N_k)
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
    
    #return interp_band_list,bandID
    return interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID

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

def cosFitTestConstrained_3D(fitParams,constStr,bandStr,gradStr,xyz,E):
    kx,ky,kz,kx2,ky2,kz2 = xyz
    E,E2 = E
    #constStr = 1+4./(1.+np.exp(-.02*constStrParam))
    #gradStr = 2./(1.+np.exp(-.02*gradStrParam))

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
    """
    ret1 = np.sign(E - cosFitTest_3D(kx,ky,kz,fitParams,5))*np.abs(E - cosFitTest_3D(kx,ky,kz,fitParams,5))**(2./4.)
    ret2 = np.sign(dE/dk - vx*ux - vy*uy - vz*uz)*np.abs(dE/dk - vx*ux - vy*uy - vz*uz)**(2./4.)
    ret1 = ret1 + constStr*deltaGaussHerm(0,E,.08)*np.sign(E - cosFitTest_3D(kx,ky,kz,fitParams,5))*np.abs(E - cosFitTest_3D(kx,ky,kz,fitParams,5))**(2./4.)
    ret2 = ret2 + constStr*deltaGaussHerm(0,E,.08)*np.sign(dE/dk - vx*ux - vy*uy - vz*uz)*np.abs(dE/dk - vx*ux - vy*uy - vz*uz)**(2./4.)
    ret3 = np.sign(E2 - cosFitTest_3D(kx2,ky2,kz2,fitParams,5))*np.abs(E2 - cosFitTest_3D(kx2,ky2,kz2,fitParams,5))**(2./4.)
    ret3 = ret3 + constStr*deltaGaussHerm(0,E2,.08)*np.sign(E2 - cosFitTest_3D(kx2,ky2,kz2,fitParams,5))*np.abs(E2 - cosFitTest_3D(kx2,ky2,kz2,fitParams,5))**(2./4.)
    """
    ret1 = (E - cosFitTest_3D(kx,ky,kz,fitParams,5))
    ret2 = (dE/dk - vx*ux - vy*uy - vz*uz)
    #deltaGaussHerm was set to 0.08 for best results so far
    #ret1 = ret1 + constStr*deltaGaussHerm(0,E,.12)*(E - cosFitTest_3D(kx,ky,kz,fitParams,5))
    ret1 = ret1 + constStr*fermi_deriv(E,8.617e-5*300)*(E - cosFitTest_3D(kx,ky,kz,fitParams,5))
    #ret2 = ret2 + constStr*deltaGaussHerm(0,E,.12)*(dE/dk - vx*ux - vy*uy - vz*uz)
    ret2 = ret2 + constStr*fermi_deriv(E,8.617e-5*300)*(dE/dk - vx*ux - vy*uy - vz*uz)
    ret3 = (E2 - cosFitTest_3D(kx2,ky2,kz2,fitParams,5))
    #ret3 = ret3 + constStr*deltaGaussHerm(0,E2,.12)*(E2 - cosFitTest_3D(kx2,ky2,kz2,fitParams,5))
    ret3 = ret3 + constStr*fermi_deriv(E2,8.617e-5*300)*(E2 - cosFitTest_3D(kx2,ky2,kz2,fitParams,5))
    #return ret1 + np.sign(ret1)*np.abs(ret2)
    ret = []
    for i in range(0,len(kx)):
        #ret.append((ret1[i],gradStr*ret2[i]))
        ret.append(bandStr*ret1[i]*np.sqrt((436./896.)*(len(kx2)/len(kx))))
        ret.append(gradStr*ret2[i]*np.sqrt((436./896.)*(len(kx2)/len(kx))))
    for i in range(0,len(kx2)):
        ret.append(ret3[i])
    #return np.abs(ret1) + gradStr*np.abs(ret2)
    #print(np.sum(ret))
    return ret

def readBandFile(EBANDSFilename,EBANDSFilename2,nbands):
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
    
    os.system('./abinitEBANDSReader.sh ' + EBANDSFilename2 + ' kpts_tmp bands_tmp')
    with open('kpts_tmp','r') as myFile:
        read = csv.reader(myFile,delimiter = ',')
        kx2 = []
        ky2 = []
        kz2 = []
        for row in read:
            kx2.append(float(row[0]))
            ky2.append(float(row[1]))
            kz2.append(float(row[2]))
    
    kx2 = np.array(kx2)
    ky2 = np.array(ky2)
    kz2 = np.array(kz2)
    
    E2 = []
    for i in range(0,2*nbands):
        E2.append([])
    
    with open('bands_tmp','r') as myFile:
        read = csv.reader(myFile,delimiter = ' ')
        for row in read:
            for i in range(0,2*nbands):
                E2[i].append(float(row[i]))
    E2 = np.array(E2)
    return kx,ky,kz,E,kx2,ky2,kz2,E2

def genFitParams(kx,ky,kz,E,kx2,ky2,kz2,E2,nbands,E_cutoff):
    print(len(kx),len(kx2))
    bandID = []
    fitParamsList = []
    strFitParamsList = []
    for i in range(0,2*nbands):
        if ((len(E[i][np.abs(E[i]) < E_cutoff]) > 0) or (len(E2[i][np.abs(E2[i]) < E_cutoff]) > 0)):# and (i == 19):
            """
            E_c = E_cutoff
            mask = np.abs(E[i]) < E_c
            while len(E[i][mask]) < 75:
                E_c = 1.2 * E_c
                mask = np.abs(E[i]) < E_c
            """
            #sort = np.argsort(np.abs(E[i]))
            print(i,len(E[i][np.abs(E[i]) < E_cutoff]),len(E2[i][np.abs(E2[i]) < E_cutoff]))
            bandID.append(i)
            #poptInit = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(0,(kx,ky,kz),E[i]))[0]
            #popt = leastsq(cosFitTestConstrainedWrapper_3D,x0 = [-30,0],args=((kx,ky,kz,kx2,ky2,kz2),(E[i],E2[i])),epsfcn=.05,maxfev=20)[0]
            #strFitParamsList.append(popt)
            #popt = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(strFitParamsList[-1][0],strFitParamsList[-1][1],(kx,ky,kz,kx2,ky2,kz2),(E[i],E2[i])))[0]
            popt = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(10.,1.5,1.5,(kx,ky,kz,kx2,ky2,kz2),(E[i],E2[i])))[0]
            #popt = leastsq(cosFitTestConstrained_3D,x0 = np.ones(75),args=(7.9,.396,(kx,ky,kz),E[i]))[0]
            #popt,pcurve = curve_fit(cosFitTestWrapper_3D,(kx[sort],ky[sort],kz[sort]),E[i][sort])#,p0 = guess)
            fitParamsList.append(popt)
    fitParamsList = np.array(fitParamsList)
    return fitParamsList,bandID

def genEquipotentialFsolve(interp_band_func,interp_vx_func,interp_vy_func,interp_vz_func,E,dk,dkz_guess):
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
    
    Kz_grid = np.zeros(Kx_grid.shape)
    for i in range(0,len(Kx_grid)):
        for j in range(0,len(Kx_grid[i])):
            for l in range(0,j+1):
                Kz_grid[i][j][l] = .5/(1+np.exp(-.01*fsolve(lambda k: (E-interp_band_func(Kx_grid[i][j][l],Ky_grid[i][j][l],.5/(1+np.exp(-.01*k))))**2,x0=[100*np.log(2*Kz_guess[i][j][l]/(1-2*Kz_guess[i][j][l]))],factor=0.1)))
                
                Kz_grid[i][l][j] = Kz_grid[i][j][l]
                
    for i in range(0,len(Kx_grid[0])):
        for j in range(0,len(Kx_grid[0][i])):
            sort = np.argsort(Kz_grid[:,i,j])
            Kz_grid[:,i,j] = Kz_grid[:,i,j][sort]
    kx = []
    ky = []
    kz = []
    #dSk = []
    for i in range(0,len(kz_guess)):
        if i == 0:
            kx.append(Kx_grid[i].flatten())
            ky.append(Ky_grid[i].flatten())
            kz.append(Kz_grid[i].flatten())
            
        else:
            mask = (np.round(Kz_grid[0],3) != np.round(Kz_grid[i],3))
            
            for j in range(1,i):
                if i != j:
                    mask = mask & (np.round(Kz_grid[j],3) != np.round(Kz_grid[i],3))
            kx.append(Kx_grid[i][mask])
            ky.append(Ky_grid[i][mask])
            kz.append(Kz_grid[i][mask])
            
    kx_1d = []
    ky_1d = []
    kz_1d = []

    for i in range(0,len(kx)):
        for j in range(0,len(kx[i])):
            kx_1d.append(kx[i][j])
            ky_1d.append(ky[i][j])
            kz_1d.append(kz[i][j])
            
    kx = np.array(kx_1d)
    ky = np.array(ky_1d)
    kz = np.array(kz_1d)
    
    kx_midpoint = kx + dk/2
    ky_midpoint = ky + dk/2
    kz_midpoint = np.zeros(len(kx))

    for i in range(0,len(kx)):
        kz_midpoint[i] = .5/(1+np.exp(-.01*fsolve(lambda k: (E-interp_band_func(kx_midpoint[i],ky_midpoint[i],.5/(1+np.exp(-.01*k))))**2,x0=[100*np.log(2*kz[i]/(1-2*kz[i]))],factor=0.1)))

    vx = interp_vx_func(kx,ky,kz)
    vy = interp_vy_func(kx,ky,kz)
    vz = interp_vz_func(kx,ky,kz)
    dSk = dk*dk*np.sqrt(1. + (vx/vz)**2 + (vy/vz)**2)
    
    vx = interp_vx_func(kx_midpoint,ky_midpoint,kz_midpoint)
    vy = interp_vy_func(kx_midpoint,ky_midpoint,kz_midpoint)
    vz = interp_vz_func(kx_midpoint,ky_midpoint,kz_midpoint)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    mask = (kz_midpoint >= 0) & (kz_midpoint <= .5) & (kx_midpoint <= ky_midpoint) & (~np.isnan(v)) & (np.abs(v) > 1e-3) & ((E-interp_band_func(kx_midpoint,ky_midpoint,kz_midpoint))**2 < 1e-3) & (dSk < 10*dk*dk)
    
    kx_midpoint_symm = np.concatenate((kx_midpoint[mask],ky_midpoint[mask]))
    ky_midpoint_symm = np.concatenate((ky_midpoint[mask],kx_midpoint[mask]))
    kz_midpoint_symm = np.concatenate((kz_midpoint[mask],kz_midpoint[mask]))
    vx_symm = np.concatenate((vx[mask],vy[mask]))
    vy_symm = np.concatenate((vy[mask],vx[mask]))
    vz_symm = np.concatenate((vz[mask],vz[mask]))
    v_symm = np.concatenate((v[mask],v[mask]))
    dSk_symm = np.concatenate((dSk[mask],dSk[mask]))
    
    return kx_midpoint_symm,ky_midpoint_symm,kz_midpoint_symm,dSk_symm,vz_symm,v_symm
"""
kx,ky,kz,E = readBandFile('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr',20)
print('interpolating bands')
fitParamsList,bandID = genFitParams(kx,ky,kz,E,20,.6)
genEquipotentialFsolve(fitParamsList[0],0,.02,.0625)
#"""

def genEquipotentialGrid(interp_band_func_list,interp_vx_list,interp_vy_list,interp_vz_list,E_cutoff,N_E,N_theta,N_phi):
    E_grid = np.linspace(-E_cutoff,E_cutoff,N_E)

    kx = []
    ky = []
    kz = []
    dS = []
    vz = []
    v = []
    
    for i in range(0,len(interp_band_func_list)):
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
            kx_tmp,ky_tmp,kz_tmp,dS_tmp,vz_tmp,v_tmp = genEquipotentialFsolve(interp_band_func_list[i],interp_vx_list[i],interp_vy_list[i],interp_vz_list[i],E_grid[j],.015625,.125)
            print(i,E_grid[j],len(kx_tmp))
            kx[i][j] = kx_tmp
            ky[i][j] = ky_tmp
            kz[i][j] = kz_tmp
            dS[i][j] = dS_tmp
            vz[i][j] = vz_tmp
            v[i][j] = v_tmp
            
    return E_grid,dS,vz,v

#def genTetrahedronDOS(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID,nbands):
    

def genDOS(bandID,nbands,dS_k,v_k,E_k,E_cut,N_E):
    dE = np.abs(E_k[1] - E_k[0])
    E = np.linspace(-E_cut,E_cut,N_E)
    N_up = np.zeros(E.shape)
    N_down = np.zeros(E.shape)
    for n in range(0,N_E):
        for i in range(0,len(bandID)):
            NF = 0.
            for j in range(0,len(E_k)):
                NF = NF + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*deltaGaussHerm(E_k[j],E[n],.05*E_k[-1])/v_k[i][j][v_k[i][j] != 0.])
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

def integrateBands(E_k,dS_k,vz_k,v_k,bandID,nbands,T):
    NF_up = 0.
    NF_down = 0.
    
    sigmaz_up0 = 0.
    sigmaz_down0 = 0.
    
    sigmaz_up1 = 0.
    sigmaz_down1 = 0.
    
    sigmaz_up2 = 0.
    sigmaz_down2 = 0.
    
    sigmaz_up5 = 0.
    sigmaz_down5 = 0.
    
    sigmaz_up7 = 0.
    sigmaz_down7 = 0.
    
    sigmaz_up10 = 0.
    sigmaz_down10 = 0.
    
    sigmaz_up15 = 0.
    sigmaz_down15 = 0.
    
    sigmaz_up20 = 0.
    sigmaz_down20 = 0.
    
    sigmaz_up30 = 0.
    sigmaz_down30 = 0.
    
    sigmaz_up40 = 0.
    sigmaz_down40 = 0.
    
    sigmaz_up50 = 0.
    sigmaz_down50 = 0.

    dE = np.abs(E_k[1] - E_k[0])
    
    tau = []
    """
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
    """

    print('calculating better conductivity')
    for i in range(0,len(bandID)):
        NF = 0.
        #sigma = 0.
        for j in range(0,len(E_k)):
            NF = NF + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*deltaGaussHerm(E_k[j],0.,.05*E_k[-1])/v_k[i][j][v_k[i][j] != 0.])
            #sigma = sigma + dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*tau[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/v_k[i][j][v_k[i][j] != 0.])
            #print(i,j,NF,sigma)
        if bandID[i] < nbands:
            NF_up = NF_up + NF
            #sigmaz_up = sigmaz_up + sigma
        else:
            NF_down = NF_down + NF
            #sigmaz_down = sigmaz_down + sigma
    for i in range(0,len(bandID)):
        for j in range(0,len(E_k)):
            if bandID[i] < nbands:
                tau_up0 = 1./(NF_up)
                sigmaz_up0 = sigmaz_up0 + tau_up0*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up1 = 1./(.99*NF_up+.01*NF_down)
                sigmaz_up1 = sigmaz_up1 + tau_up1*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up2 = 1./(.98*NF_up+.02*NF_down)
                sigmaz_up2 = sigmaz_up2 + tau_up2*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up5 = 1./(.95*NF_up+.05*NF_down)
                sigmaz_up5 = sigmaz_up5 + tau_up5*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up7 = 1./(.93*NF_up+.07*NF_down)
                sigmaz_up7 = sigmaz_up7 + tau_up7*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up10 = 1./(.9*NF_up+.1*NF_down)
                sigmaz_up10 = sigmaz_up10 + tau_up10*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up15 = 1./(.85*NF_up+.15*NF_down)
                sigmaz_up15 = sigmaz_up15 + tau_up15*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up20 = 1./(.8*NF_up+.2*NF_down)
                sigmaz_up20 = sigmaz_up20 + tau_up20*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up30 = 1./(.7*NF_up+.3*NF_down)
                sigmaz_up30 = sigmaz_up30 + tau_up30*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up40 = 1./(.6*NF_up+.4*NF_down)
                sigmaz_up40 = sigmaz_up40 + tau_up40*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_up50 = 1./(.5*NF_up+.5*NF_down)
                sigmaz_up50 = sigmaz_up50 + tau_up50*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
            else:
                tau_down0 = 1./(NF_down)
                sigmaz_down0 = sigmaz_down0 + tau_down0*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down1 = 1./(.99*NF_down+.01*NF_up)
                sigmaz_down1 = sigmaz_down1 + tau_down1*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down2 = 1./(.98*NF_down+.02*NF_up)
                sigmaz_down2 = sigmaz_down2 + tau_down2*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down5 = 1./(.95*NF_down+.05*NF_up)
                sigmaz_down5 = sigmaz_down5 + tau_down5*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down7 = 1./(.93*NF_down+.07*NF_up)
                sigmaz_down7 = sigmaz_down7 + tau_down7*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down10 = 1./(.9*NF_down+.1*NF_up)
                sigmaz_down10 = sigmaz_down10 + tau_down10*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down15 = 1./(.85*NF_down+.15*NF_up)
                sigmaz_down15 = sigmaz_down15 + tau_down15*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down20 = 1./(.8*NF_down+.2*NF_up)
                sigmaz_down20 = sigmaz_down20 + tau_down20*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down30 = 1./(.7*NF_down+.3*NF_up)
                sigmaz_down30 = sigmaz_down30 + tau_down30*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down40 = 1./(.6*NF_down+.4*NF_up)
                sigmaz_down40 = sigmaz_down40 + tau_down40*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
                
                tau_down50 = 1./(.5*NF_down+.5*NF_up)
                sigmaz_down50 = sigmaz_down50 + tau_down50*dE*np.sum(dS_k[i][j][v_k[i][j] != 0.]*fermi_deriv(E_k[j],T)*(vz_k[i][j][v_k[i][j] != 0.])**2/(v_k[i][j][v_k[i][j] != 0.]))
    
    return NF_up,NF_down,(NF_up-NF_down)/(NF_up+NF_down),sigmaz_up0,sigmaz_down0,(sigmaz_up0-sigmaz_down0)/(sigmaz_up0+sigmaz_down0),sigmaz_up1,sigmaz_down1,(sigmaz_up1-sigmaz_down1)/(sigmaz_up1+sigmaz_down1),sigmaz_up2,sigmaz_down2,(sigmaz_up2-sigmaz_down2)/(sigmaz_up2+sigmaz_down2),sigmaz_up5,sigmaz_down5,(sigmaz_up5-sigmaz_down5)/(sigmaz_up5+sigmaz_down5),sigmaz_up7,sigmaz_down7,(sigmaz_up7-sigmaz_down7)/(sigmaz_up7+sigmaz_down7),sigmaz_up10,sigmaz_down10,(sigmaz_up10-sigmaz_down10)/(sigmaz_up10+sigmaz_down10),sigmaz_up15,sigmaz_down15,(sigmaz_up15-sigmaz_down15)/(sigmaz_up15+sigmaz_down15),sigmaz_up20,sigmaz_down20,(sigmaz_up20-sigmaz_down20)/(sigmaz_up20+sigmaz_down20),sigmaz_up30,sigmaz_down30,(sigmaz_up30-sigmaz_down30)/(sigmaz_up30+sigmaz_down30),sigmaz_up40,sigmaz_down40,(sigmaz_up40-sigmaz_down40)/(sigmaz_up40+sigmaz_down40),sigmaz_up50,sigmaz_down50,(sigmaz_up50-sigmaz_down50)/(sigmaz_up50+sigmaz_down50)

def main_plotBands(EBANDSFilename,EBANDSFilename2,nbands,a,c,E_cutoff,N_theta,N_phi,N_E,N_k = 70,checkFit = False):
    n_fit = 5
    print('importing bands from file')
    kx,ky,kz,E,kx2,ky2,kz2,E2 = readBandFile(EBANDSFilename,EBANDSFilename2,nbands)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(kx2,ky2,kz2)
    plt.show()

    Kx = np.concatenate((kx,kx2))
    Ky = np.concatenate((ky,ky2))
    Kz = np.concatenate((kz,kz2))
    En = []
    for i in range(0,len(E)):
        En.append(np.concatenate((E[i],E2[i])))
    En = np.array(En)
    print('interpolating bands')
    interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID = genInterpFuncs(Kx,Ky,Kz,En,nbands,a,a,c,E_cutoff,int(N_k))
    
    #fitParamsList,bandID = genFitParams(kx,ky,kz,E,kx2,ky2,kz2,E2,nbands,E_cutoff)
    """
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
    """
    print('generating integration grid')
    #kx,ky,kz,dkx,dky,dkz,E_k,vz = genEquipotentialAdaptiveCartesianGrid(fitParamsList,E_cutoff,NUM)
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,E_cutoff,N_E,N_theta,N_phi)
    #print('calculating DOS')
    #E,N_up,N_down = genDOSAdaptiveCartesian(bandID,nbands,dkx,dky,dkz,E_k,E_cutoff,2*N_E)
    E,N_up,N_down = genDOS(bandID,nbands,dS_k,v_k,E_k,E_cutoff,2*N_E)
    
    print('DOS Results: ')
    print(' ')
    for i in range(0,len(E)):
        print(E[i],N_up[i],N_down[i])
    
#main_plotBands('abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS2_EBANDS.agr','abinitOutputFiles/Co_FM_bandStruct/bct_a_291_c_249o_DS1_EBANDS.agr',20,2.91,2.49,.24,15,15,101,N_k=50,checkFit=False)
#main_plotBands('abinitOutputFiles/Co_FM_bandStruct2/bct_a_276_c_280o_DS2_EBANDS.agr','abinitOutputFiles/Co_FM_bandStruct2/bct_a_276_c_280o_DS1_EBANDS.agr',20,2.76,2.80,.24,15,15,101,N_k=50,checkFit=False)

#main_plotBands('abinitOutputFiles/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_c_250o_DS2_EBANDS.agr','abinitOutputFiles/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_c_250o_DS1_EBANDS.agr',160,2.92,2.50,.24,15,15,101,N_k=50,checkFit=False)
#main_plotBands('abinitOutputFiles/Co75Mn25I_symm_FM_bandStruct2/bct_a_280_c_267o_DS2_EBANDS.agr','abinitOutputFiles/Co75Mn25I_symm_FM_bandStruct2/bct_a_280_c_267o_DS1_EBANDS.agr',160,2.80,2.67,.24,15,15,101,N_k=50,checkFit=False)

def DSP(EBANDSFilename,EBANDSFilename2,nbands,a,c,E_cutoff,T,N_theta,N_phi,N_E,N_k = 50):
    print('importing bands from file')
    kx,ky,kz,E,kx2,ky2,kz2,E2 = readBandFile(EBANDSFilename,EBANDSFilename2,nbands)
    Kx = np.concatenate((kx,kx2))
    Ky = np.concatenate((ky,ky2))
    Kz = np.concatenate((kz,kz2))
    En = []
    for i in range(0,len(E)):
        En.append(np.concatenate((E[i],E2[i])))
    En = np.array(En)
    print('interpolating bands')
    interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,bandID = genInterpFuncs(Kx,Ky,Kz,En,nbands,a,a,c,E_cutoff,int(N_k))
    #fitParamsList,bandID = genFitParams(kx,ky,kz,E,kx2,ky2,kz2,E2,nbands,E_cutoff)
    #E_k,dS_k,vz_k,v_k = genEquipotentialGrid(fitParamsList,E_cutoff,N_E,N_theta,N_phi)
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(interp_band_list,interp_vx_list,interp_vy_list,interp_vz_list,E_cutoff,N_E,N_theta,N_phi)
    NF_up,NF_down,PN,sigmaz_up0,sigmaz_down0,PNv20,sigmaz_up1,sigmaz_down1,PNv21,sigmaz_up2,sigmaz_down2,PNv22,sigmaz_up5,sigmaz_down5,PNv25,sigmaz_up7,sigmaz_down7,PNv27,sigmaz_up10,sigmaz_down10,PNv210,sigmaz_up15,sigmaz_down15,PNv215,sigmaz_up20,sigmaz_down20,PNv220,sigmaz_up30,sigmaz_down30,PNv230,sigmaz_up40,sigmaz_down40,PNv240,sigmaz_up50,sigmaz_down50,PNv250 = integrateBands(E_k,dS_k,vz_k,v_k,bandID,nbands,T)
    return [NF_up, NF_down, PN, sigmaz_up0, sigmaz_down0, PNv20,sigmaz_up1,sigmaz_down1,PNv21,sigmaz_up2,sigmaz_down2,PNv22,sigmaz_up5,sigmaz_down5,PNv25,sigmaz_up7,sigmaz_down7,PNv27,sigmaz_up10,sigmaz_down10,PNv210,sigmaz_up15,sigmaz_down15,PNv215,sigmaz_up20,sigmaz_down20,PNv220,sigmaz_up30,sigmaz_down30,PNv230,sigmaz_up40,sigmaz_down40,PNv240,sigmaz_up50,sigmaz_down50,PNv250]

def main(EBANDSDirectory,T,nbands,N_theta,N_phi,N_E,E_cut):
    EBANDSFilenameList = []
    os.system('ls ' + EBANDSDirectory + '*_DS2_EBANDS.agr > tmpFileList')
    with open('tmpFileList','r') as myFile:
        read = csv.reader(myFile)
        for row in read:
            EBANDSFilenameList.append(row[0])
    os.system('rm tmpFileList')
    EBANDSFilenameList2 = []
    os.system('ls ' + EBANDSDirectory + '*_DS1_EBANDS.agr > tmpFileList')
    with open('tmpFileList','r') as myFile:
        read = csv.reader(myFile)
        for row in read:
            EBANDSFilenameList2.append(row[0])
    os.system('rm tmpFileList')
    
    EBANDSFilenameList = np.array(EBANDSFilenameList)
    EBANDSFilenameList2 = np.array(EBANDSFilenameList2)
    
    a = np.array([2.71,2.72,2.73,2.75,2.78,2.79,2.8,2.81,2.82,2.83,2.84,2.85,2.86,2.87,2.88,2.89,2.90,2.91,2.92,2.93,2.94,2.95])
    c = np.array([3.06,3.03,3.00,2.92,2.75,2.70,2.67,2.64,2.62,2.60,2.59,2.57,2.56,2.55,2.54,2.53,2.52,2.51,2.50,2.49,2.49,2.48])
    
    #a = np.round(np.arange(2.71,2.96,.01),2)
    #c = np.array([3.03,3.00,2.96,2.90,2.85,2.80,2.77,2.69,2.67,2.64,2.61,2.59,2.58,2.56,2.55,2.54,2.53,2.51,2.50,2.50,2.49,2.48,2.47,2.46,2.46])
    
    #EBANDSFilenameList = EBANDSFilenameList[4:]
    #EBANDSFilenameList2 = EBANDSFilenameList2[4:]
    #a = a[4:]
    #c = c[4:]
    
    idx = np.where((a==2.87) | (a==2.88))
    EBANDSFilenameList = EBANDSFilenameList[idx]
    EBANDSFilenameList2 = EBANDSFilenameList2[idx]
    a = a[idx]
    c = c[idx]
    """
    EBANDSFilenameList = EBANDSFilenameList[7:]
    EBANDSFilenameList2 = EBANDSFilenameList2[7:]
    a = a[7:]
    c = c[7:]
    """
    for i in range(0,len(EBANDSFilenameList)):
        print(EBANDSFilenameList[i],EBANDSFilenameList2[i],a[i],c[i])
    #NOTE: should find a way to extract the a- & c-values from the 
    # filenames, this won't really matter though since I'm not 
    # plotting bands
    #T = 8.617e-5*np.linspace(10,300,30)
    tol = 1e-8
    #E_cut = T*np.arccosh(1/np.sqrt(4*T*tol))
    
    NF_up = np.zeros(len(EBANDSFilenameList))
    NF_down = np.zeros(len(EBANDSFilenameList))
    PN = np.zeros(len(EBANDSFilenameList))
    sigmaz_up0 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down0 = np.zeros(len(EBANDSFilenameList))
    PNv20 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up1 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down1 = np.zeros(len(EBANDSFilenameList))
    PNv21 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up2 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down2 = np.zeros(len(EBANDSFilenameList))
    PNv22 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up5 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down5 = np.zeros(len(EBANDSFilenameList))
    PNv25 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up7 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down7 = np.zeros(len(EBANDSFilenameList))
    PNv27 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up10 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down10 = np.zeros(len(EBANDSFilenameList))
    PNv210 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up15 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down15 = np.zeros(len(EBANDSFilenameList))
    PNv215 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up20 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down20 = np.zeros(len(EBANDSFilenameList))
    PNv220 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up30 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down30 = np.zeros(len(EBANDSFilenameList))
    PNv230 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up40 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down40 = np.zeros(len(EBANDSFilenameList))
    PNv240 = np.zeros(len(EBANDSFilenameList))
    sigmaz_up50 = np.zeros(len(EBANDSFilenameList))
    sigmaz_down50 = np.zeros(len(EBANDSFilenameList))
    PNv250 = np.zeros(len(EBANDSFilenameList))
    
    for i in range(0,len(EBANDSFilenameList)):
        results = DSP(EBANDSFilenameList[i],EBANDSFilenameList2[i],nbands,a[i],c[i],E_cut,T,N_theta,N_phi,N_E)
        NF_up[i] = results[0]
        NF_down[i] = results[1]
        PN[i] = results[2]
        sigmaz_up0[i] = results[3]
        sigmaz_down0[i] = results[4]
        PNv20[i] = results[5]
        sigmaz_up1[i] = results[6]
        sigmaz_down1[i] = results[7]
        PNv21[i] = results[8]
        sigmaz_up2[i] = results[9]
        sigmaz_down2[i] = results[10]
        PNv22[i] = results[11]
        sigmaz_up5[i] = results[12]
        sigmaz_down5[i] = results[13]
        PNv25[i] = results[14]
        sigmaz_up7[i] = results[15]
        sigmaz_down7[i] = results[16]
        PNv27[i] = results[17]
        sigmaz_up10[i] = results[18]
        sigmaz_down10[i] = results[19]
        PNv210[i] = results[20]
        sigmaz_up15[i] = results[21]
        sigmaz_down15[i] = results[22]
        PNv215[i] = results[23]
        sigmaz_up20[i] = results[24]
        sigmaz_down20[i] = results[25]
        PNv220[i] = results[26]
        sigmaz_up30[i] = results[27]
        sigmaz_down30[i] = results[28]
        PNv230[i] = results[29]
        sigmaz_up40[i] = results[30]
        sigmaz_down40[i] = results[31]
        PNv240[i] = results[32]
        sigmaz_up50[i] = results[33]
        sigmaz_down50[i] = results[34]
        PNv250[i] = results[35]
        print(EBANDSFilenameList[i], a[i], c[i], NF_up[i], NF_down[i], PN[i], sigmaz_up0[i], sigmaz_down0[i], PNv20[i], sigmaz_up1[i], sigmaz_down1[i], PNv21[i], sigmaz_up2[i], sigmaz_down2[i], PNv22[i], sigmaz_up5[i], sigmaz_down5[i], PNv25[i], sigmaz_up7[i], sigmaz_down7[i], PNv27[i], sigmaz_up10[i], sigmaz_down10[i], PNv210[i], sigmaz_up15[i], sigmaz_down15[i], PNv215[i], sigmaz_up20[i], sigmaz_down20[i], PNv220[i], sigmaz_up30[i], sigmaz_down30[i], PNv230[i], sigmaz_up40[i], sigmaz_down40[i], PNv240[i], sigmaz_up50[i], sigmaz_down50[i], PNv250[i])
        
    print('final results')
    for i in range(0,len(EBANDSFilenameList)):
        print(EBANDSFilenameList[i], a[i], c[i], NF_up[i], NF_down[i], PN[i], sigmaz_up0[i], sigmaz_down0[i], PNv20[i], sigmaz_up1[i], sigmaz_down1[i], PNv21[i], sigmaz_up2[i], sigmaz_down2[i], PNv22[i], sigmaz_up5[i], sigmaz_down5[i], PNv25[i], sigmaz_up7[i], sigmaz_down7[i], PNv27[i], sigmaz_up10[i], sigmaz_down10[i], PNv210[i], sigmaz_up15[i], sigmaz_down15[i], PNv215[i], sigmaz_up20[i], sigmaz_down20[i], PNv220[i], sigmaz_up30[i], sigmaz_down30[i], PNv230[i], sigmaz_up40[i], sigmaz_down40[i], PNv240[i], sigmaz_up50[i], sigmaz_down50[i], PNv250[i])
        #print(EBANDSFilenameList[i], NF_up[i], NF_down[i], PN[i], sigmaz_up[i], sigmaz_down[i], PNv2[i])
    
#main('abinitOutputFiles/Co_FM_bandStruct2/',8.617e-5*100,20,15,15,101,0.1)
main('abinitOutputFiles/Co75Mn25I_symm_FM_noCon_bandStruct2/',8.617e-5*100,160,15,15,101,0.1)
