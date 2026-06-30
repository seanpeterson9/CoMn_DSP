from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
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

def vk(interp_func,kx,ky,kz,dk = 1e-6):
    vx = np.zeros(kx.shape)
    vy = np.zeros(kx.shape)
    vz = np.zeros(kx.shape)
    
    mask_x1 = kx < dk
    mask_x2 = kx > .5 - dk
    mask_y1 = ky < dk
    mask_y2 = ky > .5 - dk
    mask_z1 = kz < dk
    mask_z2 = kz > .5 - dk
    
    vx = (interp_func(np.column_stack([kx+dk,ky,kz])) - interp_func(np.column_stack([kx-dk,ky,kz]))) / (2*dk)
    vy = (interp_func(np.column_stack([kx,ky+dk,kz])) - interp_func(np.column_stack([kx,ky-dk,kz]))) / (2*dk)
    vz = (interp_func(np.column_stack([kx,ky,kz+dk])) - interp_func(np.column_stack([kx,ky,kz-dk]))) / (2*dk)
    
    vx[mask_x1] = (interp_func(np.column_stack([kx[mask_x1]+dk,ky[mask_x1],kz[mask_x1]])) - interp_func(np.column_stack([kx[mask_x1],ky[mask_x1],kz[mask_x1]]))) / (dk)
    vx[mask_x2] = (interp_func(np.column_stack([kx[mask_x2],ky[mask_x2],kz[mask_x2]])) - interp_func(np.column_stack([kx[mask_x2]-dk,ky[mask_x2],kz[mask_x2]]))) / (dk)
    
    vy[mask_y1] = (interp_func(np.column_stack([kx[mask_y1],ky[mask_y1]+dk,kz[mask_y1]])) - interp_func(np.column_stack([kx[mask_y1],ky[mask_y1],kz[mask_y1]]))) / (dk)
    vy[mask_y2] = (interp_func(np.column_stack([kx[mask_y2],ky[mask_y2],kz[mask_y2]])) - interp_func(np.column_stack([kx[mask_y2],ky[mask_y2]-dk,kz[mask_y2]]))) / (dk)
    
    vz[mask_z1] = (interp_func(np.column_stack([kx[mask_z1],ky[mask_z1],kz[mask_z1]+dk])) - interp_func(np.column_stack([kx[mask_z1],ky[mask_z1],kz[mask_z1]]))) / (dk)
    vz[mask_z2] = (interp_func(np.column_stack([kx[mask_z2],ky[mask_z2],kz[mask_z2]])) - interp_func(np.column_stack([kx[mask_z2],ky[mask_z2],kz[mask_z2]-dk]))) / (dk)
    return vx, vy, vz

def genInterpFuncs(kpoints,kx,ky,kz,E,nbands,a,b,c,E_cutoff,N_k):
    """
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
    """
    E = np.array(E)
    bandID = []
    interp_band_list = []
    
    kx_vals = np.unique(np.round(kx, 6))
    ky_vals = np.unique(np.round(ky, 6))
    kz_vals = np.unique(np.round(kz, 6))
    
    for i in range(0,2*nbands):
        if len(E[i][np.abs(E[i]) < E_cutoff]) > 1:
            print(i,len(E[i][np.abs(E[i]) < E_cutoff]))
            bandID.append(i)
            #interp_band_list.append(RegularGridInterpolator((Kx[sort],Ky[sort],Kz[sort]),E_k[i][sort]))
            
            E_3d = np.zeros((len(kx_vals), len(ky_vals), len(kz_vals)))

            for idx, (Kx, Ky, Kz) in enumerate(kpoints):
                ix = np.searchsorted(kx_vals, round(Kx, 6))
                iy = np.searchsorted(ky_vals, round(Ky, 6))
                iz = np.searchsorted(kz_vals, round(Kz, 6))
                E_3d[ix, iy, iz] = E[i][idx]
            
            #interp_band_list.append(RegularGridInterpolator((np.array(kx),np.array(ky),np.array(kz)),np.array(E[i])))
            interp_band_list.append(RegularGridInterpolator((kx_vals,ky_vals,kz_vals),E_3d,method='linear',bounds_error=False,fill_value=None))
    
    return interp_band_list,bandID

def readBandFile(Filename):
    bandStruct = nlread(Filename)[-1]
    bands = bandStruct.evaluate().inUnitsOf(eV)
    
    kx = np.array(bandStruct.kpoints()[:,0])
    ky = np.array(bandStruct.kpoints()[:,1])
    kz = np.array(bandStruct.kpoints()[:,2])
    E = []
    nbands = len(bands[0][0])
    for i in range(0,2*nbands):
        if i < nbands:
            E.append(bands[0][:,i])
        else:
            E.append(bands[1][:,i-nbands])
    E = np.array(E)
    return np.array(bandStruct.kpoints()),kx,ky,kz,E,nbands

def genEquipotentialFsolve(interp_band_func,E,dk,dkz_guess):
    
    kx = np.arange(0.,0.5,dk) + dk/2
    ky = np.arange(0.,0.5,dk) + dk/2
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
                Kz_grid[i][j][l] = .5/(1+np.exp(-.01*fsolve(lambda k: (E-interp_band_func((Kx_grid[i][j][l],Ky_grid[i][j][l],.5/(1+np.exp(-.01*k)))))**2,x0=[100*np.log(2*Kz_guess[i][j][l]/(1-2*Kz_guess[i][j][l]))],factor=0.1)))
                
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
            mask = (np.round(Kz_grid[0],5) != np.round(Kz_grid[i],5))
            
            for j in range(1,i):
                if i != j:
                    mask = mask & (np.round(Kz_grid[j],5) != np.round(Kz_grid[i],5))
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
    """
    kx_midpoint = kx + dk/2
    ky_midpoint = ky + dk/2
    kz_midpoint = np.zeros(len(kx))

    for i in range(0,len(kx)):
        kz_midpoint[i] = .5/(1+np.exp(-.01*fsolve(lambda k: (E-interp_band_func((kx_midpoint[i],ky_midpoint[i],.5/(1+np.exp(-.01*k)))))**2,x0=[100*np.log(2*kz[i]/(1-2*kz[i]))],factor=0.1)))
    """
    vx,vy,vz = vk(interp_band_func,kx,ky,kz)
    dSk = dk*dk*np.sqrt(1. + (vx/vz)**2 + (vy/vz)**2)
    
    #vx,vy,vz = vk(interp_band_func,kx_midpoint,ky_midpoint,kz_midpoint)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    #mask = (kz_midpoint >= 0) & (kz_midpoint <= .5) & (kx_midpoint <= ky_midpoint) & (~np.isnan(v)) & (np.abs(v) > 1e-5) & ((E-interp_band_func(np.column_stack([kx_midpoint,ky_midpoint,kz_midpoint])))**2 < 1e-5) & (dSk < 10*dk*dk)
    mask = (kz >= 0) & (kz <= .5) & (~np.isnan(v)) & (np.abs(v) > 1e-5) & ((E-interp_band_func(np.column_stack([kx,ky,kz])))**2 < 1e-5) & (dSk < 10*dk*dk)
    
    #kx_midpoint_symm = np.concatenate((kx_midpoint[mask],ky_midpoint[mask]))
    #ky_midpoint_symm = np.concatenate((ky_midpoint[mask],kx_midpoint[mask]))
    #kz_midpoint_symm = np.concatenate((kz_midpoint[mask],kz_midpoint[mask]))
    kx_symm = np.concatenate((kx[mask],ky[mask]))
    ky_symm = np.concatenate((ky[mask],kx[mask]))
    kz_symm = np.concatenate((kz[mask],kz[mask]))
    vx_symm = np.concatenate((vx[mask],vy[mask]))
    vy_symm = np.concatenate((vy[mask],vx[mask]))
    vz_symm = np.concatenate((vz[mask],vz[mask]))
    v_symm = np.concatenate((v[mask],v[mask]))
    dSk_symm = np.concatenate((dSk[mask],dSk[mask]))
    
    #return kx_midpoint_symm,ky_midpoint_symm,kz_midpoint_symm,dSk_symm,vz_symm,v_symm
    #return kx_symm,ky_symm,kz_symm,dSk_symm,vz_symm,v_symm
    return kx[mask],ky[mask],kz[mask],dSk[mask],vz[mask],v[mask]

def genEquipotentialGrid(interp_band_func_list,E_cutoff,N_E,N_theta,N_phi):
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
            #kx_tmp,ky_tmp,kz_tmp,dS_tmp,vz_tmp,v_tmp = genEquipotentialFsolve(interp_band_func_list[i],E_grid[j],.015625,.125)
            kx_tmp,ky_tmp,kz_tmp,dS_tmp,vz_tmp,v_tmp = genEquipotentialFsolve(interp_band_func_list[i],E_grid[j],.01,.125)
            print(i,E_grid[j],len(kx_tmp))
            kx[i][j] = kx_tmp
            ky[i][j] = ky_tmp
            kz[i][j] = kz_tmp
            dS[i][j] = dS_tmp
            vz[i][j] = vz_tmp
            v[i][j] = v_tmp
            
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

def main_plotBands(Filename,a,c,E_cutoff,N_theta,N_phi,N_E,N_k = 50,checkFit = False):
    n_fit = 5
    print('importing bands from file')
    kpoints,kx,ky,kz,E,nbands = readBandFile(Filename)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(kx,ky,kz)
    plt.show()

    print('interpolating bands')
    interp_band_list,bandID = genInterpFuncs(kpoints,kx,ky,kz,E,nbands,a,a,c,E_cutoff,int(N_k))
    
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
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(interp_band_list,E_cutoff,N_E,N_theta,N_phi)
    E,N_up,N_down = genDOS(bandID,nbands,dS_k,v_k,E_k,E_cutoff,2*N_E)
    
    print('DOS Results: ')
    print(' ')
    for i in range(0,len(E)):
        print(E[i],N_up[i],N_down[i])
    
#main_plotBands('./qatkDatFiles/atomicMag/Co_bulk_new/a_277.hdf5',2.77,2.77,.2,15,15,51)

def DSP(Filename,a,c,E_cutoff,T,N_theta,N_phi,N_E,N_k = 50):
    print('importing bands from file')
    kpoints,kx,ky,kz,E,nbands = readBandFile(Filename)
    print('interpolating bands')
    interp_band_list,bandID = genInterpFuncs(kpoints,kx,ky,kz,E,nbands,a,a,c,E_cutoff,int(N_k))
    
    E_k,dS_k,vz_k,v_k = genEquipotentialGrid(interp_band_list,E_cutoff,N_E,N_theta,N_phi)
    NF_up,NF_down,PN,sigmaz_up0,sigmaz_down0,PNv20,sigmaz_up1,sigmaz_down1,PNv21,sigmaz_up2,sigmaz_down2,PNv22,sigmaz_up5,sigmaz_down5,PNv25,sigmaz_up7,sigmaz_down7,PNv27,sigmaz_up10,sigmaz_down10,PNv210,sigmaz_up15,sigmaz_down15,PNv215,sigmaz_up20,sigmaz_down20,PNv220,sigmaz_up30,sigmaz_down30,PNv230,sigmaz_up40,sigmaz_down40,PNv240,sigmaz_up50,sigmaz_down50,PNv250 = integrateBands(E_k,dS_k,vz_k,v_k,bandID,nbands,T)
    return [NF_up, NF_down, PN, sigmaz_up0, sigmaz_down0, PNv20,sigmaz_up1,sigmaz_down1,PNv21,sigmaz_up2,sigmaz_down2,PNv22,sigmaz_up5,sigmaz_down5,PNv25,sigmaz_up7,sigmaz_down7,PNv27,sigmaz_up10,sigmaz_down10,PNv210,sigmaz_up15,sigmaz_down15,PNv215,sigmaz_up20,sigmaz_down20,PNv220,sigmaz_up30,sigmaz_down30,PNv230,sigmaz_up40,sigmaz_down40,PNv240,sigmaz_up50,sigmaz_down50,PNv250]

#print(DSP('./qatkDatFiles/atomicMag/Co_bulk_new/a_277.hdf5',2.77,2.77,0.1,8.617e-5*100,15,15,31))

def main(EBANDSDirectory,T,nbands,N_theta,N_phi,N_E,E_cut):
    EBANDSFilenameList = []
    os.system('ls ' + EBANDSDirectory + '*.hdf5 > tmpFileList')
    with open('tmpFileList','r') as myFile:
        read = csv.reader(myFile)
        for row in read:
            EBANDSFilenameList.append(row[0])
    os.system('rm tmpFileList')
    
    EBANDSFilenameList = np.array(EBANDSFilenameList)
    a = np.zeros(len(EBANDSFilenameList))
    c = np.zeros(len(EBANDSFilenameList))
    for i in range(0,len(EBANDSFilenameList)):
        bulkConfig = nlread(EBANDSFilenameList[i])[0]
        a[i] = bulkConfig.primitiveVectors()[0][0].inUnitsOf(Angstrom)
        c[i] = bulkConfig.primitiveVectors()[2][2].inUnitsOf(Angstrom)
    
    #a = np.array([2.71,2.72,2.73,2.74,2.75,2.76,2.77,2.78,2.79,2.8,2.81,2.82,2.83,2.84,2.85,2.86,2.87,2.88,2.89,2.90,2.91,2.92,2.93,2.94,2.95])
    #c = np.array([3.05,3.02,2.99,2.95,2.89,2.83,2.77,2.72,2.68,2.65,2.62,2.6,2.58,2.56,2.54,2.53,2.52,2.51,2.5,2.49,2.48,2.48,2.47,2.46,2.46])
    
    #a = np.round(np.arange(2.71,2.96,.01),2)
    #c = np.array([3.03,3.00,2.96,2.90,2.85,2.80,2.77,2.69,2.67,2.64,2.61,2.59,2.58,2.56,2.55,2.54,2.53,2.51,2.50,2.50,2.49,2.48,2.47,2.46,2.46])
    
    idx = np.where((a>2.72))
    EBANDSFilenameList = EBANDSFilenameList[idx]
    a = a[idx]
    c = c[idx]
    
    for i in range(0,len(EBANDSFilenameList)):
        print(EBANDSFilenameList[i],a[i],c[i])
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
        results = DSP(EBANDSFilenameList[i],a[i],c[i],E_cut,T,N_theta,N_phi,N_E)
        
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
        f = open('DSP_Co_auto_01.dat','a')
        f.write(str(EBANDSFilenameList[i]) + '\t' + str(a[i]) + '\t' +  str(c[i]) + '\t' + str(NF_up[i]) + '\t' + str(NF_down[i]) + '\t' +  str(PN[i]) + '\t' + str(sigmaz_up0[i]) + '\t' + str(sigmaz_down0[i]) + '\t' + str(PNv20[i]) + '\t' + str(sigmaz_up1[i]) + '\t' + str(sigmaz_down1[i]) + '\t' + str(PNv21[i]) + '\t' + str(sigmaz_up2[i]) + '\t' + str(sigmaz_down2[i]) + '\t' + str(PNv22[i]) + '\t' + str(sigmaz_up5[i]) + '\t' + str(sigmaz_down5[i]) + '\t' + str(PNv25[i]) + '\t' + str(sigmaz_up7[i]) + '\t' + str(sigmaz_down7[i]) + '\t' + str(PNv27[i]) + '\t' + str(sigmaz_up10[i]) + '\t' + str(sigmaz_down10[i]) + '\t' + str(PNv210[i]) + '\t' + str(sigmaz_up15[i]) + '\t' + str(sigmaz_down15[i]) + '\t' + str(PNv215[i]) + '\t' + str(sigmaz_up20[i]) + '\t' + str(sigmaz_down20[i]) + '\t' + str(PNv220[i]) + '\t' + str(sigmaz_up30[i]) + '\t' + str(sigmaz_down30[i]) + '\t' + str(PNv230[i]) + '\t' + str(sigmaz_up40[i]) + '\t' + str(sigmaz_down40[i]) + '\t' + str(PNv240[i]) + '\t' + str(sigmaz_up50[i]) + '\t' + str(sigmaz_down50[i]) + '\t' + str(PNv250[i]) + '\n')
        f.close()
        
    print('final results')
    for i in range(0,len(EBANDSFilenameList)):
        print(EBANDSFilenameList[i], a[i], c[i], NF_up[i], NF_down[i], PN[i], sigmaz_up0[i], sigmaz_down0[i], PNv20[i], sigmaz_up1[i], sigmaz_down1[i], PNv21[i], sigmaz_up2[i], sigmaz_down2[i], PNv22[i], sigmaz_up5[i], sigmaz_down5[i], PNv25[i], sigmaz_up7[i], sigmaz_down7[i], PNv27[i], sigmaz_up10[i], sigmaz_down10[i], PNv210[i], sigmaz_up15[i], sigmaz_down15[i], PNv215[i], sigmaz_up20[i], sigmaz_down20[i], PNv220[i], sigmaz_up30[i], sigmaz_down30[i], PNv230[i], sigmaz_up40[i], sigmaz_down40[i], PNv240[i], sigmaz_up50[i], sigmaz_down50[i], PNv250[i])
    
    
main('qatkDatFiles/atomicMag/Co_bulk_new/',8.617e-5*100,20,15,15,101,0.1)
