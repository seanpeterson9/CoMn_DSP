from matplotlib import pyplot as plt
import numpy as np
import os
import csv

def getTetDist(filename,cellSize):
    bulkConfig = nlread(filename)[0]
    a = bulkConfig.primitiveVectors()[0][0].inUnitsOf(Angstrom)/cellSize
    c = bulkConfig.primitiveVectors()[2][2].inUnitsOf(Angstrom)/cellSize
    return a,c

def getAtomMag(filename):
    bulkConfig = nlread(filename)[0]
    elements = np.array(bulkConfig.elements())
    m_pop = nlread(filename)[1]
    atomMag = .5*2.002319*(np.array(m_pop.atoms(spin=Spin.Up)) - np.array(m_pop.atoms(spin=Spin.Down)))
    Co_mask = elements == Cobalt
    Mn_mask = elements == Manganese
    atomMag_Co = np.sum(atomMag[Co_mask])/len(atomMag[Co_mask])
    if len(atomMag[Mn_mask]) > 0:
        atomMag_Mn = np.sum(atomMag[Mn_mask])/len(atomMag[Mn_mask])
    return atomMag_Co,atomMag_Mn

def getNF(filename):
    projDOS = nlread(filename)[-2]
    N_up = np.array(projDOS.evaluate().inUnitsOf(eV**-1)[0])
    N_down = np.array(projDOS.evaluate().inUnitsOf(eV**-1)[1])
    E = np.array(projDOS.energies().inUnitsOf(eV))
    NF_up = np.interp(np.array([0]),E,N_up)[0]
    NF_down = np.interp(np.array([0]),E,N_down)[0]
    return NF_up,NF_down

def getDSPFromFile(filename):
    with open(filename,'r') as file:
        read = csv.reader(file,delimiter = ' ')
        a = []
        c = []
        Pn = []
        Pnv2_0 = []
        Pnv2_1 = []
        Pnv2_2 = []
        Pnv2_5 = []
        Pnv2_7 = []
        Pnv2_10 = []
        Pnv2_15 = []
        Pnv2_20 = []
        Pnv2_30 = []
        Pnv2_40 = []
        Pnv2_50 = []
        for row in read:
            a.append(float(row[1]))
            c.append(float(row[2]))
            Pn.append(float(row[5]))
            Pnv2_0.append(float(row[8]))
            Pnv2_1.append(float(row[11]))
            Pnv2_2.append(float(row[14]))
            Pnv2_5.append(float(row[17]))
            Pnv2_7.append(float(row[20]))
            Pnv2_10.append(float(row[23]))
            Pnv2_15.append(float(row[26]))
            Pnv2_20.append(float(row[29]))
            Pnv2_30.append(float(row[32]))
            Pnv2_40.append(float(row[35]))
            Pnv2_50.append(float(row[38]))
    
    a = np.array(a)
    c = np.array(c)
    Pn = np.array(Pn)
    Pnv2_0 = np.array(Pnv2_0)
    Pnv2_1 = np.array(Pnv2_1)
    Pnv2_2 = np.array(Pnv2_2)
    Pnv2_5 = np.array(Pnv2_5)
    Pnv2_7 = np.array(Pnv2_7)
    Pnv2_10 = np.array(Pnv2_10)
    Pnv2_15 = np.array(Pnv2_15)
    Pnv2_20 = np.array(Pnv2_20)
    Pnv2_30 = np.array(Pnv2_30)
    Pnv2_40 = np.array(Pnv2_40)
    Pnv2_50 = np.array(Pnv2_50)
    
    return a,c,Pn

def plotTetDistAtomMag(directory,cellSize):
    filenameList = []
    os.system('ls ' + directory + 'a_*.hdf5 > tmpFileList')
    with open('tmpFileList','r') as myFile:
        read = csv.reader(myFile)
        for row in read:
            filenameList.append(row[0])
    os.system('rm tmpFileList')
    
    a = np.zeros(len(filenameList))
    c = np.zeros(len(filenameList))
    atomMag_Co = np.zeros(len(filenameList))
    atomMag_Mn = np.zeros(len(filenameList))
    for i in range(0,len(filenameList)):
        print(filenameList[i])
        a_tmp,c_tmp = getTetDist(filenameList[i],cellSize)
        m_Co,m_Mn = getAtomMag(filenameList[i])
        a[i] = a_tmp
        c[i] = c_tmp
        atomMag_Co[i] = m_Co
        atomMag_Mn[i] = m_Mn
    
    plt.scatter(a,c)
    plt.show()
    
    plt.scatter(c/a,atomMag_Co,c = 'blue')
    plt.scatter(c/a,atomMag_Mn,c = 'red')
    plt.show()

#plotTetDistAtomMag('qatkDatFiles/atomicMag/Co75Mn25_bulk/',2)
#plotTetDistAtomMag('qatkDatFiles/atomicMag/Co875Mn125_bulk/',2)
#plotTetDistAtomMag('qatkDatFiles/atomicMag/Co_bulk_new/',1)

def plotNF(directory,DSP_filename,color='red'):
    filenameList = []
    os.system('ls ' + directory + 'a_*.hdf5 > tmpFileList')
    with open('tmpFileList','r') as myFile:
        read = csv.reader(myFile)
        for row in read:
            filenameList.append(row[0])
    os.system('rm tmpFileList')
    
    a = np.zeros(len(filenameList))
    c = np.zeros(len(filenameList))
    NF_up = np.zeros(len(filenameList))
    NF_down = np.zeros(len(filenameList))
    for i in range(0,len(filenameList)):
        print(filenameList[i])
        a_tmp,c_tmp = getTetDist(filenameList[i],1)
        tmp_up,tmp_down = getNF(filenameList[i])
        a[i] = a_tmp
        c[i] = c_tmp
        NF_up[i] = tmp_up
        NF_down[i] = tmp_down
    
    A,C,Pn = getDSPFromFile(DSP_filename)
    
    #plt.figure(figsize=(5,5))
    plt.scatter(c/a,(NF_up-NF_down)/(NF_up+NF_down),c = 'blue',label = 'accepted')
    plt.scatter(C/A,Pn,c = color,label = 'calculated')
    plt.legend(loc = 'lower right',frameon=False,fontsize = 14)
    plt.ylim([1,-1])
    plt.ylabel(r'$(N_F^\uparrow - N_F^\downarrow) / N_F$',fontsize=14)
    plt.xlabel(r'c/a',fontsize=14)
    plt.show()

#plotNF('qatkDatFiles/atomicMag/Co875Mn125_bulk/','DSP_Co875Mn125_mpi.dat')
plotNF('qatkDatFiles/atomicMag/Co75Mn25_bulk_new/','DSP_Co75Mn25_mpi.dat')
#plotNF('qatkDatFiles/atomicMag/Co75Mn25_bulk/','DSP_Co75Mn25_mpi_70x70_lessBands.dat',color = 'green')
#plt.show()
#plotNF('qatkDatFiles/atomicMag/Co_bulk_new/','DSP_Co_mpi_spaceDelim.dat')
