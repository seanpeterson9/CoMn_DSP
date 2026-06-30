import numpy as np
import csv
import os

def convHDF5toDAT(HDF5Filename,DatFilename):
    bulkConfig = nlread(HDF5Filename)[0]
    
    a = bulkConfig.primitiveVectors()[0][0].inUnitsOf(Angstrom)
    c = bulkConfig.primitiveVectors()[2][2].inUnitsOf(Angstrom)
    
    bandStruct = nlread(HDF5Filename)[-1]
    bands = bandStruct.evaluate().inUnitsOf(eV)
    
    kx = np.array(bandStruct.kpoints()[:,0])
    ky = np.array(bandStruct.kpoints()[:,1])
    kz = np.array(bandStruct.kpoints()[:,2])
    E = []
    N_k = len(kx)
    nbands = len(bands[0][0])
    for i in range(0,2*nbands):
        if i < nbands:
            E.append(bands[0][:,i])
        else:
            E.append(bands[1][:,i-nbands])
    E = np.array(E)
    
    f = open(DatFilename,'w')
    f.write(str(N_k) + '\t' + str(nbands) + '\t' + str(a) + '\t' + str(c) + '\n')
    for i in range(0,N_k):
        datString = str(kx[i]) + '\t' + str(ky[i]) +'\t' +  str(kz[i]) + '\t'
        for j in range(0,2*nbands):
            datString = datString + str(E[j][i]) + '\t'
        datString = datString + '\n'
        f.write(datString)
    f.close()

def convDirHDF5toDAT(directory):
    HDF5FilenameList = []
    DATFilenameList = []
    os.system('ls ' + directory + '*.hdf5 > tmpFileList')
    with open('tmpFileList','r') as myFile:
        read = csv.reader(myFile)
        for row in read:
            HDF5FilenameList.append(row[0])
    os.system('rm tmpFileList')
    for i in range(0,len(HDF5FilenameList)):
        DATFilenameList.append(HDF5FilenameList[i][:-4] + 'dat')
        convHDF5toDAT(HDF5FilenameList[i],DATFilenameList[i])

def readDatFile(Filename):
    with open(Filename) as f:
        F = csv.reader(f,delimiter='\t')
        j = -1
        for row in F:
            
            if j == -1:
                N_k = int(row[0])
                nbands = int(row[1])
                a = float(row[2])
                c = float(row[3])
                
                kx = np.zeros(N_k)
                ky = np.zeros(N_k)
                kz = np.zeros(N_k)
                
                E = np.zeros((2*nbands,N_k))
                print(N_k)
            
            else:
                kx[j] = float(row[0])
                ky[j] = float(row[1])
                kz[j] = float(row[2])
                for i in range(0,2*nbands):
                    print(i+3,len(row))
                    E[i][j] = float(row[i+3])
            j = j + 1
    return kx,ky,kz,E,nbands,a,c

#convHDF5toDAT('qatkDatFiles/atomicMag/Co_bulk_new/a_271_c_305.hdf5','tmp')
#readDatFile('tmp')
convDirHDF5toDAT('qatkDatFiles/atomicMag/Co_bulk_new/')
