from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import csv

def fermi_deriv(E,T):
    return 1/(2.*np.cosh(E/(2.*T)))**2.

def importMyDOS(filename):
    data = np.loadtxt(filename)
    Energy = data[:,0]
    N_up = data[:,1]
    N_down = data[:,2]
    return Energy,N_up,N_down

def importDOS(upFile,downFile):
    upData = np.loadtxt(upFile)
    downData = np.loadtxt(downFile)

    Ef = upData[0][0]
    Energy = upData[1:,0]
    upDOS = upData[1:,1]
    downDOS = downData[1:,1]

    return Energy - Ef,upDOS,downDOS

def main(Filename,calcDOSFile,NF,NF_calc):
    E = []
    N_up = []
    N_down = []
    projDOS = nlread(Filename)[-2]
    N_up = projDOS.evaluate().inUnitsOf(eV**-1)[0]
    N_down = projDOS.evaluate().inUnitsOf(eV**-1)[1]
    E = projDOS.energies().inUnitsOf(eV)
    E_calc,N_up_calc,N_down_calc = importMyDOS(calcDOSFile)
    """
    E_calc,N_up_calc,N_down_calc = importMyDOS(calcDOSFile)
    E,N_up,N_down = importDOS(spinUpDOSFile,spinDownDOSFile)
    dE = E[1]-E[0]
    dE_calc = E_calc[1] - E_calc[0]
    DOS_func = interp1d(E,N_down)
    
    print('weighted residuals: ',dE_calc*np.sum(np.abs((DOS_func(E_calc))-(N_down_calc/NF))*fermi_deriv(E_calc,8.617e-5*100))/(dE*np.sum(N_down*fermi_deriv(E,8.617e-5*100))))
    """
    #print(E)
    #plt.plot(-N_up/.28828,E,c='red')
    #plt.plot(N_down/.28828,E,c='blue')
    plt.plot(-N_up/NF,E,c='red')
    plt.plot(N_down/NF,E,c='blue')
    plt.plot(-N_up_calc/NF_calc,E_calc)
    plt.plot(N_down_calc/NF_calc,E_calc)
    plt.plot([0,0],[-1,1],ls = ':',c = 'k')
    #plt.plot([-5,22],[0,0],ls = ':',c = 'k')
    plt.ylim([-.5,.5])
    #plt.xlim([-2.5,22])
    plt.show()

main('qatkDatFiles/atomicMag/Co_bulk_new/a_277.hdf5','Co_277_DOS.dat',.297,.01529)
main('qatkDatFiles/atomicMag/Co_bulk_new/a_277.hdf5','Co_277_DOS_new.dat',.297,.0383)
