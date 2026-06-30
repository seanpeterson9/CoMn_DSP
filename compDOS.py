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

def main(spinUpDOSFile,spinDownDOSFile,calcDOSFile,NF):
    E = []
    N_up = []
    N_down = []
    """
    with open(spinUpDOSFile,'r') as myFile:
        read = csv.reader(myFile,delimiter = '\t')
        for row in read:
            E.append(float(row[0]))
            N_up.append(float(row[1]))
    with open(spinDownDOSFile,'r') as myFile:
        read = csv.reader(myFile,delimiter = '\t')
        for row in read:
            N_down.append(float(row[1]))
    E = np.array(E)
    N_up = np.array(N_up)
    N_down = np.array(N_down)
    """
    E_calc,N_up_calc,N_down_calc = importMyDOS(calcDOSFile)
    E,N_up,N_down = importDOS(spinUpDOSFile,spinDownDOSFile)
    E = E * 27.2114
    #N_up = N_up / 137.
    #N_down = N_down / 137.
    N_up = 3.4 * N_up / 27.2114 # for Co
    N_down = 3.4 * N_down / 27.2114 # for Co
    dE = E[1]-E[0]
    dE_calc = E_calc[1] - E_calc[0]
    DOS_func = interp1d(E,N_down)
    
    #print('weighted residuals: ',(dE*np.sum(N_down*fermi_deriv(E,8.617e-5*300)/.28828)-dE_calc*np.sum(N_down_calc*fermi_deriv(E_calc,8.617e-5*300)/NF))/(dE*np.sum(N_down*fermi_deriv(E,8.617e-5*300)/.28828)))
    #print('weighted residuals: ',dE_calc*np.sum(np.abs((DOS_func(E_calc)/.28828)-(N_down_calc/NF))*fermi_deriv(E_calc,8.617e-5*100))/(dE*np.sum(N_down*fermi_deriv(E,8.617e-5*100)/.28828)))
    print('weighted residuals: ',dE_calc*np.sum(np.abs((DOS_func(E_calc))-(N_down_calc/NF))*fermi_deriv(E_calc,8.617e-5*100))/(dE*np.sum(N_down*fermi_deriv(E,8.617e-5*100))))
    #print(E)
    #plt.plot(-N_up/.28828,E,c='red')
    #plt.plot(N_down/.28828,E,c='blue')
    plt.plot(-N_up/17.26,E,c='red')
    plt.plot(N_down/17.26,E,c='blue')
    #plt.plot(-N_up,E,c='red')
    #plt.plot(N_down,E,c='blue')
    plt.plot(-N_up_calc/NF,E_calc)
    plt.plot(N_down_calc/NF,E_calc)
    plt.plot([0,0],[-1,1],ls = ':',c = 'k')
    plt.plot([-5,22],[0,0],ls = ':',c = 'k')
    plt.ylim([-.5,.5])
    #plt.xlim([-2.5,22])
    plt.show()

main('abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinUpDOS.dat','abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinDownDOS.dat','myDOS_bct_Co75Mn25_a_292_interp_101.dat',0.67)
#main('abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinUpDOS.dat','abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinDownDOS.dat','myDOS_bct_Co75Mn25_a_292_fsolve_fitTwo_10_1.5_1_gauss02_fermiDeriv(2).dat',1.314)
#main('abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinUpDOS.dat','abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinDownDOS.dat','myDOS_bct_Co75Mn25_a_292_fsolve_fitTwo_10_1.5_1_gauss02_fermiDeriv.dat',1.314)
#main('abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinUpDOS.dat','abinitDatFiles/DOS/Co75Mn25I_symm_FM_bandStruct2/bct_a_292_spinDownDOS.dat','myDOS_bct_Co75Mn25_a_292_fsolve_fitTwo_8_1.5_1_gauss02_fermiDeriv.dat',1.326)

main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_interp_101.dat',.01353)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_295_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_295_spinDownDOS.dat','myDOS_bct_Co_a_295_interp_101.dat',.1327)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_interp_101.dat',.01634)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_interp_101(1).dat',.01637)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_51_01.dat')
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_101_02.dat',.000309)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_16_4_0.75_gauss02_fermiDeriv.dat',.052)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_16_4_1_gauss02_fermiDeriv.dat',.0506)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_12_4_1_gauss02_fermiDeriv.dat',.0507)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_14_4lenKx_1_gauss05_dk015625_fermiDeriv.dat',.0479)
"""
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_4lenKx_0.5_gauss05_dk015625_fermiDeriv(2).dat',.0498)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_1lenKx_1_gauss05_dk015625_fermiDeriv(2).dat',.0463)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_1.5lenKx_1_gauss05_dk015625_fermiDeriv(2).dat',.0445)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_8_2lenKx_1_gauss05_dk015625_fermiDeriv(2).dat',.0445)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_2lenKx_1_gauss05_dk015625_fermiDeriv(2).dat',.0453)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_4lenKx_1_gauss05_dk015625_fermiDeriv(2).dat',.0459)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_14_4lenKx_1_gauss05_dk015625_fermiDeriv(2).dat',.0483)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_4_1_gauss05_dk015625_fermiDeriv.dat',.0432)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_4lenKx_1_gauss05_dk015625_fermiDeriv.dat',.058)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_14_4lenKx_1_gauss05_dk015625_fermiDeriv.dat',.048)
main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_276_spinDownDOS.dat','myDOS_bct_Co_a_276_fsolve_fitTwo_10_4_0.5_gauss05_dk015625_fermiDeriv.dat',.0471)
"""
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_14_4_0.5_gauss05_dk015625_fermiDeriv.dat',.0515)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_14_4_1_gauss05_dk015625_fermiDeriv.dat',.0483)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_14_4_1_gauss02_fermiDeriv.dat',.0512)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_14_4_0.25_gauss02_fermiDeriv.dat',.0623)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_14_4_0.5_gauss02_fermiDeriv.dat',.0537)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_12_4_0.5_gauss02_fermiDeriv.dat',.0523)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_8_4_0.5_gauss02_fermiDeriv.dat',.0504)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_10_4_0.5_gauss02_fermiDeriv.dat',.051)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_10_2_0.5_gauss02_fermiDeriv.dat',.0516)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_0.5_4_0.5_gauss02_f0.12.dat',.053)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_0.75_4_0.5_gauss02_f0.12.dat',.0542)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_1_4_0.5_gauss02_f0.12.dat',.057)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_1.5_4_0.5_gauss02_f0.14.dat',.061)
#main('abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinUpDOS.dat','abinitDatFiles/DOS/Co_FM_DOS/bct_a_291_spinDownDOS.dat','myDOS_bct_Co_a_291_fsolve_fitTwo_1.5_4_0.5_gauss02_f0.12.dat',.061)
