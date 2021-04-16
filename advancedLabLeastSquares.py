#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:05:54 2020

@author: reeseboucher

Method takes two arrays of data points of the same length and finds
least squares approximation line based on error analysis 
"""
import math
import numpy as np
import matplotlib.pyplot as plt

class Uncertain:
     def __init__(self, value, err):
         self.value = value
         self.err   = err
         
     def __add__(self, other):
         totalVal = self.value + other.value
         totalErr = math.sqrt(self.err**2 + other.err**2)
         return Uncertain(totalVal, totalErr)
     
     def __mul__(self, other):
         totalVal = (self.value)*(other.value)
         totalErr = math.sqrt((totalVal**2)*(((self.err**2)/(self.value**2))+((other.err**2)/(other.value**2))))
         return Uncertain(totalVal, totalErr)
     
     def __truediv__(self, other):
         totalVal = (self.value)/(other.value)
         totalErr = math.sqrt((totalVal**2)*(((self.err**2)/(self.value**2))+((other.err**2)/(other.value**2))))
         return Uncertain(totalVal, totalErr)
            
     def __eq__(self, other):
         return (self.value, self.err) == (other.value,other.err)
         
     def __repr__(self):
         return str(self.value)+" "+ (u"\u00B1")+" "+str(self.err) 
         
     def __str__(self):
         return '{self.value}'.format(self=self) 
         
     

def meanError(inVals,error):
    inVals = np.array(inVals)
    mean   = np.mean(inVals)   
    uncertainty = 1/(np.sqrt(len(inVals)*(1/(error**2))))
    return mean, uncertainty

def percentError(observedVal,expectedVal):
    error = abs((observedVal-expectedVal)/expectedVal)*100
    print("Percent Error = "+str(error)+"%")
    return error
    

def leastSquares(x, y, outPrint=True): #y=intercept+(slope)x
    assert len(x) == len(y)
    N    = len(x)
    x    = np.array(x)
    y    = np.array(y)
    xSum = np.sum(x)
    ySum = np.sum(y)
    xSquaredSum = np.sum(x**2)
    xyProdSum   = np.sum(x*y)

    delta     = (N * xSquaredSum) - xSum**2
    intercept = ((xSquaredSum * ySum) - (xSum * xyProdSum))/delta
    slope     = ((N * xyProdSum) - (xSum * ySum))/delta
    
    residual  = (y - (intercept + slope*x))**2 
 
    commonUncertainty = np.sqrt((1/(N-2))*np.sum((y - slope * x - intercept)**2))
    interceptError    = commonUncertainty*np.sqrt(xSquaredSum/delta)
    gradientError     = commonUncertainty*np.sqrt(N/delta)
    
    if outPrint == True:

        print("N                 = "+str(N))
        print("xSum              = "+str(xSum ))
        print("ySum              = "+str(ySum))
        print("xSquaredSum       = "+str(xSquaredSum))
        print("xyProdSum         = "+str(xyProdSum))
        print("delta             = "+str(delta))
        print("===========================================")
        print("Slope             = "+str(slope))
        print("intercept         = "+str(intercept))
        print("===========================================")
        print("commonUncertainty = "+str(commonUncertainty))
        print("interceptError    = "+str(interceptError))
        print("gradientError     = "+str(gradientError))
        print("Total Error       = "+ str(interceptError+gradientError))
        # print("Least Square y Values = " + str(intercept + slope*x))
        # print("Residual          = "+str(residual))


    return(slope, intercept, gradientError, interceptError)




def plotData(slope, intercept,error=None):
    x = np.arange(0,10)
    y    = slope*np.array(x) + intercept

    # plt.xlim()
    # plt.ylim()
    if error == None: 
        plt.plot(y)
    else:
        print("fkbdsvnkjfgnksf")
        plt.errorbar(x,y,yerr=error)
    
    # plt.title("Sample Data Analysis - Temperature")
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("ln(Current) (ln(uA))")
    plt.savefig("70eight",dpi=300)
 
    
DischargeCurrent = 8
DischargeVoltage=70
probeVoltage1       = np.arange(-10,6.1,0.5)
probeVoltage2       = np.arange(-10,15.1,0.5)
probeCurrent1       = [-0.0246,-0.0232,-0.0218,-0.0204,-0.0190,-0.0176,-0.0162,-0.01148,-0.0133,-0.0188,-0.0102,-0.0086,-0.0067,-0.0041,-0.0006,0.0052,0.0164,0.0385,0.0787,0.1401,0.2089,0.2718,0.3351,0.4001,0.4678,0.5354,0.6052,0.6773,0.7505,0.8257,0.9013,0.9794,1.046]
probeCurrent1Append = np.array([1.168,1.252,1.339,1.426,1.517,1.609,1.704,1.802,1.902,2.004,2.110,2.218,2.332,2.448,2.571,2.697,2.837,2.999])-0.04
probeCurrent2       = np.append(probeCurrent1,probeCurrent1Append)
# plt.plot(probeVoltage2,probeCurrent2)
expCurrent = probeCurrent2[16:20]
expVoltage = probeVoltage1[16:20]
eV_Te=0.9566700327



# DischargeCurrent = 14
# DischargeVoltage = 77
# probeVoltage1       = np.arange(-10,15.1,0.5)
# probeCurrent1       = [-0.0328,-0.0311,-0.0294,-0.0276,-0.0258,-0.0241,-0.0233,-0.0204,-0.0186,-0.0167,-0.0147,-0.0126,-0.0102,-0.0073,-0.0031,0.0032,0.0150,0.0383,0.0831,0.1597,0.2734,0.4039,0.5198,0.6323,0.7474,0.8651,0.9885]
# probeCurrent1Append = np.array([1.137,1.277,1.425,1.581,1.747,1.924,2.109,2.304,2.509,2.722,2.943,3.172,3.409,3.651,3.899,4.159,4.423,4.694,4.973,5.262,5.559,5.869,6.192,6.548])-0.016499999999999848
# probeCurrent2       = np.append(probeCurrent1,probeCurrent1Append)
# # plt.plot(probeVoltage1,probeCurrent2)
# expCurrent = probeCurrent2[16:24]
# expVoltage = probeVoltage1[16:24]
# eV_Te=1.110129303240033


# DischargeCurrent = 12 
# DischargeVoltage = 73.5
# probeVoltage    = np.arange(-10,15.1,0.5)
# probeCurrent2    = [-0.02890,-0.0273,-0.0257,-0.0241,-0.0225,-0.0209,-0.0193,-0.0177,-0.0160,-0.0143,\
#                     -0.0125,-0.0106,-0.0084,-0.0056,-0.0016,0.0046,0.0165,0.0400,0.0841,0.1568,0.2551,\
#                       0.3498,0.4382,0.5268,0.6165,0.7085,0.8019,0.8979,0.9965, 1.095, 1.223,1.333,1.447,\
#                       1.565,1.688,1.815,1.948,2.085,2.231,2.379,2.534,2.695,2.863,3.036,3.214,3.398,
#                       3.589,3.787,3.994,4.222,4.473]
# plt.plot(probeVoltage,probeCurrent2)
# expCurrent = probeCurrent2[16:20]
# expVoltage = probeVoltage[16:20]

# eV_Te=0.9023



# DischargeCurrent = 8
# DischargeVoltage = 67.5
# probeCurrent2   = [-0.0204,-0.0192,-0.0180,-0.0168,-0.0156,-0.0144,-0.0131,-0.0119,-0.0106,-0.0093,\
#                   -0.0079,-0.0064,-0.0047,-0.0023,0.0010,0.0067,0.0178,0.0393,0.0759,0.1233,0.1691,\
#                   0.2141,0.2607,0.3092,0.3591,0.4105,0.4636,0.5179,0.5734,0.6312,0.6890,0.7496,0.8105,\
#                   0.8728,0.9364,1.001,1.066,1.1334,1.2010,1.2700,1.3410,1.4130,1.488,1.563,1.640,1.718,\
#                   1.799,1.885,1.972,2.065,2.169,2.493,3.169,3.291, 3.419, 3.573, 3.735, 3.93 , 4.15 , 4.405, 4.699,\
#                   5.044, 5.41,47.117,81.854,89.430,97.581,99.986,101.632,102.981,103.851,104.76,105.340]

# voltageAppend  = [16,17,17.1,17.2,17.3,17.4,17.5,17.6,17.7,17.8,17.9,18,19,19.1,19.2,19.3,19.4,19.5,19.6,19.7,19.8,19.9]
# probeVoltage2  = np.arange(-10,15.1,0.5)

# probeVoltage2=np.append(probeVoltage2,voltageAppend)
# expCurrent = probeCurrent2[54:64]
# expVoltage = probeVoltage2[54:64]
# eV_Te   = 0.92070025328




#######IV Trace plot
# plt.plot(probeVoltage2,probeCurrent2)

# plt.xlabel("Probe Voltage (V)")
# plt.ylabel("Probe Current (muA)")
# plt.title("Discharge Current = "+str(DischargeCurrent)+"(uA), Discharge Voltage = "+str(DischargeVoltage)+"(V)")
# plt.savefig("voltage",dpi=300)



 

###Log plot 

expCurrent = np.log(expCurrent+np.abs(np.min(expCurrent))+0.0001)

slope,intercept,slopeError, interceptError = leastSquares(expVoltage,expCurrent)
plotData(slope,intercept,slopeError)


########electron temp and density

e  = Uncertain(1.602*10**-19,.001*10**-19)
# kBoltzmann = Uncertain(1.38*10**-23,.01*10**-23)
kBoltzmann = Uncertain(8.617*10**-5,.001*10**-5)
m  = Uncertain(slope,slopeError)
Te = e/(kBoltzmann*m*e)

# eV_Te   = Te*Uncertain(8.6217*10**-5,.0001*10**-5)
I_e     = np.array(probeCurrent2)
A       = 1.71*10**-3 ##cm**2
density = (((4.02*10**9)*I_e)/(A*np.sqrt(eV_Te)))

print("temp =",Te)
print("eV temp =",eV_Te)
print("density =",density)
print("********************************************")


plt.title("Data Analysis: Discharge Current = "+str(DischargeCurrent)+" uA, Discharge Voltage = "+str(DischargeVoltage)+" V")








##########Langmuir sample data Uncertainty Analysis
# current                   = [0.1,0.15,0.23,0.3,0.45,0.75,1,2,4,6,9]#,11,11.5]
# lnCurrent                 = np.log(current)
# electronSaturationCurrent =   11
# probeBias                 = [-5.66,-5.49,-5.33,-5.22,-5,-4.78,-4.71,-4.57,-4.33,-4.25,-4.11]#,-3.66,-1]
# probeBias                 = np.array(probeBias)+np.log(electronSaturationCurrent)

# # plt.plot(probeBias,lnCurrent)
# slope,intercept,slopeError, interceptError = leastSquares(probeBias,lnCurrent)

# plotData(slope,intercept,slopeError)



# e          = Uncertain(1.602*10**-19,.001*10**-19)
# kBoltzmann = Uncertain(1.38*10**-23,.01*10**-23)
# kBoltzmann = Uncertain(8.617*10**-5,.001*10**-5)
# m          = Uncertain(slope,slopeError)
# Te         = e/(kBoltzmann*m*e)
# eV_Te   = Te*Uncertain(8.6217*10**-5,.0001*10**-5)
# eV_Te   = 0.7702213501309977
# I_e     = np.array(current)
# A       = 1.71*10**-3 ##cm**2
# density = (((4.02*10**9)*I_e)/(A*np.sqrt(eV_Te)))


# slope      = Uncertain(slope, slopeError)  




#######Plotting Langmuir probe test data only plasma in circuit    
# volts = np.arange(0,52,2)
# voltsSup =np.arange(50,76,5)

# volts=np.append(volts,voltsSup)

# blackDischarge=[0]
# blacksup=np.arange(3,32)
# blackDischarge=np.append(blackDischarge,blacksup)
# extra=[34,36]
# blackDischarge=np.append(blackDischarge,extra)
# # plt.plot(volts,blackDischarge)

# # extra = [78,75.5,77.5,80,82,84,85]
# # volts = np.append(volts,extra)

# redDischarge=[0]
# redDischargesup=np.arange(3,24,2)
# redDischarge=np.append(redDischarge,redDischargesup)
# extra=[26,31,34,36,38,190,186,239,232,234,220]
# redDischarge=np.append(redDischarge,extra)

# redVolts=[2]
# redVoltsSUp=np.arange(8,49,4)
# redVolts=np.append(redVolts,redVoltsSUp)
# # print(redVolts)
# redVoltsSUp=[55,65,70,75,78,75.5,77.5,80,82,84,85]
# redVolts=np.append(redVolts,redVoltsSUp)

# # plt.plot(redVolts,redDischarge)
   

   
#########pg 43 lab notebook Initial reading 77.2V 14mA 1000 ohms
# probeCurrent = [1.05,1.05,1.05,1.05,0.96,0.72,0.5,0.26,.07,.01,-0.0025,-0.0093,-0.013,-0.017,-0.020,-0.023,-0.026,-0.0297]
# probeVoltage = [10,8,6,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
# plt.plot(probeVoltage,probeCurrent,'bo')
# plt.savefig("dot_trace")


    
##########pg 43 lab notebook Second reading 
'''
probeCurrent2=[-.021,-0.019,-0.016,-0.014,-0.011,-0.008,-0.00051,-0.00081,0.0041,0.020,0.072,0.26,0.396,0.53,0.796,1.049,1.049]
probeVoltage2=[-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,0,2,3,4,6,8,10]

# plt.plot(probeVoltage2,probeCurrent2)

expCurrent=[0.0041,0.020,0.072]#,0.26]
expCurrent=np.log(expCurrent+np.abs(np.min(expCurrent))+0.0001)
expVoltage=[-2,  -1,0]#,2]#,-3,-4]
# plt.plot(expCurrent,expVoltage)



slope,intercept,slopeError, interceptError = leastSquares(expVoltage,expCurrent)


e  = Uncertain(1.602*10**-19,.001*10**-19)
# kBoltzmann = Uncertain(1.38*10**-23,.01*10**-23)
kBoltzmann = Uncertain(8.617*10**-5,.001*10**-5)
m  = Uncertain(slope,slopeError)
Te = e/(kBoltzmann*m*e)


eV_Te=Te*Uncertain(8.6217*10**-5,.0001*10**-5)
eV_Te=0.9025
I_e = np.array(probeCurrent)
A   = 1.71*10**-3 ##cm**2
density =(((4.02*10**9)*I_e)/(A*np.sqrt(eV_Te)))


print(Te)
print(eV_Te)
print(density)
print("********************************************")
# plt.xlabel("ln(Probe Current)")
# plt.ylabel("Probe Voltage")
 
'''









'''
#Photoelectric Effect My data
# wavelength      = [631,605,593,525,472,420, 400]  #400/380
# frequency       = (3*10**8)/(np.array(wavelength)*10**-9)
# stoppingVoltage = [.207,.237,.317,.519,.772,1.136,1.225]
# slope,intercept,residual = leastSquares(frequency,stoppingVoltage)
# plotData(slope,intercept)


#Photoelectric Effect Book Data
# wavelength      = [593,525,505,472,420]  #400/380
# frequency       = (3*10**8)/(np.array(wavelength)*10**-9)
# stoppingVoltage = [.25, .4, .7, .82, 1.15]
# slope,intercept,residual = leastSquares(frequency,stoppingVoltage, outPrint=False)
# plotData(slope,intercept)



# estimate_h = slope*1.602*10**-19
# print("Planck's Constant by Least Squares = "+str(estimate_h))
# percentError(estimate_h,6.62607004*10**-34)





#Franck Hertz
#Prelim Data - Franck Hertz
# current=[.317,.35,.5,.54,.52,.42,.63]
# voltage=[3.4,4.3,4.33,4.332,4.316,4.347,4.309]
# print(leastSquares(current,voltage))


# Franck Hertz Least Squares
# peak=[1,2,3]
# voltage=[7.4216, 12.0551, 17.2927]
# slope, intercept,residual = leastSquares(peak,voltage)
# test = np.arange(0,4)
# y = slope*np.array(test)+intercept
# plt.xlabel("Peaks")
# plt.ylabel("Voltage(V)")
# # plt.savefig("frankHertz_leastSquares.pdf")
# plt.plot(y)


#Franck Hertz main Data
# franckData = np.loadtxt("franckHertzData.txt",skiprows=1)
# slope, intercept,residual = leastSquares(franckData[:,0],franckData[:,1])

# plt.figure(1)
# plt.plot(franckData[:,0],franckData[:,1])
# plt.xlabel("Voltage (V)")
# plt.ylabel("Current (picoamps)")
# plt.title("Franck-Hertz Data")
# plt.savefig("frankHertz.pdf")
# test = np.arange(0,20)
# y = slope*np.array(test)+intercept
# plt.plot(y)


'''
'''
# Advanced Lab Hw3
# test = np.arange(0,100)
# wavelength = 1/(np.array([380,400,420,472,505,525,593,605,631])*1*10**-9)
# print(wavelength)
# vStopped   = np.array([-1.805,-1.557,-1.249,-1.063, -0.873, -0.748, -0.585, -0.458, -0.324])

# wavelength = 1/(np.array([420,472,505,525,593])*1*10**-9) #middle 5
# vStopped   = [-1.249,-1.063, -0.873, -0.748, -0.585] #middle 5

# wavelength = 1/(np.array([380,400,605,631])*1*10**-9) # all but middle 5
# vStopped   = [-1.805,-1.557, -0.458, -0.324] # all but middle 5

# slope,intercept,residual = leastSquares(wavelength,vStopped)

# print("Planck's Constant = "+str(slope*(-1.60217662*10**-19)/(2.99*10**8)))

# y = slope*np.array(test)+intercept
# plt.xlabel("wavelength")
# plt.ylabel("V_Stopped")
# plt.plot(y)
# plt.plot(vStopped,wavelength)
# plt.savefig("")
'''

'''
# Plot for Hw2 tempvTime
# time = [53115, 53175, 53235, 53295, 53355, 53415, 53475, 53535, 53595]
# time = [115, 175, 235, 295, 355, 415,475, 535, 595]
# temp = [98.51, 98.50, 98.50, 98.49, 98.52, 98.49, 98.52, 98.45, 98.47]

# test = np.arange(0,60000)
# plt.figure(1)
# slope, intercept,residual = leastSquares(time,temp)
# y = slope*np.array(test)+intercept
# plt.plot(y)
# plt.plot(time,temp)
# plt.xlabel("Time(s)")
# plt.ylabel("Temp(C)")
# plt.title("Temp v. Time")
# plt.savefig("K1plot.pdf")

'''

'''
#Hw1
#Plot for K1
# mass   = np.arange(200,1000,100)
# mass   = mass**2
# mass   = np.arange(.2,1,.1)

length = [5.1, 5.5, 5.9, 6.8, 7.4, 7.5, 8.6, 9.4]
# length= [0.051, 0.055, 0.059, 0.068, 0.074, 0.075, 0.086, 0.094]
# test = np.arange(0,1000)
# print(leastSquares(mass,length))

# plt.figure(1)

# slope, intercept,residual = leastSquares(mass,length)
# y = slope*np.array(test)+intercept
# plt.plot(y)
# plt.plot(mass,length)
# plt.xlabel("Mass(g)")
# plt.ylabel("Length(cm)")
# plt.title("Mass(g) v. Length(cm)")
# plt.savefig("K1plot.pdf")


#Residual Plot1
# plt.figure(1)
# slope, intercept,residual = leastSquares(mass,length)
# plt.plot(mass,residual)
# plt.xlabel("Mass(g)")
# plt.ylabel("Residual(cm^2)")
# plt.title("Mass(g) v. Residual(cm^2)")
# plt.savefig("Residual1.pdf")


#Plot for K2
# test = np.arange(0,1000000)
# plt.figure(1)

# slope, intercept,residual = leastSquares(mass,length)
# y = slope*np.array(test)+intercept
# plt.plot(y)
# plt.plot(mass,length)
# plt.xlabel("Mass(g^2)")
# plt.ylabel("Length(cm)")
# plt.title("Mass(g^2) v. Length(cm)")
# plt.savefig("K2plot.pdf")


#Residual Plot2
# plt.figure(1)
# slope, intercept,residual = leastSquares(mass,length)
# plt.plot(mass,residual)
# plt.xlabel("Mass^2(g^2)")
# plt.ylabel("Residual(cm^2)")
# plt.title("Mass^2(g^2) v. Residual(cm^2)")
# plt.savefig("Residual2.pdf")
'''