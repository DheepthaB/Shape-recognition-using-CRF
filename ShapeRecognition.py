
# @authors: Dheeptha, Deeksha, Akash

'''
* performs shape detection in grayscale images
* split data into train and test
* Performs feature selection
* Performs MLE to learn parameters
* Perfroms inference by directly plugging in updtaed parameters in conditional probability equation
* Finds accuarcy of the CRF model
'''


import re,sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


file=open('E:\Dheeptha\MS\Sem2\Projects\Stats\shapeset\shapeset1_1cs_2p_3o.5000.valid.amat',"r");   #.amat file containing feature information

lines=[]     #contains each line of .amat file


for line in file.readlines():
    lines.append(line)

first_line=lines[0].split()   # gives the size of data set and no of values in each line

M=int(first_line[1]);  



pixels={}    #dictionary to maintain pixel information of every image of size 32x32. 
#dictionaries to maintain feature values
shape={}     
color={}
xpos={}
ypos={}
angle={}
size={}
elongation={}

#extract individual feature and store it in appropriate dictionaries
for i in range(1,M+1):
    features=lines[i].split();
    for f in range(0,1024):
        pixels[i,f]=features[f]
    f=1024
    shape[i]=features[f]
    f=f+1
    color[i]=int(features[f])
    f=f+1
    xpos[i]=features[f]
    f=f+1
    ypos[i]=features[f]
    f=f+1
    angle[i]=features[f]
    f=f+1
    size[i]=features[f]
    f=f+1
    elongation[i]=features[f]



###################combines all features to a single dictionary
features={}
features['pixels']=pixels
features['color']=color
features['shape']=shape




#########################split into train and test data
train_features={}
train_features['shape']={}
train_features['color']={}
train_features['pixels']={}
test_features={}
test_features['shape']={}
test_features['pixels']={}
test_features['color']={}

count0=0
count1=0
count2=0
j=0
t=0



for i in range(1,M+1):
    if features['shape'][i]=='0' and count0<1250:
        train_features['color'][j]=round(features['color'][i]/(7*1.0),2)
        p=[]
        for k in range(0,1024):
            p.append(int((round(float(features['pixels'][i,k]),2)==train_features['color'][j])))
        train_features['pixels'][j]=p
        train_features['shape'][j]=features['shape'][i]
        count0=count0+1
        j=j+1
    elif features['shape'][i]=='0' and count0>=1250:
        test_features['color'][t]=round(features['color'][i]/(7*1.0),2)
        p=[]
        for k in range(0,1024):
            p.append(int((round(float(features['pixels'][i,k]),2)==test_features['color'][t])))
        test_features['pixels'][t]=p
        test_features['shape'][t]=features['shape'][i]
        count0=count0+1
        t=t+1
    elif features['shape'][i]=='1' and count1<1250:
        train_features['color'][j]=round(features['color'][i]/(7*1.0),2)
        p=[]
        for k in range(0,1024):
            p.append(int((round(float(features['pixels'][i,k]),2)==train_features['color'][j])))
        train_features['pixels'][j]=p
        train_features['shape'][j]=features['shape'][i]
        count1=count1+1
        j=j+1
    elif features['shape'][i]=='1' and count1>=1250:
        test_features['color'][t]=round(features['color'][i]/(7*1.0),2)
        p=[]
        for k in range(0,1024):
            p.append(int((round(float(features['pixels'][i,k]),2)==test_features['color'][t])))
        test_features['pixels'][t]=p
        test_features['shape'][t]=features['shape'][i]
        count1=count1+1
        t=t+1
    elif features['shape'][i]=='2' and count2<1250:
        train_features['color'][j]=round(features['color'][i]/(7*1.0),2)
        p=[]
        for k in range(0,1024):
            p.append(int((round(float(features['pixels'][i,k]),2)==train_features['color'][j])))
        train_features['pixels'][j]=p
        train_features['shape'][j]=features['shape'][i]
        count2=count2+1
        j=j+1
    elif features['shape'][i]=='2' and count2>=1250:
        test_features['color'][t]=round(features['color'][i]/(7*1.0),2)
        p=[]
        for k in range(0,1024):
            p.append(int((round(float(features['pixels'][i,k]),2)==test_features['color'][t])))
        test_features['pixels'][t]=p
        test_features['shape'][t]=features['shape'][i]
        count2=count2+1
        t=t+1
MTrain=len(train_features['shape'])   #Number of training samples
MTest=len(test_features['shape'])     #Number of testing samples



#obtains the pixels along the border blocks that contain just the shape borders to facilitate easy shape detection
def getPixels(M,allPixels):
    img={}
    for i in range(0,M):
        img[i]=[]
    pix=np.zeros((32,32))
    for m in range(0,M):
        pi=0
        for i in range(0,32):
            for j in range(0,32):
                pix[i,j]=allPixels[m][pi]
                pi=pi+1
        img[m]=pix

    blocks={}
    for i in range(0,M):
        blocks[i]={}
    for m in range(0,M):
        k=0
        r=0
        c=0
        for i in range(0,8):
            c=0
            for j in range(0,8):
                blocks[m][k]=img[m][r:r+4,c:c+4]
                k=k+1
                c=c+4
            r=r+4
               
    pixels={}
    for i in range(0,M):
        pixels[i]=[]
    for m in range(0,M):
        for b in range(0,64):
            if 1 in blocks[m][b] :  
                pixels[m].append(blocks[m][b].flatten())

    for m in range(0,M):
        l =[] 
        for b in range(0,len(pixels[m])): 
            for p in pixels[m][b]:
                l.append(p)
        if(m==0):
            pix= np.array(l)
        else:
            pix = np.vstack((pix,np.array(l)))
    return pix



##Initialising weight parameters to random float values
wts_fpis=6*np.random.random_sample((1,6))
wts_fs=3*np.random.random_sample((1,3))

## finds the number of pixel values having ones of every sample. done as a pre computation to decrease the running time
def countZeros(pix):
    count=0
    count= list(pix).count(0)
    return count

## finds the number of pixel values having zeros of every sample. done as a pre computation to decrease the running time
def countOnes(pix):
    count=0
    count=list(pix).count(1)
    return count


#inference subroutine of MLE algorithm
def conditional(s,pix,wts_fs,wts_fpis):
    wts_fpis[0]=wts_fpis[0]/np.linalg.norm(wts_fpis[0])
    wts_fs[0]=wts_fs[0]/np.linalg.norm(wts_fs[0])
    prod=joint(s,pix,wts_fs,wts_fpis)
    sum_t=prod
    for shape in range(0,3):
        if shape!=s:
            sum_t=sum_t+joint(shape,pix,wts_fs,wts_fpis)
    return prod/float(sum_t)

def joint(s,pix,wts_fs,wts_fpis):
    pixLength = len(pix)
    shape = np.exp(wts_fs[0,s])
    prod=pow(shape,pixLength)
    count0 = countZeros(pix)
    count1 = countOnes(pix)
    wtfpis_zero = np.exp(wts_fpis[0,s*2+0])
    wtfpis_ones = np.exp(wts_fpis[0,s*2+1])
    prod = prod*pow(wtfpis_zero, count0)*pow(wtfpis_ones, count1)
    return prod


############# MLE Algorithm for 1st parameter that combines pixels and shape labels ######################

###calculates empirical probabilities
emp_prob=np.zeros((1,6))

pix=getPixels(MTrain,train_features['pixels'])
for m in range(0,MTrain):
    for pix1 in pix[m]:
        for s in range(0,3):
            for x in range(0,2):
                if s==int(train_features['shape'][m]) and x==pix1:
                    emp_prob[0,2*s+x]=emp_prob[0,2*s+x]+1

emp_prob=emp_prob/MTrain*1.0
########################################


###calculates derivative of log(Z) wrt first parameter that combines pixels and shape labels
pix=getPixels(MTrain,train_features['pixels'])
cond=np.zeros((1,3))
count = np.zeros((1,2))
its=100

pix=getPixels(MTrain,train_features['pixels'])
for t in range(0,its):
    print t
    exp_prob=np.zeros((1,6))
    for m in range(0,MTrain):
        pixInfo = pix[m]
        count[0,0] = countZeros(pixInfo)
        count[0,1] = countOnes(pixInfo)
        for s in range(0,3): 
            cond[0,s] = conditional(s,pixInfo,wts_fs,wts_fpis)
            for x in range(0,2):
                exp_prob[0,2*s+x]= exp_prob[0,2*s+x]+(count[0,x]*cond[0,s])
    expProb = exp_prob/float(MTrain)
    grad=emp_prob-expProb
    normGrad = np.linalg.norm(grad[0])
    wts_fpis[0]+=grad[0]*(2.0/(2.0+t))
    if normGrad<1e-3:
        break
###############################################################################################
######################################### end of MLE for 1st parameter ####################################################################
print '1st done'

############### MLE for second parameter for shape labels #########################
#####calculate empirical probability for gradient wrt 2nd parameter
emp_prob=np.zeros((1,3))
for m in range(0,MTrain):
    for s in range(0,3):
        if s==int(train_features['shape'][m]):
            emp_prob[0,s]=emp_prob[0,s]+1


emp_prob=emp_prob/MTrain*1.0
##############################################################


########calculate derivative of log(Z) wrt 2nd parameter
cond=np.zeros((1,3))

its=100
pix=getPixels(MTrain,train_features['pixels'])

for t in range(0,its):
    exp_prob=np.zeros((1,3))
    for m in range(0,MTrain):
        pixInfo = pix[m]
        for s in range(0,3):
            cond[0,s] = conditional(s,pixInfo,wts_fs,wts_fpis)
            exp_prob[0,s]=exp_prob[0,s]+cond[0,s]
    grad=emp_prob-(exp_prob/float(MTrain))
    wts_fs=wts_fs+grad[0]*(2.0/(2.0+t))
    if np.linalg.norm(grad[0])<1e-3:
        break
############################################################
############################################# end of MLE for 2nd parameter ##########################################
print '2nd done'



################ Testing using test dataset ##############################
def test(wts_fpis,wts_fs,pix):
    cond=np.zeros((1,3))
    maxi=0
    maxClass=-1
    for s in range(0,3):
        cond[0,s]=conditional(s,pix,wts_fs,wts_fpis)
        if cond[0,s]>maxi:
            maxi=cond[0,s]
            maxClass=s
    return maxClass


pix=getPixels(MTest,test_features['pixels'])
print(len(pix[0]))
count=0
for m in range(0,MTest):
    maxLikelyClass=test(wts_fpis,wts_fs,pix[m])
    if maxLikelyClass==int(test_features['shape'][m]):
        count+=1

print (count*1.0/MTest)   ############# prints accuaracy of the model


