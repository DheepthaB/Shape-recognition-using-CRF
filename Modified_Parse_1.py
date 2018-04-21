
# coding: utf-8

# In[1]:


import re,sys
import pandas as pd
import numpy as np
import math

file=open('C:\Users\mesth\Documents\SPRING2018\Stats for AI and ML\Project\shapeset.tar\shapeset\shapeset1_1cs_2p_3o.5000.valid.amat',"r");   #.amat file containing feature information

lines=[]     #contains each line of .amat file


for line in file.readlines():
    lines.append(line)

first_line=lines[0].split()   # gives the size of data set and no of values in each line

M=int(first_line[1]);   #Number of training instances

pixels={}    #dictionary to maintain pixel information of every image of size 32x32. 
#dictionaries to maintain feature values
shape={}     
color={}
xpos={}
ypos={}
angle={}
size={}
elongation={}


# In[2]:


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

data=pd.DataFrame([features])

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


# In[3]:


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
            p.append(int((round(float(features['pixels'][i,k]),2)==train_features['color'][t])))
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
            p.append(int((round(float(features['pixels'][i,k]),2)==train_features['color'][t])))
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
            p.append(int((round(float(features['pixels'][i,k]),2)==train_features['color'][t])))
        test_features['pixels'][t]=p
        test_features['shape'][t]=features['shape'][i]
        count2=count2+1
        t=t+1


# In[157]:


M=len(train_features['shape'])

wts_fpi=np.ones((1,2))
wts_fpis=np.ones((1,6))
wts_fs=np.ones((1,3))


###Factor graph
N=1024
T=2*N+1
k=3
neighbors={}
n=[]
for i in range(N+1,T):
    n.append(i)
neighbors[N]=n

for i in range(0,N):
    neighbors[i]=[N+1+i]

for i in range(N+1,T+1):
    n=[]
    n.append(N)
    n.append(i-1-N)
    neighbors[i]=n


# In[158]:


def conditional(s,pixels,wts_fs,wts_fpis):
    prod=joint(s,pixels,wts_fs,wts_fpis)
    sum_t=prod
    for shape in range(0,3):
        if shape!=s:
            sum_t=sum_t+joint(shape,pixels,wts_fs,wts_fpis)
    #print "in condi.. pls work"
    #print prod/sum_t
    return prod/sum_t

def joint(s,pixels,wts_fs,wts_fpis):
    ws = 0
    wfpis = 0
    prod=0
    for pi in range(0,N):
        prod=prod*math.exp(wts_fpis[0,s*2+pixels[pi]])*math.exp(wts_fs[0,s])
    for in1 in wts_fs[0]:
        ws = ws + math.pow(in1,2)
    for in2 in wts_fpis[0]:
        wfpis = wfpis + math.pow(in2,2)
    prod= prod - math.exp(1*(ws + wfpis))
    #print "in joint"
    #print prod
    return prod


# In[159]:


emp_prob=np.zeros((M,6))

for m in range(0,M):
    for pi in range(0,1024):
        for s in range(0,3):
            for x in range(0,2):
                if s==int(train_features['shape'][m]) and x==train_features['pixels'][m][pi]:
                    emp_prob[m,2*s+x]=emp_prob[m,2*s+x]+1

emp_prob=emp_prob/M*1.0
print emp_prob


# In[160]:



#############calculate marginal probability for gradient wrt 2nd parameter
prob=np.zeros((M,6))

for m in range(0,M):
    pixels=[]
    for pi in range(0,N):
        pixels.append(train_features['pixels'][m][pi])
    for s in range(0,3):
        cond=conditional(s,pixels,wts_fs,wts_fpis)
        for x in range(0,2):
            for pi in range(0,N):
                if x==train_features['pixels'][m][pi] and s==int(train_features['shape'][m]):
                    prob[m,2*s+x]=prob[m,2*s+x]+cond
                    print cond
            t=1
            #print emp_prob[m,2*s+x]-prob[m,2*s+x]-(0.5*wts_fpis[0,2*s+x])
            while (abs(emp_prob[m,2*s+x]-prob[m,2*s+x]-(1*wts_fpis[0,2*s+x])))>1e-1:
                wts_fpis[0,2*s+x]=wts_fpis[0,2*s+x]+(emp_prob[m,2*s+x]-prob[m,2*s+x]-(1*wts_fpis[0,2*s+x]))*(2.0/(2.0+t))
                print wts_fpis
                cond=conditional(s,pixels,wts_fs,wts_fpis)
                #print (emp_prob[m,2*s+x]-prob[m,2*s+x]-(math.exp(-2)*wts_fpis[0,2*s+x]))
                for pi in range(0,N):
                    if x==train_features['pixels'][m][pi] and s==int(train_features['shape'][m]):
                        prob[m,2*s+x]=cond
                t=t+1
print wts_fpis


# In[109]:


emp_prob=np.zeros((M,3))

for m in range(0,M):
    for s in range(0,3):
        if s==int(train_features['shape'][m]):
            emp_prob[m,s]=emp_prob[m,s]+1


emp_prob=emp_prob/M*1.0
#############calculate marginal probability for gradient wrt 2nd parameter
prob=np.zeros((M,3))

for m in range(0,M):
    pixels=[]
    for pi in range(0,1024):
        pixels.append(train_features['pixels'][m][pi])
    for s in range(0,3):
        if s==int(train_features['shape'][m]):
            cond=conditional(s,pixels,wts_fs,wts_fpis)
            prob[m,s]=cond
        t=1
        while (emp_prob[m,s]-prob[m,s]-(math.exp(-2)*wts_fs[0,s]))>1e-3:
            wts_fs[0,s]=wts_fs[0,s]+(emp_prob[m,s]-prob[m,s]-(math.exp(-2)*wts_fs[0,s]))*(2.0/(2.0+t))
            print (emp_prob[m,s]-prob[m,s]-(math.exp(-2)*wts_fs[0,s]))
            cond=conditional(s,pixels,wts_fpi,wts_fs,wts_fpis,5)
            prob[m,s]=cond
            t=t+1
print wts_fs            

