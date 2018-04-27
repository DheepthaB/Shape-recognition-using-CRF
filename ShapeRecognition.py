
# coding: utf-8

# In[1]:


import re,sys
import pandas as pd
import numpy as np
import math

file=open('E:\Dheeptha\MS\Sem2\Projects\Stats\shapeset\shapeset1_1cs_2p_3o.5000.valid.amat',"r");   #.amat file containing feature information

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


# In[2]:


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



# In[2]:


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
M=len(train_features['shape'])


# In[3]:


img={}
for i in range(0,M):
    img[i]=[]
pix=np.zeros((32,32))
for m in range(0,M):
    pi=0
    for i in range(0,32):
        for j in range(0,32):
            pix[i,j]=train_features['pixels'][m][pi]
            pi=pi+1
    img[m]=pix
    
blocks={}
for i in range(0,M):
    blocks[i]={}
for m in range(0,M):
    k=0
    r=0
    c=0
    for i in range(0,4):
        c=0
        for j in range(0,4):
            blocks[m][k]=img[m][r:r+8,c:c+8]
            k=k+1
            c=c+8
        r=r+8


# flag={}
# for i in range(0,M):
#     flag[i]={}
    
# for m in range(0,M):
#     for b in range(0,16):
#         for i in range(0,7):
#             for j in range(0,7):
#                 if blocks[m][b][i,j]==1:
#                     if blocks[m][b][i,j]==blocks[m][b][i+1,j] or blocks[m][b][i,j]==blocks[m][b][i,j+1]:
#                         flag[m][b]=1
#                     else:
#                         flag[m][b]=0


# In[4]:


pixels={}
for i in range(0,M):
    pixels[i]=[]
for m in range(0,M):
    for b in range(0,16):
        if 0 in blocks[m][b] and 1 in blocks[m][b]:
            pixels[m].append(blocks[m][b].flatten())



# In[29]:


lam=0.5
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


# In[30]:


N = 1024
def conditional(s,pix,wts_fs,wts_fpis):
    prod=joint(s,pix,wts_fs,wts_fpis)
    sum_t=prod
    for shape in range(0,3):
        if shape!=s:
            sum_t=sum_t+joint(shape,pix,wts_fs,wts_fpis)
    return prod/sum_t

def joint(s,pix,wts_fs,wts_fpis):
    ws = 0
    wfpis = 0
    prod=1
    
    wts_fpis[0]=wts_fpis[0]/np.linalg.norm(wts_fpis[0])
    wts_fs[0]=wts_fs[0]/np.linalg.norm(wts_fs[0])
    
    for pi in range(0,len(pix)):
        prod=prod*np.exp(wts_fpis[0,s*2+int(pix[pi])])*np.exp(wts_fs[0,s])
#     for in1 in wts_fs[0]:
#         ws = ws + math.pow(in1,2)
#     for in2 in wts_fpis[0]:
#         wfpis = wfpis + math.pow(in2,2)
#     prod=prod/np.exp((lam/2)*(ws+wfpis))
    #prod= prod - math.exp(1*(ws + wfpis))
    return prod


# In[12]:


#####calculate empirical probability for gradient wrt 2nd parameter
emp_prob=np.zeros((1,6))

for m in range(0,M):
    for b in range(0,len(pixels[m])):
        for pix in pixels[m][b]:
            for s in range(0,3):
                for x in range(0,2):
                    if s==int(train_features['shape'][m]) and x==pix:
                        emp_prob[0,2*s+x]=emp_prob[0,2*s+x]+1

emp_prob=emp_prob/M*1.0
print emp_prob


# In[31]:


#############calculate marginal probability for gradient wrt 2nd parameter
cond=np.zeros((1,3))

its=100

for m in range(0,M):
    for t in range(0,its):
        pix=[]
        for b in range(0,len(pixels[m])): 
            for p in pixels[m][b]:
                pix.append(p)
        exp_prob=np.zeros((1,6))
        for s in range(0,3):
            for x in range(0,2):
                exp_prob[0,2*s+x]=exp_prob[0,2*s+x]+conditional(s,pix,wts_fs,wts_fpis)
        grad=emp_prob-exp_prob
        wts_fpis=wts_fpis+grad*(2/(2+t))
        if np.linalg.norm(grad)<1e-3:
            break
            
        
# for m in range(0,M):
#     pix=[]
#     for b in range(0,len(pixels[m])): 
#         for p in pixels[m][b]:
#             pix.append(p)
#     for s in range(0,3):
#         for x in range(0,2):
#             cond=conditional(s,pix,wts_fs,wts_fpis)
#             #print cond
#             for pi in range(0,len(pix)):
#                 if x==pix[pi] and s==int(train_features['shape'][m]):
#                     prob[m,2*s+x]=prob[m,2*s+x]+cond
                    
#     while (emp_prob[m]-prob[m]-(lam*wts_fpis[0])<-1e-1) or (emp_prob[m]-prob[m]-(lam*wts_fpis[0])>1e-1):
#         wts_fpis[0]=wts_fpis[0]+((emp_prob[m]-prob[m]-(lam*wts_fpis[0]))*(2.0/(2.0+t)))
#         cond=conditional(s,pix,wts_fs,wts_fpis)
#         for pi in range(0,len(pix)):
#                     if x==pix[pi] and s==int(train_features['shape'][m]):
#                         prob[m,2*s+x]=prob[m,2*s+x]+cond
#                         #print "hi1"
#                 t=t+1
#             t=1
#            # print "emp ",emp_prob[m,2*s+x]
#            # print "prob ",prob[m,2*s+x]
#             #print "diff ",emp_prob[m,2*s+x]-prob[m,2*s+x]
           
        
print wts_fpis 


# In[ ]:


#####calculate empirical probability for gradient wrt 3rd parameter
emp_prob=np.zeros((1,3))

for m in range(0,M):
    for s in range(0,3):
        if s==int(train_features['shape'][m]):
            emp_prob[0,s]=emp_prob[0,s]+1


emp_prob=emp_prob/M*1.0
print emp_prob


# In[ ]:


#############calculate marginal probability for gradient wrt 2nd parameter
cond=np.zeros((1,3))

its=100

for m in range(0,M):
    for t in range(0,its):
        pix=[]
        for b in range(0,len(pixels[m])): 
            for p in pix[m][b]:
                pix.append(p)
        exp_prob=np.zeros((1,3))
        for s in range(0,3):
            exp_prob[0,s]=exp_prob[0,s]+conditional(s,pixels,wts_fs,wts_fpis)
        grad=emp_prob-exp_prob
        wts_fs=wts_fs+grad*(2/(2+t))
        if np.linalg.norm(grad)<1e-3:
            break
        

# for m in range(0,M):
#     pix=[]
#     for b in range(0,len(pixels[m])): 
#         for p in pix[m][b]:
#             pix.append(p)
#     for s in range(0,3):
#         if s==int(train_features['shape'][m]):
#             cond=conditional(s,pixels,wts_fs,wts_fpis)
#             #print "cond ",cond
#             prob[m,s]=cond
#         t=1
#         #print emp_prob[m,s]-prob[m,s]-(lam*wts_fs[0,s])
#         while (emp_prob[m,s]-prob[m,s]-(lam*wts_fs[0,s])<-1e-1) or (emp_prob[m,s]-prob[m,s]-(lam*wts_fs[0,s])>1e-1):
#             wts_fs[0,s]=wts_fs[0,s]+(emp_prob[m,s]-prob[m,s]-(lam*wts_fs[0,s]))*(2.0/(2.0+t))
#             #print wts_fs
#             if s==int(train_features['shape'][m]):
#                 cond=conditional(s,pix,wts_fs,wts_fpis)
#                 prob[m,s]=cond
#             t=t+1
print wts_fs


# In[ ]:


cond=np.zeros((1,3))
m=0
pix=[]
    for b in range(0,len(pixels[m])): 
        for p in pix[m][b]:
            pix.append(p)
for s in range(0,3):
    cond[0,s]=conditional(s,pix,wts_fs,wts_fpis)
    print cond[0,s]

