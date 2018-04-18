import re,sys
import pandas as pd
import numpy as np
import math

file=open('shapeset\shapeset1_1cs_2p_3o.5000.valid.amat',"r");   #.amat file containing feature information

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

#####################converts feature dictionart to one single data frame
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



'''def lbp(pixels,wts_fpi,wts_fs,wts_fpis,its):
	
	Msgs=np.ones((k,T,T))
	for t in range(0,1):
		for i in range(0,T):
			for j in neighbors[i]:
				Msgs[:,i,j]=calc_message(pixels,wts_fpi,wts_fs,wts_fpis,Msgs,i,j)

	node_marg=np.ones((N+1,k))
	for i in range(0,N+1):
		nbr_msg=np.ones((1,k))
		if i==1024:
			for j in range(1025,1031):
				a=np.array(nbr_msg)
				b=np.array(np.squeeze(Msgs[:,i,j]))
				nbr_msg=np.multiply(a,b)
		else:
			for j in neighbors[i]:
				a=np.array(nbr_msg)
				b=np.array(np.squeeze(Msgs[:,i,j]))
				nbr_msg=np.multiply(a,b)

		nbr_msg=nbr_msg/np.sum(nbr_msg)
		node_marg[i,:]=nbr_msg


	beliefs={}
	for i in range(0,N+1):
		beliefs[i]=node_marg[i,:]
	# phi=np.exp(wts_fs)
	# nbrs=neighbors[N]
	# for shape in range(0,k):
	# 	for nid in range(1025,1031):
	# 		phi[shape]=phi[shape]*Msgs[shape,nid,N]

	# phi=phi/np.sum(phi)
	# beliefs[N]=phi

	return beliefs

def calc_message(pixels,wts_fpi,wts_fs,wts_fpis,Msgs,i,j):
	
	msg=np.zeros((1,k));
	sum_t=0
	##### message from node i to clique j
	if i<N:
		nbrs=neighbors[i]
		prod=np.exp(wts_fpi[0,pixels[i]])
		for shape in range(0,k):
			msg[0,shape]=prod
		#print str(i)+"      "+str(msg)
	##### meeage from shape node to clique j
	elif i==N:
		nbrs=neighbors[i]
		prod=np.exp(wts_fs)
		for shape in range(0,k):
			for nid in nbrs:
				if nid!=j:
					prod[0,shape]=prod[0,shape]*Msgs[shape,nid,i]
		msg=prod
	elif i>N:
		###### message from clique i to node j
		nbrs=neighbors[i]
		nbrs_without_j=nbrs[1]
		prod=np.ones((1,k))
		sum_t=0
		for shape in range(0,k):
			prod[0,shape]=prod[0,shape]*np.exp(wts_fpis[0,2*shape+pixels[nbrs_without_j]])*Msgs[shape,nbrs_without_j,i]
		msg=prod

	msg=msg/np.sum(msg)
	
	return msg'''

def conditional(s,pixels,wts_fs,wts_fpis):
	prod=joint(s,pixels,wts_fs,wts_fpis)
	sum_t=prod
	for shape in range(0,3):
		if shape!=s:
			sum_t=sum_t+joint(shape,pixels,wts_fs,wts_fpis)
	return prod/sum_t

def joint(s,pixels,wts_fs,wts_fpis):
	prod=1
	for pi in range(0,N):
		prod=prod*np.exp(wts_fpis[0,s*2+pixels[pi]])*np.exp(wts_fs[0,s])
	return prod
	

'''#####calculate empirical probability for gradient wrt 1st parameter
emp_prob=np.zeros((M,2))

for m in range(0,M):
	for x in range(0,2):
		for pi in range(0,1024):
			if x==train_features['pixels'][m][pi]:
				emp_prob[m,x]=emp_prob[m,x]+1

emp_prob=emp_prob/M*1.0
#############calculate marginal probability for gradient wrt 1st parameter

prob=np.zeros((M,2))

for m in range(0,M):
	for x in range(0,2):
		pixels=[]
		for pi in range(0,1024):
			pixels.append(train_features['pixels'][m][pi])
		be=lbp(pixels,wts_fpi,wts_fs,wts_fpis,5)
		prob=np.zeros((M,2))
		for pi in range(0,N):
			if train_features['pixels'][m][pi]==x:				
				prob[m,x]=be[pi][x]
		print "gfdgfd"+"    "+str(prob[m,x])
		t=1
		while abs(emp_prob[m,x]-prob[m,x])>1e-3:
			print emp_prob[m,x]-prob[m,x]
			wts_fpi[0,x]=wts_fpi[0,x]+(emp_prob[m,x]-prob[m,x])*(2.0/(2.0+t))
			be=lbp(pixels,wts_fpi,wts_fs,wts_fpis,5)
			prob=np.zeros((M,2))
			for pi in range(0,N):
				if train_features['pixels'][m][pi]==x:
					prob[m,x]=be[pi][x]
			print prob[m,x]
			t=t+1



		# cond_prob[x]=lbp(s,pixels,wts_fpi,wts_fs,wts_fpis,5)
		# for x in range(0,2):
		# 	count=0
		# 	for pi in range(0,1024):
		# 		if x==train_features['pixels'][m][pi]:
		# 			count=count+1
		# 	prob[x]=count*cond_prob[s]
		# 	t=0
		# 	while emp_prob[x]-prob[x]>1e-3:
		# 		wts_fpi[x]=wts_fpi[x]+(emp_prob[x]-prob[x])*(2/(2+t))
		# 		cond_prob[s]=lbp(s,pixels,wts_fpi,wts_fs,wts_fpis,5)
		# 		count=0
		# 		for pi in range(0,1024):
		# 			if x==train_features['pixels'][m][pi]:
		# 				count=count+1
		# 		prob[x]=count*cond_prob[s]
		# 		t=t+1'''


#####calculate empirical probability for gradient wrt 2nd parameter
emp_prob=np.zeros((M,6))

for m in range(0,M):
	for pi in range(0,1024):
		for s in range(0,3):
			for x in range(0,2):
				if s==int(train_features['shape'][m]) and x==train_features['pixels'][m][pi]:
					emp_prob[m,2*s+x]=emp_prob[m,2*s+x]+1

emp_prob=emp_prob/M*1.0
#############calculate marginal probability for gradient wrt 2nd parameter
prob=np.zeros((M,6))

for m in range(0,M):
	pixels=[]
	for pi in range(0,N):
		pixels.append(train_features['pixels'][m][pi])
	for s in range(0,3):
		for x in range(0,2):
			cond=conditional(s,pixels,wts_fs,wts_fpis)
			for pi in range(0,N):
				if x==train_features['pixels'][m][pi] and s==train_features['shape'][m]:
					prob[m,2*s+x]=cond
			t=1
			while emp_prob[m,2*s+x]-prob[m,2*s+x]>1e-3:
				wts_fpis[0,2*s+x]=wts_fpis[0,2*s+x]+(emp_prob[m,2*s+x]-prob[m,2*s+x])*(2.0/(2.0+t))
				cond=conditional(s,pixels,wts_fs,wts_fpis)
				for pi in range(0,N):
					if x==train_features['pixels'][m][pi] and s==train_features['shape'][m]:
						prob[m,2*s+x]=cond
				t=t+1

			# count=0
			# for pi in range(0,1024):
			# 	if x==train_features['pixels'][m][pi] and s==train_features['shape'][m]:
			# 		count=count+1
			# prob[xs]=count*cond_prob[s]
			# t=0
			# while emp_prob[xs]-prob[xs]>1e-3:
			# 	wts_fpis[xs]=wts_fpis[xs]+(emp_prob[xs]-prob[xs])*(2/(2+t))
			# 	cond_prob[s]=lbp(s,pixels,wts_fpi,wts_fs,wts_fpis,5)
			# 	count=xs=xs+1

			# 	for pi in range(0,1024):
			# 		if x==train_features['pixels'][m][pi] and s==train_features['shape'][m]:
			# 			count=count+1
			# 	prob[xs]=count*cond_prob[s]
			# 	t=t+1

#####calculate empirical probability for gradient wrt 3rd parameter
emp_prob=np.zeros((M,3))

for m in range(0,M):
	for s in range(0,3):
		if s==train_features['shape'][m]:
			emp_prob[m,s]=emp_prob[m,s]+1


emp_prob=emp_prob/M*1.0
#############calculate marginal probability for gradient wrt 2nd parameter
prob=np.zeros((M,3))

for m in range(0,M):
	pixels=[]
	for pi in range(0,1024):
		pixels.append(train_features['pixels'][m][pi])
	for s in range(0,3):
		if s==train_features['shape'][m]:
			cond=conditional(s,pixels,wts_fpi,wts_fs,wts_fpis,1)
			prob[m,s]=cond
		t=1
		while emp_prob[m,s]-prob[m,s]>1e-3:
			wts_fs[0,s]=wts_fs[0,s]+(emp_prob[m,s]-prob[m,s])*(2.0/(2.0+t))
			cond=conditional(s,pixels,wts_fpi,wts_fs,wts_fpis,5)
			prob[m,s]=cond
			t=t+1
			

