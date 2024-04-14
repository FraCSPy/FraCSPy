import numpy as np
def ksvd(X,param):
	"""
	KSVD: the KSVD algorithm
	BY Yangkang Chen
	Jan, 2020
	Ported to Python in May, 2022
	
	INPUT
	X:     input training samples
	param: parameter struct
	  param.mode=1;   	#1: sparsity; 0: error
	  param.niter=10; 	#number of SGK iterations to perform; default: 10
	  param.D=DCT;    	#initial D
	  param.T=3;      	#sparsity level
	
	OUTPUT
	D:    learned dictionary
	G:    sparse coefficients
	
	for X=DG
	size of X: MxN
	size of D: MxK
	size of G: KxN
	
	DEMO
	demos/test_pyseisdl_sgk3d.py
	"""
	import scipy
	import scipy.linalg
	T=param['T'];	#T=1; 	#requred by SGK
	niter=param['niter'];
	mode=param['mode'];
	if 'K' in param:
		K=param['K'];
	else:
		K=param['D'].shape[1];	#dictionary size: number of atoms

	D=param['D'][:,0:K];

	for iter in range(0,niter):
	
		if mode==1:
			G=ompN(D,X,T);
			# exact form
		else:
			#error defined sparse coding
			pass;
			
		E0=X-np.matmul(D,G);#error before updating
		for ik in range(0,K): 	#KSVD iteration, K times SVD
			E=E0+np.matmul(np.expand_dims(D[:,ik],1),np.expand_dims(G[ik,:],0));
			inds,=np.where(G[ik,:]!=0);
			R=E[:,inds];
			[u,s,v]=scipy.linalg.svd(R);
			
			#scipy.sparse.linalg.svds  ?
			if u.size!=0:
				D[:,ik]=u[:,0];
				G[ik,inds]=s[0]*v[0,:];

	# extra step
	G=ompN(D,X,T);

	return D,G

def ompN( D, X, K ):
	"""
	multi-column sparse coding
	BY Yangkang Chen
	Jan, 2020
	"""
	[n1,n2]=X.shape
	[n1,n3]=D.shape
	G=np.zeros([n3,n2]);

	for i2 in range(0,n2):
		G[:,i2]=omp0(D,X[:,i2],K);

	return G

def omp0( D, x, K ):
	"""
	omp0: Most basic orthogonal matching pursuit for sparse coding
	this simple tutorial code use the sparseity-constrained sparse coding
	model
	
	two version of sparse coding problems:
	1) Sparseity-constrained sparse coding problem
	  gamma = min |x-Dg|_2^2  s.t. |g|_0 <= K 
	2) Error-constrained sparse coding problem
	  gamma = min |g|_0	   s.t. |x-Dg|_2^2 <=eps
	
	Author: Yangkang Chen
	Oct 25, 2016
	"""
	[n1,n2]=D.shape;
	I=[];
	r=x;
	g=np.zeros(n2);
	for ik in range(0,K):
		k=[];
		mmax=0;	                  	#initialize max=0
		for i2 in range(0,n2):	             	#loop over all atoms (greedy algorithm)       
			if not (i2 in I):  	#search among the other atoms
				dtr=np.abs(np.sum(D[:,i2]*r));
				if mmax<dtr:
					mmax=dtr;
					k=i2;
		I.append(k);
		g[I]=np.matmul(np.linalg.inv(np.matmul(D[:,I].transpose(),D[:,I])),np.matmul(D[:,I].transpose(),x)); 	#g_I = D_I^{+}x, D_I^TD_I is guaranteed to be non-singular 
# 		print('Dg',D[:,I].shape,g[I].shape)
		r=x-np.matmul(D[:,I],g[I]);

	return g