import numpy as np
def sgk(X,param):
	"""
	sgk: SGK algorithm
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

	T=param['T'];	#T=1; 	#requred by SGK
	niter=param['niter'];
	mode=param['mode'];
	if 'K' in param:
		K=param['K'];
	else:
		K=param['D'].shape[1];	#dictionary size: number of atoms

	D=param['D'][:,0:K];

	for iter in range(1,niter+1):
	
		if mode==1:
			G=ompN(D,X,1);
			# exact form
		else:
			#error defined sparse coding
			pass;
			
		for ik in range(0,K): 	#SGK iteration, K times means 
			inds,=np.where(G[ik,:]!=0);
			if inds.shape[0]!=0:		#empty array
				D[:,ik]=np.sum(X[:,inds],1);	#better using a weighted summation ? NO, equivalent 
				D[:,ik]=D[:,ik]/np.linalg.norm(D[:,ik]); 

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
	if K==1:
		for i2 in range(0,n2):
			G[:,i2]=omp_e(D,X[:,i2]);
	else:
		for i2 in range(0,n2):
			G[:,i2]=omp0(D,X[:,i2],K);

	return G

def omp_e( D, x ):
	"""
	omp_e: Most basic orthogonal matching pursuit for sparse coding
	this simple tutorial code use the sparseity-constrained sparse coding
	model
	#
	two version of sparse coding problems:
	1) Sparseity-constrained sparse coding problem
	  gamma = min |x-Dg|_2^2  s.t. |g|_0 <= K
	2) Error-constrained sparse coding problem
	  gamma = min |g|_0	   s.t. |x-Dg|_2^2 <=eps
	#
	Author: Yangkang Chen
	Oct 25, 2016
	"""
	[n1,n2]=D.shape;
	g=np.zeros(n2);

	k=0;
	mmax=0;                      	#initialize max=0
	for i2 in range(0,n2):               	#loop over all atoms (greedy algorithm)
		dtr=np.abs(np.sum(D[:,i2]*x));
		if mmax<dtr:
			mmax=dtr;
			k=i2;

	g[k]=np.sum(D[:,k]*x)/np.sum(D[:,k]*D[:,k]); 	
	#option 1: more accurate sparse coding (according to definition of sparse coding)
	# g(k)=1.0; 	
	#option 2: according to the requirement of equation (8) in Chen (2017), GJI.
	# Note: option1 and option2 result in exactly the same result. 
	return g


def spray( G, n ):
	"""
	multi-column sparse coding
	BY Yangkang Chen
	Jan, 2020
	G: row or column vector
	axis: 1 or 2
	n:  size
	"""
	[n1,n2]=G.shape;
	if n1==1: 	#row vector, axis=1
		G2=np.zeros([n,n2]);
		for i1 in range(0,n):
			G2[i1,:]=G;
	else: 	#column vector, axis=2
		G2=np.zeros([n1,n]);
		for i2 in range(0,n):
			G2[:,i2]=G;
	return G2

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