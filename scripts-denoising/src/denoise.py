import numpy as np
def sgk_denoise(din,mode,l,s,perc,param):
	"""
	sgk_denoise: SGK algorithm for 2D and 3D denoising
	BY Yangkang Chen
	Jan, 2020
	Modified on Mar, 2020
	
	INPUT
	  din: input data
	  mode: patching mode
	  l: [l1,l2,l3]
	  l1: first patch size
	  l2: second patch size
	  l3: third patch size
	  s: [s1,s2,s3]
	  s1: first shifting size
	  s2: second shifting size
	  s3: third shifting size
	  perc: percentage
	  param: parameter struct for DL
	  param.mode=1;   	#1: sparsity; 0: error
	  param.niter=10; 	#number of K-SVD iterations to perform; default: 10
	  param.D=DCT;    	#initial D
	  param.T=3;      	#sparsity level
	
	OUTPUT
	dout:
	Dsgk,Gsgkc: learned dictionary and coefficients
	DCT: initial dictionary
	
	Key Reference
	Chen, Y., 2020, Fast dictionary learning for noise attenuation of multidimensional seismic data, Geophysical Journal International, 222, 1717-1727.
	
	Other related references (e.g., introducing the subroutines)
	Siahsar, M. A. N., Gholtashi, S., Kahoo, A. R., W. Chen, and Y. Chen, 2017, Data-driven multi-task sparse dictionary learning for noise attenuation of 3D seismic data, Geophysics, 82, V385-V396.
	Siahsar, M. A. N., V. Abolghasemi, and Y. Chen, 2017, Simultaneous denoising and interpolation of 2D seismic data using data-driven non-negative dictionary learning, Signal Processing, 141, 309-321.
	Chen, Y., M. Zhang, M. Bai, and W. Chen, 2019, Improving the signal-to-noise ratio of seismological datasets by unsupervised machine learning, Seismological Research Letters, 90, 1552-1564.
	Chen, Y., S. Zu, W. Chen, M. Zhang, and Z. Guan, 2020, Learning the blending spikes using sparse dictionaries, Geophysical Journal International, 222, 1846–1863.  
	Wang, H., Q. Zhang, G. Zhang, J. Fang, and Y. Chen, 2020, Self-training and learning the waveform features of microseismic data using an adaptive dictionary, Geophysics, 85, KS51?KS61.
	Chen, Y., S. Fomel, 2015, Random noise attenuation using local signal-and-noise orthogonalization, Geophysics, 80, WD1-WD9.
	Chen, Y., J. Ma, and S. Fomel, 2016, Double-sparsity dictionary for seismic noise attenuation, Geophysics, 81, V17-V30.
	Zu, S., H. Zhou, R. Wu, M. Jiang, and Y. Chen, 2019, Dictionary learning based on dip patch selection training for random noise attenuation, Geophysics, 84, V169?V183.
	Zu, S., H. Zhou, R. Wu, and Y. Chen, 2019, Hybrid-sparsity constrained dictionary learning for iterative deblending of extremely noisy simultaneous-source data, IEEE Transactions on Geoscience and Remote Sensing, 57, 2249-2262.
	etc. 
	"""
	
	from .sgk import sgk
	from .patch import patch2d,patch2d_inv,patch3d,patch3d_inv
	from .threshold import pthresh
	[n1,n2,n3]=din.shape;
	l1=l[0];
	l2=l[1];
	l3=l[2];

	s1=s[0];
	s2=s[1];
	s3=s[2];

	#initialization
	#[c1,c2,c3]: redundancy of the initial atom in 1st,2nd,3rd dimensions
	#[l1,l2,l3]: patch sizes and the atom sizes in each dimension
	if not ('D' in param):
		if 'K' in param:
			if n3==1:
				c1=np.ceil(np.sqrt(param['K']));c1=int(c1);
				c2=c1;
			else:
				c1=np.round(np.power(param['K'],(1/3.0)));c1=int(c1);
				c2=c1;
				c3=c1;
		else:
			c1=l1;
			c2=l2;
			c3=l3;

		dct1=np.zeros([l1,c1]);
		for k in range(0,c1):
			tmp=np.arange(l1);
			tmp=tmp[...,None]; #column vector
			V=np.cos(tmp*k*np.pi/c1); 
			if k>0:
				V=V-np.mean(V);
			dct1[:,k]=V.squeeze()/np.linalg.norm(V);
	
		dct2=np.zeros([l2,c2]);
		for k in range(0,c2):
			tmp=np.arange(l2);
			tmp=tmp[...,None]; #column vector
			V=np.cos(tmp*k*np.pi/c2);
			if k>0:
				V=V-np.mean(V);
			dct2[:,k]=V.squeeze()/np.linalg.norm(V);
	
		if n3!=1:
			dct3=np.zeros([l3,c3]);
			for k in range(0,c3):
				tmp=np.arange(l3);
				tmp=tmp[...,None]; #column vector
				V=np.cos(tmp*k*np.pi/c3);
				if k>0:
					V=V-np.mean(V);
				dct3[:,k]=V.squeeze()/np.linalg.norm(V);
	
		if n3==1:
			DCT=np.kron(dct1,dct2);	#2D DCT dictionary (l1*l2,c1*c2)
		else:
			DCT=np.kron(np.kron(dct1,dct2),dct3);	#3D DCT dictionary (l1*l2*l3,c1*c2*c3)
		param['D']=DCT;

	#SGK
	if n3==1:
		X=patch2d(din,mode,l1,l2,s1,s2);
		[Dsgk,Gsgk]=sgk(X,param);
		Gsgkc=Gsgk;
		Gsgk,thr=pthresh(Gsgkc,'ph',perc);
		X2=np.matmul(Dsgk,Gsgk);
		dout=patch2d_inv(X2,mode,n1,n2,l1,l2,s1,s2);
	else:
		X=patch3d(din,mode,l1,l2,l3,s1,s2,s3);
		[Dsgk,Gsgk]=sgk(X,param);
		Gsgkc=Gsgk;
		Gsgk,thr=pthresh(Gsgkc,'ph',perc);
		X2=np.matmul(Dsgk,Gsgk);
		dout=patch3d_inv(X2,mode,n1,n2,n3,l1,l2,l3,s1,s2,s3);

	return dout,Dsgk,Gsgkc,DCT



def ksvd_denoise(din,mode,l,s,perc,param):
	"""
	ksvd_denoise: SGK algorithm for 2D and 3D denoising
	BY Yangkang Chen
	Jan, 2020
	Modified on Mar, 2020
	
	INPUT
	  din: input data
	  mode: patching mode
	  l: [l1,l2,l3]
	  l1: first patch size
	  l2: second patch size
	  l3: third patch size
	  s: [s1,s2,s3]
	  s1: first shifting size
	  s2: second shifting size
	  s3: third shifting size
	  perc: percentage
	  param: parameter struct for DL
	  param.mode=1;   	#1: sparsity; 0: error
	  param.niter=10; 	#number of K-SVD iterations to perform; default: 10
	  param.D=DCT;    	#initial D
	  param.T=3;      	#sparsity level
	
	OUTPUT
	dout:
	Dksvd,Gsgkc: learned dictionary and coefficients
	DCT: initial dictionary
	
	Related Reference:
	Chen, Y., 2020, Fast dictionary learning for noise attenuation of multidimensional seismic data, Geophysical Journal International, 222, 1717-1727.
	Siahsar, M. A. N., Gholtashi, S., Kahoo, A. R., W. Chen, and Y. Chen, 2017, Data-driven multi-task sparse dictionary learning for noise attenuation of 3D seismic data, Geophysics, 82, V385-V396.
	Siahsar, M. A. N., V. Abolghasemi, and Y. Chen, 2017, Simultaneous denoising and interpolation of 2D seismic data using data-driven non-negative dictionary learning, Signal Processing, 141, 309-321.
	Chen, Y., M. Zhang, M. Bai, and W. Chen, 2019, Improving the signal-to-noise ratio of seismological datasets by unsupervised machine learning, Seismological Research Letters, 90, 1552-1564.
	Chen, Y., S. Zu, W. Chen, M. Zhang, and Z. Guan, 2020, Learning the blending spikes using sparse dictionaries, Geophysical Journal International, 222, 1846–1863.  
	Wang, H., Q. Zhang, G. Zhang, J. Fang, and Y. Chen, 2020, Self-training and learning the waveform features of microseismic data using an adaptive dictionary, Geophysics, 85, KS51?KS61.
	Chen, Y., S. Fomel, 2015, Random noise attenuation using local signal-and-noise orthogonalization, Geophysics, 80, WD1-WD9.
	Chen, Y., J. Ma, and S. Fomel, 2016, Double-sparsity dictionary for seismic noise attenuation, Geophysics, 81, V17-V30.
	Zu, S., H. Zhou, R. Wu, M. Jiang, and Y. Chen, 2019, Dictionary learning based on dip patch selection training for random noise attenuation, Geophysics, 84, V169?V183.
	Zu, S., H. Zhou, R. Wu, and Y. Chen, 2019, Hybrid-sparsity constrained dictionary learning for iterative deblending of extremely noisy simultaneous-source data, IEEE Transactions on Geoscience and Remote Sensing, 57, 2249-2262.
	etc. 
	"""
	from .ksvd import ksvd
	from .patch import patch2d,patch2d_inv,patch3d,patch3d_inv
	from .threshold import pthresh
	[n1,n2,n3]=din.shape;
	l1=l[0];
	l2=l[1];
	l3=l[2];

	s1=s[0];
	s2=s[1];
	s3=s[2];

	#initialization
	#[c1,c2,c3]: redundancy of the initial atom in 1st,2nd,3rd dimensions
	#[l1,l2,l3]: patch sizes and the atom sizes in each dimension
	if not ('D' in param):
		if 'K' in param:
			if n3==1:
				c1=np.ceil(np.sqrt(param['K']));c1=int(c1);
				c2=c1;
			else:
				c1=np.round(np.power(param['K'],(1/3.0)));c1=int(c1);
				c2=c1;
				c3=c1;
		else:
			c1=l1;
			c2=l2;
			c3=l3;

		dct1=np.zeros([l1,c1]);
		for k in range(0,c1):
			tmp=np.arange(l1);
			tmp=tmp[...,None]; #column vector
			V=np.cos(tmp*k*np.pi/c1); 
			if k>0:
				V=V-np.mean(V);
			dct1[:,k]=V.squeeze()/np.linalg.norm(V);
	
		dct2=np.zeros([l2,c2]);
		for k in range(0,c2):
			tmp=np.arange(l2);
			tmp=tmp[...,None]; #column vector
			V=np.cos(tmp*k*np.pi/c2);
			if k>0:
				V=V-np.mean(V);
			dct2[:,k]=V.squeeze()/np.linalg.norm(V);
	
		if n3!=1:
			dct3=np.zeros([l3,c3]);
			for k in range(0,c3):
				tmp=np.arange(l3);
				tmp=tmp[...,None]; #column vector
				V=np.cos(tmp*k*np.pi/c3);
				if k>0:
					V=V-np.mean(V);
				dct3[:,k]=V.squeeze()/np.linalg.norm(V);
	
		if n3==1:
			DCT=np.kron(dct1,dct2);	#2D DCT dictionary (l1*l2,c1*c2)
		else:
			DCT=np.kron(np.kron(dct1,dct2),dct3);	#3D DCT dictionary (l1*l2*l3,c1*c2*c3)
		param['D']=DCT;

	#KSVD
	if n3==1:
		X=patch2d(din,mode,l1,l2,s1,s2);
		[Dsgk,Gsgk]=ksvd(X,param);
		Gksvdc=Gksvd;
		Gksvd,thr=pthresh(Gksvdc,'ph',perc);
		X2=np.matmul(Dksvd,Gksvd);
		dout=patch2d_inv(X2,mode,n1,n2,l1,l2,s1,s2);
	else:
		X=patch3d(din,mode,l1,l2,l3,s1,s2,s3);
		[Dksvd,Gksvd]=ksvd(X,param);
		Gksvdc=Gksvd;
		Gksvd,thr=pthresh(Gksvdc,'ph',perc);
		X2=np.matmul(Dksvd,Gksvd);
		dout=patch3d_inv(X2,mode,n1,n2,n3,l1,l2,l3,s1,s2,s3);

	return dout,Dksvd,Gksvdc,DCT