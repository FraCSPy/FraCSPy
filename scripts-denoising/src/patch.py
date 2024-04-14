import numpy as np
def patch2d( A,mode=1,l1=8,l2=8,s1=4,s2=4 ):
	"""
	patch2d: decompose the image into patches:
	 
	by Yangkang Chen
	Oct, 2017
	
	Input 
	  D: input image
	  mode: patching mode
	  l1: first patch size
	  l2: second patch size
	  s1: first shifting size
	  s2: second shifting size
	  
	Output
	  X: patches
	
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			      Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	
	Example
	sgk_denoise() in pyseisdl/denoise.py
	"""
	[n1,n2]=A.shape;

	if mode==1: 	#possible for other patching options
		
		tmp=np.mod(n1-l1,s1);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([s1-tmp,n2])),axis=0); 
		tmp=np.mod(n2-l2,s2);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],s2-tmp])),axis=2); 
		
		[N1,N2]=A.shape;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
					if i1==0 and i2==0:
						X=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3],[l1*l2*l3,1],order='F');
					else:
						tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],[l1*l2,1],order='F');
						X=np.concatenate((X,tmp),axis=1); 
	else:
		#not written yet
		pass;
	return X


def patch3d(A, mode,l1=4,l2=4,l3=4,s1=2,s2=2,s3=2):
	"""
	patch3d: decompose 3D data into patches:
	
	by Yangkang Chen
	March, 2020
	
	Input
	  D: input image
	  mode: patching mode
	  l1: first patch size
	  l2: second patch size
	  l3: third patch size
	  s1: first shifting size
	  s2: second shifting size
	  s3: third shifting size
	
	Output
	  X: patches
	
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			      Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	
	Example
	sgk_denoise() in pyseisdl/denoise.py
	"""

	[n1,n2,n3]=A.shape;

	if mode==1: 	#possible for other patching options
	
		tmp=np.mod(n1-l1,s1);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([s1-tmp,n2,n3])),axis=0);

		tmp=np.mod(n2-l2,s2);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],s2-tmp,n3])),axis=1);

		tmp=np.mod(n3-l3,s3);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],A.shape[1],s3-tmp])),axis=2);	#concatenate along the third dimension

		[N1,N2,N3]=A.shape;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					if i1==0 and i2==0 and i3==0:
						X=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3],[l1*l2*l3,1],order='F');
					else:
						tmp=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3],[l1*l2*l3,1],order='F');
						X=np.concatenate((X,tmp),axis=1);
	else:
		#not written yet
		pass;
		
	return X


def patch2d_inv( X,mode,n1,n2,l1=8,l2=8,s1=4,s2=4 ):
	"""
	patch2d_inv: insert patches into the image
	 
	by Yangkang Chen
	Oct, 2017
	
	Input 
	  D: input image
	  mode: patching mode
	  l1: first patch size
	  l2: second patch size
	  s1: first shifting size
	  s2: second shifting size
	  
	Output
	  X: patches
	
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
				  Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	
	Example
	sgk_denoise() in pyseisdl/denoise.py

	"""

	if mode==1: 	#possible for other patching options

		tmp1=np.mod(n1-l1,s1);
		tmp2=np.mod(n2-l2,s2);
		if tmp1!=0 and tmp2!=0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2]); 
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2]); 

		if tmp1!=0 and tmp2==0:
			A=np.zeros([n1+s1-tmp1,n2]); 
			mask=np.zeros([n1+s1-tmp1,n2]);

		if tmp1==0 and tmp2!=0:
			A=np.zeros([n1,n2+s2-tmp2]);   
			mask=np.zeros([n1,n2+s2-tmp2]);   


		if tmp1==0 and tmp2==0:
			A=np.zeros([n1,n2]); 
			mask=np.zeros([n1,n2]);

		[N1,N2]=A.shape;
		id=-1;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				id=id+1;
				A[i1:i1+l1,i2:i2+l2]=A[i1:i1+l1,i2:i2+l2]+np.reshape(X[:,id],[l1,l2],order='F');
				mask[i1:i1+l1,i2:i2+l2]=mask[i1:i1+l1,i2:i2+l2]+np.ones([l1,l2]);

		A=A/mask; 
		A=A[0:n1,0:n2];
	else:
		#not written yet
		pass;
	return A


def patch3d_inv( X,mode,n1,n2,n3,l1=4,l2=4,l3=4,s1=2,s2=2,s3=2 ):
	"""
	patch3d_inv: insert patches into the 3D data
	
	by Yangkang Chen
	March, 2020
	
	Input
	  D: input image
	  mode: patching mode
	  n1: first dimension size
	  n1: second dimension size
	  n3: third dimension size
	  l1: first patch size
	  l2: second patch size
	  l3: third patch size
	  s1: first shifting size
	  s2: second shifting size
	  s3: third shifting size
	
	Output
	  X: patches
	
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			      Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	            Marich, 31, 2020, 2D->3D
	
	Example
	sgk_denoise() in pyseisdl/denoise.py

	"""

	if mode==1: 	#possible for other patching options
	
		tmp1=np.mod(n1-l1,s1);
		tmp2=np.mod(n2-l2,s2);
		tmp3=np.mod(n3-l3,s3);
		if tmp1!=0 and tmp2!=0 and tmp3!=0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3]);
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3]);

		if tmp1!=0 and tmp2!=0 and tmp3==0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3]);
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3]);
	
		if tmp1!=0 and tmp2==0 and tmp3==0:
			A=np.zeros([n1+s1-tmp1,n2,n3]);
			mask=np.zeros([n1+s1-tmp1,n2,n3]);
	
		if tmp1==0 and tmp2!=0 and tmp3==0:
			A=np.zeros([n1,n2+s2-tmp2,n3]);
			mask=np.zeros([n1,n2+s2-tmp2,n3]);
	
		if tmp1==0 and tmp2==0 and tmp3!=0:
			A=np.zeros([n1,n2,n3+s3-tmp3]);
			mask=np.zeros([n1,n2,n3+s3-tmp3]);
	
		if tmp1==0 and tmp2==0  and tmp3==0:
			A=np.zeros([n1,n2,n3]);
			mask=np.zeros([n1,n2,n3]);
	
		[N1,N2,N3]=A.shape;
		id=-1;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					id=id+1;
					A[i1:i1+l1,i2:i2+l2,i3:i3+l3]=A[i1:i1+l1,i2:i2+l2,i3:i3+l3]+np.reshape(X[:,id],[l1,l2,l3],order='F');
					mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]=mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]+np.ones([l1,l2,l3]);
		A=A/mask;
	
		A=A[0:n1,0:n2,0:n3];
	else:
		#not written yet
		pass;
	return A


def patch5d(A,mode,l1=4,l2=4,l3=4,l4=4,l5=4,s1=2,s2=2,s3=2,s4=2,s5=2):
	"""
	patch5d: decompose 4D/5D data into patches:
	
	by Yangkang Chen
	March, 2020
	
	Input
	  D: input image
	  mode: patching mode
	  l1: first patch size
	  l2: second patch size
	  l3: third patch size
	  l4: fourth patch size
	  l5: fifth patch size	(when n5=1, l5=1, s5=0)
	  s1: first shifting size
	  s2: second shifting size
	  s3: third shifting size
	  s4: fourth shifting size
	  s5: fifth shifting size (when n5=1, l5=1, s5=0)
	
	Output
	  X: patches
	
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
				  Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	            April 2, 2020 (3D-5D)
	
	Example
	sgk_denoise() in pyseisdl/denoise.py

	"""

	[n1,n2,n3,n4,n5]=A.shape;

	if mode==1: 	#possible for other patching options
	
		tmp=np.mod(n1-l1,s1);
		if tmp!=0:
			A=np.concatenate((A,zeros(s1-tmp,n2,n3,n4,n5)),axis=0);
	
		tmp=np.mod(n2-l2,s2);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],s2-tmp,n3,n4,n5)),axis=1);
	
		tmp=np.mod(n3-l3,s3);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],A.shape[1],s3-tmp,n4,n5)),axis=2);	#concatenate along the third dimension
	
		tmp=np.mod(n4-l4,s4);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],A.shape[1],A.shape[2],s4-tmp,n5)),axis=3);	#concatenate along the forth dimension

		tmp=np.mod(n5-l5,s5);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],A.shape[1],A.shape[2],A.shape[3],s5-tmp)),axis=4);	#concatenate along the fifth dimension  
	
		[N1,N2,N3,N4,N5]=A.shape;
		X=np.array([]);
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					for i4 in range(0,N4-l4+1,s4):
						for i5 in range(0,N5-l5+1,s5):
							tmp=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5],[l1*l2*l3*l4*l5,1],order='F');
							X=np.concatenate((X,tmp),axis=1);
	else:
		#not written yet
		pass;
	return X

def patch5d_inv( X,mode,n1,n2,n3,n4,n5,l1=4,l2=4,l3=4,l4=4,l5=4,s1=2,s2=2,s3=2,s4=2,s5=2):
	"""
	patch5d_inv: insert patches into the 4D/5D data
	
	by Yangkang Chen
	April, 2020
	
	Input
	  D: input image
	  mode: patching mode
	  n1: first dimension size
	  n1: second dimension size
	  n3: third dimension size
	  n4: forth dimension size
	  n5: fifth dimension size
	  l1: first patch size
	  l2: second patch size
	  l3: third patch size
	  l4: fourth patch size
	  l5: fifth patch size	(when n5=1, l5=1, s5=0)
	  s1: first shifting size
	  s2: second shifting size
	  s3: third shifting size
	  s4: fourth shifting size
	  s5: fifth shifting size (when n5=1, l5=1, s5=0)
	  
	Output
	  X: patches
	
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
				  Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	            Marich, 31, 2020, 2D->3D
	            April 2, 2020 (3D-5D)
	
	Example
	sgk_denoise() in pyseisdl/denoise.py

	"""

	if mode==1: 	#possible for other patching options
		tmp1=np.mod(n1-l1,s1);
		tmp2=np.mod(n2-l2,s2);
		tmp3=np.mod(n3-l3,s3);
		tmp4=np.mod(n4-l4,s4);
		tmp5=np.mod(n5-l5,s5);
	
		if tmp1!=0 and tmp2!=0 and tmp3!=0 and tmp4!=0 and tmp5!=0:
			A=zeros(n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3,n4+s4-tmp4,n5+s5-tmp5);
			mask=zeros(n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3,n4+s4-tmp4,n5+s5-tmp5);

	
		if tmp1!=0 and tmp2==0 and tmp3==0 and tmp4==0 and tmp5==0:
			A=zeros(n1+s1-tmp1,n2,n3,n4,n5);
			mask=zeros(n1+s1-tmp1,n2,n3,n4,n5);

	
		if tmp1==0 and tmp2!=0 and tmp3==0 and tmp4==0 and tmp5==0:
			A=zeros(n1,n2+s2-tmp2,n3,n4,n5);
			mask=zeros(n1,n2+s2-tmp2,n3,n4,n5);

	
		if tmp1==0 and tmp2==0 and tmp3!=0 and tmp4==0 and tmp5==0:
			A=zeros(n1,n2,n3+s3-tmp3,n4,n5);
			mask=zeros(n1,n2,n3+s3-tmp3,n4,n5);


		if tmp1==0 and tmp2==0 and tmp3==0 and tmp4!=0 and tmp5==0:
			A=zeros(n1,n2,n3,n4+s4-tmp4,n5);
			mask=zeros(n1,n2,n3,n4+s4-tmp4,n5);

	
		if tmp1==0 and tmp2==0 and tmp3==0 and tmp4==0 and tmp5!=0:
			A=zeros(n1,n2,n3,n4,n5+s5-tmp5);
			mask=zeros(n1,n2,n3,n4,n5+s5-tmp5);

	
		if tmp1==0 and tmp2==0  and tmp3==0 and tmp4==0 and tmp5==0:
			A=zeros(n1,n2,n3,n4,n5);
			mask=zeros(n1,n2,n3,n4,n5);


		[N1,N2,N3,N4,N5]=A.shape;
		id=-1;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					for i4 in range(0,N4-l4+1,s4):
						for i5 in range(0,N5-l5+1,s5):
							A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]=A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]+np.reshape(X[:,id],[l1,l2,l3,l4,l5],order='F');
							mask[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]=mask[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]+np.ones([l1,l2,l3,l4,l5]);
		A=A/mask;
		A=A[0:n1,0:n2,0:n3,0:n4,0:n5];
	else:
		#not written yet
		pass;
	return A



