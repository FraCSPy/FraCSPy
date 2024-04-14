def pthresh(x,sorh,t):
	"""
	PTHRESH Perform soft or hard thresholding or percentile
		    soft or hard thresholding.  
	   Y = WTHRESH(X,SORH,T) returns soft (if SORH = 's') 
	   or hard (if SORH = 'h') T-thresholding  of the input  
	   vector or matrix X. T is the threshold value. 
	 
	   Y = WTHRESH(X,'s',T) returns Y = SIGN(X).(|X|-T)+, soft  
	   thresholding is shrinkage. 
	 
	   Y = WTHRESH(X,'h',T) returns Y = X.1_(|X|>T), hard 
	   thresholding is cruder. 
	 
	   See also WDEN, WDENCMP, WPDENCMP. 
 
	   M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96. 
	
	   Yangkang Chen, The University of Texas at Austin
	"""
	import numpy as np

	if sorh == 's':
		tmp = (np.abs(x)-t); 
		tmp = (tmp+np.abs(tmp))/2; 
		y   = np.sign(x)*tmp; 
	elif sorh=='h':
		y=x*(np.abs(x)>t);
	
	elif sorh=='ps':
		tmp=np.abs(x).flatten();
		t=np.percentile(tmp,100-t);
		thr=t;
		tmp = (np.abs(x)-t); 
		tmp = (tmp+np.abs(tmp))/2; 
		y   = np.sign(x)*tmp; 
	
	elif sorh=='ph':
		tmp=np.abs(x).flatten();
		t=np.percentile(tmp,100-t);
		thr=t;
		y   = x*(np.abs(x)>t);
	else:
		print('Invalid argument value.'); 

	return y,thr