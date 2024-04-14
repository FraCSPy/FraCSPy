def localsimi(d1,d2,rect,niter=50,eps=0.0,verb=1):
	#LOCALSIMI: calculate local similarity between two datasets
	#
	#IN   d1:   	input data 1
	#     d2:     input data 2
	#     rect:   3-D vector denoting smooth radius
	#     niter:  number of CG iterations
	#     eps:    regularization parameter, default 0.0
	#     verb:   verbosity flag (default: 0)
	#
	#OUT  simi:  	calculated local similarity, which is of the same size as d1 and d2
	#
	#Copyright (C) 2016 Yangkang Chen
	#Ported to Python in 2022 by Yangkang Chen 
	#
	#This program is free software: you can redistribute it and/or modify
	#it under the terms of the GNU General Public License as published
	#by the Free Software Foundation, either version 3 of the License, or
	#any later version.
	#
	#This program is distributed in the hope that it will be useful,
	#but WITHOUT ANY WARRANTY; without even the implied warranty of
	#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#GNU General Public License for more details: http://www.gnu.org/licenses/
	#
	#Reference:   
	#				1. Chen, Y. and S. Fomel, 2015, Random noise attenuation using local signal-and-noise orthogonalization, Geophysics, , 80, WD1-WD9. (Note that when local similarity is used for noise suppression purposes, this reference must be cited.)
	#             2. Local seismic attributes, Fomel, Geophysics, 2007
	#
	# DEMO
	# demos/test_pyortho_localortho2d.py
	# demos/test_pyortho_localortho3d.py
	
	import numpy as np
	from .divne import divne
	
	if d1.ndim==2:
		d1=np.expand_dims(d1, axis=2)
	if d2.ndim==2:
		d2=np.expand_dims(d2, axis=2)
	[n1,n2,n3]=d1.shape

	nd=n1*n2*n3;
	ndat=[n1,n2,n3];
	eps_dv=eps;
	eps_cg=0.1; 
	tol_cg=0.000001;

	ratio = divne(d2, d1, niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb);
	ratio1 = divne(d1, d2, niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb);
	simi=np.sqrt(np.abs(ratio*ratio1));
	return simi



