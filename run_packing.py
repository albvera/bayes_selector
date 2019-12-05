#!/usr/bin/python
import numpy as np
from algos import *

if __name__ == '__main__':
	
	#----- Input data begins -----
	T = 200
	B = np.array([40,40])
	A = np.array([[1,1,0,0,1,1],[0,0,1,1,1,1]])
	p = np.array([0.2,0.2,0.2,0.2,0.1,0.1])
	r = np.array([10,6,10,5,9,8])

	n_samples = 10							
	#----- Input data ends -----

	d = len(A[:,0])
	n = len(A[0,:])
	resources = np.array(range(0,d))
	types = np.array(range(0,n))

	J = sample_arrival(T,types,p,n_samples)
	r_offline = offline_packing(T,B,A,types,resources,p,r,J,n_samples)
	r_bayes = bayes_packing(T,B,A,types,resources,p,r,J,n_samples)
	r_irt = irt(T,B,A,types,resources,p,r,J,n_samples)
	r_rr = resolve_randomize(T,B,A,types,resources,p,r,J,n_samples)
	r_sr = static_randomize(T,B,A,types,resources,p,r,J,n_samples)

	print r_offline
	print r_bayes
	print r_irt
	print r_rr
	print r_sr
