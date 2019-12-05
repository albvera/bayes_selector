#!/usr/bin/python
import random
from datetime import datetime
import numpy as np
from algos import *
from scipy.stats import bayes_mvs

if __name__ == '__main__':
	
	#----- Input data begins -----
	T = 20
	B = np.array([4,5])
	A = np.array([[1,1,0,0,1,1],[0,0,1,1,1,1]])
	p = np.array([0.2,0.2,0.2,0.2,0.1,0.1])
	r = np.array([[10,6,10,5,9,8],[15,1,5,10,20,20]])

	n_samples = 10		
	#----- Input data ends -----

	resources = np.array(range(0,len(A[:,0])))
	types = np.array(range(0,len(A[0,:])))
	
	J = sample_arrival(T,types,p,n_samples)
	r_offline = offline_matching(T,B,A,types,resources,p,r,J,n_samples)
	r_bayes = bayes_matching(T,B,A,types,resources,p,r,J,n_samples)
	r_mar = marginal(T,B,A,types,resources,p,r,J,n_samples)
	r_com = competitive(T,B,A,types,resources,p,r,J,n_samples)
		
	print r_offline
	print r_bayes
	print r_mar
	print r_com
