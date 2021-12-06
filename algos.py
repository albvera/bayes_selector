import numpy as np
from gurobipy import *

"""
Draw n_samples of arrivals over T periods
J[t,s] is the arriving type at time t=1,...,T for sample s=0,...,n_samples-1
"""
def sample_arrival(T,types,p,n_samples):
	J = np.random.choice(types,(T+1,n_samples), p=p)			#Types of arrivals
	return J

"""
Run the bayes selector on the given sample paths J
types = [0,..,n-1]
resources = [0,...,d-1]
B = initial budgets
"""
def bayes_packing(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_packing(A,r,types,resources,B,T*p)

	for s in range(0,n_samples):
		Bt = np.copy(B) 										#Budget over time
		for t in range(T,0,-1):
			if all(Bt[i] < 1 for i in resources):
				break

			j = J[t,s]
			if not all(Bt[i] >= A[i,j] for i in resources):		#Not enough resources
				continue

			#Update LP and optimize
			mod.setAttr("RHS", bud_const, Bt)
			mod.setAttr("RHS", dem_const, t*p)
			mod.optimize()

			if x[j].X >= t*p[j]/2:								#Accept
				reward[s] = reward[s] + r[j]
				Bt = Bt - A[:,j]

	return reward

"""
Run the bayes selector on the given sample paths J
types = [0,..,n-1]
resources = [0,...,d-1]
B = initial budgets
"""
def bayes_matching(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_matching(A,r,types,resources,B,T*p)

	for s in range(0,n_samples):
		Bt = np.copy(B) 									#Budget over time
		for t in range(T,0,-1):
			if all(Bt[i] < 1 for i in resources):
				break

			j = J[t,s]
			if all(A[i,j]*Bt[i] < 1 for i in resources):	#Can't match to any i s.t. A[i,j]=1
				continue

			#Update LP and optimize
			mod.setAttr("RHS", bud_const, Bt)
			mod.setAttr("RHS", dem_const, t*p)
			mod.optimize()

			xt = [x[i,j].X for i in resources]
			k = xt.index(max(xt))

			if xt[k] >= t*p[j]-sum(xt):						#Compare against the slack variable (reject option), which is t*p[j]-sum(x[i,j] for i in resources)
				#match to k
				reward[s] = reward[s] + r[k,j]
				Bt[k] = Bt[k] - 1

	return reward

"""
Run competitive algo for online matching on the given sample paths J
types = [0,..,n-1]
resources = [0,...,d-1]
B = initial budgets
"""
from itertools import product as prod

def competitive(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Create graph
	L = range(0,max(B))
	Kj = [int(np.ceil(p[j]*T)) for j in types]					#Number of copies of each j
	K = range(0,max(Kj))
	U = [(i,l) for i in resources for l in L if B[i]>l]
	V = [(j,k) for j in types for k in K if k<Kj[j]]
	N = {}														#Neighbours of j
	N_range = {}
	for j in types:
		N[j] = [(i,l) for (i,l) in U if A[i,j]==1]
		N_range[j] = range(0,len(N[j]))							#Used to sample from N

	edges = [(i,l,j,k) for (i,l) in U for (j,k) in V if A[i,j]==1]


	#Set up the LP
	print("Creating the LP model")
	mod = Model("LP")
	mod.setParam('OutputFlag', False )							#No output message
	x = mod.addVars(edges,name="x")
	mod.setObjective(sum(r[i,j]*x[i,l,j,k] for (i,l,j,k) in edges),GRB.MAXIMIZE)
	edges = None												#Free memory
	mod.addConstrs((quicksum(x[i,l,j,k] for (j,K) in V if A[i,j]==1) <= 1 for (i,l) in U))
	U = None													#Free memory
	mod.addConstrs((quicksum(x[i,l,j,k] for (i,l) in N[j]) <= 1 for (j,k) in V))
	print("Solving the LP")
	mod.optimize()

	probs = {}													#Sampling probabilities

	print("Running Competitive")
	for s in range(0,n_samples):
		Used = Set()											#Static nodes that have been used
		for t in range(T,0,-1):
			j = J[t,s]
			k = np.random.randint(0,high=Kj[j])					#Which copy of j?

			if (j,k) not in probs:								#We haven't computed the probabilities
				aux = np.array([x[i,l,j,k].X for (i,l) in N[j]])
				aux[aux<0] = 0									#Because of floating point error some x could be negative
				if sum(aux)>1:									#Because of floating point error they could add up to >1
					aux = aux/sum(aux)
				probs[j,k] = aux

			u = sample(N_range[j],probs[j,k])
			if u == None:										#The sampling returned no edge
				continue
			(i,l) = N[j][u]
			if (i,l) in Used:									#Returned node was taken
				continue

			Used.add((i,l))
			reward[s] = reward[s] + r[i,j]

	return reward


"""
Generates a random sample from the list a with probabilities q
If sum(q)<1, then it returns None with probability 1-sum(q)
"""
def sample(a,q):
	if sum(q)<1:
		return np.random.choice(a+[None],p=np.append(q,1-sum(q)))
	else:
		return np.random.choice(a,p=q)



"""
Run the Marginal Allocation Algorithm
"""
def marginal(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_matching(A,r,types,resources,B,T*p)
	mod.optimize()

	#Compute bid prices
	print("Computing bid prices")
	f = np.zeros((len(resources),T+1,max(B)+1))

	XX = np.zeros((len(resources),len(types)))				#Create array for x[i,j] since it will be accessed a lot
	for i in resources:
		for j in types:
			XX[i,j] = x[i,j].X

	for t in range(2,T+1):
		for i in resources:
			for b in range(1,B[i]+1):
				aux = r[i,:] -f[i,t-1,b]+f[i,t-1,b-1]
				aux[aux<0] = 0								#Zero-out negative components
				f[i,t,b] = f[i,t-1,b]+np.dot(XX[i,:],aux)/T

	print("Running Marginal")
	for s in range(0,n_samples):
		Bt = np.copy(B) 									#Budget over time
		for t in range(T,0,-1):
			if all(Bt[i] < 1 for i in resources):
				break

			j = J[t,s]
			if all(A[i,j]*Bt[i] < 1 for i in resources):	#Can't match to any i s.t. A[i,j]=1
				continue

			the_max = 0										#Find resource with largest marginal
			i_max = None
			for i in resources:
				if Bt[i]>0 and A[i,j]==1:
					aux = r[i,j] -f[i,t,Bt[i]] +f[i,t,Bt[i]-1]
					if aux>the_max or i_max==None:
						the_max = aux
						i_max = i

			if i_max!=None and the_max>0:					#Match to i_max
				reward[s] = reward[s] + r[i_max,j]
				Bt[i_max] = Bt[i_max] - 1

	return reward


"""
Run Infrequent Re-solving with Thresholding
"""
def irt(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_packing(A,r,types,resources,B,T*p)

	p_accept = np.zeros(len(types))							#Probability of acceptance

	for s in range(0,n_samples):
		Bt = np.copy(B) 									#Budget over time
		u = 0
		tu = T												#tu = T^((5/6)^u) for u = 0,1...
		for t in range(T,0,-1):
			if all(Bt[i] < 1 for i in resources):
				break

			if t == tu: 									#Should we optimize?
				mod.setAttr("RHS", bud_const, Bt)
				mod.setAttr("RHS", dem_const, t*p)
				mod.optimize()

				for j in types:								#Update acceptance probabilities
					p_accept[j] = x[j].X/(t*p[j])
					if p_accept[j] <= np.power(t,-0.25):
						p_accept[j] = 0
					if p_accept[j] >= 1 - np.power(t,-0.25):
						p_accept[j] = 1

				u = u+1										#Find next optimization point
				tu = int(np.power(T,np.power(5.0/6,u)))

			j = J[t,s]
			if not all(Bt[i] >= A[i,j] for i in resources):	#Not enough resources
				continue

			Unif = np.random.uniform()						#Draw algo randomization

			if p_accept[j] >= Unif:							#Accept
				reward[s] = reward[s] + r[j]
				Bt = Bt - A[:,j]

	return reward


"""
Run Frequent Re-solving with Thresholding
"""
def frt(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_packing(A,r,types,resources,B,T*p)

	p_accept = np.zeros(len(types))							#Probability of acceptance

	for s in range(0,n_samples):
		Bt = np.copy(B) 									#Budget over time
		u = 0
		tu = T												#tu = T^((5/6)^u) for u = 0,1...
		for t in range(T,0,-1):
			if all(Bt[i] < 1 for i in resources):
				break

			if True:  # always optimize
				mod.setAttr("RHS", bud_const, Bt)
				mod.setAttr("RHS", dem_const, t*p)
				mod.optimize()

				for j in types:								#Update acceptance probabilities
					p_accept[j] = x[j].X/(t*p[j])
					if p_accept[j] <= np.power(t,-0.25):
						p_accept[j] = 0
					if p_accept[j] >= 1 - np.power(t,-0.25):
						p_accept[j] = 1

				u = u+1										#Find next optimization point
				tu = int(np.power(T,np.power(5.0/6,u)))

			j = J[t,s]
			if not all(Bt[i] >= A[i,j] for i in resources):	#Not enough resources
				continue

			Unif = np.random.uniform()						#Draw algo randomization

			if p_accept[j] >= Unif:							#Accept
				reward[s] = reward[s] + r[j]
				Bt = Bt - A[:,j]

	return reward

"""
Run Re-solve and randomize
"""
def resolve_randomize(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_packing(A,r,types,resources,B,T*p)

	for s in range(0,n_samples):
		Bt = np.copy(B) 									#Budget over time
		for t in range(T,0,-1):
			if all(Bt[i] < 1 for i in resources):
				break

			j = J[t,s]
			if not all(Bt[i] >= A[i,j] for i in resources):	#Not enough resources
				continue

			mod.setAttr("RHS", bud_const, Bt)
			mod.setAttr("RHS", dem_const, t*p)
			mod.optimize()

			Unif = np.random.uniform()						#Draw algo randomization

			if x[j].X/(t*p[j]) >= Unif:						#Accept
				reward[s] = reward[s] + r[j]
				Bt = Bt - A[:,j]

	return reward

"""
Run Static randomized allocation
"""
def static_randomize(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_packing(A,r,types,resources,B,T*p)
	mod.optimize()
	p_accept = [1.0*x[j].X/(T*p[j]) for j in types]

	for s in range(0,n_samples):
		Bt = np.copy(B) 									#Budget over time
		for t in range(T,0,-1):
			if all(Bt[i] < 1 for i in resources):
				break

			j = J[t,s]
			if not all(Bt[i] >= A[i,j] for i in resources):	#Not enough resources
				continue

			Unif = np.random.uniform()						#Draw algo randomization

			if p_accept[j] >= Unif:							#Accept
				reward[s] = reward[s] + r[j]
				Bt = Bt - A[:,j]

	return reward

"""
Run offline problem
"""
def offline_packing(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_packing(A,r,types,resources,B,T*p)

	for s in range(0,n_samples):
		Z = []
		for j in types:
			Z.append(np.count_nonzero(J[1:,s]==j))		#Time t=0 doesn't count, so we start from index 1
		#Update LP and optimize
		mod.setAttr("RHS", dem_const, Z)
		mod.optimize()
		reward[s] = mod.ObjVal

	return reward

"""
Run offline problem
"""
def offline_matching(T,B,A,types,resources,p,r,J,n_samples):
	reward = np.zeros(n_samples)

	#Set up the LP
	mod,x,bud_const,dem_const = create_model_matching(A,r,types,resources,B,T*p)

	for s in range(0,n_samples):
		Z = []
		for j in types:
			Z.append(np.count_nonzero(J[1:,s]==j))		#Time t=0 doesn't count, so we start from index 1
		#Update LP and optimize
		mod.setAttr("RHS", dem_const, Z)
		mod.optimize()
		reward[s] = mod.ObjVal

	return reward

"""
Create gurobi model max{r'x:Ax<=B, 0<= x <= Z}
"""
def create_model_packing(A,r,types,resources,B,Z):
	mod = Model("LP")
	mod.setParam('OutputFlag', False )						#No output message
	x = mod.addVars(types,name="x")
	mod.setObjective(sum(r[j]*x[j] for j in types),GRB.MAXIMIZE)
	bud_const = mod.addConstrs((quicksum(A[i,j]*x[j] for j in types) <= B[i] for i in resources))
	dem_const = mod.addConstrs((x[j] <= Z[j] for j in types))

	return mod,x,bud_const,dem_const

"""
Create gurobi model for matching problem
"""
def create_model_matching(A,r,types,resources,B,Z):
	pairs = list(prod(resources,types))
	mod = Model("LP")
	mod.setParam('OutputFlag', False )						#No output message
	x = mod.addVars(pairs,name="x")
	mod.setObjective(sum(r[i,j]*A[i,j]*x[i,j] for (i,j) in pairs),GRB.MAXIMIZE)
	bud_const = mod.addConstrs((quicksum(x[i,j] for j in types) <= B[i] for i in resources))
	dem_const = mod.addConstrs((quicksum(x[i,j] for i in resources)<= Z[j] for j in types))

	return mod,x,bud_const,dem_const
