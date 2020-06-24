import sys
import numpy as np
from scipy.stats import poisson
import random
# ================================================
# population N = 5; 
# mutation u = 1; 
# generations 10N = 50;
# ================================================
n = 5
u = 1

parents = [0]*n
parents
offspring = []
for i in range(n):
	off = poisson.rvs(1)
	offspring.append(off)
	
offspring
# offspring = np.random.poisson(1)
# offspring = np.random.poisson(lam = 1.0, size = 5)
# offspring
# offspring[1]
# mutation = np.random.poisson(lam=1.0, size = 10)
# mutation
# mutation = np.random.poisson(10)
# mutation

# from scipy.stats import poisson
# n = poisson.rvs(1)
# n

# if __name__ == '__main__':
# 	population = sys.argv[1]
# 	main(population)



# ================================================
# population N = 5; 
# mutation u = 1; 
# generations 10N = 50;
# ================================================
n, u = 5, 2
population = np.array([[None]* n])
population = 
np.zeros([100, 10], dtype=int)
population

# step 1 
# select parents for each offspring 
def select_paraents(n):
	return np.random.randint(0, n, size=n)  # Select N offspring from N parents randomly

def add_new_mutation(N, u):
	# TODO: poisson dist 
	np.random.randint(1, u+1, N)
	return 
	
np.random.randint(1, u+1, n)
new_pop = population[parents]
new_pop


# try
n, u = 100, 10
g = 10*n
g
population = np.array([[0]*u]* n)
population = np.zeros([n, u], dtype=int)
population
population[0][1]
def select_paraents(n):
	return np.random.randint(0, n, size=n)

def add_new_mutation(population, u):
	mutation_num = np.random.poisson(1,u)
	for i in range(u):
		if mutation_num[i] != 0:
			a = np.random.randint(0,5, size = mutation_num[i])
			for j in range(len(a)):
				population[j][i] = '1'
	return population
# s = select_paraents(n)
# s
# new_pop = population[s]
# new_pop
i = 1
mutation = 0
while i <= g:
	offspring = population[select_paraents(n)]
	new_off = add_new_mutation(offspring, u)
	population = new_off
	mutation += np.sum(population, axis = 1)
	i += 1
		
print(new_off)
print(mutation/g)
np.sum(new_off, axis = None)
b = np.random.poisson(1,2)
b
a = np.random.randint(0,5, size = b[0])
a
range(len(a))
population[a][0] = '1'
population
population[a[0]][0] = '1'

population
a = np.random.randint(0,5, size = 3)
a
a[0] == 0
new_pop
population[2][0] = '1'
population

a, b = np.random.poisson(1,2)
a, b
np.random.randint(1, 3, 5)
np.random.randint(0,5)
help(np.random.randint)
