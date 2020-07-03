import numpy as np

# ================================================
# Part 1
# Neutral selection simulation
# ================================================

# ================================================
# Functions
# ================================================
def select_parents(n):
	"""
	Args:
		n: (int) number of individuals in the population
	Returns:
		array: indexs for offspring
	"""
	# randomly select n individuals as offsprings in next generation
	return np.random.randint(0, n, size=n)

def add_new_mutation(population, n, u, nummuts):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		n: (int) number of individuals in the population
		u: (int) number of sites subject to mutations in each individual
	Returns:
		population: (numpy.ndarray)
	"""
	# randomly select which individual and which position has mutation 
	row = np.random.randint(0, n, size = nummuts) # list of row indexs
	col = np.random.randint(0, u, size = nummuts) # list of column indexs
	# Fancy Indexing
	population[row, col] += 1
	return population

def check_mut(population, fixnum, losnum):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		fixnum: (int) number of mutaiton got fixed
		losnum: (int) number of mutaiton got lost
	Returns:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		fixnum: (int) number of mutaiton got fixed
		losnum: (int) number of mutaiton got lost
		index: index of fixed columns
	"""
	n = population.shape[0]
	# u = population.shape[1]
	# 1. fixation: column all non-zero
	# 2. loss: column all zero
	check_matrix = population != 0 # if the site not equal to zero, return True; else ruturn False
	check_sum = np.sum(check_matrix, axis=0) # count number of True for each columns (have mutation)
	fix_col = np.where(check_sum==n)[0]  # fixation column index
	los_col = np.where(check_sum==0)[0]   # loss column index
	# seg_col = list(set(range(u)) - set(fix_col) - set(los_col))
	if fix_col.size > 0:
		population[:, fix_col] = 0 # convert fixed columns into 0
		fixnum += len(fix_col) # count fixation number
	elif los_col.size > 0:
		losnum += len(los_col) # count loss number
	return population, fixnum, losnum, fix_col

# ================================================


# ================================================
# population N = 100;
# positions u = 10;
# generations 10N = 1000;
# mutation rate v  = 0.01; number of mutations per chromosome position per generation
#    then expected number of mutations in population each generation will be v*N*u
# ================================================

n, u = 100, 10
g = 100*n
v = 0.01

# create a n*u array, each row is one individual, each column is one site for each mutation
population = np.zeros([n, u], dtype=int)

mutation_seg = [] # a list of number of sites carrying mutations over g generations
totalmuts = 0 # number of total mutation occurs
fix_num = 0 # number of fixation
los_num = 0 # number of loss
time = [] # list of recording fixation time

# run for g times as g gerations
i = 1
while i <= g:
	# 1. get offspring from parents
	offspring = population[select_parents(n)]
	# nummuts = np.random.randint(v*n*u)
	# 2. add mutations
	nummuts = np.random.poisson(v*n*u) # expected number of mutations in population each generation
	totalmuts += nummuts
	if nummuts != 0:
		new_off = add_new_mutation(offspring, n, u, nummuts)
		# 3. check mutations get fixed or lost
		population, fix_num, los_num, index = check_mut(new_off, fix_num, los_num)
		# 3.1 if get fixed, record the time
		if index.size > 0:
			time.append(i)
	else:
		population, fix_num, los_num, index = check_mut(offspring, fix_num, los_num)
		# 3.1 if get fixed, record the time
		if index.size > 0:
			time.append(i)
	mutation = np.sum(population != 0, axis = None) # calculate the number of sites carrying mutations
	mutation_seg.append(mutation)
	i += 1

print('Mean time to fixation: ', np.sum(time)/fix_num)
print('Probability of fixation in neutral selection: ', fix_num/(fix_num+los_num))
print(1/n)



# ================================================
# Part 2
# add beneficial mutations
# ================================================

# ================================================
# Functions
# ================================================
def new_offspring(population, n, u, s):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		n: (int) number of individuals in the population
		u: (int) number of sites subject to mutations in each individual
		s: (float) selection coefficient
	Returns:
		(numpy.ndarray) indexs of offspring from parents
	"""
	# 1. weights, probability distribution => p 
	p = np.ones(n)
	# 1.1 read beneficial column, find non-zero element => index
	ben_cols = population[:, u:]
	rows, _ = np.where(ben_cols!=0)
	if rows.size > 0:
		# 1.2 modify p 
		p[rows] += s
		# 1.3 normalization (to sum up to 1)
		p = p / np.sum(p)
		return np.random.choice(np.arange(n), size = n, p = p)
	else:
		return np.random.choice(np.arange(n), size = n)

def add_new_mutation_b(population, n, u, nummuts):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		n: (int) number of individuals in the population
		u: (int) number of sites subject to mutations in each individual
	Returns:
		population: (numpy.ndarray)
	"""
	# randomly select which individual and which position has mutation 
	row = np.random.randint(0, n, size = nummuts) # list of row indexs
	col = np.random.randint(0, u, size = nummuts) # list of column indexs
	b_muts = 0
	for i in col:
		if i == (u-1):
			b_muts += 1
	population[row, col] += 1
	return population, b_muts

def check_beneficial_mutation(population, u):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		u: (int) number of sites subject to mutations in each individual
	Returns:
		index of beneficial mutation column for fixation or loss, start from 0
	"""
	ben_cols = population[:, u:]
	n = population.shape[0]  # number of rows i.e. individuals  
	# 1. fixation: column all non-zero
	# 2. loss: column all zero
	check_matrix = ben_cols != 0 # if the site not equal to zero, return True; else ruturn False
	check_sum = np.sum(check_matrix, axis=0) # count number of True for each columns (beneficial mutation)
	fix_col = np.where(check_sum==n)[0]  # fixation column index
	los_col = np.where(check_sum==0)[0]   # loss column index
	if fix_col.size > 0:
		population[:, fix_col+u] = 0
	return population, fix_col, los_col


# ================================================
# selection coefficient s
# ================================================

diff_s = np.array([0.01, 0.05, 0.1, 0.5, 0.8, 1, 2])

harmful_s = np.array([-0.8, -0.6, -0.4, -0.2, -0.1, -0.05, -0.01, -0.001])

list_p_fix = [] # list of probability of fixation among different selection coefficients
list_p_loss = [] # list of probability of loss among different selection coefficients

for s in diff_s:
# for s in harmful_s:
	# 1. add first beneficial mutation
	# 1.1 # add a new column for the first beneficial mutation site
	first_beneficial_site = np.zeros(n, dtype = int)
	# 1.2 add it into population
	pop = np.column_stack((population, first_beneficial_site))
	b = 1 # number of total beneficial mutations
	j = 1
	time_to_fix = {} # key is b; value is time to get fixed
	time_to_loss = {} # key is b; value is time to get lost
	total_b_muts = 0 # total number of beneficial mutations happened
	while j <= g:
		# 1.3 add mutations;
		num_muts = np.random.poisson(v*n*(u+b)) # expected number of mutations in population each generation
		add_mutation_pop, b_muts = add_new_mutation_b(pop, n, u+b, num_muts)
		total_b_muts += b_muts
		# 2. get offspring for next generation;
		offspring = add_mutation_pop[new_offspring(add_mutation_pop, n, u+b-1, s)]
		# 2.1 use check function to figure when it got fixed or lost; if it did, convert that column into 0 and 
		pop, fix_col, los_col = check_beneficial_mutation(offspring, u+b-1)
		# 2.2 record its time;
		# if b_muts > 0 and fix_col.size > 0:
		if fix_col.size > 0:
			time_to_fix[b] = j # add time for beneficial muation got fixed
			# 2.3 add another beneficial site into population;
			new_beneficial_site = np.zeros(n, dtype = int)
			pop = np.column_stack((pop, new_beneficial_site))
			b += 1
		elif b_muts > 0 and los_col.size > 0:
		# elif los_col.size > 0:
			time_to_loss[b] = j # add time for beneficial muation got lost 
			# 2.3 add another beneficial site into population;
			new_beneficial_site = np.zeros(n, dtype = int)
			pop = np.column_stack((pop, new_beneficial_site))
			b += 1
		j += 1
	prob_of_fix = len(time_to_fix.keys()) / (len(time_to_fix.keys()) + len(time_to_loss.keys()))
	prob_of_loss = len(time_to_loss.keys()) / (len(time_to_fix.keys()) + len(time_to_loss.keys()))
	list_p_fix.append(prob_of_fix)
	list_p_loss.append(prob_of_loss)


import matplotlib.pyplot as plt
# plot
fig, ax = plt.subplots(figsize=(8, 6))
x, y = diff_s, list_p_fix
# x, y = harmful_s, list_p_fix
ax.plot(x, y, label='Probability of fixation')
x, y = diff_s, list_p_loss
# x, y = harmful_s, list_p_loss
ax.plot(x, y, label='Probability of loss')
# legend and grid
ax.grid(alpha=0.3)
ax.legend()
# axis and title
ax.set_title('Probability of fixation and loss')
ax.set_xlabel('Selection Coefficient')
ax.set_ylabel('Probability')
fig.tight_layout()
# fig.savefig('Probability of fixation and loss in harmful mutation.png', dpi=300, transparent=True)
fig.savefig('Probability of fixation and loss in beneficial mutation.png', dpi=300, transparent=True)
exit()

# ================================================
# Plot by matplotlib
# ================================================
import matplotlib.pyplot as plt
# plot
fig, ax = plt.subplots(figsize=(8, 6))
x, y = range(g), mutation_seg
ax.plot(x, y, label='Number of Segregating Mutations')
# legend and grid
ax.grid(alpha=0.3)
ax.legend()
# axis and title
ax.set_title(f'Population of {n} Individual {u} Sites, Mutation Rate {v}, {g} Generations')
ax.set_xlabel('Generation')
ax.set_ylabel('Number of Segregating Mutations')
fig.tight_layout()
fig.savefig('Number of Mutations Segregating40.png', dpi=300, transparent=True)
# fig.savefig('filename.pdf', transparent=True)
# fig.savefig('filename.svg', transparent=True)
exit()

# ================================================
# end
# ================================================


# print(b)
# print(time_to_fix)
# print(time_to_loss)
# mean_time_fix = sum(time_to_fix.values()) / len(time_to_fix.keys())
# mean_time_loss = sum(time_to_loss.values()) / len(time_to_loss.keys())
# print(mean_time_fix)
# print(mean_time_loss)
# prob_of_fix = len(time_to_fix.keys()) / total_b_muts
# prob_of_loss = len(time_to_loss.keys()) / total_b_muts
# print(prob_of_fix)
# print(prob_of_loss)
