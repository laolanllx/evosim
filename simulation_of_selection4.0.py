import numpy as np

# ================================================
# Part 1
# Neutral selection simulation
# ================================================

# ================================================
# Functions
# ================================================
def select_paraents(n):
	"""
	Args:
		n: (int) number of individuals in the population
	Returns:
		array: indexs for offsprin
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
	# for i in range(nummuts):
	# 	population[row[i]][col[i]] += 1
	# Fancy Indexing
	population[row, col] += 1
	return population


def check_site(population, u, fixnum):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
	Returns:
		None
	"""
	_, col = np.where(population == 0)  # find columns that contain 0 i.e. no mutation
	if col.size > 0:
		index = set(range(u)) - set(col)  # find columns that do not contain 0
		index = list(index)    # convert set to list
		if len(index) > 0:
			fixnum += len(index)
		population[:, index] = 0 # convert sites that has become fixed to 0
	return population, fixnum, index
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
v = 0.0001

# create a n*u array, each row is one individual, each column is one site for each mutation
population = np.zeros([n, u], dtype=int)
# population
# run for g times as g gerations
i = 1
mutation_seg = [] # a list of number of sites carrying mutations over g generations
totalmuts = 0
fix_num = 0
time = []

help(np.random.poisson())

while i <= g:
	offspring = population[select_paraents(n)]
	# nummuts = np.random.randint(v*n*u)
	nummuts = np.random.poisson(v*n*u) # expected number of mutations in population each generation
	totalmuts += nummuts
	# for i in range(nummuts):
	if nummuts != 0:
		new_off = add_new_mutation(offspring, n, u, nummuts)
		population, fix_num, index = check_site(new_off, u, fix_num)
		if len(index) > 0:
			time.append(i)
		# else:
		# 	time.append(0)
	else:
		population, fix_num, index = check_site(offspring, u, fix_num)
		if len(index) > 0:
			time.append(i)
		# else:
		# 	time.append(0)
	mutation = np.sum(population != 0, axis = None) # calculate the number of sites carrying mutations
	mutation_seg.append(mutation)
	i += 1

len(time)
np.sum(time)/fix_num

totalmuts
fix_num
fix_num/totalmuts
population
1/n*5


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
	rows, occurrence = np.unique(rows, return_counts=True)
	if rows.size > 0:
		# 1.2 modify p 
		p[rows] *= [(1+s)**i for i in occurrence]
		# 1.3 normalization (to sum up to 1)
		p = p / np.sum(p)
		return np.random.choice(np.arange(n), size = n, p = p)
	else:
		return np.random.choice(np.arange(n), size = n)


def check_beneficial_mutation(population, u):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		u: (int) number of sites subject to mutations in each individual
	Returns:
		index of beneficial mutation column for fixation or loss, start from 0
	"""
	ben_cols = population[:, u:] 
	# 1. fixation: column all non-zero
	# 2. loss: column all zero
	check_matrix = ben_cols != 0 # if the site not equal to zero, return True; else ruturn False
	check_sum = np.sum(check_matrix, axis=0) # count number of True for each columns (beneficial mutation)
	fix_col = np.where(check_sum==10)[0]  # fixation 
	los_col = np.where(check_sum==0)[0]   # loss
	return fix_col, los_col 


# ================================================
# selection coefficient s
# ================================================
s = 0.1
# time for beneficial muation got fixed or lost:
# there should be 10 beneficial mutation for 100*N generations
# shape 10*2, first column is fixation time, second column is loss time
time_to_fix_or_loss_b = np.zeros([10,2], dtype=int)
time_to_fix_or_loss_b
b = 0 # number of beneficial mutations
j = 1

while j < g:
	#  every 10*N generations add a new beneficial mutation
	if j % (10*n) == 0:
		b += 1
		new_position = np.zeros(n, dtype = int)
		population = np.column_stack((population,new_position)) # add a new column for a beneficial mutation site
	num_muts = np.random.poisson(v*n*(u+b)) # expected number of mutations in population each generation
	add_ben_population = add_new_mutation(population, n, u+b, num_muts) # add mutation
	if b > 0: # population contains beneficial mutaion
		offspring = add_ben_population[new_offspring(add_ben_population, n, u, s)]
		fix_col, los_col = check_beneficial_mutation(offspring, u)
		if fix_col != 'NAN':
			time_to_fix_or_loss_b[fix_col, 0] = j # add time for beneficial muation got fixed
		elif los_col != 'NAN':
			time_to_fix_or_loss_b[los_col, 1] = j # add time for beneficial muation got lost 
		population = offspring
	else: 
		offspring = add_ben_population[select_paraents(n)]
		population = offspring
	j += 1
time_to_fix_or_loss_b

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
fig.savefig('Number of Mutations Segregating30.png', dpi=300, transparent=True)
# fig.savefig('filename.pdf', transparent=True)
# fig.savefig('filename.svg', transparent=True)
exit()

# ================================================
# Plot by plotly
# ================================================
import plotly.graph_objects as go
from plotly.offline import iplot, plot

fig = go.Figure()
# data to plot
x, y, labels = list(range(g)), mutation_seg, [f'Generation: {i}<br>Mutation: {j}' for i,j in zip(range(g), mutation_seg)]
fig.add_trace(go.Scatter(x=x, y=y,
    mode='markers+lines', name='Number of Segregating Mutations',
	text=labels,
    # line=dict(width=1),
	marker=dict(size=2.5),
	))
# add figure layout
fig.update_layout(
    title=dict(
        text=f'Population of {n} Individual {u} Sites, Mutation Rate {v}',
        x=0.5, y=0.9,
        xanchor='center', yanchor='top',
        font=dict(size=15),
        ),
    xaxis=dict(
        title='Generation',
        # range=[0, g]
        ),
    yaxis=dict(
        title='Number of Segregating Mutations',
        # range=[min, max],
        ),
    width=1000,
    height=600,
    showlegend=True,
    )
iplot(fig)
plot(fig, filename='plotly_segregating_mutation3.html')


# print(new_off)
# mutation_seg
# print(mutation/g)


# ================================================
# test
# ================================================
# a = np.random.randint(0, 5, size=(10, 2))
# a
# a
# ben_cols = a[:, 2:]
# ben_cols
# row, col  = check_beneficial_mutation(a, 2)
# row, col
# row != 'NAN'
# a = np.column_stack((a, np.ones(10), np.zeros(10)))
# a
# a[3][1] = 1
# a
# b = a!=0
# b
# 
# ben_cols = a[:, 2:] 
# ben_cols
# 
# check_matrix = ben_cols != 0 # if the site not equal to zero, return True; else ruturn False
# check_matrix
# 
# check_sum = np.sum(check_matrix, axis=0)
# check_sum
# # check_sum = np.sum(check_matrix, axis=0) # count number of True for each columns (beneficial mutation)
# fix_col = np.where(check_sum==10)[0]  # fixation 
# los_col = np.where(check_sum==0)[0]
# fix_col
# los_col
# 
# np.arange(0)[check_sum==10]
# np.arange(1)
# np.arange(2)
# 
# check_sum = np.array([1, 2, 9, 0])
# rows = np.where(check_sum==10)[0]
# rows
#


# a  = np.random.randint(0,10, size = (10,2))
# a
# new_position = np.ones(10, dtype = int)
# new_position
# b = np.zeros(10)
# b[2] = 1
# new_population = np.column_stack((a,np.ones(10),np.zeros(10),b)) # add a new column for a beneficial mutation
# new_population
# # num_muts = np.random.poisson(v*n*(u+b))
# # add_ben_population = add_new_mutation(new_population, n, u+b, num_muts)
# p = np.ones(10)
# p
# ben_cols = new_population[:, 2:]
# ben_cols
# rows, _ = np.where(ben_cols!=0)
# rows
# rows, occurrence = np.unique(rows, return_counts=True)
# rows, occurrence
# p[rows] *= [(1+0.2)**i for i in occurrence]
# p
# # 1.3 normalization (to sum up to 1)
# p = p / np.sum(p)
# p
# np.random.choice(np.arange(10), size = 10, p = p)
# np.random.choice(np.arange(10), size = 10)
# new_offspring(new_population, 10, 2, 0.2)
# new_population[new_offspring(new_population, 10, 2, 0.2)]
# parents
# new_offspring = new_population[parents]
# new_offspring
# p = np.ones(10)

# a = np.random.poisson(v*n*u)
# if a != 0:
# 	print(a)
# 
# np.where(a == 0)
# np.random.randint(0, 10)
# row= np.random.randint(0, n, size = a)
# row
# col = np.random.randint(0, u, size = a)
# col
# population = np.zeros([n, u], dtype=int)
# population
# for i in range(a):
# 	population[row[i]][col[i]] += 1
# population
# a = np.random.randint(5, size=(10,5 ))
# a
# a != 0
# np.sum(a!=0, axis=None)
# mutation_num = np.random.poisson(1, 10)
# mutation_num
# np.random.seed = 123
# a = np.random.randint(10, size=(10, 5))
# a
# check_site(a, 5) is None
# a
# _, col = np.where(a == 0)
# index = set(range(5)) - set(col)
#
#
# row_index, col_index = np.where(a == 0)
# row_index
# i = set(range(5)) - set(col_index)
# i
# len(i)
# for c in i:
# 	a[:, c] = 0
# a
# a[:, [0, 2, 4]] = 0
# a

# p = np.random.random()
# p
# s = select_paraents(n)
# s
# new_pop = population[s]
# new_pop
# np.random.seed=1
# t = np.random.randint(0, 5, [5, 3])
# t
# np.sum(t, axis=0)
# mutation_num = np.random.poisson(1,10)
# mutation_num
# np.sum(mutation_num)
# a = np.random.randint(0, 10, size = 3)
# a
# s = np.zeros([10,2], dtype = int)
# s
# len(a)
# s
# a
# s[a]
# a[0]
# for i in range(len(a)):
# 	print(i)
# 	s[a[i]][0] = '1'

# ================================================
# end
# ================================================