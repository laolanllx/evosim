import numpy as np

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

def add_new_mutation(population, n, u):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites 
		n: (int) number of individuals in the population 
		u: (int) number of sites subject to mutations in each individual 
	Returns:
		population: (numpy.ndarray) 
		mutation: (int) number of sites carrying mutations

	"""
	# randomly decide the mutation numbers of each site
	mutation_num = np.random.poisson(1, u) # mutation_num is the # of mutation for each site in all individuals
	# randomly select which individual has mutation for each mutation
	for i in range(u):
		if mutation_num[i] != 0:
			a = np.random.randint(0, n, size = mutation_num[i]) # i.e. mutation_num[i] = 3, a: array([10, 42, 1])
			for j in range(len(a)):
				population[a[j]][i] += 1
	return population


def check_site(population, u):
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
		population[:, index] = 0 # convert sites that has become fixed to 0
	return population
# ================================================


# ================================================
# population N = 100; 
# mutation u = 10; 
# generations 10N = 1000;
# mutation rate v = 0.01;
# ================================================

n, u = 100, 10
g = 10*n
v = 0.5

# create a n*u array, each row is one individual, each column is one site for each mutation
population = np.zeros([n, u], dtype=int)
population
# run for g times as g gerations
i = 1
mutation_seg = [] # a list of number of sites carrying mutations over g generations
while i <= g:
	offspring = population[select_paraents(n)]
	# Metropolis Hasting 
	if np.random.random() <= v: # have v possibility to add new mutation in next generation
		new_off = add_new_mutation(offspring, n, u)
		population = check_site(new_off, u)
		# population = new_off
	else:
		population = check_site(offspring, u)
		
	mutation = np.sum(population != 0, axis = None) # calculate the number of sites carrying mutations
	mutation_seg.append(mutation)
	i += 1

# population
# len(mutation_seg)

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
fig.savefig('Number of Mutations Segregating.png', dpi=300, transparent=True)
# fig.savefig('filename.pdf', transparent=True)
# fig.savefig('filename.svg', transparent=True)


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
np.random.poisson(10, size = 100)
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
a = np.random.randint(0, 10, size = 3)
a
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