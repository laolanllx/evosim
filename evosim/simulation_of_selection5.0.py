import numpy as np
import matplotlib.pyplot as plt


# ================================================
# Part 1
# Neutral selection simulation
# ================================================

# Object-oriented language 
class Population:
	def __init__(self, n, u, v):
		"""
		Create a new population with n individuals and u mutation sites
		Mutation rate of v
		:param n: (int) number of individuals in a population
		:param u:
		:param v: 
		"""
		self.population = np.zeros([n, u], dtype=int)
		self.n = n
		self.u = u
		self.v = v

	def __repr__(self):
		s = f"Number of individuals: {self.n}\n" \
			f"Number of mutation sites: {self.u}\n" \
			f"Mutation rate: {self.v}"
		return s

	@property  #  decrator
	def v(self):
		return self._v

	@v.setter
	def v(self, v):
		try:
			assert 0 <= v <=1
			self._v = v
		except ValueError:
			print(f"Invalid input mutation rate {v} which should be (0, 1)")


class NeutralSelectionSimulator(Population):
	def __init__(self, n, u, v, g):
		super(NeutralSelectionSimulator, self).__init__(n, u, v)
		self.num_generation = g

	def select_parents(self):
		return np.random.randint(0, self.n, size=self.n)


p1 = NeutralSelectionSimulator(1000, 10, 0.01, 100 * 1000)
p1_offspring = p1.select_parents()


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
	row = np.random.randint(0, n, size=nummuts)  # list of row indexs
	col = np.random.randint(0, u, size=nummuts)  # list of column indexs
	# Fancy Indexing
	population[row, col] += 1
	return population


def check_site(population, u, fixnum):
	"""
	Args:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
	Returns:
		population: (numpy.ndarray) shape n*u, n individuals and u sites
		fixnum: number of mutaiton got fixed
		index: index of fixed columns
	"""
	_, col = np.where(population == 0)  # find columns that contain 0 i.e. no mutation
	if col.size > 0:
		index = set(range(u)) - set(col)  #  find columns that do not contain 0
		index = list(index)  #  convert set to list
		if len(index) > 0:
			fixnum += len(index)
		population[:, index] = 0  # convert sites that has become fixed to 0
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
g = 100 * n
v = 0.001

# create a n*u array, each row is one individual, each column is one site for each mutation
population = np.zeros([n, u], dtype=int)
# population
# run for g times as g gerations
i = 1
mutation_seg = []  # a list of number of sites carrying mutations over g generations
totalmuts = 0
fix_num = 0
time = []

while i <= g:
	offspring = population[select_paraents(n)]
	# nummuts = np.random.randint(v*n*u)
	nummuts = np.random.poisson(v * n * u)  # expected number of mutations in population each generation
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
	mutation = np.sum(population != 0, axis=None)  # calculate the number of sites carrying mutations
	mutation_seg.append(mutation)
	i += 1

len(time)
np.sum(time) / fix_num

totalmuts
fix_num / g
fix_num / totalmuts
population
1 / (n * 2)
1 / n


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
	rows, _ = np.where(ben_cols != 0)
	if rows.size > 0:
		# 1.2 modify p
		p[rows] += s
		# 1.3 normalization (to sum up to 1)
		p = p / np.sum(p)
		return np.random.choice(np.arange(n), size=n, p=p)
	else:
		return np.random.choice(np.arange(n), size=n)


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
	row = np.random.randint(0, n, size=nummuts)  # list of row indexs
	col = np.random.randint(0, u, size=nummuts)  # list of column indexs
	b_muts = 0
	for i in col:
		if i == (u - 1):
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
	check_matrix = ben_cols != 0  # if the site not equal to zero, return True; else ruturn False
	check_sum = np.sum(check_matrix, axis=0)  # count number of True for each columns (beneficial mutation)
	fix_col = np.where(check_sum == n)[0]  #  fixation column index
	los_col = np.where(check_sum == 0)[0]  # loss column index
	if fix_col.size > 0:
		population[:, fix_col + u] = 0
	return population, fix_col, los_col


# ================================================
# selection coefficient s
# ================================================
# s = 0.1
diff_s = np.array([0.01, 0.05, 0.1, 0.5, 0.8, 1, 2])
diff_s
# 1. add first beneficial mutation
# 1.1 # add a new column for the first beneficial mutation site
first_beneficial_site = np.zeros(n, dtype=int)
# 1.2 add it into population
population = np.column_stack((population, first_beneficial_site))
b = 1  # number of total beneficial mutations
j = 1
time_to_fix = {}
time_to_loss = {}
total_b_muts = 0

list_p_fix = []
list_p_loss = []

while j <= g:
	# 1.3 add mutations;
	num_muts = np.random.poisson(v * n * (u + b))  # expected number of mutations in population each generation
	add_mutation_pop, b_muts = add_new_mutation_b(population, n, u + b, num_muts)
	total_b_muts += b_muts
	# 2. get offspring for next generation;
	offspring = add_mutation_pop[new_offspring(add_mutation_pop, n, u + b - 1, s)]
	# 2.1 use check function to figure when it got fixed or lost; if it did, convert that column into 0 and
	population, fix_col, los_col = check_beneficial_mutation(offspring, u + b - 1)
	# 2.2 record its time;
	# if b_muts > 0 and fix_col.size > 0:
	if fix_col.size > 0:
		time_to_fix[b] = j  # add time for beneficial muation got fixed
		# 2.3 add another beneficial site into population;
		new_beneficial_site = np.zeros(n, dtype=int)
		population = np.column_stack((population, new_beneficial_site))
		b += 1
	elif b_muts > 0 and los_col.size > 0:
		# elif los_col.size > 0:
		time_to_loss[b] = j  # add time for beneficial muation got lost
		# 2.3 add another beneficial site into population;
		new_beneficial_site = np.zeros(n, dtype=int)
		population = np.column_stack((population, new_beneficial_site))
		b += 1
	j += 1
prob_of_fix = len(time_to_fix.keys()) / total_b_muts
prob_of_loss = len(time_to_loss.keys()) / total_b_muts
list_p_fix.append(prob_of_fix)
list_p_loss.append(prob_of_loss)

list_p_loss
list_p_fix

print(b)
print(time_to_fix)
print(time_to_loss)
mean_time_fix = sum(time_to_fix.values()) / len(time_to_fix.keys())
mean_time_loss = sum(time_to_loss.values()) / len(time_to_loss.keys())
print(mean_time_fix)
print(mean_time_loss)
prob_of_fix = len(time_to_fix.keys()) / total_b_muts
prob_of_loss = len(time_to_loss.keys()) / total_b_muts
print(prob_of_fix)
print(prob_of_loss)

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
fig.savefig('Number of Mutations Segregating30.png', dpi=300, transparent=True)
# fig.savefig('filename.pdf', transparent=True)
# fig.savefig('filename.svg', transparent=True)
exit()
