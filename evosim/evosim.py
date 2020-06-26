import numpy as np
import matplotlib.pyplot as plt


# ================================================
# Part 1
# Neutral selection simulation
# ================================================

# Object-oriented language
class Population:
	def __init__(self, n: int, u: int, v: float):
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
	def v(self, v: float):
		try:
			assert 0 <= v <=1
			self._v = v
		except ValueError:
			print(f"Invalid input mutation rate {v} which should be (0, 1)")


class NeutralSelectionSimulator(Population):
	def __init__(self, n, u, v, g):
		super(NeutralSelectionSimulator, self).__init__(n, u, v)
		self.g = g
		self.fixation_count = 0
		self.mutation_count = 0
		self.fixation_timepoint = []
		self.loss_timepoint = []
		self.segregating_mutation = []

	@property
	def g(self):
		return self._g

	@g.setter
	def g(self, g):
		self._g = g

	def get_mutations(self):
		return np.random.poisson(self.v * self.n * self.u)

	def update_population(self):
		parents = np.random.randint(0, self.n, size=self.n)
		self.population = self.population[parents]

	def add_new_mutation(self, num_mutations, verbose=False):
		# randomly select which individual and which position has mutation
		row = np.random.randint(0, self.n, size=num_mutations)  # list of row indexs
		col = np.random.randint(0, self.u, size=num_mutations)  # list of column indexs
		# Fancy Indexing
		self.population[row, col] += 1
		if verbose:
			self.__repr__()

	def check_fixation(self):
		"""
		In a simulation
		1. Detect any mutation sites that become fixed
		2. Update number of fixation i.e. self.fixation

		:return:
		"""
		_, col = np.where(self.population == 0)  # find columns that contain 0 i.e. no mutation
		if col.size > 0:
			idx = set(range(self.u)) - set(col)  #  find columns that do not contain 0
			if len(idx) > 0:
				self.fixation_count += len(idx)
				self.population[:, list(idx)] = 0  # convert sites that has become fixed to 0
				return True
			return False
		return False

	def simulate_neutral_selection(self):
		i = 1
		while i <= self.g:
			self.update_population()
			nummuts = self.get_mutations()
			self.mutation_count += nummuts

			# if new mutation, add it to population
			if nummuts != 0:
				self.add_new_mutation(nummuts)

			# check fixation
			r = self.check_fixation()
			if r:
				self.fixation_timepoint.append(i)

			num = np.sum(self.population != 0, axis=None)  # calculate the number of sites carrying mutations
			self.segregating_mutation.append(num)
			i += 1

	def plot_segregating_mutation_generation(self, savefig=False):
		fig, ax = plt.subplots(figsize=(8, 6))
		x, y = range(self.g), self.segregating_mutation
		ax.plot(x, y, label='Number of Segregating Mutations')
		# legend and grid
		ax.grid(alpha=0.3)
		ax.legend()
		# axis and title
		ax.set_title(f'Population of {self.n} Individual {self.u} Sites, Mutation Rate {self.v}, {self.g} Generations')
		ax.set_xlabel('Generation')
		ax.set_ylabel('Number of Segregating Mutations')
		fig.tight_layout()
		if savefig:
			fig.savefig('Number of Mutations Segregating30.png', dpi=300, transparent=True)