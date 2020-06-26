from evosim import NeutralSelectionSimulator


n, u, v = 100, 10, 0.01
g = 100*n
p1 = NeutralSelectionSimulator(n, u, v, g)
p1.simulate_neutral_selection()
p1.plot_segregating_mutation_generation(savefig=False)
