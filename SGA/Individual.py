import numpy as np

class Individual:
	def __init__(self, genotype = [] ):
		self.genotype = np.array(genotype)
		self.fitness = 0
	
	def initialize_uniform_at_random(genotype_length):
		individual = Individual()
		individual.genotype = np.random.choice((0,1), p=(0.5, 0.5), size=genotype_length)
		return individual
	
	def from_sets(A, B, genotype_length):
		genome = [0] * genotype_length
		for vertex in A:
			genome[vertex] = 1
		for vertex in B:
			genome[vertex] = 0
		return Individual(genome)
