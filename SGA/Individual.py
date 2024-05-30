import numpy as np

class Individual:
	def __init__(self, genotype = [] ):
		self.genotype = np.array(genotype)
		self.fitness = 0
	
	def initialize_uniform_at_random(genotype_length):
		individual = Individual()
		individual.genotype = np.random.choice((0,1), p=(0.5, 0.5), size=genotype_length)
		return individual

	def initialize_from_square_assumption(genotype_length, adjacency_list):
		passed_over = [False] * len(adjacency_list)
		individual = Individual()
		individual.genotype = np.zeros(genotype_length)
		recursive_init_for_square(0, adjacency_list, passed_over, individual.genotype, 1)
		return individual


def recursive_init_for_square(idx: int, adjacency_list, passed_over: list[bool], genotype, include):
	if passed_over[idx] == True:
		return

	genotype[idx] = include
	passed_over[idx] = True
	for i in adjacency_list[idx]:
		recursive_init_for_square(i, adjacency_list, passed_over, genotype, flip(include))


def flip(v):
	# if 0: 0 * -2 + 1 = 1
	# if 1: 1 * -2 + 1 = 0
	return v * -2 + 1