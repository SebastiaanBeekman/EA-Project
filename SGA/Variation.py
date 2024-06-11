import random
import numpy as np
import random

from Individual import Individual
from FitnessFunction import FitnessFunction

def uniform_crossover(individual_a: Individual, individual_b: Individual, p = 0.5 ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
    
	m = np.random.choice((0,1), p=(p, 1-p), size=l)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)
	
	return [offspring_a, offspring_b]

def one_point_crossover(individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
    
	l = len(individual_a.genotype)
	m = np.arange(l) < np.random.randint(l+1)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)
	
	return [offspring_a, offspring_b]

def two_point_crossover(individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	offspring_a = Individual()
	offspring_b = Individual()
    
	l = len(individual_a.genotype)
	m = (np.arange(l) < np.random.randint(l+1)) ^ (np.arange(l) < np.random.randint(l+1))
	offspring_a.genotype = np.where(m, individual_b.genotype, individual_a.genotype)
	offspring_b.genotype = np.where(~m, individual_b.genotype, individual_a.genotype)
	
	return [offspring_a, offspring_b]



def edge_crossover(edges, individual_a: Individual, individual_b: Individual):
    # Initialize offspring and edge map
	num_nodes = len(individual_a.genotype)
	offspring_a = Individual(np.full(num_nodes, -1))
	offspring_b = Individual(np.full(num_nodes, -1))
	edge_map = {i: [] for i in range(num_nodes)}

	
    # Collect edges from both parents
	for edge in edges:
		for parent in [individual_a, individual_b]:
			node1, node2 = edge
			if parent.genotype[node1] != parent.genotype[node2]:  # if nodes are on different sides
				edge_map[node1].append(parent.genotype[node2])
				edge_map[node2].append(parent.genotype[node1])
                
    # Assign nodes to offspring
	for i in range(num_nodes):
		if offspring_a.genotype[i] == -1:
			if edge_map[i]:
				offspring_a.genotype[i] = random.choice(edge_map[i])
			else:
				offspring_a.genotype[i] = random.choice([0, 1])

    # Ensure both offspring have complementary sets
	for i in range(num_nodes):
		if offspring_b.genotype[i] == -1:
			offspring_b.genotype[i] = 1 - offspring_a.genotype[i]

	return [offspring_a, offspring_b]

      
    


if __name__ == "__main__":
	# Example usage:
	parent1 = [1, 0, 0, 1, 0, 1, 1, 0, 0]
	parent1 = Individual(parent1)
	parent2 = [0, 1, 1, 0, 1, 0, 0, 1, 1]
	parent2 = Individual(parent2)
	edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6), (4, 5), (4, 7), (5, 8), (6, 7), (7, 8)]

	child = edge_crossover(edges, parent1, parent2)
	print("Parent 1:", parent1)
	print("Parent 2:", parent2)
	print("Child:", child[0])
	print("Child:", child[1])
