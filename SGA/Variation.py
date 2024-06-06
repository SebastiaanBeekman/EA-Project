import numpy as np

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

def custom_crossover( fitness: FitnessFunction, individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
   
   	# Implement your custom crossover here
	offspring_a.genotype = individual_a.genotype.copy()
	offspring_b.genotype = individual_b.genotype.copy()
	
	return [offspring_a, offspring_b]

def block_crossover( fitness: FitnessFunction, individual_a: Individual, individual_b: Individual , k = 5, p = 0.5):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	assert len(individual_a.genotype) % k == 0, "solutions should be divisible by k"

	l = len(individual_a.genotype)
	blocks = l // k
	offspring_a = Individual(l)
	offspring_b = Individual(l)
   
	m = np.random.choice((0,1), p=(p, 1-p), size=blocks)
	m = np.repeat(m, k)

	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)

	return [offspring_a, offspring_b]


def ROM_variation(fitness: FitnessFunction, subsets, individual_a: Individual , population: list):
	individual_b = population[np.random.randint(len(population))]
	while individual_a == individual_b:
		individual_b = population[np.random.randint(len(population))]
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
	
	offspring_a.genotype = individual_a.genotype.copy()
	offspring_b.genotype = individual_b.genotype.copy()
	
	offspring_a.fitness = individual_a.fitness

	for subset in subsets:
		# crossover the genotype for the genes in the subset
		offspring_a.genotype[subset] = individual_b.genotype[subset].copy()
		offspring_b.genotype[subset] = individual_a.genotype[subset].copy()
		
		if (offspring_a.genotype != offspring_b.genotype).any():
			fitness.evaluate(offspring_a)
			if offspring_a.fitness > individual_a.fitness:
				individual_a.genotype = offspring_a.genotype.copy()
				individual_a.fitness = offspring_a.fitness
				individual_b.genotype = offspring_b.genotype.copy()
			else:
				offspring_a.genotype = individual_a.genotype.copy()
				offspring_a.fitness = individual_a.fitness
				offspring_b.genotype = individual_b.genotype.copy()
	return [offspring_a]

def GOM_variation(fitness: FitnessFunction, subsets, individual_a: Individual, population: list):
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)

	offspring_a.genotype = individual_a.genotype.copy()
	offspring_b.genotype = individual_a.genotype.copy()

	offspring_a.fitness = individual_a.fitness
	offspring_b.fitness = individual_a.fitness

	n = len(population)

	for subset in subsets:
		individual_b = population[np.random.randint(n)]
		
		# crossover the genotype for the genes in the subset
		offspring_a.genotype[subset] = individual_b.genotype[subset].copy()
		
		if (offspring_a.genotype != offspring_b.genotype).any():
			fitness.evaluate(offspring_a)
			if offspring_a.fitness > offspring_b.fitness:
				offspring_b.genotype = offspring_a.genotype.copy()
				offspring_b.fitness = offspring_a.fitness
			else:
				offspring_a.genotype = offspring_b.genotype.copy()
				offspring_a.fitness = offspring_b.fitness
	return [offspring_a]

