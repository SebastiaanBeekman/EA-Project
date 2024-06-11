import numpy as np
import time
from functools import partial 

import Variation
import Selection
from FitnessFunction import FitnessFunction
from Individual import Individual
from Utils import ValueToReachFoundException

class GeneticAlgorithm:
	def __init__(self, fitness: FitnessFunction, population_size, **options ): 
		self.fitness = fitness
		self.evaluation_budget = 1000000
		self.variation_operator = Variation.uniform_crossover
		self.selection_operator = Selection.tournament_selection
		self.evaluation_operator = "evaluate"
		self.population_size = population_size
		self.population = []
		self.number_of_generations = 0
		self.verbose = False
		self.print_final_results = True 
		self.are_all_equal = False # Added to stop if all individuals are equal
		self.print_final_results = True
		self.heuristic_fraction = 0.5
		self.local_search_fraction = 0.1
		self.local_search_epsilon = 0.01

		if "verbose" in options:
			self.verbose = options["verbose"]

		if "evaluation_budget" in options:
			self.evaluation_budget = options["evaluation_budget"]

		if "variation" in options:
			if options["variation"] == "UniformCrossover":
				self.variation_operator = Variation.uniform_crossover
			elif options["variation"] == "OnePointCrossover":
				self.variation_operator = Variation.one_point_crossover
			elif options["variation"] == "TwoPointCrossover":
				self.variation_operator = Variation.two_point_crossover
			elif options["variation"] == "CustomCrossover":
				self.variation_operator = partial(Variation.custom_crossover, self.fitness)
			elif options["variation"] == "EdgeCrossover":
				self.variation_operator = partial(Variation.edge_crossover, self.fitness.edge_list)
    
		if "evaluation" in options:
			if options["evaluation"] == "evaluate":
				self.evaluation_operator = "evaluate"
			elif options["evaluation"] == "partial_evaluate":
				self.evaluation_operator = "partial_evaluate"
    
		if "heuristic_fraction" in options:
			self.heuristic_fraction = options["heuristic_fraction"]
   
		if "local_search_fraction" in options:
			self.local_search_fraction = options["local_search_fraction"]
   
		if "local_search_epsilon" in options:
			self.local_search_epsilon = options["local_search_epsilon"]

	def calculate_cut_weight(self, edge_list, A, B):
		self.fitness.number_of_evaluations += 1
		
		cut_weight = 0
		for e in edge_list:
			if (e[0] in A and e[1] in B) or (e[0] in B and e[1] in A):
				cut_weight += self.fitness.weights[e]
		return cut_weight
 
	def greedy_maxcut_initialization(self, graph):
		egdes = self.fitness.edge_list
		vertices = list(graph.keys())
		degrees = {v: len(graph[v]) for v in vertices}
		sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)
		
		A, B = set(), set()
		
		for vertex in sorted_vertices:
			set_A = A | {vertex}
			set_B = B | {vertex}
			if self.calculate_cut_weight(egdes, set_A, B) > self.calculate_cut_weight(egdes, A, set_B):
				A.add(vertex)
			else:
				B.add(vertex)
		
		return A, B
 
	def initialize_population( self ):
		heuristic_population_size = int(self.population_size * self.heuristic_fraction)
		random_population_size = self.population_size - heuristic_population_size

		population = []
		
		# Generate heuristic-based individuals
		for _ in range(heuristic_population_size):
			A, B = self.greedy_maxcut_initialization(self.fitness.adjacency_list)
			individual = Individual.from_sets(A, B, self.fitness.dimensionality)
			self.fitness.evaluate(individual)
			population.append(individual)
		
		# Generate random individuals
		for _ in range(random_population_size):
			individual = Individual.initialize_uniform_at_random(self.fitness.dimensionality)
			self.fitness.evaluate(individual)
			population.append(individual)
		
		self.population = population

	def make_offspring( self ):
		offspring = []
		order = np.random.permutation(self.population_size)
		for i in range(len(order)//2):
			offspring = offspring + self.variation_operator(self.population[order[2*i]],self.population[order[2*i+1]])
		for individual in offspring:
			if self.evaluation_operator == "evaluate":
				self.fitness.evaluate(individual)
			else:
				parents_genotype = np.stack([self.population[order[2*i]].genotype,self.population[order[2*i+1]].genotype, 1-self.population[order[2*i]].genotype, 1-self.population[order[2*i+1]].genotype])
				parents_fitness = np.stack([self.population[order[2*i]].fitness,self.population[order[2*i+1]].fitness, self.population[order[2*i]].fitness, self.population[order[2*i+1]].fitness])
    			#distance from parent to individual can be calculated as the absolute difference between the two
				distance = np.sum(np.abs(parents_genotype - individual.genotype), axis=1)
				#the closest parent is the one with the smallest distance
				self.fitness.partial_evaluate(individual, parents_genotype[np.argmin(distance)], parents_fitness[np.argmin(distance)])
		return offspring

	def make_selection( self, offspring ):
		return self.selection_operator(self.population, offspring)		

	# Based on https://courses.engr.illinois.edu/cs598csc/sp2011/Lectures/lecture_12.pdf
	def local_search( self, individual, epsilon ):
		improved = True
		while improved:
			improved = False
			n = len(individual.genotype)
			for i in range(n):
				individual.genotype[i] = 1 - individual.genotype[i]
				old_fitness = individual.fitness
				self.fitness.evaluate(individual, is_local_search=True)
				if individual.fitness > (1 + epsilon / n) * old_fitness:
					improved = True
				else:
					individual.genotype[i] = 1 - individual.genotype[i]
					individual.fitness = old_fitness
    
	def apply_local_search(self):
		epsilon = self.local_search_epsilon
		num_local_search = int(self.population_size * self.local_search_fraction)
		selected_individuals = np.random.choice(self.population, num_local_search, replace=False)
		for individual in selected_individuals:
			self.local_search(individual, epsilon)
	
	def print_statistics( self ):
		fitness_list = [ind.fitness for ind in self.population]
		print("Generation {}: Best_fitness: {:.1f}, Avg._fitness: {:.3f}, Nr._of_evaluations: {}, sGA evalutations: {}, Local search evaluations {}".format(self.number_of_generations,max(fitness_list),np.mean(fitness_list),self.fitness.number_of_evaluations,self.fitness.number_of_sga_evaluations,self.fitness.number_of_local_search_evaluations))

	def get_best_fitness( self ):
		return max([ind.fitness for ind in self.population])

	def all_equal(self):
		return all(x == self.population[0] for x in self.population)

	def run( self ):
		try:
			self.initialize_population()
			while( self.fitness.number_of_evaluations < self.evaluation_budget): # stop if all individuals have the same fitness
				self.number_of_generations += 1
				if( self.verbose and self.number_of_generations%100 == 0 ):
					self.print_statistics()

				offspring = self.make_offspring()
				selection = self.make_selection(offspring)
				self.population = selection
				
				self.apply_local_search()			
    
				if self.all_equal():
					print("All individuals have the same fitness")
					break
			if( self.verbose and self.are_all_equal):
				self.print_statistics()
		except ValueToReachFoundException as exception:
			if( self.print_final_results ):
				print(exception)
				print("Best fitness: {:.1f}, Nr._of_evaluations: {}, Nr._of_sGA_evaluations: {}, Nr._of_LS_evaluations: {}".format(exception.individual.fitness, self.fitness.number_of_evaluations, self.fitness.number_of_sga_evaluations, self.fitness.number_of_local_search_evaluations))
			return exception.individual.fitness, self.fitness.number_of_evaluations, self.fitness.number_of_sga_evaluations, self.fitness.number_of_local_search_evaluations
		if( self.print_final_results ):
			self.print_statistics()
		return self.get_best_fitness(), self.fitness.number_of_evaluations, self.fitness.number_of_sga_evaluations, self.fitness.number_of_local_search_evaluations

