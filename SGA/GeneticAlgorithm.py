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
		self.population_size = population_size
		self.population = []
		self.number_of_generations = 0
		self.verbose = False
		self.print_final_results = True
		self.heuristic_fraction = 0.5
		self.local_search_fraction = 0.1

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
    
		if "heuristic_fraction" in options:
			self.heuristic_fraction = options["heuristic_fraction"]
   
		if "local_search_fraction" in options:
			self.local_search_fraction = options["local_search_fraction"]

	def calculate_cut_weight(self, graph, A, B):
		cut_weight = 0
		for u in A:
			for v in B:
				if v in graph[u]:
					cut_weight += graph[u][graph[u].index(v)]
		return cut_weight
 
	def greedy_maxcut_initialization(self, graph):
		vertices = list(graph.keys())
		degrees = {v: len(graph[v]) for v in vertices}
		sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)
		
		A, B = set(), set()
		
		for vertex in sorted_vertices:
			set_A = A | {vertex}
			set_B = B | {vertex}
			if self.calculate_cut_weight(graph, set_A, B) > self.calculate_cut_weight(graph, A, set_B):
				A.add(vertex)
			else:
				B.add(vertex)
		
		return A, B
 
	def initialize_population( self ):
		# self.population = [Individual.initialize_uniform_at_random(self.fitness.dimensionality) for i in range(self.population_size)]
		# for individual in self.population:
		# 	self.fitness.evaluate(individual)
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
			self.fitness.evaluate(individual)
		return offspring

	def make_selection( self, offspring ):
		return self.selection_operator(self.population, offspring)

	def local_search(self, individual):
		improved = True
		while improved:
			improved = False
			for i in range(len(individual.genotype)):
				individual.genotype[i] = 1 - individual.genotype[i]
				old_fitness = individual.fitness
				self.fitness.evaluate(individual)
				if individual.fitness > old_fitness:
					improved = True
				else:
					individual.genotype[i] = 1 - individual.genotype[i]
					individual.fitness = old_fitness
    
	def apply_local_search(self):
		num_local_search = int(self.population_size * self.local_search_fraction)
		selected_individuals = np.random.choice(self.population, num_local_search, replace=False)
		for individual in selected_individuals:
			self.local_search(individual)
	
	def print_statistics( self ):
		fitness_list = [ind.fitness for ind in self.population]
		print("Generation {}: Best_fitness: {:.1f}, Avg._fitness: {:.3f}, Nr._of_evaluations: {}".format(self.number_of_generations,max(fitness_list),np.mean(fitness_list),self.fitness.number_of_evaluations))

	def get_best_fitness( self ):
		return max([ind.fitness for ind in self.population])

	def run( self ):
		try:
			self.initialize_population()
			while( self.fitness.number_of_evaluations < self.evaluation_budget ):
				self.number_of_generations += 1
				if( self.verbose and self.number_of_generations%100 == 0 ):
					self.print_statistics()

				offspring = self.make_offspring()
				selection = self.make_selection(offspring)
				self.population = selection
				
				self.apply_local_search()
    
			if( self.verbose ):
				self.print_statistics()
		except ValueToReachFoundException as exception:
			if( self.print_final_results ):
				print(exception)
				print("Best fitness: {:.1f}, Nr._of_evaluations: {}".format(exception.individual.fitness, self.fitness.number_of_evaluations))
			return exception.individual.fitness, self.fitness.number_of_evaluations
		if( self.print_final_results ):
			self.print_statistics()
		return self.get_best_fitness(), self.fitness.number_of_evaluations

