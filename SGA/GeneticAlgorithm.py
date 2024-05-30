import numpy as np
import time
from functools import partial 

import Variation
import Selection
from FitnessFunction import FitnessFunction
from Individual import Individual
from Utils import ValueToReachFoundException

import networkx as nx
import cvxgraphalgs as cvxgr

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
	
	def goemans_williamson( self, graph ):
		G = nx.Graph()
		adj_list = self.fitness.adjacency_list

		for u in adj_list.keys():
			for v in adj_list[u]:
				if( v > u ):
					G.add_edge(u, v, weight=self.fitness.get_weight(u,v))
		
		sol = cvxgr.algorithms.goemans_williamson_weighted(G)
		return sol.left, sol.right

	def initialize_population( self ):
		# self.population = [Individual.initialize_uniform_at_random(self.fitness.dimensionality) for i in range(self.population_size)]
		# for individual in self.population:
		# 	self.fitness.evaluate(individual)
		heuristic_population_size = int(self.population_size * self.heuristic_fraction)
		random_population_size = self.population_size - heuristic_population_size

		population = []
		
		# Generate heuristic-based individuals
		for _ in range(heuristic_population_size):
			A, B = self.goemans_williamson(self.fitness.adjacency_list)
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

