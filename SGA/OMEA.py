import numpy as np
import Variation
import Selection
from FitnessFunction import FitnessFunction
from Individual import Individual
from Utils import ValueToReachFoundException
import FamilyOfSubsets 

class OMEA:
	def __init__(self, fitness: FitnessFunction, population_size, **options ): 
		self.fitness = fitness
		self.evaluation_budget = 1000000
		self.variation_operator = Variation.GOM_variation
		self.selection_operator = Selection.tournament_selection_for_OMEA
		self.population_size = population_size
		self.population = []
		self.number_of_generations = 0
		self.verbose = False
		self.print_final_results = True 
		self.FOS_operator = FamilyOfSubsets.univariate_subsets
		self.FOS = None

		if "verbose" in options:
			self.verbose = options["verbose"]

		if "evaluation_budget" in options:
			self.evaluation_budget = options["evaluation_budget"]
			
		if "FOS" in options:
			if options["FOS"] == "univariate":
				self.FOS_operator = FamilyOfSubsets.univariate_subsets
			elif options["FOS"] == "linkage_tree":
				self.FOS_operator = FamilyOfSubsets.linkage_tree_fos_learning
			elif options["FOS"] == "marginal_product":
				self.FOS_operator = FamilyOfSubsets.marginal_product_fos_learning

		if "variation" in options:
			if options["variation"] == "GOM_variation":
				self.variation_operator = Variation.GOM_variation
			elif options["variation"] == "ROM_variation":
				self.variation_operator = Variation.ROM_variation
		


				
		
	def initialize_population( self ):
		self.population = [Individual.initialize_uniform_at_random(self.fitness.dimensionality) for i in range(self.population_size)]
		for individual in self.population:
			self.fitness.evaluate(individual)

	def make_offspring( self ):
		offspring = []
		order = np.random.permutation(self.population_size)
		for i in range(len(order)):
			offspring = offspring + self.variation_operator(self.fitness, self.FOS,self.population[order[i]],self.population)
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
				self.FOS = self.FOS_operator(self.population,self.fitness)
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

