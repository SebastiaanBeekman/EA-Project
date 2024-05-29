import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import FitnessFunction
import Individual as individual
import GeneticAlgorithm as ga
import numpy as np

class TestPartialEvaluation(unittest.TestCase):
    def set_Up(self):
        pass
    
    def test_one_difference(self):
        # Test that the partial evaluation function works correctly for maxcut when only one gene is different
        inst = "SGA/maxcut-instances/setA/n0000006i00.txt"
        individual1 = individual.Individual()
        individual2 = individual.Individual()
        individual3 = individual.Individual()
        individual1.genotype = np.array([0,0,0,0,0,0])
        individual2.genotype = np.array([0,0,0,0,0,1])
        individual3.genotype = np.array([0,0,0,0,0,1])
        fitness = FitnessFunction.MaxCut(inst)
        genetic_algorithm = ga.GeneticAlgorithm(fitness, 500, variation="CustomCrossover", evaluation_budget=100000, verbose=False)
        genetic_algorithm.fitness = fitness
        fitness.evaluate(individual1)
        fitness.evaluate(individual2)
        fitness.partial_evaluate(individual3, individual1)
        self.assertEqual(individual3.fitness, individual2.fitness)
        
    def test_all_differences(self):
        # Test that the partial evaluation function works correctly for maxcut when all genes are different
        inst = "SGA/maxcut-instances/setA/n0000006i00.txt"
        individual1 = individual.Individual()
        individual2 = individual.Individual()
        individual3 = individual.Individual()
        individual1.genotype = np.array([0,0,0,0,0,0])
        individual2.genotype = np.array([1,1,1,1,1,1])
        individual3.genotype = np.array([1,1,1,1,1,1])
        fitness = FitnessFunction.MaxCut(inst)
        genetic_algorithm = ga.GeneticAlgorithm(fitness, 500, variation="CustomCrossover", evaluation_budget=100000, verbose=False)
        genetic_algorithm.fitness = fitness
        fitness.evaluate(individual1)
        fitness.evaluate(individual2)
        fitness.partial_evaluate(individual3, individual1)
        self.assertEqual(individual3.fitness, individual2.fitness)
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    
    suite.addTest(TestPartialEvaluation('test_one_difference'))
    suite.addTest(TestPartialEvaluation('test_all_differences'))
    
    #create a test runner
    runner = unittest.TextTestRunner()
    
    #Run the test suite
    result = runner.run(suite)
    