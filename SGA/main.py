import numpy as np

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

if __name__ == "__main__":
	crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
	for cx in crossovers:
		inst = "SGA/maxcut-instances/setA/n0000006i00.txt"
		with open("output-{}.txt".format(cx),"w") as f:
			population_size = 500
			num_evaluations_list = []
			evaluation_time_list = []
			num_runs = 10
			num_success = 0
			for i in range(num_runs):
				fitness = FitnessFunction.MaxCut(inst)
				genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False, evaluation="evaluate")
				best_fitness, num_evaluations, generation = genetic_algorithm.run()
				if best_fitness == fitness.value_to_reach:
					num_success += 1
				num_evaluations_list.append(num_evaluations)
				evaluation_time_list.append(fitness.evaluation_time)
			print("{}/{} runs successful".format(num_success,num_runs))
			print("{} evaluations (median)".format(np.median(num_evaluations_list)))
			# print average time per evaluation
			print("Average time per evaluation: {}".format(np.mean(np.array(evaluation_time_list)/np.array(num_evaluations_list))))
			percentiles = np.percentile(num_evaluations_list,[10,50,90])
			f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))
