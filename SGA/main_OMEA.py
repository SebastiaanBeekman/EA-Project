import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GeneticAlgorithm import GeneticAlgorithm
from OMEA import OMEA
import FitnessFunction

if __name__ == "__main__":
	crossovers = ["GOM_variation", "ROM_variation", "block_crossover", "uniform_crossover"]
	for cx in crossovers:
		inst = "EA-Project/SGA/maxcut-instances/setD/n0000020i00.txt"
		with open("output-{}.txt".format(cx),"w") as f:
			for population_size in [20,50,80,100,150,200,400]:
				num_evaluations_list = []
				num_runs = 10
				num_success = 0
				for i in range(num_runs):
					fitness = FitnessFunction.MaxCut(inst)	
					if cx == "GOM_variation" or cx == "ROM_variation":
						genetic_algorithm = OMEA(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False,FOS="linkage_tree")
					else:
						genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False)
					best_fitness, num_evaluations = genetic_algorithm.run()
					if best_fitness == fitness.value_to_reach:
						num_success += 1
						# adapted to only store when successful
						num_evaluations_list.append(num_evaluations)
				if num_evaluations_list == []:
					num_evaluations_list = [0]
				print("{}/{} runs successful".format(num_success,num_runs))
				print("{} evaluations (median)".format(np.median(num_evaluations_list)))
				percentiles = np.percentile(num_evaluations_list,[10,50,90])
				f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))

	