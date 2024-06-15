import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GeneticAlgorithm import GeneticAlgorithm
from OMEA import OMEA
import FitnessFunction

if __name__ == "__main__":
	crossovers = ["uniform_crossover", "GOM_variation", "ROM_variation"]#, "block_crossover", "uniform_crossover"]
	for cx in crossovers:
		with open("output-{}-set-D-mp-report-2.txt".format(cx),"w") as f:
			for population_size in [200,1200,1500]:
				num_success = 0
				num_evaluations_list = []
				num_runs = 30
				for instance in range(10):
					inst = "SGA/maxcut-instances/setD/n0000040i0{}.txt".format(instance)
					for i in range(3): # 3 runs per instance
						fitness = FitnessFunction.MaxCut(inst)	
						if cx == "GOM_variation" or cx == "ROM_variation":
							genetic_algorithm = OMEA(fitness,population_size,variation=cx,evaluation_budget=200000,verbose=False,FOS="marginal_product")
						else:
							genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=200000,verbose=False)
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
				standard_deviation = np.std(num_evaluations_list)
				f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2],standard_deviation))

	