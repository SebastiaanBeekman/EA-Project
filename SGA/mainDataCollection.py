import numpy as np
import os
import time

from GeneticAlgorithm import GeneticAlgorithm
import GraphGeneration
import FitnessFunction

if __name__ == "__main__":
	crossovers = ["UniformCrossover", "OnePointCrossover", "TwoPointCrossover"] # add later: "CustomCrossover"
	for set in ["setA","setB","setC","setD","setE"]:

		for cx in crossovers:
			directory = "maxcut-instances/{}".format(set)
			files = {file for file in os.listdir(directory) if file.endswith(".txt")}
			with open("output-{}-{}.txt".format(cx,set),"w") as f:
				f.write("instance number_vertices number_edges population_size runtime num_success/num_runs num_evaluations_10_percentile num_evaluations_50_percentile num_evaluations_90_percentile\n")
				for file in files:
					inst = "maxcut-instances/{}/{}".format(set,file)

					with open( inst, "r" ) as f_in:
						lines = f_in.readlines()
						first_line = lines[0].split()
						number_of_vertices = int(first_line[0])
						number_of_edges = int(first_line[1])
					
					# Remove this line to run all instances
					if number_of_vertices > 10:
						continue

					population_size = 500
					num_evaluations_list = []
					num_runs = 30
					num_success = 0
					
					start_time = time.time()
					for i in range(num_runs):
						fitness = FitnessFunction.MaxCut(inst)
						genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False)
						best_fitness, num_evaluations = genetic_algorithm.run()
						if best_fitness == fitness.value_to_reach:
							num_success += 1
						num_evaluations_list.append(num_evaluations)
					end_time = time.time()
					runtime = end_time - start_time
					print("{}/{} runs successful".format(num_success,num_runs))
					print("{} evaluations (median)".format(np.median(num_evaluations_list)))
					percentiles = np.percentile(num_evaluations_list,[10,50,90])
					f.write("{} {} {} {} {} {} {} {} {}\n".format(file,number_of_vertices,number_of_edges, population_size, runtime, num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))

		GraphGeneration.plot_runtime(set)
		GraphGeneration.plot_population_size(set)
		GraphGeneration.plot_num_evaluations(set)
		GraphGeneration.plot_success_rate(set)
