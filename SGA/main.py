from multiprocessing import Process
import numpy as np
import threading
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def run(cx, fraction, inst):
	with open(f"output/output-{cx}-{fraction}-{inst.replace('/', '_')}","w") as f:
		population_size = 200
		num_evaluations_list = []
		best_fitnesses = []
		num_runs = 30
		num_success = 0
		for i in range(num_runs):
			fitness = FitnessFunction.MaxCut(inst)
			genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False, pop_frac=fraction)
			best_fitness, num_evaluations = genetic_algorithm.run()
			if best_fitness == fitness.value_to_reach:
				num_success += 1
			num_evaluations_list.append(num_evaluations)
			best_fitnesses.append(best_fitness)
		print("{}/{} runs successful".format(num_success,num_runs))
		print("{} evaluations (median)".format(np.median(num_evaluations_list)))
		percentiles = np.percentile(num_evaluations_list,[10,50,90])
		f.write("{} {} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2], np.average(best_fitnesses)))


if __name__ == "__main__":
	# Example usage:
	# crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
	# instances = ["maxcut-instances/setA/n0000025i09.txt", "maxcut-instances/setC/n0000050i08.txt", "maxcut-instances/setD/n0000040i09.txt", "maxcut-instances/setE/n0000040i08.txt"]
	# population_fractions = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1]
	
	crossovers = ["UniformCrossover", "OnePointCrossover", "TwoPointCrossover"]
	instances = ["maxcut-instances/setE/n0000020i00.txt"]
	population_fractions = [0.05]
	procs = []
	for cx in crossovers:
		for inst in instances:
			for fraction in population_fractions:
				print(f"Submitting run: {cx} - {fraction} - {inst}")
				proc = Process(target=run, args=(cx, fraction, inst))
				procs.append(proc)
				proc.start()
	print("Done submitting tasks to the multi processor")
	alive = len(procs)
	for proc in procs:
		proc.join()
		alive = alive - 1
		print(f"Job done, left: [{alive}/{len(procs)}]")
	print("Done")