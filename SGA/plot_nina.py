import numpy as np
import os

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import matplotlib.pyplot as plt


def plot_fitness_population_size(graphs, number_vertices):
    population_size = [2, 10, 20, 30, 40, 50]
    runs = 10
    instances, answers = preprocess(graphs, number_vertices)
    inst = instances[0]
    ans = answers[0]
    crossovers = {"UniformCrossover": [], "OnePointCrossover": [], "TwoPointCrossover": []}
    print(ans)
    for crossover in crossovers:
        resulting_fitness = []
        for pop_size in population_size:
            run_result = []
            for _ in range(runs):
                fitness = FitnessFunction.MaxCut(inst)
                fitness.value_to_reach = ans
                genetic_algorithm = GeneticAlgorithm(fitness, pop_size, variation=crossover, verbose=False, evaluation_budget = 100000)
                best_fitness, num_evaluations, generations = genetic_algorithm.run()
                run_result.append(best_fitness/ans)
            resulting_fitness.append(run_result)
        crossovers[crossover] = resulting_fitness
    
    means_one = np.mean(crossovers["OnePointCrossover"], axis=1)
    st_dev_one = np.std(crossovers["OnePointCrossover"], axis=1)
    means_two = np.mean(crossovers["TwoPointCrossover"], axis=1)
    st_dev_two = np.std(crossovers["TwoPointCrossover"], axis=1)
    means_uniform = np.mean(crossovers["UniformCrossover"], axis=1)
    st_dev_uniform = np.std(crossovers["UniformCrossover"], axis=1)
    
    
    plt.errorbar(population_size, means_one, yerr=st_dev_one, fmt='o', capsize=5, color='blue')
    plt.plot(population_size, means_one, color='blue', label='OnePointCrossover')
    plt.errorbar(population_size, means_two, yerr=st_dev_two, fmt='o', capsize=5, color='red')
    plt.plot(population_size, means_two, color='red', label='TwoPointCrossover')
    plt.errorbar(population_size, means_uniform, yerr=st_dev_uniform, fmt='o', capsize=5, color='green')
    plt.plot(population_size, means_uniform, color='green', label='UniformCrossover')
    plt.xlabel("Population size")
    plt.ylabel("Best fitness upon termination")
    plt.title(f"Fitness vs Population size for a {graphs} graph with {number_vertices} vertices, evaluation budget of 100000")
    plt.legend()
    plt.show()


def preprocess(graphs, number_vertices):
    instances = []
    answers = []
    directory = "maxcut-instances/{}".format(graphs)
    files = [file for file in os.listdir(directory) if file.endswith(".txt") and f"{number_vertices}i" in file]
    for i in range(len(files)):
        inst = "maxcut-instances/{}/{}".format(graphs,files[i])
        ans = inst.replace(".txt",".bkv")
        
        print(inst, ans)
        
        with open( inst, "r" ) as f_in:
            lines = f_in.readlines()
            first_line = lines[0].split()
            number_of_vertices = int(first_line[0])
            
        with open( ans, "r" ) as f_in:
            lines = f_in.readlines()
            a = int(lines[0])
            answers.append(a)
            
        if number_of_vertices == number_vertices:
            instances.append(inst)
    return instances, answers


if __name__ == "__main__":
    # preprocess("setA", "UniformCrossover", 6)
    plot_fitness_population_size("setA", 12)