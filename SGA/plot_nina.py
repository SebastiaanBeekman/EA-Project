import numpy as np
import os

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import matplotlib.pyplot as plt


def fitness_population_size(graphs, number_vertices):
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
                genetic_algorithm = GeneticAlgorithm(fitness, pop_size, variation=crossover, verbose=False, evaluation_budget = 100000, are_all_equal = True)
                best_fitness, num_evaluations, generations = genetic_algorithm.run()
                run_result.append(best_fitness/ans)
            resulting_fitness.append(run_result)
        crossovers[crossover] = resulting_fitness
    
    plot_fitness_population_size(crossovers, population_size, graphs, number_vertices)
    
    
def plot_fitness_population_size(crossovers, population_size, graphs, number_vertices):    
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
    
    
def number_of_generations_vs_fitness(graphs, number_vertices):
    population_size = 500
    number_of_generations = [2, 3, 4, 5]
    runs = 10
    instances, answers = preprocess(graphs, number_vertices)
    inst = instances[0]
    ans = answers[0]
    crossovers = {"UniformCrossover": [], "OnePointCrossover": [], "TwoPointCrossover": []}
    print(ans)
    for crossover in crossovers:
        resulting_fitness = []
        for gen in number_of_generations:
            run_result = []
            for _ in range(runs):
                fitness = FitnessFunction.MaxCut(inst)
                genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=crossover, verbose=False, evaluation_budget = gen * population_size + population_size)
                best_fitness, num_evaluations, generations = genetic_algorithm.run()
                print(generations)
                run_result.append(best_fitness/ans)
            resulting_fitness.append(run_result)
        crossovers[crossover] = resulting_fitness
    
    
    plot_number_of_generations_vs_fitness(crossovers, population_size, graphs, number_vertices, number_of_generations)


def plot_number_of_generations_vs_fitness(crossovers, population_size, graphs, number_vertices, number_of_generations):    
    means_one = np.mean(crossovers["OnePointCrossover"], axis=1)
    st_dev_one = np.std(crossovers["OnePointCrossover"], axis=1)
    means_two = np.mean(crossovers["TwoPointCrossover"], axis=1)
    st_dev_two = np.std(crossovers["TwoPointCrossover"], axis=1)
    means_uniform = np.mean(crossovers["UniformCrossover"], axis=1)
    st_dev_uniform = np.std(crossovers["UniformCrossover"], axis=1)


    plt.errorbar(number_of_generations, means_one, yerr=st_dev_one, fmt='o', capsize=5, color='blue')
    plt.plot(number_of_generations, means_one, color='blue', label='OnePointCrossover')
    plt.errorbar(number_of_generations, means_two, yerr=st_dev_two, fmt='o', capsize=5, color='red')
    plt.plot(number_of_generations, means_two, color='red', label='TwoPointCrossover')
    plt.errorbar(number_of_generations, means_uniform, yerr=st_dev_uniform, fmt='o', capsize=5, color='green')
    plt.plot(number_of_generations, means_uniform, color='green', label='UniformCrossover')
    plt.xlabel("Number of generations")
    plt.ylabel("Best fitness upon termination")
    plt.title(f"Fitness vs number of generations for a {graphs} graph with {number_vertices} vertices, population size: {population_size}")
    plt.legend()
    plt.show()
    


def preprocess(graphs, number_vertices):
    instances = []
    answers = []
    directory = "SGA/maxcut-instances/{}".format(graphs)
    files = [file for file in os.listdir(directory) if file.endswith(".txt") and f"{number_vertices}i" in file]
    for i in range(len(files)):
        inst = "SGA/maxcut-instances/{}/{}".format(graphs,files[i])
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
    # fitness_population_size("setA", 12)
    # number_of_generations_vs_fitness("setA", 25)
    preprocess("setA", 6)