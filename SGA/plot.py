import numpy as np
import os
import matplotlib.pyplot as plt

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def plot_evaluations_population_size(mixed: bool = True):
    figure, ax = plt.subplots(1,2)
    sets = ["setA","setB","setC","setD","setE"]
    groups = {"setA": [], "setB": [], "setC": [], "setD": [], "setE": []}
    if mixed:
        pop_type = "mixed"
        name = "GW"
    else:
        pop_type = "random"
        name = "random"
    for group in sets:
        population = []
        success_rate = []
        num_evaluations_10_percentile = []
        num_evaluations_50_percentile = []
        num_evaluations_90_percentile = []
        with open("output-{}-{}.txt".format(group,name), "r") as f:
            labels = f.readline().split()
            for line in f:
                data = line.split()
                population.append(int(data[0]))
                success_rate.append(float(data[1]))
                num_evaluations_10_percentile.append(float(data[2]))
                num_evaluations_50_percentile.append(float(data[3]))
                num_evaluations_90_percentile.append(float(data[4]))
        ax[0].plot(population,num_evaluations_50_percentile,label=group)
        ax[1].plot(population,success_rate,label=group)
    
    ax[0].legend()
    ax[0].set(xlabel="Population size", ylabel="Median number of evaluations", title="Number of evaluations for different population sizes ({} population)".format(pop_type))
    ax[1].legend()
    ax[1].set(xlabel="Population size", ylabel="Success rate", title="Success rate for different population sizes ({} population)".format(pop_type))
    plt.show()

def preprocess(graphs, number_vertices):
    instances = []
    answers = []
    directory = "SGA/maxcut-instances/{}".format(graphs)
    files = [file for file in os.listdir(directory) if file.endswith(".txt") and f"{number_vertices}i" in file]
    for i in range(len(files)):
        inst = "SGA/maxcut-instances/{}/{}".format(graphs,files[i])
        ans = inst.replace(".txt",".bkv")
        
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

def get_data(GW: bool = False):
    sets = ["setA","setB","setC","setD","setE"]
    population_sizes = [20,50,80,100,150,200,400]
    nodes = [25,25,25,20,20]
    if GW:
        heuristic_fraction = 0.5
        name = "GW"
    else:
        heuristic_fraction = 0.0
        name = "random"
    for group, vertices in zip(sets,nodes):
        print("Running for {}".format(group))
        with open("output-{}-{}.txt".format(group,name), "w") as f:
            f.write("population_size num_success/num_runs num_evaluations_10_percentile num_evaluations_50_percentile num_evaluations_90_percentile\n")
            instances, answers = preprocess(group, vertices)
            inst = instances[0]
            ans = answers[0]
            for population_size in population_sizes:
                print("Running for population size {}".format(population_size))
                num_evaluations_list = []
                num_runs = 10
                num_success = 0

                for i in range(num_runs):
                    fitness = FitnessFunction.MaxCut(inst)
                    fitness.value_to_reach = ans
                    genetic_algorithm = GeneticAlgorithm(fitness,population_size,evaluation_budget=100000,verbose=False,heuristic_fraction=heuristic_fraction)
                    best_fitness, num_evaluations = genetic_algorithm.run()
                    if best_fitness == fitness.value_to_reach:
                        num_success += 1
                    num_evaluations_list.append(num_evaluations)
                print("{}/{} runs successful".format(num_success,num_runs))
                print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                percentiles = np.percentile(num_evaluations_list,[10,50,90])
                f.write("{} {} {} {} {}\n".format(population_size, num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))

if __name__ == "__main__":
    # True for GW, False for random
    # get_data(True)
    # plot_evaluations_population_size(True)
    get_data(False)
    plot_evaluations_population_size(False)

