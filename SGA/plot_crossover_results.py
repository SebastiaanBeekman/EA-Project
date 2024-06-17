import numpy as np
import os

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import matplotlib.pyplot as plt


def number_of_evaluations():
    sets = {"setA": 50, "setB": 49, "setC": 50, "setD": 40, "setE": 40}
    crossovers = ["UniformCrossover", "OnePointCrossover", "TwoPointCrossover", "EdgeCrossover"]
    population_size = 2000
    results = {}
    evaluation_budget = 1000000
    runs = 5
    for set in sets:
        results[set] = {}
        instances, answers = get_files(set, sets[set])
        for crossover in crossovers:
            results[set][crossover] = {"success_rate": [], "evaluations_median": [], "evaluations_mean": [], "evaluations_std": [], "total_runs": 0}
            success = 0
            evaluations = []
            total = 0
            for c, instance in enumerate(instances):
                for i in range(runs):
                    fitness = FitnessFunction.MaxCut(instance)
                    fitness.value_to_reach = answers[c]
                    genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=crossover, evaluation_budget=evaluation_budget, verbose=False)
                    best_fitness, evaluation, generations = genetic_algorithm.run()
                    if best_fitness == answers[c]:
                        evaluations.append(evaluation)
                        success += 1
                    total += 1
            results[set][crossover]["success_rate"] = success/total
            results[set][crossover]["evaluations_median"] = np.median(evaluations)
            results[set][crossover]["evaluations_mean"] = np.mean(evaluations)
            results[set][crossover]["evaluations_std"] = np.std(evaluations)
            results[set][crossover]["total_runs"] = total
        with open("number_of_evaluations.txt", "w") as f:
            f.write(str(results))
    

def population_size_success_crossovers(graphs, number_vertices):
    crossovers = ["UniformCrossover", "OnePointCrossover", "TwoPointCrossover", "EdgeCrossover"]
    population_sizes = [50, 100, 200, 400, 800, 1200, 1600]
    runs = 3
    evaluation_budget = 10
    instances, answers = get_files(graphs, number_vertices)
    results = []
    for crossover in crossovers:
        results.append([])
        result_pop = []
        for pop in population_sizes:
            count  = 0
            success = 0
            for c, instance in enumerate(instances):
                for i in range(runs):
                    fitness = FitnessFunction.MaxCut(instance)
                    fitness.value_to_reach = answers[c]
                    genetic_algorithm = GeneticAlgorithm(fitness, pop, variation=crossover, evaluation_budget=evaluation_budget, verbose=False)
                    best_fitness, evaluation, generations = genetic_algorithm.run()
                    if best_fitness == answers[c]:
                        success += 1
                    count += 1
            result_pop.append(success/count)
        results[-1].append(result_pop)
        with open(f"SGA/population_size_success_rate_crossovers_{graphs}.txt", "w") as f:
            f.write(str(results))
    plot_population_size_success_rate(results, population_sizes, f"Success rate of different crossovers for {graphs}", crossovers)


def population_size_succes_sets():
    sets = {"setA": 50, "setB": 49, "setC": 50, "setD": 40, "setE": 40}
    population_size = [50, 100, 200, 400, 800, 1200, 1600]
    runs = 3
    crossover = "EdgeCrossover"
    evaluation_budget = 100000
    results = []
    for set in sets:
        results.append([])
        instances, answers = get_files(set, sets[set])
        result_pop = []
        for pop in population_size:
            count  = 0
            success = 0
            for c, instance in enumerate(instances):
                for i in range(runs):
                    fitness = FitnessFunction.MaxCut(instance)
                    fitness.value_to_reach = answers[c]
                    genetic_algorithm = GeneticAlgorithm(fitness, pop, variation=crossover, evaluation_budget=evaluation_budget, verbose=False)
                    best_fitness, evaluation, generations = genetic_algorithm.run()
                    if best_fitness == answers[c]:
                        success += 1
                    count += 1
            result_pop.append(success/count)
        results[-1].append(result_pop)
        with open("SGA/population_size_success_rate.txt", "w") as f:
            f.write(str(results))
    plot_population_size_success_rate(results, population_size, "Success rate of different sets with EdgeCrossover", list(sets.keys()))
        
        
def plot_population_size_success_rate(results, population_size, title, labels):
    for i, result in enumerate(results):
        plt.plot(population_size, result[0], label=labels[i], marker="o")
    plt.title(title)
    plt.xlabel("Population size")
    plt.ylabel("Success rate")
    plt.legend()
    plt.savefig(f"SGA/{title}.png")
    

def get_files(graphs, number_vertices):
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
                


if __name__ == "__main__":
    sets = sets = {"setA": 50, "setB": 49, "setC": 50, "setD": 40, "setE": 40}
    # population_size_succes_sets()
    # number_of_evaluations()
    for s in sets:
        population_size_success_crossovers(s, sets[s])