import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_runtime(set):
    for crossover in ["UniformCrossover","OnePointCrossover","TwoPointCrossover"]:
        data = pd.read_csv("output-{}-{}.txt".format(crossover,set),sep=" ")
        number_vertices = data["number_vertices"]
        runtime = data["runtime"]
        plt.scatter(number_vertices,runtime,label=crossover)
    plt.xlabel("Number of vertices")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.title("Runtime for different crossovers for {}".format(set))
    #plt.savefig("runtime-{}.png".format(set))
    plt.show()

def plot_population_size(set):
    for crossover in ["UniformCrossover","OnePointCrossover","TwoPointCrossover"]:
        data = pd.read_csv("output-{}-{}.txt".format(crossover,set),sep=" ")
        number_vertices = data["number_vertices"]
        population_size = data["population_size"]
        plt.scatter(number_vertices,population_size,label=crossover)
    plt.xlabel("Number of vertices")
    plt.ylabel("Population size")
    plt.legend()
    plt.title("Population size for different crossovers for {}".format(set))
    #plt.savefig("population_size-{}.png".format(set))
    plt.show()

def plot_num_evaluations(set):
    for crossover in ["UniformCrossover","OnePointCrossover","TwoPointCrossover"]:
        data = pd.read_csv("output-{}-{}.txt".format(crossover,set),sep=" ")
        number_vertices = data["number_vertices"]
        num_evaluations_50_percentile = data["num_evaluations_50_percentile"]
        plt.scatter(number_vertices,num_evaluations_50_percentile,label=crossover)
    plt.xlabel("Number of vertices")
    plt.ylabel("Number of evaluations (50th percentile)")
    plt.legend()
    plt.title("Number of evaluations for different crossovers for {}".format(set))
    #plt.savefig("num_evaluations-{}.png".format(set))
    plt.show()

def plot_success_rate(set):
    for crossover in ["UniformCrossover","OnePointCrossover","TwoPointCrossover"]:
        data = pd.read_csv("output-{}-{}.txt".format(crossover,set),sep=" ")
        number_vertices = data["number_vertices"]
        success_rate = data["num_success/num_runs"]
        plt.scatter(number_vertices,success_rate,label=crossover)
    plt.xlabel("Number of vertices")
    plt.ylabel("Success rate")
    plt.legend()
    plt.title("Success rate for different crossovers for {}".format(set))
    #plt.savefig("success_rate-{}.png".format(set))
    plt.show()
    
    