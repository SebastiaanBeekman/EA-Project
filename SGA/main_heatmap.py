import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import time

def run_experiment(instance, cx, k_fraction, epsilon, population_size=500, num_runs=30, evaluation_budget=100000):
    success_rates = [0] * num_runs
    num_evaluations_list = []
    num_sga_evaluations_list = []
    num_ls_evaluations_list = []
    for i in range(num_runs):
        fitness = FitnessFunction.MaxCut(instance)
        genetic_algorithm = GeneticAlgorithm(
            fitness,
            population_size,
            variation=cx,
            evaluation_budget=evaluation_budget,
            verbose=False,
            heuristic_fraction=0.0,
            local_search_fraction=k_fraction,
            local_search_epsilon=epsilon,
            partial_local_search=False
        )
        best_fitness, num_evaluations, num_sga_evaluations, num_ls_evaluations = genetic_algorithm.run()
        if best_fitness == fitness.value_to_reach:
            success_rates[i] = 1
            num_evaluations_list.append(num_evaluations)
            num_sga_evaluations_list.append(num_sga_evaluations)
            num_ls_evaluations_list.append(num_ls_evaluations)
        
    success_rate = np.mean(success_rates) if len(success_rates) > 0 else 0
    median_num_evaluations = np.median(num_evaluations_list) if len(num_evaluations_list) > 0 else 0
    median_num_sga_evaluations = np.median(num_sga_evaluations_list) if len(num_sga_evaluations_list) > 0 else 0
    median_num_ls_evaluations = np.median(num_ls_evaluations_list) if len(num_ls_evaluations_list) > 0 else 0
    
    return success_rate, median_num_evaluations, median_num_sga_evaluations, median_num_ls_evaluations

if __name__ == "__main__":
    instances = ["setA/n0000025i00.txt"]
    crossovers = ["UniformCrossover"]
    k_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    epsilon_values = [0.0, 0.001, 0.01, 0.1, 0.5]

    for instance in instances:
        instance_set = instance.split("/")[0]
        for cx in crossovers:
            fig, axes = plt.subplots(2, 1, figsize=(8, 12))
            fig.suptitle(f"Heatmap of Success Rate and Median Evaluations over Local Search Parameters for {instance_set} with {cx}", fontsize=16)
            
            success_rate_results = []
            median_evaluations_results = []
            for k in k_values:
                for epsilon in epsilon_values:
                    if k == 0.0 and epsilon != 0.0:
                        success_rate_results.append((k, epsilon, 0.0))
                        median_evaluations_results.append((k, epsilon, 0.0, 0.0, 0.0))
                        continue
                    
                    print(f"Running experiment for {instance} with {cx}, k={k}, epsilon={epsilon}")
                    success_rate, median_num_evaluations, median_num_sga_evaluations, median_num_ls_evaluations = run_experiment(f"SGA/maxcut-instances/{instance}", cx, k, epsilon, num_runs=10)
                    
                    success_rate_results.append((k, epsilon, success_rate))
                    median_evaluations_results.append((k, epsilon, median_num_evaluations, median_num_sga_evaluations, median_num_ls_evaluations))
            
            success_rate_df = pd.DataFrame(success_rate_results, columns=["k%", "epsilon", "success_rate"])
            success_rate_pivot_table = success_rate_df.pivot(index="k%", columns="epsilon", values="success_rate")
            success_rate_pivot_table = success_rate_pivot_table.iloc[::-1] # Reverse the order of k%
            
            sns.heatmap(success_rate_pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[0])
            axes[0].set_title(f"{cx}")
            axes[0].set_xlabel("Epsilon")
            axes[0].set_ylabel("k%")
            
            # success_rate_df.to_csv(f"data/heatmap/data/partial_{instance_set}_{cx}_heatmap.csv", index=False)
            success_rate_df.to_csv(f"partial_{instance_set}_{cx}_heatmap.csv", index=False)
            
            median_evaluations_df = pd.DataFrame(median_evaluations_results, columns=["k%", "epsilon", "median_num_evaluations", "median_num_sga_evaluations", "median_num_ls_evaluations"])
            median_evaluations_pivot_table = median_evaluations_df.pivot(index="k%", columns="epsilon", values="median_num_evaluations")
            median_evaluations_pivot_table = median_evaluations_pivot_table.iloc[::-1]
            
            sns.heatmap(median_evaluations_pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[1])
            axes[1].set_title(f"{cx}")
            axes[1].set_xlabel("Epsilon")
            axes[1].set_ylabel("k%")
            
            plt.tight_layout()
            # plt.savefig(f"data/heatmap/figs/partial_{instance_set}_{cx}_heatmap_median_evaluations.png")
            plt.savefig(f"partial_{instance_set}_{cx}_heatmap_median_evaluations.png")
            
            # median_evaluations_df.to_csv(f"data/heatmap/data/partial_{instance_set}_{cx}_median_evaluations.csv", index=False)
            median_evaluations_df.to_csv(f"partial_{instance_set}_{cx}_median_evaluations.csv", index=False)
