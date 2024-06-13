import numpy as np
import pandas as pd

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

if __name__ == "__main__":
    instances = ["setA/n0000050i00.txt", "setB/n0000049i00.txt", "setC/n0000050i00.txt", "setD/n0000040i00.txt", "setE/n0000040i00.txt"]
    initialization_percentages = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5]
    crossovers = ["UniformCrossover"]
    
    results = []
    for instance in instances:
        set_name = instance.split("/")[0]
        for cx in crossovers:
            for initialization_percentage in initialization_percentages:
                    inst = f"SGA/maxcut-instances/{instance}"
                    population_size = 500
                    num_evaluations_list = []
                    num_sga_evaluations_list = []
                    num_ls_evaluations_list = []
                    num_runs = 30
                    num_success = 0

                    for i in range(num_runs):
                        fitness = FitnessFunction.MaxCut(inst)
                        genetic_algorithm = GeneticAlgorithm(
                            fitness,
                            population_size,
                            variation=cx,
                            evaluation_budget=100000,
                            verbose=False,
                            heuristic_fraction=initialization_percentage,
                            local_search_fraction=0.0,
                            local_search_epsilon=0.0,
                            partial_local_search=False
                        )
                        best_fitness, num_evaluations, num_sga_evaluations, num_ls_evaluations = genetic_algorithm.run()
                        if best_fitness == fitness.value_to_reach:
                            num_success += 1
                        num_evaluations_list.append(num_evaluations)
                        num_sga_evaluations_list.append(num_sga_evaluations)
                        num_ls_evaluations_list.append(num_ls_evaluations)
                        
                    success_rate = num_success / num_runs
                    
                    median_evaluations = np.median(num_evaluations_list)
                    # lower_percentile, upper_percentile = np.percentile(num_evaluations_list, [10, 90])
                    
                    # median_sga_evaluations = np.median(num_sga_evaluations_list)
                    # lower_sga_percentile, upper_sga_percentile = np.percentile(num_sga_evaluations_list, [10, 90])
                    
                    # median_ls_evaluations = np.median(num_ls_evaluations_list)
                    # lower_ls_percentile, upper_ls_percentile = np.percentile(num_ls_evaluations_list, [10, 90])

                    results.append({
                        "Percentage": initialization_percentage,
                        "Success Rate": success_rate,
                        "Median Evaluations": median_evaluations,
                        # "10th Percentile Evaluations": lower_percentile,
                        # "90th Percentile Evaluations": upper_percentile,
                        # "Median SGA Evaluations": median_sga_evaluations,
                        # "10th Percentile SGA Evaluations": lower_sga_percentile,
                        # "90th Percentile SGA Evaluations": upper_sga_percentile,
                        # "Median LS Evaluations": median_ls_evaluations,
                        # "10th Percentile LS Evaluations": lower_ls_percentile,
                        # "90th Percentile LS Evaluations": upper_ls_percentile
                    })
                    
                    print(f"Instance: {set_name}")
                    print(f"Crossover: {cx}, Initialization Percentage: {initialization_percentage}")
                    print(f"{num_success}/{num_runs} runs successful")
                    print(f"{median_evaluations} evaluations (median)")
                    # print(f"{median_sga_evaluations} SGA evaluations (median)")
                    # print(f"{median_ls_evaluations} LS evaluations (median)")

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"greedy_set{set_name}.csv", index=False)