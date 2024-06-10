import numpy as np
import matplotlib.pyplot as plt

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

if __name__ == "__main__":
    crossovers = ["OnePointCrossover", "TwoPointCrossover", "UniformCrossover"]
    local_search_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Example fractions to test
    results = {}

    for cx in crossovers:
        for fraction in local_search_fractions:
            inst = "SGA/maxcut-instances/setA/n0000040i00.txt"
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
                    heuristic_fraction=0.0,
                    local_search_fraction=fraction,  # Set the local search fraction here
                    local_search_epsilon=0.0
                )
                best_fitness, num_evaluations, num_sga_evaluations, num_ls_evaluations = genetic_algorithm.run()
                if best_fitness == fitness.value_to_reach:
                    num_success += 1
                num_evaluations_list.append(num_evaluations)
                num_sga_evaluations_list.append(num_sga_evaluations)
                num_ls_evaluations_list.append(num_ls_evaluations)

            # Store results for plotting
            results[(cx, fraction)] = {
                "num_success": num_success,
                "num_evaluations_list": num_evaluations_list,
                "num_sga_evaluations_list": num_sga_evaluations_list,
                "num_ls_evaluations_list": num_ls_evaluations_list,
                "percentiles": np.percentile(num_evaluations_list, [10, 50, 90])
            }
            
            print(f"Crossover: {cx}, Local Search Fraction: {fraction}")
            print(f"{num_success}/{num_runs} runs successful")
            print(f"{np.median(num_evaluations_list)} evaluations (median)")

    # Plotting the results
    fig, ax = plt.subplots(len(local_search_fractions), 2, figsize=(10, 5 * len(local_search_fractions)))

    for idx, fraction in enumerate(local_search_fractions):
        crossovers = [key[0] for key in results.keys() if key[1] == fraction]
        success_rates = [results[(cx, fraction)]["num_success"] / num_runs for cx in crossovers]
        medians = np.array([results[(cx, fraction)]["percentiles"][1] for cx in crossovers])
        lower_percentiles = np.array([results[(cx, fraction)]["percentiles"][0] for cx in crossovers])
        upper_percentiles = np.array([results[(cx, fraction)]["percentiles"][2] for cx in crossovers])

        # Plot success rates
        ax[idx, 0].bar(crossovers, success_rates, color='b')
        ax[idx, 0].set_title(f'Success Rates (Local Search Fraction: {fraction})')
        ax[idx, 0].set_xlabel('Crossover Method')
        ax[idx, 0].set_ylabel('Success Rate')

        # Plot number of evaluations
        error_bars = [medians - lower_percentiles, upper_percentiles - medians]
        ax[idx, 1].bar(crossovers, medians, color='g', yerr=error_bars)
        ax[idx, 1].set_title(f'Number of Evaluations (Local Search Fraction: {fraction})')
        ax[idx, 1].set_xlabel('Crossover Method')
        ax[idx, 1].set_ylabel('Number of Evaluations (Median)')

    plt.tight_layout()
    plt.savefig('results_ls_set.png')
    plt.show()