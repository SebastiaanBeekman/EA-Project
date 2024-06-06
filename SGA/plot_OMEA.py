import pandas as pd
import matplotlib.pyplot as plt

df_block = pd.read_csv("output-block_crossover.txt", sep=" ", header=None)
df_block.columns = ["population_size","success_rate","10th","50th","90th"]
df_uniform = pd.read_csv("output-uniform_crossover.txt", sep=" ", header=None)
df_uniform.columns = ["population_size","success_rate","10th","50th","90th"]
df_gom = pd.read_csv("output-GOM_variation.txt", sep=" ", header=None)
df_gom.columns = ["population_size","success_rate","10th","50th","90th"]
df_rom = pd.read_csv("output-ROM_variation.txt", sep=" ", header=None)
df_rom.columns = ["population_size","success_rate","10th","50th","90th"]

plt.plot(df_block["population_size"],df_block["success_rate"],label="BlockCrossover")
plt.plot(df_uniform["population_size"],df_uniform["success_rate"],label="UniformCrossover")
plt.plot(df_gom["population_size"],df_gom["success_rate"],label="GOM_variation")
plt.plot(df_rom["population_size"],df_rom["success_rate"],label="ROM_variation")
plt.legend()
plt.xlabel("Population size")
plt.ylabel("Success rate")
plt.title("Success rate of instance with 20 nodes of set D")
plt.show()

plt.plot(df_block["population_size"],df_block["50th"],label="BlockCrossover")
plt.plot(df_uniform["population_size"],df_uniform["50th"],label="UniformCrossover")
plt.plot(df_gom["population_size"],df_gom["50th"],label="GOM_variation")
plt.plot(df_rom["population_size"],df_rom["50th"],label="ROM_variation")
plt.legend()
plt.xlabel("Population size")
plt.ylabel("Number of evaluations")
plt.title("Number of evaluations of instance with 20 nodes of set D")
plt.show()
