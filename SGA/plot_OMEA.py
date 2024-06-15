import pandas as pd
import matplotlib.pyplot as plt

df_uniform_D = pd.read_csv("output-uniform_crossover-set-D-report.txt", sep=" ", header=None)
df_uniform_D.columns = ["population_size","success_rate","10th","50th","90th"]
df_gom_D_LT = pd.read_csv("output-GOM_variation-set-D-LT-report.txt", sep=" ", header=None)
df_gom_D_LT.columns = ["population_size","success_rate","10th","50th","90th"]
df_rom_D_LT = pd.read_csv("output-ROM_variation-set-D-LT-report.txt", sep=" ", header=None)
df_rom_D_LT.columns = ["population_size","success_rate","10th","50th","90th"]
df_gom_D_MP = pd.read_csv("output-GOM_variation-set-D-MP-report.txt", sep=" ", header=None)
df_gom_D_MP.columns = ["population_size","success_rate","10th","50th","90th"]
df_rom_D_MP = pd.read_csv("output-ROM_variation-set-D-MP-report.txt", sep=" ", header=None)
df_rom_D_MP.columns = ["population_size","success_rate","10th","50th","90th"]
df_gom_D_uni = pd.read_csv("output-GOM_variation-set-D-uni-report.txt", sep=" ", header=None)
df_gom_D_uni.columns = ["population_size","success_rate","10th","50th","90th"]
df_rom_D_uni = pd.read_csv("output-ROM_variation-set-D-uni-report.txt", sep=" ", header=None)
df_rom_D_uni.columns = ["population_size","success_rate","10th","50th","90th"]


plt.plot(df_uniform_D["population_size"],df_uniform_D["success_rate"],marker="o",label="Uniform CO")
plt.plot(df_gom_D_LT["population_size"],df_gom_D_LT["success_rate"],marker="o",label="GOM LT")
plt.plot(df_rom_D_LT["population_size"],df_rom_D_LT["success_rate"],marker="o",label="ROM LT")
plt.plot(df_gom_D_MP["population_size"],df_gom_D_MP["success_rate"],marker="o",label="GOM MP")
plt.plot(df_rom_D_MP["population_size"],df_rom_D_MP["success_rate"],marker="o",label="ROM MP")
plt.plot(df_gom_D_uni["population_size"],df_gom_D_uni["success_rate"],marker="o",label="GOM uni")
plt.plot(df_rom_D_uni["population_size"],df_rom_D_uni["success_rate"],marker="o",label="ROM uni")


plt.legend()
plt.xlabel("Population size")
plt.ylabel("Success rate")
#plt.title("Success rate of instance with 40 nodes of set D")
plt.show()

plt.plot(df_uniform_D["population_size"],df_uniform_D["50th"],marker="o",label="Uniform CO")
plt.plot(df_gom_D_LT["population_size"],df_gom_D_LT["50th"],marker="o",label="GOM LT")
plt.plot(df_rom_D_LT["population_size"],df_rom_D_LT["50th"],marker="o",label="ROM LT")
plt.plot(df_gom_D_MP["population_size"],df_gom_D_MP["50th"],marker="o",label="GOM MP")
plt.plot(df_rom_D_MP["population_size"],df_rom_D_MP["50th"],marker="o",label="ROM MP")
plt.plot(df_gom_D_uni["population_size"],df_gom_D_uni["50th"],marker="o",label="GOM uni")
plt.plot(df_rom_D_uni["population_size"],df_rom_D_uni["50th"],marker="o",label="ROM uni")

plt.legend()
plt.xlabel("Population size")
plt.ylabel("Number of evaluations")
#plt.title("Number of evaluations of instance with 40 nodes of set D")
plt.show()
