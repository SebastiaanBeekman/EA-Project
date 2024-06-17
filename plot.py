import matplotlib.pyplot as plt
import numpy as np

population_size = [50, 500, 1000, 1500, 2000, 3000, 4000, 5000]
setA = [0.0, 0.1, 0.13333333333333333, 0.1, 0.2, 0.23333333333333334, 0.26666666666666666, 0.5]
setB = [0.43333333333333335, 0.9333333333333333, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
setC = [0.0, 0.03333333333333333, 0.36666666666666664, 0.36666666666666664, 0.5, 0.5, 0.5666666666666667, 0.5666666666666667]
setD = [0.0, 0.26666666666666666, 0.3, 0.3333333333333333, 0.6, 0.5666666666666667, 0.6, 0.7333333333333333]
setE = [0.0, 0.23333333333333334, 0.3, 0.3333333333333333, 0.5, 0.6333333333333333, 0.6333333333333333, 0.8333333333333334]

plt.plot(population_size, setA, label='setA', color='red', marker='o')
plt.plot(population_size, setB, label='setB', color='blue', marker='o')
plt.plot(population_size, setC, label='setC', color='green', marker='o')
plt.plot(population_size, setD, label='setD', color='purple', marker='o')
plt.plot(population_size, setE, label='setE', color='orange', marker='o')
plt.title("Success rate of different set using Edge Crossover for different population sizes")
plt.legend()
plt.xlabel("Population size")
plt.ylabel("Success rate")
plt.show()