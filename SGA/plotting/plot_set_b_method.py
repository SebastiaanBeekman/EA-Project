import matplotlib.pyplot as plt
import numpy as np

# Define your data arrays
setA = np.array([528.0,527.7666666666667,527.1,526.9666666666667,527.3333333333334,525.9666666666667,525.1666666666666,492.0])
setC = np.array([26438.966666666667,26519.8,26181.866666666665,25980.333333333332,25453.266666666666,24892.433333333334,23689.8,22059.0])
setD = np.array([177.2,178.66666666666666,175.36666666666667,173.93333333333334,170.06666666666666,169.4,162.76666666666668,155.0])
setE = np.array([192.6,190.76666666666668,192.7,192.13333333333333,189.13333333333333,186.86666666666667,182.1,174.0])

# Define your x-axis labels
x_labels = ['0', '0.05', '0.1', '0.2', '0.35', '0.5', '0.7', '1']

# Calculate differences with respect to the first element
def calculate_relative_changes(array):
    return array / array[0]

rel_setA = calculate_relative_changes(setA)
rel_setC = calculate_relative_changes(setC)
rel_setD = calculate_relative_changes(setD)
rel_setE = calculate_relative_changes(setE)

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(rel_setA, marker='o', label='Set A')
plt.plot(rel_setC, marker='s', label='Set C')
plt.plot(rel_setD, marker='^', label='Set D')
plt.plot(rel_setE, marker='x', label='Set E')

# Adding labels and title
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
plt.xlabel('Fraction of the population initialized with SetB optimal method')
plt.ylabel('Relative Fitness')
plt.title('Relative Fitness compared to fraction of population initialized with SetB optimal method')
plt.legend()

# Display the plot
plt.show()
