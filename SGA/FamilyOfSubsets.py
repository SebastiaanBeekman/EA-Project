import FitnessFunction
import numpy as np

def univariate_subsets(population, fitness: FitnessFunction ):
    subsets = []
    for i in range(fitness.dimensionality):
        subsets.append([i])
    return subsets

def chain_of_cliques_subsets_linkage_tree(population,  fitness: FitnessFunction ):
    subsets = [[0,1,2,3,4]]
    i = 5
    while i < fitness.dimensionality-4:
        subsets.append([i,i+1,i+2,i+3,i+4])
        subsets.append(subsets[-2]+[i,i+1,i+2,i+3,i+4])
        i += 5
 
    return subsets

def chain_of_cliques_subsets(population,  fitness: FitnessFunction ):
    subsets = [[0,1,2,3,4]]
    i = 5
    while i < fitness.dimensionality-4:
        subsets.append([i,i+1,i+2,i+3,i+4])
        subsets.append([i-1,i])
        i += 5
 
    return subsets

def block_subsets(population,  fitness: FitnessFunction, block_size = 5):
    assert fitness.dimensionality % block_size == 0, "Dimensionality should be a multiple of block size"
    subsets = []
    for i in range(fitness.dimensionality):
        if i % block_size == 0:
            subsets.append([i,i+1,i+2,i+3,i+4])
    return subsets

def linkage_tree_fos_learning(population, fitness: FitnessFunction): 

    entropy_values = entropy_matrix(population, fitness)
 
    subsets = []
    # create the initial subsets
    for i in range(fitness.dimensionality):
        subsets.append([i])
    active_subsets = subsets.copy()
    # now iterate over the subsets and create new subsets by combining them based on the UPGMA
    for _ in range(fitness.dimensionality-1):
        # find the two closest subsets
        mutual_information = np.ones((len(active_subsets),len(active_subsets))) * -np.inf
        for i in range(len(active_subsets)):
            for j in range(i+1,len(active_subsets)):
                mutual_information[i,j] = UPGMA(active_subsets[i],active_subsets[j], entropy_values)         
        # find the maximum mutual information
        max_mutual_information = np.max(mutual_information)
        # find the indices of the maximum mutual information
        indices = np.where(mutual_information == max_mutual_information)
        # create a new subset
        new_subset = active_subsets[indices[0][0]] + active_subsets[indices[1][0]]
        # remove the two old subsets
        if indices[0][0] < indices[1][0]:
            active_subsets.pop(indices[1][0])
            active_subsets.pop(indices[0][0])
        else:
            active_subsets.pop(indices[0][0])
            active_subsets.pop(indices[1][0])
        # add the new subset
        active_subsets.append(new_subset)
        # add the new subset to the list of subsets
        subsets.append(new_subset)
    subsets.pop()
    subsets.reverse()
    return subsets

def marginal_product_fos_learning(population, fitness: FitnessFunction):
    entropy_values = entropy_matrix(population, fitness)
    subsets = []
    for i in range(fitness.dimensionality):
        subsets.append([i])
    # shuffle subsets
    np.random.shuffle(subsets)
    n = len(population)
    while len(subsets) > 1:
        mdl_matrix = minimum_description_length_matrix(subsets, entropy_values, n)

        # find the maximum value in the mdl matrix
        max_value = np.max(mdl_matrix)
        if max_value <= 0:
            break
        # find the indices of the maximum value
        indices = np.where(mdl_matrix == max_value)
        # create a new subset
        new_subset = subsets[indices[0][0]] + subsets[indices[1][0]]
        # remove the two old subsets
        if indices[0][0] < indices[1][0]:
            subsets.pop(indices[1][0])
            subsets.pop(indices[0][0])
        else:
            subsets.pop(indices[0][0])
            subsets.pop(indices[1][0])
        # add the new subset
        subsets.append(new_subset)
    #print(subsets)
    return subsets

def entropy_matrix(population, fitness):
    p = np.zeros(fitness.dimensionality)
    for i in range(fitness.dimensionality):
        p[i] = np.sum([individual.genotype[i] for individual in population]) / len(population)
    entropy_values = np.zeros((fitness.dimensionality, fitness.dimensionality))
    for i in range(fitness.dimensionality):
        if p[i] == 0 or p[i] == 1:
            entropy_values[:,i] = 0
            entropy_values[i,:] = 0
            continue
        for j in range(i,fitness.dimensionality):
            if p[j] == 0 or p[j] == 1:
                entropy_values[i,j] = 0
                entropy_values[j,i] = 0
                continue
            if i == j:
                entropy_values[i,j] = -p[i] * np.log2(p[i]) - (1-p[i]) * np.log2(1-p[i])
                continue
            p_1_1 = p[i] * p[j]
            p_1_0 = p[i] * (1-p[j])
            p_0_1 = (1-p[i]) * p[j]
            p_0_0 = (1-p[i]) * (1-p[j])
            entropy_values[i,j] = -p_1_1 * np.log2(p_1_1) - p_1_0 * np.log2(p_1_0) - p_0_1 * np.log2(p_0_1) - p_0_0 * np.log2(p_0_0)
            entropy_values[j,i] = entropy_values[i,j]

    return entropy_values

def UPGMA (subset_a, subset_b, entropy_values):
    value = 0
    for X in subset_a:
        for Y in subset_b:
            value += entropy_values[X,X] + entropy_values[Y,Y] - entropy_values[X,Y]
    value /= len(subset_a) * len(subset_b)
    return value

def minimum_description_length_matrix(active_subsets, entropy_values, n):
    mdl_matrix = np.zeros((len(active_subsets),len(active_subsets)))
    for i in range(len(active_subsets)):
        for j in range(i+1,len(active_subsets)):
            entropy_term = n *( entropy_values[i,i] + entropy_values[j,j] - entropy_values[i,j])
            l_i = len(active_subsets[i])
            l_j = len(active_subsets[j])
            log_term = np.log2(n+1) * ((2**l_i - 1) + (2**l_j - 1) - (2**(l_i+l_j) - 1))
            mdl_matrix[i,j] = entropy_term + log_term

    return mdl_matrix
    
