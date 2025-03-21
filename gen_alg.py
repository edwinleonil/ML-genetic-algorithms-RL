import random

# GA Parameters
# The size of the population
POPULATION_SIZE = 10
# THe length of the gene
GENE_LENGTH = 6
# The mutation rate
MUTATION_RATE = 0.1
# The number of generations
GENERATIONS = 50


# Fitness function (maximise sum of genes)
def fitness(individual):
    return sum(individual)

# Create an individual (random list of numbers between 0 and 10)


def create_individual():
    return [random.randint(0, 10) for _ in range(GENE_LENGTH)]

# Generate initial population


def create_population(size):
    return [create_individual() for _ in range(size)]

# Selection (Tournament Selection)


def select(population):
    # Randomly select 3 individuals and return the best
    return max(random.sample(population, 3), key=fitness)

# Crossover (Single Point Crossover)


def crossover(parent1, parent2):
    # Randomly select a crossover point and swap the genes
    point = random.randint(1, GENE_LENGTH - 1)
    # Return the new children
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Mutation (Randomly change one gene)


def mutate(individual):
    # Randomly select a gene and change it
    if random.random() < MUTATION_RATE:
        # Randomly select a gene and change it
        index = random.randint(0, GENE_LENGTH - 1)
        # Change the gene to a random value between 0 and 10
        individual[index] = random.randint(0, 10)
    return individual

# Main GA loop


def genetic_algorithm():
    # Generate initial population
    population = create_population(POPULATION_SIZE)
    # Evolve the population
    for generation in range(GENERATIONS):
        # Sort the population by fitness
        population = sorted(population, key=fitness, reverse=True)
        new_population = [population[0]]  # Keep the best solution (Elitism)

        # Create the next generation
        while len(new_population) < POPULATION_SIZE:
            # Select the parents
            parent1, parent2 = select(population), select(population)
            # Crossover and mutate the children
            child1, child2 = crossover(parent1, parent2)
            # Add the mutated children to the new population
            new_population.extend([mutate(child1), mutate(child2)])
        # Replace the old population with the new population
        population = new_population[:POPULATION_SIZE]
        # Print the best individual in the current generation
        best_individual = max(population, key=fitness)
        print(
            f"Generation {generation + 1}: Best = {best_individual}, Fitness = {fitness(best_individual)}")

    print("\nFinal Best Solution:", best_individual)


# Run the Genetic Algorithm
genetic_algorithm()
