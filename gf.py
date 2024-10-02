import random

# Problem-specific variables
weights = [10, 20, 30, 40, 50]  # Weights of items
values = [60, 100, 120, 200, 240]  # Values of items
capacity = 100  # Maximum weight the knapsack can hold
population_size = 10
generations = 100
mutation_rate = 0.1

# Fitness function to evaluate the value of a knapsack solution
def fitness(chromosome):
    total_weight = sum([chromosome[i] * weights[i] for i in range(len(weights))])
    total_value = sum([chromosome[i] * values[i] for i in range(len(values))])

    if total_weight > capacity:
        return 0  # Invalid solution
    return total_value

# Initialize population (random solutions)
def initialize_population():
    return [[random.randint(0, 1) for _ in range(len(weights))] for _ in range(population_size)]

# Selection function using roulette wheel
def select(population):
    fitness_values = [fitness(ind) for ind in population]
    total_fitness = sum(fitness_values)
    selection_probs = [fit / total_fitness for fit in fitness_values]
    return population[random.choices(range(population_size), weights=selection_probs)[0]]

# Crossover function (Single-point crossover)
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation function
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip the bit
    return chromosome

# Main GA function
def genetic_algorithm():
    population = initialize_population()

    for generation in range(generations):
        new_population = []
        for _ in range(population_size // 2):  # We will create population_size new individuals
            parent1 = select(population)
            parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

    # Get the best solution
    best_solution = max(population, key=fitness)
    best_value = fitness(best_solution)

    return best_solution, best_value

# Run the Genetic Algorithm
best_solution, best_value = genetic_algorithm()
print(f"Best solution: {best_solution}")
print(f"Best value: {best_value}")
