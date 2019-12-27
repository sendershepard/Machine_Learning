"""
Sender Shepard
Genetic Algorithm to solve the Knapsack problem.
"""
import random
import operator


""" Knapsack problem using a Genetic Algorithm to find the solution. """

def generate_individual(size):
    """ This function randomly creates an binary list, representing an invidual
        or member of the population. """
    individual = []
    
    for i in range(size):
        individual.append(random.randint(0,1))

    return individual

def generate_population(population_size, member_size):
    """ This function will generate the population of size population_size and
        each member of the population will be a binary list of size n. """
    population = []

    for i in range(population_size):
        population.append(generate_individual(member_size))

    return population
        
def fitness_score(sack, individual, max_weight):
    """ This function evaluates profit and weight of the knapsack.
        The individual with the largest profit value is a specimen that is
        genetically closer to thrive than others, however, if the sack weight
        is exceeded, profit will become 0. """
    profit = sack[0] 
    weight = sack[1] 
    total_profit = 0
    total_weight = 0
    
    for i in range(len(individual)):
        if individual[i] == 1:
            total_profit += profit[i]
            total_weight += weight[i]
    
    if total_weight > max_weight:
        total_profit = 0

    return total_profit, total_weight

def population_performance(population, sack, max_weight):
    """ This function will provide each member of the population with their
        fitness value. """
    
    def getKey(item):
        """ Helper function: returns value to be sorted by--second element. """
        return item[1][0]
    
    pop_perf = ()

    for member in population: #trailing coma (,) defines tuples in pairs
        pop_perf += ((member, fitness_score(sack, member, max_weight)), )

    return sorted(pop_perf, key=getKey, reverse=True)

def population_selection(population, sack, max_weight):
    """ This function will select the most fit genomes and kill of the two
        weakest members of the population"""
    sorted_population = population_performance(population, sack, max_weight)
    new_gen = []
    
    for fit_member in range(len(sorted_population) - 2): #killing two weakest
        new_gen.append(sorted_population[fit_member][0])

    return new_gen

def member_mutation(member, MutationPct):
    """ This function will randomly mutate one gene of the genome to avoid
        converging on just one solution and thus do more exploration. """
    ran_spot = random.randint(0, int(len(member)) - 1)

    if MutationPct:
        if member[ran_spot] == 1:
            member[ran_spot] = 0
        else:
            member[ran_spot] = 1

def member_crossover(population):
    """ This function will take two members from the population and picks a
        random spot to crossover the genes to breed two new genomes. Crossover
        leads the population to converge on a good solution--exploitation. """
    gene1 = population[random.randint(0, int(len(population) - 1))]
    gene2 = population[random.randint(0, int(len(population) - 1))]
    split = random.randint(1, int(len(population[0]) - 1))
    new_gene1 = gene1[:split] + gene2[split:]
    new_gene2 = gene2[:split] + gene1[split:]

    return new_gene1, new_gene2
       
def genetic_algorithm(population, sack, NumIterations, MaxWeight):
    """ Genetic Algorithm that solves the Knapsack problem.
        Population is random collection members. The population performance
        is computed through the fitness function.
        It loops iteratively NumIterations times, performing:
        • A pair of parents are randomly selected to breed in new_generation.
        • A random spot for crossover is chosen and creates two new children.
        • The two weakest members of the population are killled -- constant size.
        • Based on MutationPct a mutate one gene -- this has a 50% chance.
        Furntion returns the best performing population after evolution. """
    for i in range(NumIterations):
        new_generation = member_crossover(population)
        for member in new_generation:
            population.append(member)
        population = population_selection(population, sack, MaxWeight)
        MutationPct = random.randint(0, 1)
        member_mutation(population[random.randint(0,PopulationSize-1)], MutationPct)
        
    return population_performance(population, sack, MaxWeight)

def sack_helper(chosen_mem):
    """ This is a helper function that returns the best member of the chosen
        population. """
    ObjList = []
    ch = chosen_mem[0]    
    for i in range(len(ch)):
        if ch[i] == 1:
            ObjList.append(i + 1)

    return ObjList

#Dynamic Programing, temp table of knapsack
def knap_sack_total(sack_weight, profit, weight, n):
    """ Fills the table used to determine the values that are possible in the
        Knapsack. """
    knap = [[0 for x in range(sack_weight + 1)] for x in range(n + 1)]
    #This is equivalent to int K[n+1][W+1];

    for i in range(n + 1):
        for w in range(sack_weight + 1):
            if (i == 0 or w == 0):
                knap[i][w] = 0
            elif weight[i - 1] <= w:
                knap[i][w] = max(
                    knap[i - 1][w],
                    knap[i - 1][w - weight[i - 1]] + profit[i - 1])
            else:
                knap[i][w] = knap[i - 1][w]

    return knap

def print_sack(knapsack, n):
    """ Prints the Knapsack table and returns the total"""
    for i in range(n+1):
        print(knapsack[i])

def total_profit(knapsack, items, weight):
    """ Returns the total value of the items that fit in the Knapsack. """
    return knapsack[items][weight]

def chosen_items(sack, items, weight):
    """ This function returns the chosen items of the knapsack of total value. """
    total = total_profit(sack, items, weight)
    chosen = []
    
    while total != 0:
        for i in range(items + 1):
            if total in sack[i]:
                chosen.append(i) 
                total = total - profit[i - 1] 
                break 
                
    return sorted(chosen)


#Program should end by reporting the best remaining member of the population
#at the end of the run along with its weight and value
if __name__ == '__main__':
    profit = [3,5,8,10] #[10,5,15,7,6,18,3]#[1,2,5,6]#
    weight = [45,40,50,90] #[2,3,5,7,1,4,1]#[2,3,4,5]#
    MaxWeight = 100 #15#8#
    mem_len = len(weight)
    PopulationSize = 3
    NumIterations = 30000
    
    print("*** Welcome to the Knapsack Problem ***")
    print("The Knapsack has", mem_len, "objects", "each with profit", profit,
          "\nEach with weight", weight, "the sack has max weight of", MaxWeight)
    gene_sack = [profit, weight]    
    population = generate_population(PopulationSize, mem_len)

    pop_perf = population_performance(population, gene_sack, MaxWeight)
    print("\nBefore Genetic Algorithm runs the original Population is::")
    original_pop = [print(pop_perf) for pop_perf in pop_perf]
    
    pop_perf = genetic_algorithm(population, gene_sack, NumIterations, MaxWeight)
    print("\nAfter Genetic Algorithm runs the chosen Population is:")
    chosen_pop = [print(pop_perf) for pop_perf in pop_perf]

    chosen_member = pop_perf[0]
    ObjectList = sack_helper(chosen_member)
    print("\nUsing A Genetic Algorithm ")
    print("Best Remaining Member:", chosen_member[0],
          "\nIn ObjectList format:", ObjectList,
          "\nWhose total profit is", chosen_member[1][0], 
          "and total sack weight is", chosen_member[1][1])

    """Using Dynamic Programming to find the perfect solution. """
    dyn_sack = knap_sack_total(MaxWeight, profit, weight, mem_len)
    #print_sack(dyn_sack, mem_len)
    dyn_chosen = chosen_items(dyn_sack, mem_len, MaxWeight)
    print("\nUsing Dynamic Programming to compare results ")
    print("These are the chosen items:", dyn_chosen,
          "\nWhose total profit is", total_profit(dyn_sack, mem_len, MaxWeight))
    
 
    input("Exit")        
