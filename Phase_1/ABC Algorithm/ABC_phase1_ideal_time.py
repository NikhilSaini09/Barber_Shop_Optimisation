import numpy as np
import random
import matplotlib.pyplot as plt
import itertools

# Shop Parameters
n_customers = 80         # no. of customers at a particular day
n_barbers = 3            # no. of barbers available in shop
n_operations = 4         # no. of operations can be performed

# Initializing shop parameters and customers data
class Barber_shop:
    def __init__(self, n_customers, n_barbers, operation_list, customer_type, schedule, discount_fraction):   # for initialization for class barber shop
        self.n_customers = n_customers
        self.n_barbers = n_barbers
        self.operation_list = operation_list    # a list of lists containing operations of each customer as a list
        self.customer_type = customer_type    # a list for customer type (premium or standard)
        self.schedule = schedule   # a list of lists where each list represents schedule for each barber 
        self.operation_costs = [100, 50, 150, 200]
        self.weight = []   # a list of weight is assigned to each customer depending on the type of customer
        self.time_required = []   # a list of actual time that was required to complete task for a provided customer
        self.avg_time_req = []   # avg time till when operations should start
        self.price = []   # a list of profit we could gain from each customer without discount
        self.membership = 200    # fee to become a premium member
        self.discount_fraction = discount_fraction  # fraction of code given as discount if expected time runs out

    def get_weight(self):    # to find weights
        for ctype in self.customer_type:
            if ctype == 'Standard':
                self.weight.append(1)   # for standard customer weight is assigned as 1
            else:
                self.weight.append(2)   # for standard customer weight is assigned as 2 for better consideration

    def get_operation_price(self):    # to find profit from each customer
        for operation in self.operation_list:   # a list in operation list consists of 4 values as random from 0 to 1 (0 for yes 1 for no) 
            price_j = np.dot(operation, self.operation_costs)
            self.price.append(price_j)

    def normalvariate(self, mu, sigma):   # defining normal variate in class
        return random.normalvariate(mu, sigma)

    def get_operation_time(self):
        for operation in self.operation_list:
            # Generating random numbers as a variate for the time required for a particular set of operations
            operation_times = [round(self.normalvariate(10,2)), round(self.normalvariate(5,1)), 
                                round(self.normalvariate(15,3)), round(self.normalvariate(20,5))]
            avg_op_times = [10, 5, 15, 20]
            time_j = np.dot(operation, operation_times)   # to find total time required for a customer
            self.time_required.append(time_j)
            time_j = np.dot(operation, avg_op_times)   # avg time that ideal case gives
            self.avg_time_req.append(time_j)

    def initialize(self):  # initialization of these three functions below
        self.get_weight()
        self.get_operation_price()
        self.get_operation_time()

    # to find which barber(n) is assigned to a particular customer(j) in a schedule (loop for j is in objective function below)
    def get_barber(self, j, schedule):
        for n in range(len(schedule)):
            for m in range(len(schedule[n])):
                if schedule[n][m] == j:
                    return n

    # finding a start time for a customer(j) given a schedule by taking time required for previous customers in the schedule
    def get_start_time(self, j, schedule, time_required):
        i = self.get_barber(j, schedule)  # finding barber for j customer
        s_i = schedule[i]  # schedule for that barber(i)
        start_j = 0    # start time for j customer
        avg_start = 0   # ideal start time
        for k in s_i:
            if k == j:
                break
            start_j += time_required[k]
            avg_start += self.avg_time_req[k]
        return start_j, avg_start

    # for finding if we have to give discount of 20% to a particular customer or not
    def get_discount(self, j, schedule, time_required):
        start_time, avg_start_time = self.get_start_time(j, schedule, time_required)
        return self.discount_fraction if start_time > avg_start_time else 0

    # calculating total profit
    def objective(self, schedule, time_required):
        net_profit = 0
        for j in range(self.n_customers):
            wj = self.weight[j]
            pj = self.get_discount(j, schedule, time_required)   # either 0.2 or 0
            cj = self.price[j]
            net_profit += (1 - wj * pj) * cj + (wj - 1) * self.membership   # profit for each customer
        return net_profit

# Generating random operations for each customer
def generate_operations(n_customers, n_operations):
    operation_list = []              # list containing which operations a particular customer wants
    for _ in range(n_customers):
        op = [0 for _ in range(n_operations)]
        while op == [0 for _ in range(n_operations)]:        # to ensure each customer want at least one operation
            op = [random.randint(0,1) for _ in range(n_operations)]      # randomly assign operations 
        operation_list.append(op)
    return operation_list

# randomly generating 70% Standard and 30% as premium customers
def generate_customer_type(n_customers):
    std = int(0.7*n_customers)     # considering 70% are normal customers at 200 member fees
    type_list = ['Standard' for _ in range(std)]
    type_list.extend('Premium' for _ in range(n_customers - std))
    random.shuffle(type_list)      # to make list i.e. customer random
    return type_list

# Define ABC parameters
max_iterations = 1000    # no of iterations for the algorithm will run
num_employed_bees = n_customers   # no of all three type of bees are equal and the value is equal to customers
num_onlooker_bees = n_customers   # each bee representing a solution in this case each represents a schedule
n_loops = 80   # no of loops for which onlooker bees will work
num_scout_bees = n_customers
trial = [0 for _ in range(num_employed_bees)]   # trial number for each bee 
max_trials = 50   # max no of trial allowed after which scout bees will work

# Define function to generate a new solution by swapping two customer position two times
def generate_new_solution(bee, random_bee):
    new_solution = [sublist[:] for sublist in bee]
    # for a loop to run twice
    for _ in range(2):
        # to find a random position at random sublist 
        sublist_index = random.randrange(len(new_solution))
        position = random.randrange(len(new_solution[sublist_index]))
        # finding the value at that random position in the bee that is randomly selected
        new_customer_no = random_bee[sublist_index][position]
        # finding the same value in the original bee 
        n, m = 0, 0
        for i in range(len(new_solution)):
            for j in range(len(new_solution[i])):
                if new_solution[i][j] == new_customer_no:
                    n, m = i, j
        # interchanging the the values at these positions
        new_solution[sublist_index][position], new_solution[n][m] = new_solution[n][m], new_solution[sublist_index][position]
    return new_solution

# Define function to generate a new solution by randomly assigning integer values to the tasks and resources
def generate_random_solution():
    solution = [[] for _ in range(n_barbers)]  # generating empty list of lists for a solution (schedule)
    orders = list(range(n_customers))  # filling another list with all the numbers (customers)
    random.shuffle(orders)   # shuffling them for random positions
    for i, num in enumerate(orders):  
        solution[i % n_barbers].append(num)   # generating lists for each barber
    return solution

# Calling functions
operation_list = generate_operations(n_customers, n_operations)
customer_type = generate_customer_type(n_customers)
random_sol = generate_random_solution()
# Calling Class
shop_init = Barber_shop(n_customers, n_barbers, operation_list, customer_type, random_sol, 0.2)  # initialise problem parameters
shop_init.initialize()       # get weight, operation price and operation time

# Initialize the employed bees, their fitness values, and the best solution found so far
employed_bees = [generate_random_solution() for _ in range(num_employed_bees)]  # generating list of lists of lists for employed bees(n random solution for n bees)
employed_bees_obj = [shop_init.objective(bee, shop_init.time_required) for bee in employed_bees]  # list of profit for all these n schedules
# Fitness function for the objective of the ABC for a particular schedule (bee)
def fitness(bee):
    return  1 / (1 + shop_init.objective(bee, shop_init.time_required))  # 1/(1 + obj) for obj > 0
# profit function for a given fitness value assigned to a schedule
def output(fitness):
    return (1 - fitness) / fitness
employed_bees_fitness = [fitness(bee) for bee in employed_bees]   # a list of fitness value for all the employed bees
best_fitness = min(employed_bees_fitness)   # min fitness existing which corresponds to max profit
# function for finding the schedule in employed bees for a given fitness 
def best_sol(best_fitness):
    for i in range(len(employed_bees_fitness)):
        if employed_bees_fitness[i] == best_fitness:
            return employed_bees[i]
best_solution = best_sol(best_fitness)

# parameters for plot
Objective = [] 
Iteration_no = []

# Main loop of ABC algorithm
for iteration in range(max_iterations):

    # Employed bees phase
    for i in range(num_employed_bees):
        random_bee = random.choice(employed_bees)
        new_solution = generate_new_solution(employed_bees[i], random_bee)
        new_fitness = fitness(new_solution)
        if new_fitness < employed_bees_fitness[i]:  # to reduce fitness
            employed_bees[i] = new_solution
            employed_bees_fitness[i] = new_fitness
            trial[i] = 0   # resetting to 0 if an update found
        else:
            trial[i] += 1  # otherwise increase trial by 1 

    # Onlooker bees phase
    onlooker_bees = [sublist[:] for sublist in employed_bees]
    onlooker_bees_fitness = employed_bees_fitness[:]
    max_fitness = max(onlooker_bees_fitness)
    for _, i in itertools.product(range(n_loops), range(num_onlooker_bees)):  # looping n times the range(num_onlooker_bees)
        probability = 0.9*(employed_bees_fitness[i] / max_fitness) + 0.1   # probability function of onlooker bee
        rand_probability = random.random()
        if rand_probability < probability:   # condition of probability for updating an bee
            random_bee = random.choice(onlooker_bees)
            new_solution = generate_new_solution(onlooker_bees[i], random_bee)
            new_fitness = fitness(new_solution)
            if new_fitness < employed_bees_fitness[i]:
                onlooker_bees[i] = new_solution
                onlooker_bees_fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1
        else:
            trial[i] += 1

    # Scout bees phase
    scout_bees = []
    scout_bees_fitness = []
    for i in range(num_scout_bees):
        if trial[i] >= max_trials:  # condition for scout bee activation
            scout_bees.append(generate_random_solution())
            scout_bees_fitness.append(fitness(scout_bees[i]))
            trial[i] = 0
        else:      # otherwise same bee
            scout_bees.append(onlooker_bees[i])
            scout_bees_fitness.append(onlooker_bees_fitness[i])

    # Update employed bees and best solution
    for i in range(num_employed_bees):
        if onlooker_bees_fitness[i] < employed_bees_fitness[i]:
            employed_bees[i] = [sublist[:] for sublist in onlooker_bees[i]]
            employed_bees_fitness[i] = onlooker_bees_fitness[i]
        if scout_bees_fitness[i] < employed_bees_fitness[i]:
            employed_bees[i] = [sublist[:] for sublist in scout_bees[i]]
            employed_bees_fitness[i] = scout_bees_fitness[i]
        if employed_bees_fitness[i] < best_fitness:
            best_solution = [sublist[:] for sublist in employed_bees[i]]
            best_fitness = employed_bees_fitness[i]

    Objective.append(shop_init.objective(best_solution, shop_init.time_required))
    Iteration_no.append(iteration + 1)
    # Print progress
    print("Iteration:", iteration+1, "Best fitness:", best_fitness, "Max profit:", shop_init.objective(best_solution, shop_init.time_required))

Best_output = shop_init.objective(best_solution, shop_init.time_required)  # max profit
print(Best_output)
# Plot Profit vs Iteration
plt.plot(Iteration_no, Objective)
plt.show()

def var(schedule):
    obj_list = []
    for _ in range(50):
        op_times_random = []
        for operation in operation_list:
            operation_times = [round(random.normalvariate(10,2)), round(random.normalvariate(5,1)), 
                                round(random.normalvariate(15,3)), round(random.normalvariate(20,5))]
            time_j = np.dot(operation, operation_times)
            op_times_random.append(time_j)
        obj_list.append(shop_init.objective(schedule, op_times_random))
    return obj_list

variance = var(best_solution)
variance_in_best_solution = np.var(variance)
Max_deviation = max(variance) - min(variance)
print(Max_deviation)                   
plt.plot(variance)
plt.show()