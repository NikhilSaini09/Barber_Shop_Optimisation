import numpy as np
import random
from Model2 import Barber_shop as BS
class Genetic_Algorithm:
    def __init__(self, n_customers, n_barbers, operation_list, customer_type, max_obj_var, discount, membership):
        self.barber_model = BS(n_customers, n_barbers, operation_list, customer_type, discount, membership)
        self.objlist = []
        self.objolist = []
        self.time_required = self.barber_model.time_required
        self.operation_list = operation_list
        self.customer_type = customer_type
        self.n_customers = n_customers
        self.n_barbers = n_barbers
        self.max_obj_var = max_obj_var
        self.barber_model.initialize()

    def Initial_Parent_Generator(self, n_customers):
        a = np.arange(start=1, stop=n_customers + 1, step=1) 
        initial_pool = 10
        Generation = []
        objective = []
        for i in range(initial_pool):
            n1 = 0
            n2 = 0
            while n2<=n1:
                n1 = round(random.uniform(0.1, 0.9), 1)
                n2 = round(random.uniform(0.1, 0.9), 1)

            split1 = int(n1 * len(a))
            split2 = int(n2 * len(a))

            np.random.seed(42)
            np.random.shuffle(a)
            p1 = list(a[:split1])
            p2 = list(a[split1:split2])
            p3 = list(a[split2:])

            for i in range(n_customers - split1):
                p1.append(n_customers + 1 + i)

            for i in range(n_customers - split2 + split1):
                p2.append(2*n_customers - split1 + 1 + i)

            for i in range(split2):
                p3.append(3*n_customers - split2 + 1 + i)

            parent = np.concatenate((p1, p2, p3))
            obj = self.barber_model.objective(parent, self.time_required)

            Generation.append(parent)
            objective.append(obj)

        #Sorting in decending order
        for ind in range(len(objective)):
            max_index = ind
            
            for j in range(ind + 1, len(objective)):
                # select the maximum element in every iteration
                if objective[j] > objective[max_index]:
                    max_index = j
            # swapping the elements to sort the array
            (objective[ind], objective[max_index]) = (objective[max_index], objective[ind])
            (Generation[ind], Generation[max_index]) = (Generation[max_index], Generation[ind])

        return Generation, objective

    def get_obj_variance(self, schedule):
        obj_list = []
        for i in range (50):
            op_times_random = []
            for operation in self.operation_list:
                operation_times = [round(random.normalvariate(10,2)), round(random.normalvariate(5,1)), 
                                round(random.normalvariate(15,3)), round(random.normalvariate(20,5))]
                time_j = np.dot(operation, operation_times)
                op_times_random.append(time_j)

            obj_list.append(self.barber_model.objective(schedule, op_times_random))
        
        return np.var(obj_list)

    def roulette_wheel_selection(self, Generation, objective): #parent selection
  
        # Computes the totallity of the population fitness
        sum = 0
        for i in range (0,10):
            sum+= objective[i]

        population_fitness = sum
    
        # Computes for each chromosome the probability
        chromosome_probabilities = []
        for i in range (0,10):
            chromosome_probabilities.append(objective[i]/population_fitness)
        
        # Selects one chromosome based on the computed probabilities
        number = []
        for i in range (10):
            number.append(i)

        i1 = np.random.choice(number, p=chromosome_probabilities)
        i2 = np.random.choice(number, p=chromosome_probabilities)

        while(i1 == i2):
            i2 = np.random.choice(number, p=chromosome_probabilities)

        parent1 = Generation[i1]
        parent2 = Generation[i2]
        objp1 = objective[i1]
        objp2 = objective[i2]

        return parent1, parent2, objp1, objp2

    def Offspring_Generator(self, parent1, parent2, parent_string, n_customers):
        while True:
            Offspring1 = [0 for i in range(3*n_customers)]
            Offspring2 = [0 for i in range(3*n_customers)]

            for i in range(len(parent_string)):
                if parent_string[i] == 1:
                    Offspring1[i] = parent1[i]
            
            for i in range(len(parent_string)):
                if parent_string[i] == 0:
                    for j in range(len(parent2)):
                        if parent2[j] in Offspring1:
                            continue
                        else:
                            Offspring1[i] = parent2[j]

            n1 = 0
            n2 = 0 
            rn = 0

            n1 = random.randint(0, 3*n_customers - 1)
            n2 = random.randint(0, 3*n_customers - 1)

            while(n1 == n2 or (Offspring1[n1] > n_customers and Offspring1[n2] > n_customers)):
                n1 = random.randint(0, 3*n_customers - 1)
                n2 = random.randint(0, 3*n_customers - 1)

            rn = random.randint(0, 100)

            if (rn/100) > 0.8:
                temp = Offspring1[n1]
                Offspring1[n1] = Offspring1[n2]
                Offspring1[n2] = temp

            obj1 = self.barber_model.objective(Offspring1, self.time_required)

            offspring_var = self.get_obj_variance(Offspring1)

            if offspring_var <= self.max_obj_var:
                return Offspring1, obj1

    def Run_Algorithm(self):
        Generation, objective = self.Initial_Parent_Generator(self.n_customers)

        for i in range(30*self.n_customers):

            parent1, parent2, objp1, objp2 = self.roulette_wheel_selection(Generation, objective)

            parent_string = [random.randint(0, 1) for i in range(3*self.n_customers)]

            Offspring1, objo1 = self.Offspring_Generator(parent1, parent2, parent_string, self.n_customers)

            for i in range(0, 9):
                if objo1 > objective[0]:
                    for j in range(9, 0):
                        objective[j] = objective[j-1]
                        Generation[j] = Generation[j-1]

                    objective[0] = objo1
                    Generation[0] = Offspring1

                elif objo1 < objective[9]:
                    break;

                elif objo1 < objective[i] and objo1 > objective[i+1]:
                    for j in range(9, i+1):
                        objective[j] = objective[j-1]
                        Generation[j] = Generation[j-1]

                    objective[i+1] = objo1
                    Generation[i+1] = Offspring1

            self.objlist.append(objective[0])
            self.objolist.append(objo1)
                        
        return Generation[0], objective[0], self.objlist, self.objolist