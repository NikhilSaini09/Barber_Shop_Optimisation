from GA1 import Genetic_Algorithm as Gen_Al
import matplotlib.pyplot as plt
import random
import math

n_barbers = 3
n_operations = 4
max_obj_var = 500

def generate_operations(n_customers, n_operations):
    operation_list = []
    for i in range(n_customers):
        op = [random.randint(0,1) for _ in range(n_operations)]
        operation_list.append(op)

    return operation_list

def generate_customer_type(n_customers):
    std = int(0.66*n_customers)
    type_list = ['Standard' for _ in range(std)]
    type_list.extend('Premium' for _ in range(n_customers - std))
    random.shuffle(type_list)
    return type_list

def sigmoid(x):
    return 1/(1+math.exp(-x))

def get_n_customers(discount):
    x = (discount - 18)/4
    return int(49.5*(1+sigmoid(x)))

def get_optimal_discount():
    discount_list = [0.2,0.22,0.24,0.26,0.28,0.30,0.35,0.40]
    obj_list_d = []
    for discount in discount_list:
        n_customers = get_n_customers(discount)
        operation_list = generate_operations(n_customers, n_operations)
        customer_type = generate_customer_type(n_customers)
        avg_final_obj = 0
        for i in range(10):
            Algo = Gen_Al(n_customers, n_barbers, operation_list, customer_type, max_obj_var, discount)
            _, final_obj, _, _ = Algo.Run_Algorithm()
            avg_final_obj+=final_obj

        avg_final_obj = avg_final_obj/10
        obj_list_d.append(avg_final_obj)

    return obj_list_d

obj_list_d = get_optimal_discount()
plt.plot(obj_list_d)    
plt.show()