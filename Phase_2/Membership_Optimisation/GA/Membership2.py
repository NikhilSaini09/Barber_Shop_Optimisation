from GA2 import Genetic_Algorithm as Gen_Al
from Model2 import Barber_shop as BS
import matplotlib.pyplot as plt
import random
import numpy as np
import math

n_barbers = 3
n_operations = 4
max_obj_var = 500
n_customers = 85
discount = 0.22

def generate_operations(n_customers, n_operations):
    operation_list = []
    random.seed(42)
    for i in range(n_customers):
      op = []
      for j in range(n_operations):
        op.append(random.randint(0,1))
      operation_list.append(op)

    return operation_list

def generate_customer_type(n_customers, n_Premium_customers):
  std = int((1-n_Premium_customers)*n_customers)
  type_list = ['Standard' for i in range(std)]
  for i in range(n_customers - std):
    type_list.append('Premium')
  random.shuffle(type_list)
  return type_list

def get_n_Premium_customers(membership):
  n_Premium_customers = (0.3)**(membership/200)
  return n_Premium_customers

def get_optimal_membership():
  membership_list = [100,120,140,160,180,200,220,240,260,280,300]
  obj_list_m = []
  for membership in membership_list:
    n_Premium_customers = get_n_Premium_customers(membership)

    operation_list = generate_operations(n_customers, n_operations)

    customer_type = generate_customer_type(n_customers, n_Premium_customers)

    Algo = Gen_Al(n_customers, n_barbers, operation_list, customer_type, max_obj_var, discount, membership)
    _, final_obj, _, _ = Algo.Run_Algorithm()

    obj_list_m.append(final_obj)

  return membership_list, obj_list_m

membership_list, obj_list_m = get_optimal_membership()

plt.plot(membership_list, obj_list_m)    
plt.show()