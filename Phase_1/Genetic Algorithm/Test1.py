from GA1 import Genetic_Algorithm as Gen_Al
from Model1 import Barber_shop as BS
import matplotlib.pyplot as plt
import random
import numpy as np

n_customers = 80
n_barbers = 3
n_operations = 4
max_obj_var = 500

def generate_operations(n_customers, n_operations):
  operation_list = []
  for i in range(n_customers):
    op = [random.randint(0,1) for j in range(n_operations)]
    operation_list.append(op)

  return operation_list

def generate_customer_type(n_customers):
  std = int(0.7*n_customers)
  type_list = ['Standard' for i in range(std)]
  type_list.extend('Premium' for i in range(n_customers - std))
  random.shuffle(type_list)
  return type_list

operation_list = generate_operations(n_customers, n_operations)
customer_type = generate_customer_type(n_customers)

Algo = Gen_Al(n_customers, n_barbers, operation_list, customer_type, max_obj_var)
final_sch, final_obj, obj_list, objo_list = Algo.Run_Algorithm()

b1 = []
b2 = []
b3 = []

for i in range (0, n_customers):
  if final_sch[i] <= n_customers:
    b1.append(final_sch[i])

for i in range (n_customers, 2*n_customers):
  if final_sch[i] <= n_customers:
    b2.append(final_sch[i])

for i in range (2*n_customers, 3*n_customers):
  if final_sch[i] <= n_customers:
    b3.append(final_sch[i])

final_schedule = [b1, b2, b3]

print(final_schedule, final_obj)

plt.plot(objo_list)
plt.show()
plt.plot(obj_list)
plt.show()