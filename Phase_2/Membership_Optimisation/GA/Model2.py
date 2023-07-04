import numpy as np
import random
class Barber_shop:
  def __init__(self, n_customers, n_barbers, operation_list, customer_type, discount, membership):
    self.n_customers = n_customers
    self.n_barbers = n_barbers
    self.operation_list = operation_list
    self.customer_type = customer_type
    self.operation_costs = [100, 50, 150, 200]
    self.weight = []
    self.time_required = []
    self.avg_time_req = []
    self.price = []
    self.discount = 0.22
    self.membership = membership

  def get_weight(self):
    for ctype in self.customer_type:
      if ctype == 'Standard':
        self.weight.append(1)
      elif ctype == 'Premium':
        self.weight.append(2)

  def get_operation_price(self):
    for operation in self.operation_list:
      price_j = np.dot(operation, self.operation_costs)
      self.price.append(price_j)

  def get_operation_time(self):
    for operation in self.operation_list:
      operation_times = [round(random.normalvariate(10,2)), round(random.normalvariate(5,1)), 
                         round(random.normalvariate(15,3)), round(random.normalvariate(20,5))]
      avg_op_times = [10,5,15,20]
      time_j = np.dot(operation, operation_times)
      self.time_required.append(time_j)
      time_j = np.dot(operation, avg_op_times)
      self.avg_time_req.append(time_j)

  def initialize(self):
    self.get_weight()
    self.get_operation_price()
    self.get_operation_time()

  def get_discount(self, j, schedule, time_required):
    start_time, avg_start_time = self.get_start_time(j, schedule, time_required)
    if start_time > avg_start_time:
      return self.discount
    return 0

  def get_barber(self, j, schedule):
    
    for k in range(300): # parent/Offspring is a 1D array
        if schedule[k] == j + 1:
          return int(k/self.n_customers) 

  def get_start_time(self, j, schedule, time_required):
    i = self.get_barber(j, schedule)
    split1 = int(len(schedule)/3)
    split2 = int(2 * len(schedule)/3)
    if i == 0:
        s_i = list(schedule[:split1]) 
    elif i == 1:
        s_i = list(schedule[split1:split2])
    elif i == 2:
        s_i = list(schedule[split2:])

    start_j = 0
    avg_start = 0

    for k in s_i:
        if k == j + 1:
          break
        if k<=self.n_customers:
          start_j+=self.time_required[k-1]
          avg_start+=self.avg_time_req[k-1]

    return start_j, avg_start

  def objective(self, schedule, time_required):
    net_profit = 0
    for j in range(1, self.n_customers + 1):
      wj = self.weight[j-1]
      pj = self.get_discount(j-1, schedule, time_required)
      cj = self.price[j-1]

      net_profit += (1-wj*pj)*cj + (wj-1)*self.membership

    return net_profit