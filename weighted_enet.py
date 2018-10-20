import cvxpy 
import itertools
import numpy as np
import time

def make_prob(m,n):
  np.random.seed(1)
  A = np.random.randn(m,n)
  gamma = cvxpy.Parameter(nonneg=True)
  alpha = cvxpy.Parameter(nonneg=True)

  x = cvxpy.Variable(n)
  weights = np.linspace(0.1,2,num=m)
  b = weights*(A.dot(np.random.randn(n)))
  error = cvxpy.sum_squares(weights * (A*x - b))
  # enet
  obj = cvxpy.Minimize(error + gamma*cvxpy.norm(x, 1) + alpha*cvxpy.norm(x,2))
  #lasso obj = cvxpy.Minimize(error + gamma*cvxpy.norm(x, 1))
  prob = cvxpy.Problem(obj)
  return prob,error,gamma,alpha,x

def solve(prob, error, gamma, alpha, x):
  ''' Construct a trade-off curve of ||Ax-b||^2 vs. regularizers ||x||_1 and ||x||_2 '''
  sq_penalty = []
  l1_penalty = []
  l2_penalty = []
  x_values = []
  gamma_vals = np.logspace(-4, 6, 10)
  alpha_vals = np.logspace(-4, 6, 10)
  start = time.time()
  for val in itertools.product(gamma_vals,alpha_vals):
    gamma.value = val[0]
    alpha.value = val[1]
    obj = prob.solve(solver=cvxpy.SCS, eps=1.0e-3, use_indirect=True, verbose=False)
    sq_penalty.append(error.value)
    l1_penalty.append(cvxpy.norm(x, 1).value)
    l2_penalty.append(cvxpy.norm(x, 2).value)
    x_values.append(x.value)
    if obj and prob.status=='optimal':
      print("solve time {:.2f} gamma {:.3f} alpha {:.3f} objective {:.3f}".format(prob.solver_stats.solve_time, val[0],val[1], obj))
    else:
      print('prob {} try verbose=True gamma {:.3f} alpha {:.3f}'.format(prob.status,val[0],val[1]))
  end = time.time()
  print("time elapsed=", end - start)

def main():
  m = 7*60*150 
  n = 50 
  prob, error, gamma, alpha,x = make_prob(m,n)
  solve(prob,error, gamma, alpha, x)

main()
