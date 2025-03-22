import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize


# B-spline we care about
def g(x):
  tf=(abs((x))<=1).astype(int)
  return (1-abs((x)))*tf


#Define Gabor functions G(g,A,B)
def bspline_gabor(g):
    """
    Parameters:
    g (function): The function g to be applied.

    Returns:
    a function that is time-frequency modulated
    """
    return lambda x, a, b: g(x-a)*np.exp(2 * np.pi * 1j * b*x)


# Example usage
# Define a subset of a lattice axb
a = np.linspace(0,1,3)
b = np.array([0, 1, 3])
x = np.linspace(-1, 1, 1011)
phi = bspline_gabor(g)
plt.figure(figsize=(16,10))
plt.title("B-spline Gabor system")
for ai in a:
    for bj in b:
        plt.plot(x, np.real(phi(x, ai,bj)), label=f'Real part for a={ai}, b={bj}')
        plt.plot(x, np.imag(phi(x, ai, bj)), linestyle='--', label=f'Imaginary part for a={ai}, b={bj}')
plt.show()


def run_learned_gabor(n_trials=10, k=6, method='L-BFGS-B', options={}):


  def square_function_distance(f, g):
    return np.sum( (f(x) - g(x))**2 )

  def loss(beta):
    beta = beta.reshape( (k,3) )
    g = sum_of_learned_functions(phi, beta)
    return square_function_distance(target_function, g)

  def sum_of_learned_functions(phi, beta):
    beta = beta.reshape( (beta.size//3,3) )
    def f(x):
        total = np.zeros(shape=x.shape)
        for i, b in enumerate(beta):
            total += np.real(phi(x,b[1],b[2])) * b[0]
        return total
    return f

  losses=np.zeros([n_trials])

  best_loss = float('inf')
  beta_hat = np.zeros( shape=(k,3) )
  for iteration in range(n_trials):

    # initial guess
    beta_zero = np.random.normal(0, 0.01, size=(k,3))
    beta_zero[:, 1] = np.linspace(0, 1, k)
    beta_zero[:, 2] = np.ones(shape=k) * 0.2
    beta_zero = beta_zero.reshape( (3*k,) )


    print('fitting attempt', iteration)

    # minimize the loss
    best = minimize(loss, x0=beta_zero, method=method, options=options) #constrain a between [-2 2], change the tolerance so the loss has to be lower than 1e-10

    candidate_beta = best.x.reshape( (k,3) )
    candidate_loss = loss(candidate_beta)
    losses[iteration]=candidate_loss
    if candidate_loss < best_loss:
        best_loss = candidate_loss
        beta_hat = candidate_beta

  print('beta:', beta_hat)
  print("best loss:", loss(beta_hat))
  g_hat=sum_of_learned_functions(phi, beta_hat)(x)
  a = beta_hat[:, 1]
  b = beta_hat[:, 2]

  return losses, g_hat, a, b



# Gabor function phi(x,a,b)
phi = bspline_gabor(g)
#aa=.5
#bb=1
#phi_L = make this phi_L(x,i,j) = phi(x, i*aa, j*bb) where alpha and beta are fixed. need aa*bb<1
# in this case you need to do integer programming

# Target function to be approximated
def target_function(x):
    return x*(x-1) * np.sin(13*x) + (1-x) * np.cos(23*x)*(1-x) + 1*np.exp(-(x-.5)*(x-.5)*1000)


n_trials=10
x = np.linspace(start=-1, stop=1, num=1001)
y = target_function(x)
# if you want to experiment with adding noise to the target function, add this: + np.random.normal(0, 0.1, size=x.shape)


#Minimize the loss with 3k unknowns
k=4
losses_gabor, g_g, a, b=run_learned_gabor(k=4, n_trials=n_trials, method='SLSQP', options={'maxiter': 1000000})

plt.plot(x,g_g,  label="k="+str(k))
plt.plot(x, y,'--', label="target",linewidth=6, alpha=.2)
plt.legend()
plt.show()

# We are plotting the basis vectors that got chosen by the optimization routine
plt.figure(figsize=(16,10))
plt.title("B-spline Gabor system")
for k in range(len(a)):
  ai=a[k]
  bj=b[k]
  plt.plot(x, np.real(phi(x, ai,bj)))
  plt.plot(x, np.imag(phi(x, ai, bj)), linestyle='--')
plt.legend()
plt.show()
