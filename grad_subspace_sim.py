import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(d, n, s, kappa):
    Z = np.random.rand(s, n)  
    U, S, V = np.linalg.svd(Z)  
    S = np.random.uniform(1/kappa, 1, size=min(s,n)) 
    Sigma = np.zeros((s, n)) 
    Sigma[:len(S), :len(S)] = np.diag(S)
    Z = U @ Sigma @ V  

    R = np.random.rand(d, s)
    Q, _ = np.linalg.qr(R)

    X = Q @ Z
    return X, Q


# This is the simplified gradient formulas
def simulate_training(X, Y, L, num_iterations, eta, lambda_val):
  W = np.random.rand(X.shape[0], Y.shape[0])
  errors = []
  for t in range(num_iterations):
      # Simulate gradient descent step
      grad = -2 * (X - W @ Y) @ Y.T + 2 * lambda_val * W # Simplified gradient
      W -= eta * grad

      # Reconstruction error
      reconstruction_error = np.linalg.norm(W @ Y - X, 'fro') / np.linalg.norm(X, 'fro')

      errors.append(reconstruction_error)
  return errors, W

def off_subspace_error(W, Q):
  P_perp = np.eye(Q.shape[0]) - Q @ Q.T  
  off_subspace_error = np.linalg.norm(W @ P_perp, ord=2) # Operator norm
  return off_subspace_error
  
d = 100       
n = 50        
num_iterations = 10000
eta = 0.1    
lambda_val = 0.001 
kappa = 10
subspace_dimensions = [2, 4, 8, 16, 32]
network_depths = [2, 4, 8, 16, 32]

# Simulation of errors
reconstruction_errors = {}
off_subspace_errors = {}

for s in subspace_dimensions:
  X, Q = generate_synthetic_data(d, n, s, kappa) 
  Y = X + 0.01 * np.random.randn(*X.shape) 
  reconstruction_errors[s], W = simulate_training(X, Y, 2, num_iterations, eta, lambda_val)

  off_subspace_errors[s] = off_subspace_error(W, Q) 

for L in network_depths:
  X, Q = generate_synthetic_data(d, n, 8, kappa)  
  Y = X + 0.01 * np.random.randn(*X.shape) 
  _, W = simulate_training(X, Y, L, num_iterations, eta, lambda_val) 

  off_subspace_errors[L] = off_subspace_error(W, Q)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for s, errors in reconstruction_errors.items():
    plt.plot(range(num_iterations), errors, label=f's = {s}')
plt.xlabel('Iteration')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs. Iteration')
plt.legend()

plt.subplot(1, 2, 2)
for L, error in off_subspace_errors.items():
  if isinstance(L,int):
    plt.plot(L, error, marker='o', label=f'L = {L}')
  else:
    plt.scatter(L,error,marker='o',label=f"s = {L}")
plt.xlabel('L / s')
plt.ylabel('Off-Subspace Error')
plt.title('Off-Subspace Error')
plt.legend()

plt.tight_layout()
plt.show()
