import numpy as np
import matplotlib.pyplot as plt

m = 10  
L = 2   
dw = 20 
d = 5   
gamma = 0.1  

X = np.random.rand(d, m)  
Y = np.random.rand(m, 1)  

def initialize_weights(m, dw, d, L):
  W1 = np.random.normal(0, 1/m, size=(dw, m))
  WL = np.random.normal(0, 1/dw, size=(d, dw))
  return W1, WL

W1, WL = initialize_weights(m, dw, d, L)

sigma_max_X = np.linalg.norm(X, ord=2)  # Biggest Singularity Set Up
sigma_min_X = np.linalg.norm(X, ord=-2) # Smallest Singulatiry Set Up

eta = m / (L * (sigma_max_X**2))  
lambda_ = gamma * (sigma_min_X**2) * np.sqrt(m/d) 

learning_rate = eta * 0.0001  

def loss(W1, WL, X, Y, lambda_):
  output = WL @ W1 @ Y
  reconstruction_error = 0.5 * np.linalg.norm(output - X @ Y)**2 
  regularization_term = (lambda_/2) * (np.linalg.norm(W1)**2 + np.linalg.norm(WL)**2)
  return reconstruction_error + regularization_term

# Clipping Gradient Descent
def gradient_descent_step(W1, WL, X, Y, learning_rate, lambda_, clip_value=1.0):
    output = WL @ W1 @ Y
    grad_W1 = WL.T @ (output - X @ Y) @ Y.T + lambda_ * W1
    grad_WL = (output - X @ Y) @ (W1 @ Y).T + lambda_ * WL

    grad_W1 = np.clip(grad_W1, -clip_value, clip_value)
    grad_WL = np.clip(grad_WL, -clip_value, clip_value)

    W1 = W1 - learning_rate * grad_W1
    WL = WL - learning_rate * grad_WL

    return W1, WL

num_iterations = 1000000
losses = []  
for i in range(num_iterations):
    W1, WL = gradient_descent_step(W1, WL, X, Y, learning_rate, lambda_)
    current_loss = loss(W1, WL, X, Y, lambda_)
    losses.append(current_loss)
    if i % 100 == 0:
        print(f"Iteration {i+1}, Loss: {current_loss}")

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss During Training")
plt.show()

print("Finish Training")
