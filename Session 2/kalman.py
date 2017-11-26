import numpy as np
import pylab

'''
The prediction funtion recives 5 arguments:

X: The state stimate of the previous step
P: The state covariance of the previous step
A: The transition matrix
Q: The process noise covariance marix.
B: The input effect matrix.
U: The control input.
'''
def prediction(X, P, A, Q, B, U):
    X = np.dot(A, X) + np.dot(B, U) 
    P = np.dot(A, np.dot(P, A.T)) + Q # (A - no transpose because it is a scalar)
    return X, P

def update(X, P, C, K, Z, R):
    MP = np.dot(C, X) # measurement prediction
    residual = Z - MP # the residual of the prediction
    MPC = np.dot(C, np.dot(P, C.T))  + R    # measurement prediction covariance ( C- no transpose because it is a scalar)
    K = np.dot(P, np.dot(C, np.inv(MPC))) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars) 
    X = X + np.dot(K, residual) # Updated State Estimate
    P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T))) # Updated State Covariance

    return (X,P,K) 

# Initialization of the values of the variable
A = np.matrix('1 1; 0 1') # The state is a constant
U = 0 # There is no control input
B = np.matrix('0 0') # There is no control input
C = np.matrix('1 0') # The noisy measurement is direct
G = np.matrix('0.5 1')
Q = 10e-5 # Process variance
R = 0.01 # given from the problem as a constant noise
X = np.zeros(iterations) # Initializing voltage estimate the array with the number of iterations 
P = np.zeros(iterations) # Initializing error estimate the array with the number of iterations
K = np.zeros(iterations) # gain or blending factor initialization
ISE = np.zeros(iterations) # Instant Square Error
MSE = np.zeros(iterations) # Mean Square Error
x = np.matrix('0.0 0.0')
noise = np.random.normal(0.0, 0.1, iterations) # for 50 measurements simulated
#Z = x + noise #just for the first time
#https://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
print(X.shape)