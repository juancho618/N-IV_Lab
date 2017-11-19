'''
16/11/2017
by: Juan Jose Soriano Escobar

This code is an implemetation of the Kalman Filter for the first lab session of the course Navigation and Intelligent Vehicles
'''
import numpy as np
import pylab

# The idea is to recursively create a prediction givin the state vector and the prediction as parameters

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
    P = np.dot(A, np.dot(P, A)) + Q # (A - no transpose because it is a scalar)
    return X, P

def update(X, P, C, K, Z, R):
    MP = np.dot(C, X) # measurement prediction
    residual = Z - MP # the residual of the prediction
    MPC = np.dot(C, np.dot(P, C))  + R    # measurement prediction covariance ( C- no transpose because it is a scalar)
    K = np.dot(P, np.dot(C, MPC**-1)) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars) 
    X = X + np.dot(K, residual) # Updated State Estimate
    P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T))) # Updated State Covariance

    return (X,P,K) 
    

# Application with the voltage!
iterations = 50

# Initialization of the values of the variable
A = 1 # The state is a constant
U = 0 # There is no control input
B = 0 # There is no control input
C = 1 # The noisy measurement is direct
Q = 10e-5 # Process variance
R = 0.01 # given from the problem as a constant noise
X = np.zeros(iterations) # Initializing voltage estimate the array with the number of iterations 
P = np.zeros(iterations) # Initializing error estimate the array with the number of iterations
K = np.zeros(iterations) # gain or blending factor initialization
ISE = np.zeros(iterations) # Instant Square Error
MSE = np.zeros(iterations) # Mean Square Error
x = np.full(iterations, 0.26578)
noise = np.random.normal(0.0, 0.1, iterations) # for 50 measurements simulated
#Z = x + noise #just for the first time

Z = np.genfromtxt('data.csv', delimiter=',')
# to save the data in csv the first time
# data = np.asarray(Z);
# np.savetxt('data.csv', data, delimiter=',')

# initial values
X[0] = 0.0
P[0] = -0.001
    
# Simulation for 50 measurements
for i in range(1, iterations):    
    # prediction
    X[i], P[i] = prediction(X[i-1], P[i-1], A, Q, B, U)
    # update the values
    X[i], P[i], K[i] = update(X[i-1], P[i-1], C, K[i-1],Z[i-1], R)
    MSE[i] = ((x[:i] - X[:i])**2).mean() # Mean square error calculated at iteration i

ISE = (x - X)**2 # Instant Square erro calculation


pylab.figure()
pylab.plot(Z,'k+',label='noisy measurements', color='r')
pylab.plot(X,'b-',label='voltage estimate')
pylab.plot(x,color='g',label='truth value')
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Voltage [V]')
pylab.title('Simulation results')
#pylab.savefig('P001_.png')

pylab.figure()
valid_iter = range(1,iterations) 
pylab.plot(valid_iter,P[valid_iter],label='Error estimate')
pylab.xlabel('Iteration')
pylab.ylabel('covariance')
pylab.title('Estimate error covariance (P)')
#pylab.savefig('P001_P.png')

pylab.figure()
valid_iter = range(0,iterations) 
pylab.plot(valid_iter,K[valid_iter],label='a priori error estimate')
pylab.xlabel('Iteration')
pylab.title('Kalman gain (K)')
pylab.ylabel('gain')
#pylab.savefig('P001_K.png')

pylab.figure()
valid_iter = range(1,iterations) 
pylab.plot(valid_iter, MSE[valid_iter],label='MSE', color='r')
pylab.plot(valid_iter, ISE[valid_iter],label='ISE', color='b')
pylab.xlabel('Iteration')
pylab.ylabel('$Spread\ error [V^2]$')
pylab.title('Error')
#pylab.savefig('P001_E.png')

pylab.show()