import numpy as np
import pylab
from functools import reduce
'''
The prediction funtion recives 5 arguments:

X: The state stimate of the previous step
P: The state covariance of the previous step
A: The transition matrix
Q: The process noise covariance marix.
B: The input effect matrix.
U: The control input.
'''
# simulation of the data
Xreal_distance =  reduce(lambda c, x: c + [c[-1] + 10], range(100),[0])[1:]
Xreal_speed = list(map(lambda x: 10, range(100)))
real = np.vstack((Xreal_distance, Xreal_speed))

def prediction(X, P, A, Q, B, U, W): # delete U, B
    X = np.dot(A, X) + W
    P = np.dot(A, np.dot(P, A.T)) # (A - no transpose because it is a scalar) + Q
    return X, P

def update(X, P, H, K, Y, R):
    MP = np.dot(H, X) # measurement prediction
    residual = Y - MP # the residual of the prediction
    print(residual)
    MPC = np.dot(H, np.dot(P, H))  + 1 # R TODO: add r again   # measurement prediction covariance ( C- no transpose because it is a scalar)
    K = np.dot(P, np.dot(H, np.linalg.inv(MPC))) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
    X = X + np.dot(K, residual) # Updated State Estimate
    P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T))) # Updated State Covariance
    return (X,P,K)

iterations = 100
dt = 1
q = 0


# Q = np.random.normal(10, 0.0, iterations)
wk_distance = np.random.normal(0, np.sqrt(q*((dt**3)/3)),iterations)
wk_velocity = np.random.normal(0, np.sqrt(q*dt),iterations)
W = np.vstack((wk_distance, wk_velocity))
Q = np.matrix([[(dt**3)/3, (dt**2/2)],[(dt**2/2), dt]])
R = np.random.normal(0, 1.0, iterations)
#     X.append([[0.0], [0.0]])

X = np.matrix([[0.0],[0.0]]) # Initial X values
P = np.matrix([[1., 0.5], [0.5, 2]])


A = np.matrix([[1., dt], [0, 1.]]) # The state is a constant
U = 0 # There is no control input (Aceleration!).
B = np.matrix([[0.0], [0.0]]) # There is no control input
C = np.matrix([[1, 1]]) # The noisy measurement is direct
G = np.matrix([[(dt**2)/2],[ dt ] ])
B = G
H = np.matrix([[1., 0], [0, 1.]])
Hi = 1
Z = C*real + R # measurements of the position of the car + noise
Y = Z.item(0,0)  #+ R[0] TODO: modify real
#Y = np.dot(C, X) + Hi*R[0] # initial measurement
# Q = 10e-5 # Process variance
# R = 0.01 # given from the problem as a constant noise
K = np.zeros(iterations) # gain or blending factor initialization
ISE = np.zeros(iterations) # Instant Square Error
MSE = np.zeros(iterations) # Mean Square Error


p_distance = []
p_velocity = []
p_distance.append((X.item(0,0)))
p_velocity.append((X.item(1,0)))
for i in range(1, iterations):
    # prediction
    X, P = prediction(X, P, A, Q, B, U, np.matrix([[W[0][i]],[W[1][i]]]))
    p_distance.append(X.item(0,0))
    # update the values
    X, P, K = update(X, P, Hi, K, Y, R[i])
    Y = Z.item(0,i) # next measurement
    # Y = np.dot(C, X) + Hi*R[i]
    # p_dist.append((Y.item(0,0)))
   # MSE[i] = ((x[:i] - X[:i])**2).mean() # Mean square error calculated at iteration i

# velo = list(map(lambda x: x[1], dist))
# dista = list(map(lambda x: x[0], dist))
# distance = reduce(lambda c, x: c + [c[-1] + x[0]], dist,[0])[1:]
pylab.figure()
pylab.plot(real[0], real[1],label='true measurements', color='r')
#pylab.plot(p_distance, p_velocity,'b-',label='Velocity estimate')
# pylab.plot(x,color='g',label='truth value')
pylab.legend()
pylab.xlabel('Distance')
pylab.ylabel('Velocity [V]')
pylab.title('Simulation results')
pylab.show()

#Z = x + noise #just for the first time
#https://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
