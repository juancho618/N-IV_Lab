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
def prediction(X, P, A, Q, B, U):
    X = np.dot(A, X) + np.dot(B, U)
    P = np.dot(A, np.dot(P, A.T)) + Q # (A - no transpose because it is a scalar)
    return X, P

def update(X, P, H, K, Y, R):
    MP = np.dot(H, X) # measurement prediction
    residual = Y - MP # the residual of the prediction
    MPC = np.dot(1, np.dot(P, 1))  + R    # measurement prediction covariance ( C- no transpose because it is a scalar)
    K = np.dot(P, np.dot(1, np.linalg.inv(MPC))) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
    X = X + np.dot(K, residual) # Updated State Estimate
    print(np.dot(K, residual))
    P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T))) # Updated State Covariance
    return (X,P,K)
iterations = 100
# Q = np.random.normal(10, 0.0, iterations)
Q = np.matrix([[0.0, 0.0],[0.0, 1.0]])
R = np.random.normal(0, 1.0, iterations)
print(R)
dt = 1
# Initialization of the values of the variable
# R = []
# for i in range(iterations):
#     rq = np.random.normal(0, 1)
#     rdiag = np.diag([rq, rq])
#     R.append(rdiag)
#     Pi = np.diag([0.1, 0.1])
#     P.append(Pi)
#     X.append([[0.0], [0.0]])

X = np.matrix([[0.0],[10.0]])
P = np.matrix([[1., 0.5], [0.5, 2]])

vel = np.full(100,10)
# X = np.zeros((100,2,1))
#X = np.matrix([[0.0], [0.0]]) # Initializing voltage estimate the array with the number of iterations
#P = np.diag([0.1, 0.1, 0.1, 0.1]) # Initializing error estimate the array with the number of iterations
A = np.matrix([[1., dt], [0, 1.]]) # The state is a constant
U = 1 # There is no control input
B = np.matrix([[0.0], [0.0]]) # There is no control input
C = np.matrix([[1, 0]]) # The noisy measurement is direct
G = np.matrix([[(dt**2)/2],[ dt ] ])
B = G
H = np.matrix([[1., 0], [0, 1.]])
Y = C*X + R[0]
# Q = 10e-5 # Process variance
# R = 0.01 # given from the problem as a constant noise
K = np.zeros(iterations) # gain or blending factor initialization
ISE = np.zeros(iterations) # Instant Square Error
MSE = np.zeros(iterations) # Mean Square Error
x = np.matrix([[0.0], [0.0]])

dist = []
p_dist = []
dist.append((X.item(0,0),X.item(1,0)))
p_dist.append((Y.item(0,0)))
for i in range(1, iterations):
    print(X)
    # prediction
    X, P = prediction(X, P, A, Q, B, U)
    # update the values
    X, P, K = update(X, P, H, K, Y, R[i])
    dist.append((X.item(0,0),X.item(1,0)))
    Y = C*X +R[i]
    p_dist.append((Y.item(0,0)))
   # MSE[i] = ((x[:i] - X[:i])**2).mean() # Mean square error calculated at iteration i

velo = list(map(lambda x: x[1], dist))
dista = list(map(lambda x: x[0], dist))
distance = reduce(lambda c, x: c + [c[-1] + x[0]], dist,[0])[1:]
pylab.figure()
pylab.plot(vel,label='noisy measurements', color='r')
#pylab.plot(distance, velo,'b-',label='voltage estimate')
# pylab.plot(x,color='g',label='truth value')
pylab.legend()
pylab.xlabel('Distance')
pylab.ylabel('Velocity [V]')
pylab.title('Simulation results')
pylab.show()

#Z = x + noise #just for the first time
#https://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
