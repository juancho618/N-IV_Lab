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

'''
Kalman Definition functions
'''
def prediction(X, P, A, Q, B, U, W):
    X = np.dot(A, X) + W # + np.dot(B, U)
    P = np.dot(A, np.dot(P, A.T)) + Q # (A - no transpose because it is a scalar)
    return X, P

def update(X, P, H, K, Y, R):
    MP = np.dot(H, X) # measurement prediction
    residual = Y - MP # the residual of the prediction
    MPC = np.dot(H, np.dot(P, H))  + R    # measurement prediction covariance ( C- no transpose because it is a scalar)
    K = np.dot(P, np.dot(H, np.linalg.inv(MPC))) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
    X = X + np.dot(K, residual) # Updated State Estimate
    P = np.dot((np.identity(P.shape[0]) - np.dot(K, H)), P)# Updated State Covariance # old way: P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T))) 
    
    return (X,P,K)


'''
values initialization
'''
iterations = 100
dt = 1
q = 0

wk_distance = np.random.normal(0, np.sqrt(q*((dt**3)/3)),iterations) # noise in the position
wk_velocity = np.random.normal(0, np.sqrt(q*dt),iterations)         # noise in the velocity
W = np.vstack((wk_distance, wk_velocity)) # white noise function

Q = np.matrix([[(dt**3)/3, (dt**2/2)],[(dt**2/2), dt]])
R = np.random.normal(0, 1.0, iterations)
X = np.matrix([[0.0],[0.0]]) # Initial X values
P = np.matrix([[1., 0.5], [0.5, 2]])

A = np.matrix([[1., dt], [0, 1.]]) # The state is a constant
U = 0 # There is no control input (Aceleration!).
B = np.matrix([[0.0], [0.0]]) # There is no control input
C = np.matrix([[1, 0]]) # The noisy measurement is direct
G = np.matrix([[(dt**2)/2],[ dt ] ])
B = G
H = np.matrix([[1., 0], [0, 1.]])
Hi = 1
Y = np.matrix([[real[0][0]],[real[1][0]]]) + R[0]
K = np.zeros(iterations) # gain or blending factor initialization

p_distance = []
p_velocity = []
p_distance.append((X.item(0,0)))
p_velocity.append((X.item(1,0)))

'''
Simulation
'''
for i in range(1, iterations):
    # prediction
    X, P = prediction(X, P, A, Q, B, U, np.matrix([[W[0][i]],[W[1][i]]]))
    p_distance.append(X.item(0,0)) # add predicted distance
    p_velocity.append(X.item(1,0)) # add predicted velocity
    # update the values
    X, P, K = update(X, P, Hi, K, Y, R[i])
    Y = np.matrix([[real[0][i]],[real[1][i]]]) + R[i] # next measurement value

'''
Charts
'''
pylab.figure()
pylab.plot(real[0],real[1],label='true measurements', color='r')
pylab.plot(p_distance, p_velocity,'b-',label='Velocity estimate')
# pylab.plot(x,color='g',label='truth value')
pylab.legend()
pylab.xlabel('Distance')
pylab.ylabel('Velocity [V]')
pylab.title('Simulation results')
pylab.show()

#Z = x + noise #just for the first time
#https://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
