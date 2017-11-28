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
# Xreal_distance =  reduce(lambda c, x: c + [c[-1] + 10], range(100),[0])[1:]
# Xreal_speed = list(map(lambda x: 10, range(100)))
# real = np.vstack((Xreal_distance, Xreal_speed))

'''
Kalman Definition functions
'''
def prediction(X, P, A, Q, B, U):
    X = np.dot(A, X) 
    P = np.dot(A, np.dot(P, A.T)) + Q # (A - no transpose because it is a scalar)
    return (X, P)

def update(X, P, H, K, Y, R):
    MP = np.dot(H, X) # measurement prediction
    residual = Y - MP # the residual of the prediction
    MPC = np.dot(H, np.dot(P, H))  + R    # measurement prediction covariance ( C- no transpose because it is a scalar)
    print(MPC.shape)
    K = np.dot(P, np.dot(H, np.linalg.inv(MPC))) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
    X = X + np.dot(K, residual) # Updated State Estimate
    P = np.dot((np.identity(P.shape[0]) - np.dot(K, H)), P)# Updated State Covariance # old way: P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T))) 
    
    return (X,P,K)

def simulate_movement(A, X, q):            
    process_model = np.dot(A, X) + np.matrix([[np.random.normal(0, np.sqrt(q*((dt**3)/3)))], [np.random.normal(0, np.sqrt(q*dt))]]) # TODO: validate model
    return process_model



'''
values initialization
'''

iterations = 100
dt = 1 # time differential
q = 0 # process variance
r = 1. # sensor variance



Q = np.matrix([[(dt**3)/3, (dt**2/2)],[(dt**2/2), dt]])
#Q = R = np.random.normal(0, q, iterations)
R = np.random.normal(0, r, iterations)
X = np.matrix([[0.0],[0.0]]) # Initial X values
P = np.matrix([[ 1, 0], [0, 2/(dt**2)]])

A = np.matrix([[1., dt], [0, 1.]]) # The state is a constant
U = 0 # There is no control input (Aceleration!).
B = np.matrix([[0.0], [0.0]]) # There is no control input
C = np.matrix([[1, 0]]) # The noisy measurement is direct
G = np.matrix([[(dt**2)/2],[ dt ] ])
B = G
H = np.matrix([[1., 0], [0, 1.]])
Hi = 1

K = np.zeros(iterations) # gain or blending factor initialization

p_distance = []
p_velocity = []
p_distance.append((X.item(0,0)))
p_velocity.append((X.item(1,0)))

'''
Simulation
'''
# creatin simulated data
data = []
Xk = np.matrix([[0.0],[10.0]])

for i in range(100):
    data.append(Xk)
    Xk = simulate_movement(A, Xk, q)

Y = np.dot( 1, data[0]) + R[0] # measured position

for i in range(1, iterations):
    # prediction
    X, P = prediction(X, P, A, Q[i], B, U)
    p_distance.append(X.item(0,0)) # add predicted distance
    # update the values
    X, P, K = update(X, P, Hi, K, Y, R[i])
    Y = np.dot( 1, data[i]) + R[i] # next measurement value

print(p_distance)

# calculating velocity
for i in range(2, len(p_distance)):
    p_velocity.append((p_distance[i] - p_distance[i-1])/1)

print(p_distance)
print(p_velocity)

'''
Charts
'''
pylab.figure()
pylab.plot(list(map(lambda x: x[0].item(0,0), data)), list(map(lambda x: x[1].item(0,0), data)),label='true measurements', color='r')
pylab.plot(p_distance[:99], p_velocity,'b-',label='Velocity estimate')
# pylab.plot(x,color='g',label='truth value')
pylab.legend()
pylab.xlabel('Distance')
pylab.ylabel('Velocity [V]')
pylab.title('Simulation results')
pylab.show()



#Z = x + noise #just for the first time
#https://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
