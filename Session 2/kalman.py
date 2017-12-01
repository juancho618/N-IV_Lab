import numpy as np
import pylab
from functools import reduce
from filterpy.common import Q_discrete_white_noise
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
<<<<<<< HEAD
    P = np.dot(A, np.dot(P, A.T)) + Q # (A - no transpose because it is a scalar)
=======
    P = np.dot(A, np.dot(P, A.T)) + Q#0.0000000001#Q # (A - no transpose because it is a scalar)
>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc
    return (X, P)

def update(X, P, H, Y, R):
    MP = np.dot(H, X) # measurement prediction
    residual = Y - MP # the residual of the prediction
    MPC = np.dot(H, np.dot(P, H.T))  +  R #Q    # measurement prediction covariance ( C- no transpose because it is a scalar)
    K = np.dot(P, np.dot(H.T, MPC**-1)) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
<<<<<<< HEAD
=======
    print(residual.shape)

>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc
    X = X + np.dot(K, residual) # Updated State Estimate
    P = np.dot((np.identity(P.shape[0]) - np.dot(K, H)), P)# Updated State Covariance # old way: P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T)))

    return (X,P,K)
dt = 1
G = np.matrix([[(dt**2)/2],[ dt ] ])
def simulate_movement(A, X, q):            
    process_model = np.dot(A, X) + np.dot(G,np.random.normal(0, np.sqrt(q))) # TODO: validate model
    return process_model



'''
values initialization
'''

iterations = 100
<<<<<<< HEAD
q = 0 # process variance
=======
dt = 1 # time differential
q = 2 # process variance
>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc
r = 1. # sensor variance


Q = Q_discrete_white_noise(dim=2, dt=dt, var=q)
#Q = np.dot(q,np.matrix([[(dt**3)/3, (dt**2/2)],[(dt**2/2), dt]]))*q
#Q  = np.random.normal(0, q, iterations)
R = np.random.normal(0, r, iterations)
X = np.matrix([[0.0],[0.0]]) # Initial X values
<<<<<<< HEAD
#P = np.linalg.inv(np.matrix([[ r, r/dt], [r/dt, 2*r/(dt**2)]]))
P = np.matrix([[ r, 0], [0, 2*r/(dt**2)]])
=======
P = np.linalg.inv(np.matrix([[ r, r/dt], [r/dt, 2*r/(dt**2)]]))

>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc
A = np.matrix([[1., dt], [0, 1.]]) # The state is a constant
U = 0 # There is no control input (Aceleration!).
B = np.matrix([[0.0], [0.0]]) # There is no control input
C = np.matrix([[1, 0]]) # The noisy measurement is direct
<<<<<<< HEAD
B = G

NEES = []
NIS = []
klist = []
=======
G = np.matrix([[(dt**2)/2],[ dt ] ])
B = G
H = np.matrix([[1., 0], [0, 1.]])
Hi = 1
K = np.zeros(iterations) # gain or blending factor initialization

>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc
plist = []
p_distance = []
p_velocity = []
vel = []
p_distance.append((X.item(0,0)))
p_velocity.append((X.item(1,0)))
vel.append(X.item(1,0))
plist.append(P.item(0,0))
<<<<<<< HEAD
klist.append(0)
=======
>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc

'''
Simulation
'''
# creatin simulated data
data = []
Xk = np.matrix([[0.0],[10.0]])

for i in range(100):
    data.append(Xk)
    Xk = simulate_movement(A, Xk, q)

Y = np.dot( C, data[0]) + R[0] # measured position

for i in range(1, iterations):
    # prediction
    X, P = prediction(X, P, A, Q, B, U)

   
    # update the values
<<<<<<< HEAD
    X, P, K = update(X, P, C, Y, R[i])

    Xs = data[i] - X
    s = np.dot(C, np.dot(P,C.T)) + np.dot(1, np.dot(R[i],1))
    z = Y - np.dot(C,data[i])
    Zs = np.dot(z.T, np.dot(s**-1,z))
    NIS.append(Zs.item(0,0))
    NEES.append(np.dot(Xs.T, np.dot(np.linalg.inv(P),Xs)).item(0,0))
    p_distance.append(X.item(0,0)) # add predicted distance
    vel.append(X.item(1,0))
    plist.append(P.item(0,0))
    klist.append(K.item(0,0))
    Y = np.dot( C, data[i]) + R[i] # next measurement value
=======
    X, P, K = update(X, P, C, K, Y, R[i])
    p_distance.append(X.item(0,0)) # add predicted distance
    vel.append(X.item(1,0))
    plist.append(P.item(0,0))
    Y = np.dot( 1, data[i]) + R[i] # next measurement value
>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc

print(p_distance)

# calculating velocity
for i in range(2, len(p_distance)):
    p_velocity.append((p_distance[i] - p_distance[i-1])/1)

print('nis', NIS)

'''
Charts
'''
pylab.figure()
pylab.plot(list(map(lambda x: x[0].item(0,0), data)), list(map(lambda x: x[1].item(0,0), data)),label='true measurements', color='r')
#pylab.plot(p_distance[:99], p_velocity,'b-',label='Position stimate estimate')
pylab.plot(p_distance,vel,color='g',label='estimate value')
pylab.legend()
pylab.xlabel('Position X')
pylab.ylabel('Velocity [V]')
pylab.title('Simulation results')

pylab.figure()
pylab.plot(plist,color='g',label='estimate value')
<<<<<<< HEAD


pylab.figure()
pylab.plot(klist,color='r',label='estimate value')

pylab.figure()
pylab.plot(NEES,color='y',label='estimate value')

pylab.figure()
pylab.plot(NIS,color='purple',label='estimate value')



=======
>>>>>>> 38a29072eeaec433c44318e904edaa817be1dedc
pylab.show()



#Z = x + noise #just for the first time