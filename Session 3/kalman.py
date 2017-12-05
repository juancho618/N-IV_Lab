import numpy as np
import pylab
from functools import reduce
from filterpy.common import Q_discrete_white_noise
import math

class RadarSim(object):
    def __init__(self, dt, pos, velx, alt, vely):
        self.pos = pos
        self.velx = velx
        self.vely = vely
        self.alt = alt
        self.dt = dt

    def get_range(self):
        self.velx = self.velx + .1*np.random.randn() # TODO: replace with the process noise!
        self.vely = self.vely + .1*np.random.randn() # TODO: replace with the process noise!
        self.alt =self.alt + self.vely*self.dt
        self.pos =self.pos + self.velx*self.dt

        # adding some measurement noise
        err = self.pos*0.05*np.random.randn()
        r_distance = math.sqrt(self.pos**2 + self.alt**2)

        return(r_distance + err)
    


# datax = []
# datay = []
# for i in range(50):
#     r,x,y = radar.get_range()
#     datax.append(x)
#     datay.append(y)

# pylab.figure()
# pylab.plot(datax, datay, color='g')
# pylab.show()

'''
Data generation
'''

'''
initial values
'''
dt = 1
r = 5
X = np.matrix([[0.0],[0.0], [0.0], [0.0]])
A = np.matrix([[1., dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]])
Q = np.dot( np.sqrt(0.05),np.matrix([[(dt**3)/3, (dt**2)/2, 0, 0],
            [(dt**2)/2, dt, 0, 0],
            [0, 0, (dt**3)/3, (dt**2)/2],
            [0, 0, (dt**2)/2, dt]])) 
P = np.eye(4)*50           

def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """
    horiz_dist = x[0]
    altitude = x[2]
    denom = math.sqrt(horiz_dist**2 + altitude**2)
    return np.matrix([[horiz_dist/denom, 0., altitude/denom, 0.]])

def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """
    print(x)
    return(math.sqrt(x[0]**2 + x[2]**2))

def prediction(X, P, A, Q, B, U):
    X = np.dot(A, X) 
    P = np.dot(A, np.dot(P, A.T)) + Q 
    return (X, P)

def update(X, P, H, Y, r):
    MP =  hx(X) #np.dot(H, X) # measurement prediction
    print('mp', MP)
    print('y', Y)
    residual = Y - MP # the residual of the prediction
    print('residual', residual)
    MPC = np.dot(H, np.dot(P, H.T))  + 5 #r  # measurement prediction covariance ( C- no transpose because it is a scalar)
    print(MPC.shape)
    K = np.dot(P, np.dot(H.T, MPC.item(0,0)**-1)) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
    X = X + np.dot(K, residual) # Updated State Estimate
    P = np.dot((np.identity(P.shape[0]) - np.dot(K, H)), P)# Updated State Covariance # old way: P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T)))

    return (X,P,K)
radar = RadarSim(1, pos=0, velx=100,vely=100, alt=1000)

track = []
xs = []
for i in range(100):
    print(i)
    z = radar.get_range()
    track.append((radar.pos, radar.alt, radar.velx, radar.vely))
    H = HJacobian_at(X)

    X, P = prediction(X, P, A, Q, 0, 0)

    X, P, K = update(X, P, H, z, 0)
    xs.append(X)