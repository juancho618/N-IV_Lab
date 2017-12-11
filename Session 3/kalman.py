import numpy as np
import pylab
from functools import reduce
from filterpy.common import Q_discrete_white_noise
import math


dt = 0.05
r = 5
q = 1


class RadarSim(object):
    def __init__(self, dt, pos, velx, alt, vely):
        self.pos = pos
        self.velx = velx
        self.vely = vely
        self.alt = alt
        self.dt = dt

    def get_range(self):
        self.velx = self.velx + np.sqrt(q)*np.random.randn() 
        self.vely = self.vely + np.sqrt(q)*np.random.randn() 
        self.alt =self.alt + self.vely*self.dt
        self.pos =self.pos + self.velx*self.dt

        # adding some measurement noise
        err = self.pos*0.05*np.random.randn()
        r_distance = math.sqrt(self.pos**2 + self.alt**2)

        return(r_distance + err)

'''
initial values
'''

A = np.matrix([[1., dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]])
Q = np.dot( np.sqrt(q),np.matrix([[(dt**3)/3, (dt**2)/2, 0, 0],
            [(dt**2)/2, dt, 0, 0],
            [0, 0, (dt**3)/3, (dt**2)/2],
            [0, 0, (dt**2)/2, dt]])) 
P = np.linalg.inv(np.diag([r**2,(r**2/dt**2),r**2,(r**2/dt**2)]))

def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """
    horiz_dist = x.item(0,0)
    altitude = x.item(2,0)
    denom = math.sqrt(horiz_dist**2 + altitude**2)

    return np.matrix([[horiz_dist/denom, 0., altitude/denom, 0.],
                    [altitude/(horiz_dist**2 + altitude**2), 0., -1*(horiz_dist/(horiz_dist**2 + altitude**2)), 0.]])

def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """
    r = math.sqrt(x.item(0,0)**2 + x.item(2,0)**2)
    a = math.atan2(x.item(2,0),x.item(0,0))
    return(r, a)

def prediction(X, P, A, Q, B, U):
    X = np.dot(A, X) 
    P = np.dot(A, np.dot(P, A.T)) + Q 
    return (X, P)

def update(X, P, H, Y, r):
    MP =  hx(X) #np.dot(H, X) # measurement prediction
    residual = Y - MP # the residual of the prediction
    MPC = np.dot(H, np.dot(P, H.T))  + r #r  # measurement prediction covariance ( C- no transpose because it is a scalar)
    K = np.dot(P, np.dot(H.T, MPC.item(0,0)**-1)) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
    X = X + np.dot(K, residual) # Updated State Estimate
    P = np.dot((np.identity(P.shape[0]) - np.dot(K, H)), P)# Updated State Covariance # old way: P = P - np.dot(K, np.dot(K, np.dot(MPC, K.T)))

    return (X,P,K)

radar = RadarSim(dt, pos=0, velx=100,alt=1000,vely=100 )
X = np.matrix([[radar.pos],[radar.velx-10], [radar.alt+100], [radar.vely-10]])



track = []
xs = []
for i in range(int(20/dt)):
    z = radar.get_range()
    track.append((radar.pos, radar.velx, radar.alt, radar.vely))
    
    H = HJacobian_at(X)
    X, P = prediction(X, P, A, Q, 0, 0)

    X, P, K = update(X, P, H, z, 0)
    xs.append(X)

def getParameters(x):
    r = math.sqrt(x[0]**2 + x[2]**2)
    ang = math.atan2(x[2],x[0])

    return r

time = np.arange(0, len(xs)*dt, dt)
xs = np.asarray(np.squeeze(xs))
track_pos = list(map(lambda x: getParameters(x),track)) 
x_pos = list(map(lambda x: getParameters(x),xs))
pylab.title('Position of the object from the radar')
pylab.plot(time, track_pos,  color = 'g', label='Real X position')
pylab.plot(time,x_pos, color = 'r', label='Estimated X position')
pylab.xlabel('time [s]')
pylab.ylabel('Velocity V [m/s]')

track_velx = list(map(lambda x: x[1],track)) 
x_velx = list(map(lambda x: x[1], xs)) 
pylab.figure()
pylab.title('Velocity in axis X')
pylab.plot(time, track_velx, color='b', label='Real X velocity')
pylab.plot(time, x_velx, color='r', label='Predicted X velocity')
pylab.xlabel('time [s]')
pylab.ylabel('Velocity V [m/s]')

track_vely = list(map(lambda x: x[3],track)) 
x_vely = list(map(lambda x: x[3], xs)) 
pylab.figure()
pylab.title('Velocity in axis Y')
pylab.plot(time, track_vely, color='b', label='Real Y velocity')
pylab.plot(time, x_vely, color='r', label='Predicted Y velocity')
pylab.xlabel('time [s]')
pylab.ylabel('Velocity V [m/s]')


track_alt = list(map(lambda x: x[2],track)) 
x_alt = list(map(lambda x: x[2], xs)) 
pylab.figure()
pylab.title('Y position of the object')
pylab.plot(time, track_alt, color='b', label='Real y position')
pylab.plot(time, x_alt, color='r', label='Estimated y position')
pylab.xlabel('time [s]')
pylab.ylabel('Velocity V [m/s]')


pylab.show()