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
'''
Kalman Definition functions
'''
def simulation():
    
    def prediction(X, P, A, Q, B, U):
        X = np.dot(A, X) 
        P = np.dot(A, np.dot(P, A.T)) + Q 
        return (X, P)

    def update(X, P, H, Y, R):
        MP = np.dot(H, X) # measurement prediction
        residual = Y - MP # the residual of the prediction
        MPC = np.dot(H, np.dot(P, H.T))  +  r  # measurement prediction covariance ( C- no transpose because it is a scalar)
        K = np.dot(P, np.dot(H.T, MPC**-1)) # kalman (C - no transpose because it is a scalar and np.inv()no used for escalars)
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
    q = 1 # process variance
    r = 0.1 # sensor variance
    qf = 3
    qg = 3


    Q = Q_discrete_white_noise(dim=2, dt=dt, var=qf)
    R = np.random.normal(0, np.square(r), iterations)
    X = np.matrix([[0.0],[0.0]]) # Initial X values
    P = np.linalg.inv(np.matrix([[ r, r/dt], [r/dt, 2*r/(dt**2)]]))
    A = np.matrix([[1., dt], [0, 1.]]) # The state is a constant
    U = 0 # There is no control input.
    B = np.matrix([[0.0], [0.0]]) # There is no control input
    C = np.matrix([[1, 0]]) # The noisy measurement is direct
    B = G

    NEES = []
    NIS = []
    Zs_list = []
    k_dist = [] # kalman filter for the distance
    k_vel = [] # kalman filter for the velocity
    plist = []
    p_22=[] 
    p_distance = [] # array to save predicted distance
    vel = []# array to save predicted speed
    p_distance.append((X.item(0,0)))
    vel.append(X.item(1,0))
    k_dist.append(0)


    
    '''
    Simulation
    '''
    
    # creatin simulated data
    data = []
    Xk = np.matrix([[0.0],[10.0]])

    for i in range(100):
        data.append(Xk)
        Xk = simulate_movement(A, Xk, qg)


    for i in range(1, iterations):
        # prediction
        X, P = prediction(X, P, A, Q, B, U)
    
        Y = np.dot( C, data[i]) + R[i]# next measurement value

        # update the values
        X, P, K = update(X, P, C, Y, R[i])
        Xs = data[i] - X
        NEES.append(np.dot(Xs.T, np.dot(np.linalg.inv(P),Xs)).item(0,0))
        # updating values to plot later
        p_22.append(P.item(1,1))

        
        

        s = np.dot(C, np.dot(P,C.T)) + np.dot(1, np.dot(R[i],1))
        z = Y - np.dot(C,X)
        Zs = np.dot(z.T, np.dot(s**-1,z))
        Zs_list.append(Zs)
        NIS.append(Zs.item(0,0))
        
        p_distance.append(X.item(0,0)) # add predicted distance
        vel.append(X.item(1,0))
        plist.append(P.item(0,0))
        
        k_dist.append(K.item(0,0))
        k_vel.append(K.item(1,0))
    return(NEES,NIS,Zs_list)
    
   
simulation()


            


  
    # '''
    # Charts (Uncomment to create Illustrations!!!!!!)
    # '''
    # pylab.figure()
    # pylab.plot(list(map(lambda x: x[0].item(0,0), data)), list(map(lambda x: x[1].item(0,0), data)),label='True Measurements', color='r')
    # pylab.plot(p_distance[1:],vel[1:],color='g',label='Estimated values')
    # pylab.legend()
    # pylab.xlabel('Position X [m]')
    # pylab.ylabel('Velocity V [m/s]')
    # pylab.title('Simulation results for qf = ' + str(qf) + ' and qg =' + str(qg) )

    # pylab.figure()
    # pylab.plot(plist[1:],color='r',label='P11')
    # pylab.title('Estimated P Variance for the Position')

    # pylab.figure()
    # pylab.plot(p_22[1:],color='r',label='P22')
    # pylab.title('Estimated P Variance for the Velocity')



    # pylab.figure()
    # pylab.plot(k_dist[1:],color='g')
    # pylab.title('Kalman gain for distance with qf = '+ str(qf) + ' and qg =' + str(qg)  )

    # pylab.figure()
    # pylab.plot(k_vel[1:],color='g')
    # pylab.title('Kalman gain for velocity with qf = '+ str(qf) + ' and qg =' + str(qg)  )


    # pylab.figure()
    # pylab.plot(NEES[1:],color='y')
    # pylab.title('Normalized Estimation Error Squared qf = '+ str(qf) +' (NEES)' + ' and qg =' + str(qg) )


    # pylab.figure()
    # pylab.plot(NIS[1:],color='purple',label='estimate value')
    # pylab.title('Normalized Innovation Squared qf = '+ str(qf) +'(NIS)' + ' and qg =' + str(qg) )



    # pylab.show()



