import numpy as np
import pylab
from kalmanAcc import simulation

mc_iterations = 50

mc_NEES = []
mc_NIS = []
mc_Zs =[]

for x in range(mc_iterations):

    NEES, NIS,Zs_list,qf,qg,r  = simulation()
    mc_NEES.append(NEES)
    mc_NIS.append(NIS)
    mc_Zs.append(Zs_list)



gNEES = []
for times in range(99):
    sum = 0
    for sim in range(50):
        sum += mc_NEES[sim][times]
    gNEES.append(sum/50)

gNIS = []
for times in range(99):
    sum = 0
    for sim in range(50):
        sum += mc_NIS[sim][times]
    gNIS.append(sum/50)

gZs = [] # TODO: to be continued
for times in range(99):
    suma = 0
    sumb = 0
    sumc = 0
    add = 1
    for sim in range(50):
        if (sim == 49):
            add = -49
        suma += mc_Zs[sim][times]
        sumc += mc_Zs[sim+add][times]
        sumb += np.dot(mc_Zs[sim+add][times].T,mc_Zs[sim][times])
    sac = (sumb/50)*np.sqrt((1/50)*np.dot(sumc,suma))
    if (np.abs(sac.item(0,0)) > 100):
        sac = sac*.001
    elif (np.abs(sac.item(0,0)) > 1000):
        sac = sac*.00001
    elif (np.abs(sac.item(0,0)) > 10):
        sac = sac*.001
    # elif (np.abs(sac.item(0,0)) > 1):
    #     sac = sac*.01
    gZs.append(sac.item(0,0))


    

'''
Chart Monte Carlo
'''
pylab.figure()
pylab.plot(gNEES,color='y', marker='o' )
pylab.plot([1.5]*99,color='r')
pylab.plot([2.6]*99,color='r')
pylab.ylim(ymax = 15, ymin = -1)
pylab.title('Monte Carlo simulated NEES with qf = '+str(qf)+', qg = '+str(qg)+' and R ='+str(r))
pylab.savefig('./img/mc/acc/nees'+str(qf)+'_qg'+str(qg)+'_r'+str(r)+'.png')


#list(map(lambda x: x.item(0,0), gNIS))
pylab.figure()
pylab.plot([0.65]*99,color='r')
pylab.plot([1.43]*99,color='r')
pylab.ylim(ymax = 2, ymin = 0)
pylab.plot(gNIS,color='purple',label='estimate value', marker='h')
pylab.title('Monte carlo simulated NIS with qf '+str(qf)+', qg = '+str(qg)+' and R ='+str(r))
pylab.savefig('./img/mc/acc/nis'+str(qf)+'_qg'+str(qg)+'_r'+str(r)+'.png')

pylab.figure()
pylab.plot([-0.277]*99,color='r')
pylab.plot([0.277]*99,color='r')
pylab.ylim(ymax = 0.6, ymin = -0.6)
pylab.plot(gZs,color='b',label='estimate value', marker='h')
pylab.title('Monte carlo simulated SAC with qf '+str(qf)+', qg = '+str(qg)+' and R ='+str(r))
pylab.savefig('./img/mc/acc/sac'+str(qf)+'_qg'+str(qg)+'_r'+str(r)+'.png')



pylab.show()