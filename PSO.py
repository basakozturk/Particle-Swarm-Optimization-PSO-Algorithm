import math
import random
import numpy as np 
import matplotlib.pyplot as plt
import benchmark as f

def PSO(s_size,d,high,low,iterasyon,swarm):
    W = 0.8  # constant inertia weight
    c1 = 2   # cognative constant
    c2 = 2   # social constant
    """
    print('sınırlar:')
    print(low)
    print(high)
    print('*****************************************************')
    print('parametre olarak gelen sürü:')
    print(swarm)
    print('*****************************************************')
    """    
    #swarm = (high - low)*np.random.rand(s_size,d) + low
    velocity = np.zeros((s_size,d))
    obj = np.zeros((s_size,1))
    
    for i in range(s_size):
        obj[i] = f.Schaffer(swarm[i])
            
    pbestpos = swarm
    pbestval = obj
    
    sbestval = obj.min()    
    idx = obj.argmin(axis = 0)
    sbestpos = swarm[idx,:]
    objit = [sbestval]
    
    for iter in range(iterasyon):
        for i in range(s_size):
            velocity[i] = W*velocity[i]+c1*(np.random.random())*(pbestpos[i]-swarm[i]) + c2*(np.random.random())*(sbestpos-swarm[i])
        #print(velocity)
        
        # arama alanımızın dışına çıkmak istemiyoruz bu yüzden vmax oluşturup arama alanının yarısına eşitleyip kontrol gerşkleştiriyoruz
        vmax = (high - low)/2
        for i in range(s_size):
            for j in range(d):
                if (velocity[i,j] > vmax):
                    velocity[i,j] = vmax
                elif (velocity[i,j] < -vmax):
                    velocity[i,j] = -vmax
                    
        swarm = swarm + velocity
        #print(swarm)
        
        # alt ve üst degerleri aşan degerler varsa bunları sınıra çekmiş olduk
        for i in range(s_size):
            for j in range(d):
                if (swarm[i,j] > high):
                    swarm[i,j] = high
                elif (swarm[i,j] < low):
                    swarm[i,j] = low           
                    
                    
        for i in range(s_size):
            obj[i] = f.Schaffer(swarm[i])   
            
        # parçaçıklar için en iyi çözüm güncellemesi
        
        for i in range(s_size):
            if (obj[i] < pbestval[i]):
                pbestval[i] = obj[i]
                pbestpos[i,:] = swarm[i,:]
        
        # Sürünün en iyisinin güncellenmesi
        if (obj.min() < sbestval):
            sbestval = obj.min()
            idx = obj.argmin(axis = 0)
            sbestpos = swarm[idx,:]
            
        objit.append(sbestval)
        
    plt.figure()
    plt.plot(objit)
    plt.show()
        
    #print('Sürünün en iyi pozisyonu : ')
    #print(sbestpos)   
    #print('Sürünün en iyi degeri : ')
    #print(sbestval)   
    #print('En iyi degerler')
    #print(objit)
    
    return sbestpos[0], sbestval, objit
