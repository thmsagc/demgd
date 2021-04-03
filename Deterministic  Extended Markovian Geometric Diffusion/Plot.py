import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotList2D(inputSet, setReduced, setRemoved):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Deterministic E-MGD")
    ax1.set_title("Input set")
    ax2.set_title("Reduced set")
    for i in inputSet:
        x = i[0]
        y = i[1]
        ax1.plot(x,y,'bo',color='blue')
        
    for i in setReduced:
        x = i[0]
        y = i[1]
        ax2.plot(x,y,'bo',color='blue')
        
    for i in setRemoved:
        x = i[0]
        y = i[1]
        ax2.plot(x,y,'bo',color='red')
        
    plt.show()

def plotList3D(inputSet, setReduced, setRemoved):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.suptitle("Deterministic E-MGD")
    ax1.set_title("Input set")
    ax2.set_title("Reduced set")
    for i in inputSet:
        x = i[0]
        y = i[1]
        z = i[2]
        ax1.plot(x,y,z,'bo',color='blue')
        
    for i in setReduced:
        x = i[0]
        y = i[1]
        z = i[2]
        ax2.plot(x,y,z,'bo',color='blue')
        
    for i in setRemoved:
        x = i[0]
        y = i[1]
        z = i[2]
        ax2.plot(x,y,z,'bo',color='red')
        
    plt.show()
