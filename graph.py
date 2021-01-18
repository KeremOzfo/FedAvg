import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

def special_adress():
    adress=[]
    adress_loss = []
    labels = []
    #labels = ['\u03B1=0.85', '\u03B1=0.90', '\u03B1=0.95','\u03B1=1']
    #labels = ['\u03B1=0,0.01,0.01','\u03B1=0,0.01,0.05', '\u03B1=0,0.05,0.05', '\u03B1=0,0.05,0.1','\u03B1=1']
    a = 'Results/128bs/16w/16H/'
    for dir in listdir(a):
        adress.append(a+dir+'/acc')
        adress_loss.append(a+dir + '/loss')
        labels.append(dir)
    return adress,adress_loss,labels

def compile_results(adress):
    results = None
    f_results = []
    for i, dir in enumerate(listdir(adress)):
        vec = np.load(adress + '/'+dir)
        final_result = vec[len(vec)-1]
        f_results.append(final_result)
        if i==0:
            results = vec/len(listdir(adress))
        else:
           results += vec/len(listdir(adress))
    avg = np.average(f_results)
    st_dev = np.std(f_results)
    return results, [adress,avg,st_dev]

def cycle_graph_props(colors,markers,linestyles):
    randoms =[]
    randc = np.random.randint(0,len(colors))
    randm = np.random.randint(0,len(markers))
    randl = np.random.randint(0,len(linestyles))
    m = markers[randm]
    c = colors[randc]
    l = linestyles[randl]
    np.delete(colors,randc)
    np.delete(markers,randm)
    np.delete(linestyles,randl)
    print(colors,markers,linestyles)
    return c,m,l


def avgs(sets):
    avgs =[]
    for set in sets:
        avg = np.zeros_like(set[0])
        avgs.append(avg)
    return avgs

def graph(data, legends,interval):
    marker = ['s', 'v', '+', 'o', '*']
    linestyle =['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    markercycler = cycle(marker)
    epoch = 300
    for d,legend in zip(data,legends):
        x_axis = []
        psuedo_epoch = len(d)
        l = next(linecycler)
        m = next(markercycler)
        for i in range(0,len(d)):
            x_axis.append(i*epoch/psuedo_epoch)
        plt.plot(x_axis,d, marker= m ,linestyle = l ,markersize=2, label=legend)
    #plt.axis([5, 45,70 ,90])
    #plt.axis([145,155,88,92])
    #plt.axis([290, 300, 87, 95])
    #plt.axis([50, 100, 87, 95])
    plt.axis([0, 300, 85, 95])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.title('Majority Voting')
    plt.legend()
    plt.grid(True)
    plt.show()

def graph_loss(data, legends, interval):
    marker = ['s', 'v', '+', 'o', '*']
    linestyle = ['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    markercycler = cycle(marker)
    epoch = 300
    for d, legend in zip(data, legends):
        x_axis = []
        l = next(linecycler)
        psuedo_epoch = len(d)
        m = next(markercycler)
        for i in range(0, len(d)):
            x_axis.append(i*epoch/psuedo_epoch)
        plt.plot(x_axis, d, marker=m, linestyle=l, markersize=2, label=legend)
    plt.axis([0, 300, 0, 0.5])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()



def concateresults(dirsets):
    all_results =[]
    for set in dirsets:
        all_results.append(compile_results(set)[0])
    return all_results



intervels = 1
epoch  = 300
labels = special_adress()[2]
results = concateresults(special_adress()[0])
results_loss = concateresults(special_adress()[1])
#results = concateresults(locations)
graph(results,labels,intervels)
graph_loss(results_loss,labels,intervels)
#data,legends = compile_results(loc)
#graph(data,labels,intervels)