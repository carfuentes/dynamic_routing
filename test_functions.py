#imports
from datetime import *
from ESN_class_integration import ESN
from graph_analysis import *
import numpy as np
import os
from parameters import *
from mutual_info import memory_capacity_n,calc_MI_binning
from nrmse_calc import nrmse_n,nrmse_calc
from preprocessing_net import get_cyclic_net
from r_to_python import get_r_dictionaries,mapping_relabel_function

from parameters import *
from matplotlib.pyplot import *
from graph_analysis import *


#functions

    
def printTime(*args):

    print (datetime.now(),"".join(map(str,args)))




def test_readout(directory,file_path,in_size, tau_list,tau_test_list,tau_train,spectral_radius, i_scaling,beta_scaling,GOterms,readout_dict,n, noise, euler=True, save=False, notebook=True):
    #init
    printTime(file_path)
    printTime(miLen)
    filename=file_path[file_path.index("list")+5:file_path.index(".csv")]

    
    printTime("Running network...")
    G=get_cyclic_net(os.path.join(directory, file_path))
    
    GO_id_map=get_r_dictionaries("test.txt",mapping=False)
    edgeid_ezid_map=get_r_dictionaries("mapping_id_to_entrez.txt")
    mapping=mapping_relabel_function(G,edgeid_ezid_map)
   
    for key,values in GO_id_map.items():
        GO_id_map[key]=set([mapping[value] for value in values])

    
    #Run network
    net=ESN(G,in_size,1,spectral_radius)                              
    net.initialize(i_scaling,beta_scaling,edgeid_ezid_map,GOterms,GO_id_map,tau_list)
    printTime("SR", net.spectral_radius)
    
    readout_pos={gene:pos for pos,gene in enumerate(readout_dict.keys())}
    #Choose input and collect states
    printTime("Collecting states")
    net.collect_states_derivative(initLen, trainLen, testLen, tau_list,dt=0.001)
   
    if notebook and noise:
        printTime("Autocorrelation of generated noise")
        for tau in net.u_dict:
            printTime(tau)
            autocorr=autocorrelation(net.u_dict[tau])
            exponential_fitting(autocorr,exp_func)
            show()
            
    #train 
    for tau_test in tau_test_list:
        printTime("El traning se hace para {} y en el testing introducimos las taus en {}".format(tau_train,tau_test))
        net.run_simulation_readout(readout_dict,readout_pos,initLen,trainLen,testLen,n,tau_train,tau_test,tau_list,beta=1e-8)
    ## Mi in Go term
    #initialize arrays
        mi_go_by_gene={}
        nrmse_go_by_gene={}
        mi_not_go=[]
    
    #calculating Mi
        printTime("MI")
        for gene,pos in readout_pos.items():
            printTime(gene)
            for tau in tau_list:
                printTime("El input al que comparamos es tau={}".format(tau))
                plot(net.u_dict[tau][trainLen-n:trainLen+miLen-n])
                plot(net.Y[pos,0:miLen])
                show()
                printTime(miLen)
                mi=calc_MI_binning(net.u_dict[tau][trainLen-n:trainLen+miLen-n], net.Y[pos,0:miLen])
                nrmse_val=nrmse_calc(net.u_dict[tau][trainLen-n:trainLen+errorLen-n], net.Y[pos,0:errorLen])
                printTime(mi)
                printTime(nrmse_val)
                printTime()
                if tau==tau_train:
                    mi_go_by_gene[gene]=mi
                    nrmse_go_by_gene[gene]=nrmse_val
    return net,mi_go_by_gene,nrmse_go_by_gene