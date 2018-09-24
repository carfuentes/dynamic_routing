import networkx as nx
import numpy as np
from matplotlib.pyplot import *
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import scipy
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import datetime
from preprocessing_net import get_cyclic_net
from mutual_info import *
from nrmse_calc import nrmse, nrmse_n
from r_to_python import *
import entropy_estimators as ee
import scipy.spatial as ss
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import scipy

class ESN(object):
    def __init__(self, net, in_size, out_size, spectral_radius):
        self.net=net
        self.res_size= len(net.nodes())
        self.in_size=in_size
        self.out_size=out_size
        self.spectral_radius= spectral_radius
        self.W0=None
        self.W=None
        self.Win=dict()
        self.Wout=None
        self.X=dict()
        self.Y=None
        self.x=np.zeros((self.res_size,1))
        self.x0_e=np.random.rand(self.res_size)
        #NO SE SI ESTABA EL 10
        self.x0=np.insert(np.random.rand(self.res_size),0,[1.0,1.0,1.0])
        self.u0=0
        self.decay=np.random.gamma(5.22,0.017,size=self.res_size).reshape((self.res_size,1))
        self.u=None
        self.x_act=None
        self.dict_pos=None
        self.u_dict=dict()
        self.x_act_dict=dict()

   
    def build_adj_weighted_matrix(self,mapping):
        #NETWORK v2.0
        net=nx.relabel_nodes(self.net,mapping,copy=True)
        dict_pos=dict((node,pos) for (pos,node) in enumerate(net.nodes()))
        for edge in net.edges(data="mode", default=0):
            source,target,mode=edge
            if mode== "+":
                net[source][target]["weight"]= np.random.uniform(0,1) -0.5
            elif mode== "-":
                net[source][target]["weight"]= np.random.uniform(-1,0) -0.5
            elif mode== 0:
                net[source][target]["weight"]= np.random.uniform(-1,1)-0.5
        
        return nx.to_numpy_matrix(net),dict_pos
    
    def input_matrix_just_genes_GOterm(self,Win,GOterms,GO_id_map,mapping,i_scaling,tau_list):
        print(i_scaling)
        Win={} 
        Win[sum(tau_list)]=np.zeros((self.res_size,1+len(tau_list)))
        for index, GOterm in enumerate(GOterms):
            print(GOterm, tau_list[index])
            Win[tau_list[index]]=np.zeros((self.res_size,2))
            for gene in GO_id_map[GOterm]:
                Win[tau_list[index]][self.dict_pos[gene],1]=(np.random.uniform()) *i_scaling[index]
                Win[tau_list[index]][self.dict_pos[gene],0]=(np.random.uniform()) *beta_scaling[index]
                #Win[tau_list[index]][self.dict_pos[gene],index+1]=(np.random.uniform()) *i_scaling[index]
                Win[sum(tau_list)][self.dict_pos[gene],0]=(np.random.uniform()) *beta_scaling[index]
                Win[sum(tau_list)][self.dict_pos[gene],index+1]=(np.random.uniform()) *i_scaling[index]
        return Win
    
    def initialize(self,i_scaling,beta_scaling,mapping,GOterms,GO_id_map,tau_list): 
        np.random.seed(42)
        print("seed puesta")
        print("factor i= {}".format(i_scaling))
        print("factor beta={}".format(beta_scaling))
        
        self.W0, self.dict_pos=self.build_adj_weighted_matrix(mapping)
        self.res_size= self.W0.shape[0]
        self.W0 = np.squeeze(np.asarray(self.W0)) 
        radius = np.max(np.abs(np.linalg.eigvals(self.W0)))
        self.W= (self.spectral_radius/radius)*self.W0
        
        
        self.Win=self.input_matrix_just_genes_GOterm(self.Win,GOterms,GO_id_map,mapping,i_scaling,tau_list)
        return self.W
    
    
    def dx_act_dt(self, x,u,tau):
        x=x.reshape(self.res_size,1)
        x_act=self.decay*0.5*(np.tanh( np.dot( self.Win[tau], np.vstack((1,u)) ) + np.dot( self.W, x ) )+1) - (self.decay * x)
        return x_act.reshape(self.res_size)
    
    def colored_noise_euler_integration(self, x_0, u_0, t_stop, tau_list,dt=0.001):
        
    
        x={}
        u={}
        t = np.linspace(0, t_stop, int(t_stop/dt))
        
       
        
        for index,tau in enumerate(tau_list):
            np.random.seed(2+index)
            c=(1/tau)**2
            mu=np.exp(-dt/tau)
            sigma= sqrt( ((c * tau)/2) * (1-mu**2) )
            print("Generating {} noise".format(tau))
            x[tau]=np.zeros((len(t),self.res_size))
            x[tau][0,:]=x_0
            u[tau] = u_0 * np.ones_like(t)         
            for i in np.arange(0, len(t) - 1):
                u[tau][i+1] = u[tau][i]* mu + sigma * np.random.normal()
                x[tau][i+1,:] = x[tau][i,:] + dt * self.dx_act_dt(x[tau][i,:], u[tau][i],tau)
        
        print("Generating {} noise".format(sum(tau_list)))
        x[sum(tau_list)]=np.zeros((len(t),self.res_size))
        x[sum(tau_list)][0,:]=x_0
        for i in np.arange(0, len(t) - 1):
            u_concat=np.vstack(([u[tau][i] for tau in tau_list]))
            x[sum(tau_list)][i+1,:] = x[sum(tau_list)][i,:] + dt * self.dx_act_dt(x[sum(tau_list)][i,:], u_concat,sum(tau_list))
        return u,x
    

    
    def collect_states_derivative(self, init_len, train_len, test_len,tau_list,dt=0.001):
        t_stop=train_len+test_len
        indexes=[int(t/dt) for t in range(0,t_stop)]
       
        print("Collecting states with noise input...")
        u, x_act=self.colored_noise_euler_integration(self.x0_e, self.u0, t_stop, tau_list,dt)
        self.u_dict= {tau:u_tau[indexes] for tau,u_tau in u.items() }
        u=None
        self.x_act_dict={tau:x_tau[indexes] for tau,x_tau in x_act.items() }
        x_act=None
   
    

    def calculate_weights_readout(self,init_len, train_len, n, res_indexes,tau_train,tau_test, tau_list,beta=1e-8):
        print("{} a la que se esta entrando".format(tau_train))
        Y=np.array([self.u_dict[tau_train][init_len-n:train_len-n]])
        #indexes=list(range(0,self.in_size+1))+[index+self.in_size+1 for index in res_indexes]
        #indexes=[index+self.in_size+1 for index in res_indexes]
        print("xact entreno {}".format(sum(tau_list)))
        X=self.x_act_dict[sum(tau_list)][init_len-n:train_len-n,res_indexes].T
        print(X.shape)
        X_T=X.T
        Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(X,X_T) + beta * np.eye(len(res_indexes)))) #w= y*x_t*(x*x_t + beta*I)^-1
        return Wout
    
    def run_predictive_readout(self, test_len, train_len,indexes,Wout, tau_test):
        self.Win=None
        Y = np.zeros((test_len))
        for t in np.arange(train_len,train_len+test_len):
            #x_concat=self.x_act_dict[tau_test][t,indexes].reshape(self.x_act_dict[tau_test][t,indexes].shape[0],1)
            y = np.dot( Wout, self.x_act_dict[tau_test][t,indexes] )
            Y[t-train_len] = y
        return Y
    
    def run_simulation_readout(self,readout_dict,readout_pos,init_len,train_len,test_len,n,tau_train,tau_test,tau_list,beta=1e-8):
        self.Y = np.zeros((len(readout_dict),test_len))
        for gene_readout in readout_dict:
            print("{} output being calculated".format(gene_readout))
            indexes=[self.dict_pos[gene_res] for gene_res in readout_dict[gene_readout]]
            
            print("{} training to noise and neuron activation with tau_train= {}".format(gene_readout,tau_train))
            Wout=self.calculate_weights_readout(init_len, train_len, n, indexes, tau_train,tau_test,tau_list,beta=1e-8)
      
            print("{gene} testing with neuron activation from tau_test= {tau} (input tau {tau})".format(gene=gene_readout,tau=tau_test))
            Y=self.run_predictive_readout(test_len, train_len,indexes,Wout,tau_test)
            self.Y[readout_pos[gene_readout],:]=Y
            
        return self.Y

def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    result=result[int(result.size/2):]
    return result/result[0]

def test_readout(directory,file_path,in_size, tau_list,tau_test_list,tau_train,spectral_radius, i_scaling,beta_scaling,GOterms,readout_dict,n, noise, euler=True, save=False, notebook=True):
    #init
    print(file_path)
    print(miLen)
    filename=file_path[file_path.index("list")+5:file_path.index(".csv")]

    
    print("Running network...")
    G=get_cyclic_net(os.path.join(directory, file_path))
    
    GO_id_map=get_r_dictionaries("test.txt",mapping=False)
    edgeid_ezid_map=get_r_dictionaries("mapping_id_to_entrez.txt")
    mapping=mapping_relabel_function(G,edgeid_ezid_map)
   
    for key,values in GO_id_map.items():
        GO_id_map[key]=set([mapping[value] for value in values])

    
    #Run network
    net=ESN(G,in_size,1,spectral_radius)                              
    net.initialize(i_scaling,beta_scaling,edgeid_ezid_map,GOterms,GO_id_map,tau_list)
    print("SR", net.spectral_radius)
    
    readout_pos={gene:pos for pos,gene in enumerate(all_readout.keys())}
    #Choose input and collect states
    print("Collecting states")
    net.collect_states_derivative(initLen, trainLen, testLen, tau_list,dt=0.001)
   
    if notebook and noise:
        print("Autocorrelation of generated noise")
        for tau in net.u_dict:
            print(tau)
            autocorr=autocorrelation(net.u_dict[tau])
            exponential_fitting(autocorr,exp_func)
            show()
            
    #train 
    for tau_test in tau_test_list:
        print("El traning se hace para {} y en el testing introducimos las taus en {}".format(tau_train,tau_test))
        net.run_simulation_readout(readout_dict,readout_pos,initLen,trainLen,testLen,n,tau_train,tau_test,tau_list,beta=1e-8)
    ## Mi in Go term
    #initialize arrays
        mi_go_by_gene={}
        nrmse_go_by_gene={}
        mi_not_go=[]
    
    #calculating Mi
        print("MI")
        for gene,pos in readout_pos.items():
            print(gene)
            for tau in tau_list:
                print("El input al que comparamos es tau={}".format(tau))
                plot(net.u_dict[tau][trainLen-n:trainLen+miLen-n])
                plot(net.Y[pos,0:miLen])
                show()
                print(miLen)
                mi=calc_MI_binning(net.u_dict[tau][trainLen-n:trainLen+miLen-n], net.Y[pos,0:miLen])
                nrmse_val=nrmse(net.u_dict[tau][trainLen-n:trainLen+errorLen-n], net.Y[pos,0:errorLen])
                print(mi)
                print(nrmse_val)
                print()
                if tau==tau_train:
                    mi_go_by_gene[gene]=mi
                    nrmse_go_by_gene[gene]=nrmse_val
    return net,mi_go_by_gene,nrmse_go_by_gene

def exp_func(x, a, b,c):
    return a * np.exp(-1/b * x)+c

def exponential_fitting(x, func,p0=None,start_point=0,MI=False):
    if MI: #x is MI
        xdata=np.array(list(x.keys())[start_point:])
        ydata=np.array(list(x.values())[start_point:])
    else: #x is autocorr
        a=x[:np.argmax(x<0)]
        ydata=a
        xdata=np.arange(a.shape[0])
    popt, pcov = curve_fit(func, xdata, ydata,p0)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot(xdata, ydata, '-o', label='data')
    plot(xdata, func(xdata, *popt), 'r-', label="fit")
    #plot(xdata, exp_func(xdata, popt[0],tau,popt[2]), 'b-', label="expected fit")
    xlabel('n',fontsize=20)
    ylabel('MImax',fontsize=20)
    legend(loc=1, prop={'size': 20})
    print(popt)
    return popt,xdata,ydata



errorLen = 500
trainLen=2000
testLen=1000
miLen=500
initLen=200

csv_files=['network_edge_list_ENCODE.csv', 'network_edge_list_modENCODE.csv', 'network_edge_list_YEASTRACT.csv', 'network_edge_list_EcoCyc.csv', 'network_edge_list_DBTBS.csv']

file=csv_files[0]

## N range
i_max=100
n_max=100
n_range=np.array(range(n_max))


#parameters
GO_005= np.load('GO_005.npy').item()
GO_007= np.load('GO_007.npy').item()
GO_007_005= np.load('GO_007and005.npy').item()
all_readout={**GO_005,**GO_007,**GO_007_005}

i_scaling=[1,1]
beta_scaling=[0.01,0.01]
tau_list=[20,10]
mi_histogram={}
nrmse_histogram={}
nets={}

for tau_train,tau_test in zip([10,20,10,20],[[10],[20],[30],[30]]):
    print(tau_train, "tau_train", tau_test, "tau_test")
    #np.random.seed(7)
    nets[(tau_train,tau_test[0])], mi_histogram[(tau_train,tau_test[0])],nrmse_histogram[(tau_train,tau_test[0])]=test_readout("Dataset1",file, 1,tau_list,tau_test,tau_train, 0.95, i_scaling,beta_scaling,['GO:0051591','GO:0070317'],all_readout,10, noise=True, euler=True, save=False, notebook=True)

