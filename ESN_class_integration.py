import networkx as nx
import numpy as np
import random as rand
import scipy
from preprocessing_net import get_cyclic_net
from math import sqrt


from parameters import *


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
        print([Win[taau][3,0] for taau in Win])
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
            #np.random.seed(2+index)
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
    