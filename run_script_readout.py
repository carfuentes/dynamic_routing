from test_functions import test_readout
from genes_readout import *
from parameters import *

i_scaling=[1,1]
beta_scaling=[0.01,0.01]
tau_list=[20,10]
mi_histogram={}
nrmse_histogram={}
nets={}

for tau_train,tau_test in zip([10,20,10,20],[[10],[20],[30],[30]]):
    print(tau_train, "tau_train", tau_test, "tau_test")
    #np.random.seed(42)
    nets[(tau_train,tau_test[0])], mi_histogram[(tau_train,tau_test[0])],nrmse_histogram[(tau_train,tau_test[0])]=test_readout("Dataset1",file, 1,tau_list,tau_test,tau_train, 0.95, i_scaling,beta_scaling,['GO:0051591','GO:0070317'],all_readout,10, noise=True, euler=True, save=False, notebook=True)