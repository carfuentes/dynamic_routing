#noise parameters
tau=10
c_n=0.01

#Time lengths
errorLen = 500
trainLen=10000
testLen=5000
miLen=990
initLen=200



#Net parameters
i_scaling=[1,1]
beta_scaling=[0.05,0.05]
tau_list=[1,10]
tau_test=[1,10,sum(tau_list)]
tau_train=1
 
#Test parametes
nrmse= True
noise=True
spline=True
euler=True
save=False
single=True
notebook=False


# TRAINING AND TEST LENGHT

csv_files=['network_edge_list_ENCODE.csv', 'network_edge_list_modENCODE.csv', 'network_edge_list_YEASTRACT.csv', 'network_edge_list_EcoCyc.csv', 'network_edge_list_DBTBS.csv']
file=csv_files[0]