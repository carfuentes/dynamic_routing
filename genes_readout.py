import numpy as np

#Load varibales from files
GO_005= np.load('GO_005.npy').item()
GO_007= np.load('GO_007.npy').item()
GO_007_005= np.load('GO_007and005.npy').item()


all_readout={**GO_005,**GO_007,**GO_007_005}