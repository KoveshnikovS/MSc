import pandas as pd
import h5py
import numpy as np

def load_data() -> list:
    """### Loading raw data from .mat files
    Load file from the folder structured as\n
    \\IEEE
        files
    The time put as index, the data column is the current.
    #### Return:
    - list of data experiments -- data[class[0BRB-4BRB]][load[.125-1]]
    """
    
    exp_data=[[j for j in range(8)] for i in range(5)]
    load_levels=['torque05','torque10','torque15','torque20','torque25','torque30','torque35','torque40']
    states=['rs','r1b','r2b','r3b','r4b']
    i=0
    for state in states:
        j=0
        for level in load_levels:
            mat=h5py.File('../IEEE/struct_'+state+'_R1.mat','r')
            # print(mat,level,state)
            dataset_refname=mat[state][level]['Ia'][0]
            data=mat[dataset_refname[0]]
            data=np.ravel(data)
            data=data[120000:]
            d={'Current, A':data}
            exp_data[i][j]=pd.DataFrame(data=d,index=np.ravel([k/50000 for k in range (len(data))]))
            exp_data[i][j].index.name='Time, s'
            j+=1
        mat.close()
        i+=1
    
    return exp_data
