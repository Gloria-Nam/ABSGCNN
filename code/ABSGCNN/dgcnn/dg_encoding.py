import numpy as np
import pandas as pd

feature_list = {'Z' : 118, 'Vl' : 32, 'Pc' : 12, 'Pc18' : 18, 'Pr' : 6, 'Uo' : 22, 'Eg' : 10, 'Ip1' : 10, 'Ip2' : 10, 'Ea' : 10, 'Homo' : 10, 'Lumo' : 10, 'Ros' : 10, 'Rop' : 10, 'Rod' : 10, 'Ra' : 10, 'Rv' : 10, 'Rc' : 10, 'Plz' : 10, 'M' : 10, 'Mel' : 10, 'Boi' : 10, 'de' : 10, 'Zeff' : 10, 'Fus' : 10, 'Vap' : 10, 'Cap' : 10}

data_bare = pd.read_csv('ElementInfo.csv')
data = data_bare.copy()
data.fillna(data_bare.mean(numeric_only=True), inplace=True)
data_onehot = data.copy()
for j in data.columns[1:]:
    one_hot = pd.qcut(data[j], feature_list[j], labels=False, retbins=False, precision=3, duplicates='drop')
    data_onehot[j] = one_hot
atom_features = {}
for i in range(data_onehot.shape[0]):
    features_lists = {}
    for j in range(data_onehot.shape[1]-1):
        if np.isnan(data_onehot.iloc[i,j+1]):
            features_list = {data_onehot.columns[j+1] : data_onehot.iloc[i,j+1]}
        else:
            features_list = {data_onehot.columns[j+1] : int(data_onehot.iloc[i,j+1])}
        features_lists.update(features_list)
    atom_feature = {data_onehot.iloc[i,0] : features_lists}
    atom_features.update(atom_feature)

with open('ExtraSelectInfo.csv','r') as f2:  ## read CSV file and make dictionary
    features_cg = f2.readline().strip().split(sep=',')[1:]
    features_sg = f2.readline().strip().split(sep=',')[1:]
    features_dg = f2.readline().strip().split(sep=',')[1:]
    while '' in features_dg:
        features_dg.remove('')
    LAYER_DISTRIB = f2.readline().strip().split(sep=',')[1:]
    while '' in LAYER_DISTRIB: # 判断是否有空值在列表中
        LAYER_DISTRIB.remove('')
        
features = features_dg
CATEGORY_NUM = 0 ## Total number of category
for i in range(len(features)):
    CATEGORY_NUM += feature_list[features[i]]

#BOND_CATEGORY_NUM = 20
BOND_CATEGORY_NUM = 1

NEIGHBOR_CATEGORY_NUM = CATEGORY_NUM + BOND_CATEGORY_NUM  ## The total catergoty number of neighbor atom
TOTAL_CATEGORY_NUM = CATEGORY_NUM + NEIGHBOR_CATEGORY_NUM  ## atom + neighbor atom

def atom_encoding(a):
    c = 0
    a_one_hot = np.zeros(CATEGORY_NUM, dtype=np.int32)  ## make [0 0 0 .. ] vector, length = CATEGORY_NUM
    for i in range(len(features)):
        index = 0  ## The category which elements belong to
        ## atom_features = The dictonary of feature values
        if features[i] =='Z':
            index = atom_features[a]['Z'] ## Ex. atom_features[a]['Z'] = 1 -> index = 0
        elif features[i] == 'Vl':
            index = atom_features[a]['Vl']
        elif features[i] == 'Pc':
            index = atom_features[a]['Pc']
        elif features[i] == 'Pc18':
            index = atom_features[a]['Pc18']
        elif features[i] == 'Pr':
            index = atom_features[a]['Pr'] 
        elif features[i] == 'Uo':
            index = atom_features[a]['Uo'] 
        elif features[i] == 'Eg':
            index = atom_features[a]['Eg'] 
        elif features[i] == 'Ip1':
            index = atom_features[a]['Ip1'] 
        elif features[i] == 'Ip2':
            index = atom_features[a]['Ip2']    
        elif features[i] == 'Ea':
            index = atom_features[a]['Ea'] 
        elif features[i] == 'Homo':
            index = atom_features[a]['Homo']             
        elif features[i] == 'Lumo':
            index = atom_features[a]['Lumo']  
        elif features[i] == 'Ros':
            index = atom_features[a]['Ros'] 
        elif features[i] == 'Rop':
            index = atom_features[a]['Rop']             
        elif features[i] == 'Rod':
            index = atom_features[a]['Rod']             
        elif features[i] == 'Ra':
            index = atom_features[a]['Ra']             
        elif features[i] == 'Rv':
            index = atom_features[a]['Rv'] 
        elif features[i] == 'Rc':
            index = atom_features[a]['Rc']            
        elif features[i] == 'Plz':
            index = atom_features[a]['Plz'] 
        elif features[i] == 'M':
            index = atom_features[a]['M']             
        elif features[i] == 'Mel':
            index = atom_features[a]['Mel']  
        elif features[i] == 'Boi':
            index = atom_features[a]['Boi']             
        elif features[i] == 'de':
            index = atom_features[a]['de']     
        elif features[i] == 'Zeff':
            index = atom_features[a]['Zeff'] 
        elif features[i] == 'Fus':
            index = atom_features[a]['Fus']             
        elif features[i] == 'Vap':
            index = atom_features[a]['Vap'] 
        elif features[i] == 'Cap':
            index = atom_features[a]['Cap'] 
        index = index + c
        a_one_hot[index] = 1
        c = c + feature_list[features[i]]
#     print(a, a_one_hot, len(a_one_hot))
    return a_one_hot

# def bond_encoding(d):
#     d_one_hot = np.zeros(BOND_CATEGORY_NUM, dtype=np.float32)
#     for i in range(BOND_CATEGORY_NUM):
#         d_one_hot[i] = 1
#     return d_one_hot

def bond_encoding(d, cut_rds, step, var=None):
    if var is None:
        var = step
    bond_filter = np.arange(0, cut_rds+step, step)
    distance_gdf = np.exp(-(d[..., np.newaxis] - bond_filter)**2 / var**2)
    return distance_gdf