import numpy as np
from dgcnn.dg_encoding import *
from pymatgen.core.structure import Structure
import os
from collections import Counter
from math import cos, sin, atan2, sqrt, pi ,radians, degrees
import re
import string

"""This is where inter layer graph is constructed. """


# 分割结构层数
def countlayers(name, crystal, layer_thresthold, surface_layerNum, inter_layerNum):
    if name.split('_')[-1][0].upper() == 'S':
        singleAtom_Index = get_singleAtom_index(name, crystal)
        crystal_del = crystal.copy()
        crystal_del.remove_sites([singleAtom_Index]) 
        fraction_coords_p = crystal.frac_coords
        fraction_coords = crystal_del.frac_coords
    elif name.split('_')[-1][0].upper() == 'D':
        singleAtom_Index = get_singleAtom_index(name, crystal)
        crystal_del = crystal.copy()
        crystal_del.remove_sites([singleAtom_Index]) 
        fraction_coords_p = crystal.frac_coords
        fraction_coords = crystal_del.frac_coords
    elif name.split('_')[-1][0].upper() == 'B':
        fraction_coords = crystal.frac_coords
        layerscount_bare = getlayers(crystal, layer_thresthold)
        layerscount_top = list(layerscount_bare.values())[-1]
        x_cartesian, y_cartesian, z_cartesian = get_XYZ_cartesian(fraction_coords)
        singleAtom_Index = get_support_index(layerscount_top, x_cartesian, y_cartesian, z_cartesian)
        crystal_del = crystal.copy()
        crystal_del.remove_sites([singleAtom_Index]) 
        fraction_coords_p = crystal.frac_coords
        fraction_coords = crystal_del.frac_coords
    thresthold = layer_thresthold
    num_atoms = len(fraction_coords)
    layer_num = surface_layerNum + inter_layerNum
    layerscount = getlayers(crystal_del, thresthold)
    if name.split('_')[-1][0].upper() == 'S':
        surface_layerIndex_del = dict_slice(layerscount, layer_num - surface_layerNum, layer_num, 'layer')
        surface_layerIndex, crystal_new = insert_singleAtom(crystal, surface_layerIndex_del, singleAtom_Index, fraction_coords_p, fraction_coords)
    elif name.split('_')[-1][0].upper() == 'D':
        surface_layerIndex_del = dict_slice(layerscount, layer_num - surface_layerNum, layer_num, 'layer')
        surface_layerIndex, crystal_new = insert_singleAtom(crystal, surface_layerIndex_del, singleAtom_Index, fraction_coords_p, fraction_coords)
    elif name.split('_')[-1][0].upper() == 'B':
        surface_layerIndex_del = dict_slice(layerscount, layer_num - surface_layerNum, layer_num, 'layer')
        surface_layerIndex, crystal_new = insert_singleAtom(crystal, surface_layerIndex_del, singleAtom_Index, fraction_coords_p, fraction_coords)
    inter_layerIndex_del = dict_slice(layerscount, 0, inter_layerNum, 'layer')
    dot_layerIndex = dict_slice(layerscount, layer_num + 1, singleAtom_Index, 'dot')
    total_layerIndex = surface_layerIndex.copy()
    total_layerIndex.update(inter_layerIndex_del)
    return surface_layerIndex_del, inter_layerIndex_del, dot_layerIndex, total_layerIndex, crystal_new, crystal_del

# 按不同元素分割层数
def getlayers(crystal_del, layer_thresthold):
    element_list = crystal_del.formula.split(' ')
    element_list.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    element_list_noNum = list(map(lambda x: x.strip(string.digits), element_list))
    countlayers_list = {}
    for i in element_list_noNum:
        element_ = []
        dict_ = {}
        for j in range(crystal_del.num_sites):
            if i in crystal_del.sites[j]:
                element_.append(j)
        dict_ = {i : element_}
        countlayers_list.update(dict_)
    ##
    thresthold = layer_thresthold
    layerscount_total = {}
    layers_count = 0
    element_count = 0
    for i in range(len(countlayers_list)):
        list_ = list(countlayers_list.values())[i]
        num_atoms = len(crystal_del.frac_coords[list_])
        layer_num = int(LAYER_DISTRIB[i])
        x_cartesian = []
        y_cartesian = []
        z_cartesian = []
        for j in list_: 
            x_cartesian.append(float(crystal_del.frac_coords[j][0]))
            y_cartesian.append(float(crystal_del.frac_coords[j][1]))
            z_cartesian.append(float(crystal_del.frac_coords[j][2]))  
        layerscount_initial = determinelayers(z_cartesian, thresthold, layers_count, element_count)
        ##
        if len(layerscount_initial) == layer_num:
            layerscount = layerscount_initial
        elif len(layerscount_initial) > layer_num:
            while len(layerscount_initial) > layer_num:
                thresthold = thresthold + 0.001
                layerscount_initial = determinelayers(z_cartesian, thresthold, layers_count, element_count)
                if len(layerscount_initial) == layer_num:
                    layerscount = layerscount_initial
                    break
        elif len(layerscount_initial) < layer_num:
            while len(layerscount_initial) < layer_num:
                thresthold = thresthold - 0.001
                layerscount_initial = determinelayers(z_cartesian, thresthold, layers_count, element_count)
                if len(layerscount_initial) == layer_num:
                    layerscount = layerscount_initial
                    break
        layers_count = layers_count + layer_num
        element_count = element_count + num_atoms
        layerscount_total.update(layerscount_initial) 
    return layerscount_total

# 确定层数
def determinelayers(z_cartesian, thresthold, layers_count, element_count):
    seq = sorted(z_cartesian)
    min = seq[0]
    layerscount = {}
    sets = [min]
    for j in range(len(seq)):
        if abs(seq[j]-min) >= thresthold:
            min = seq[j]
            sets.append(min)
    for i in range(1,len(sets)+1):
        layerscount[i+layers_count] = []            
        if i > 1:
            for k in range(len(z_cartesian)):   
                if abs(z_cartesian[k]-sets[i-1]) <= thresthold and not abs(z_cartesian[k]-sets[i-2]) <= thresthold:
                    layerscount[i+layers_count].append(k+element_count)            
        else:
            for k in range(len(z_cartesian)):   
                if abs(z_cartesian[k]-sets[i-1]) <= thresthold:
                    layerscount[i+layers_count].append(k+element_count)
    return layerscount

# 字典切片
def dict_slice(adict, start, end, type_):
    if type_ == 'layer':
        keys = adict.keys()
        dict_slices = {}
        for k in list(keys)[start:end]:
            dict_slices[k] = adict[k]
    elif type_ == 'dot':
        dict_slices = {}
        if type(end) == list:
            dict_slices.update({start: end})
        elif type(end) == int:
            dict_slices.update({start: [end]})
    return dict_slices

#定位单原子序号
def get_singleAtom_index(name, crystal):
    if name.split('_')[0] != name.split('_')[-2]:
        singleAtom_index = sum(int(i) for i in range(len(crystal.sites)) if name.split('_')[-2] in crystal.sites[i])
    elif name.split('_')[0] == name.split('_')[-2]:
        singleAtom_list = [i for i in range(len(crystal.sites)) if name.split('_')[-2] in crystal.sites[i]]
        high = 0
        singleAtom_index = 0
        for j in singleAtom_list:
            if crystal.frac_coords[j][2] > high:
                high = crystal.frac_coords[j][2]
                singleAtom_index = j
    return singleAtom_index

#定位单原子插入表层位置
def insert_singleAtom(crystal, surface_layerIndex_p, singleAtom_Index, fraction_coords_p, fraction_coords):
    z_index = 0
    z_diff = float('inf')
    z_doping = fraction_coords_p[singleAtom_Index][2]
    crystal_del = crystal.copy()
    crystal_del.remove_sites([singleAtom_Index]) 
    max_num = len(crystal_del)
    for i in surface_layerIndex_p.keys():
        z_list = []
        for j in surface_layerIndex_p[i]:
            z_list.append(fraction_coords[j][2])
        z_list_mean = np.mean(z_list)        
        if z_diff > abs(z_list_mean-z_doping):
            z_diff = abs(z_list_mean-z_doping)
            z_index = i
    surface_layerIndex = surface_layerIndex_p.copy()
    surface_layerIndex[z_index].append(max_num)
    crystal_new = crystal_del.copy()
    crystal_new.append(crystal[singleAtom_Index].species, crystal[singleAtom_Index].frac_coords)
    return surface_layerIndex, crystal_new

# 列表索引
def list_indice(x,y):
    return [x[i] for i in y] #x为需要索引的列表，y为索引列表

#确定中心原子序号
def get_support_index(sup_layer, x_cartesian, y_cartesian, z_cartesian):
    x_cartesian_sup = list_indice(x_cartesian, sup_layer)
    y_cartesian_sup = list_indice(y_cartesian, sup_layer)
    z_cartesian_sup = list_indice(z_cartesian, sup_layer)
    position_xy = list(zip(x_cartesian_sup, y_cartesian_sup))
    position_center = list(center_geolocation(position_xy))
    min_index = get_min_index(position_center, x_cartesian_sup, y_cartesian_sup)
    center_atom_index = sup_layer[min_index]
    return center_atom_index

# 确定最上层原子中心
def center_geolocation(geolocations):
    x = 0
    y = 0
    z = 0
    lenth = len(geolocations)
    for lon, lat in geolocations:
        lon = radians(float(lon))
        lat = radians(float(lat))
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)    
    x = float(x / lenth)
    y = float(y / lenth)
    z = float(z / lenth)
    center = (degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))))    
    return center

# 确定中心位置与上层原子的最短距离
def get_min_index(position_center, x_cartesian_top, y_cartesian_top):
    position_xy = list(zip(x_cartesian_top, y_cartesian_top))
    min_distance = float('inf')
    min_index = []
    for i in range(len(position_xy)):
        distance = sqrt((position_center[0]-position_xy[i][0])**2 + (position_center[1]-position_xy[i][1])**2)
        if distance < min_distance:
            min_distance = distance
            min_index = i    
    return min_index

# 获取XYZ坐标
def get_XYZ_cartesian(fraction_coords):
    num_atoms = len(fraction_coords)
    x_cartesian = []
    y_cartesian = []
    z_cartesian = []
    for i in range(num_atoms): 
        x_cartesian.append(float(fraction_coords[i][0]))
        y_cartesian.append(float(fraction_coords[i][1]))
        z_cartesian.append(float(fraction_coords[i][2]))  
    return x_cartesian, y_cartesian, z_cartesian

def unit_cell_expansion_slab(lattices, matrix):
    num = len(lattices)
    expanded_lattices = np.zeros([3,num*9],dtype=np.float32)
    coeff = [0, -1, 1]
    count = 0

    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])

    for i in coeff:
        for j in coeff:
            new_lattices = np.copy(lattices)
            translation = v1*i + v2*j
            new_lattices = np.transpose(np.matmul(new_lattices + translation, matrix))
            expanded_lattices[:,count*num:(count+1)*num] = new_lattices
            count = count + 1
    return expanded_lattices

def find_neighbor(lattices, expanded_lattices, elements, cut_rds, nbr_type):
    lattice_num = len(lattices[0])
    expanded_lattice_num = len(expanded_lattices[0])
    connectivity = []
    distances = []
    tolerance = 1.5
    for i in range(lattice_num):
        neighbor_num = 0
        for j in range(expanded_lattice_num):
            if i==j:
                continue

            cond1 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < 6
            cond2 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < (atom_features[elements[i]]['Ra'] + atom_features[elements[j%len(elements)]]['Ra'] + tolerance)
            if nbr_type == 'T':
                if cond1 and cond2:
                    connectivity.append([i,j])
                    distance = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j])

                    distances.append(distance)
                    neighbor_num = neighbor_num + 1
            elif nbr_type == 'F':
                if cond1:
                    connectivity.append([i,j])
                    distance = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j])

                    distances.append(distance)
                    neighbor_num = neighbor_num + 1

    return connectivity, distances

def bond_construction(elements,connectivity,distances, features, CATEGORY_NUM, TOTAL_CATEGORY_NUM, NEIGHBOR_CATEGORY_NUM):
    BOND_CATEGORY_NUM = TOTAL_CATEGORY_NUM - 2 * CATEGORY_NUM

    bond_vectors = []
    neighbor_indices = []
    atom_num = len(elements)
    neighbor_num = np.zeros(atom_num, dtype=np.int32)
    for connection in connectivity:
        neighbor_num[connection[0]] += 1
    count = 0
    for i in range(len(elements)):
        bond_vector = np.zeros([neighbor_num[i], BOND_CATEGORY_NUM], dtype=np.float32)
        neighbor_index = np.zeros([neighbor_num[i], 2], dtype=np.float32)
        for j in range(neighbor_num[i]):
            # neighbor_atom = elements[connectivity[count][1]%atom_num]
            # neighbor_atom = connectivity[count][1] % atom_num
            neighbor_index[j][0] = connectivity[count][0] % atom_num
            neighbor_index[j][1] = connectivity[count][1] % atom_num
            bond_vector[j] = bond_encoding(distances[count])
            count += 1
        bond_vectors.append(bond_vector)
        neighbor_indices.append(neighbor_index)

    return bond_vectors, neighbor_indices

def poscar_to_graph(name, poscar_dir, formula, parameters, cut_rds, nbr_type):
    """This function converts POSCAR file to graph structure."""
    poscar_dir = poscar_dir + '/'
    if formula == 'cif':
        crystal_ori = Structure.from_file(os.path.join(poscar_dir,
                                           name+'.cif'))#获取cif结构
    elif formula == 'vasp':
        crystal_ori = Structure.from_file(os.path.join(poscar_dir,
                                           name+'.vasp'))#获取vasp结构 

    num_surface_layer = int(parameters[0])
    num_inter_layer = int(parameters[1])
    layer_thresthold = float(parameters[2])
    
    _, inter_layer_index, _, _, _, crystal = countlayers(name, crystal_ori, layer_thresthold, num_surface_layer, num_inter_layer)
    inter_layer_index2 = sum(sorted(inter_layer_index.values()),[])
#     inter_layer_index2 = np.array([value for value in sorted(inter_layer_index.values())]).reshape(-1)
#     print(crystal.species[i])
    element_inter = [str(crystal.species[i]) for i in inter_layer_index2]
    
    a = crystal.lattice.matrix[0].tolist()
    b = crystal.lattice.matrix[1].tolist()
    c = crystal.lattice.matrix[2].tolist()
    
    trans_matrix = np.zeros([3,3], dtype=np.float32)
    trans_matrix[0] = a
    trans_matrix[1] = b
    trans_matrix[2] = c

    inter_atom_num = [int(i) for i in Counter(element_inter).values()]
    inter_total_atom_num = sum(inter_atom_num)
    
    inter_atoms = crystal.frac_coords[inter_layer_index2]
    lattices = np.transpose(np.matmul(inter_atoms, trans_matrix))
    expanded_lattices = unit_cell_expansion_slab(inter_atoms, trans_matrix)
    connectivity, distances = find_neighbor(lattices, expanded_lattices, element_inter, cut_rds, nbr_type)
    
    atom_vectors = np.zeros([inter_total_atom_num, CATEGORY_NUM], dtype=np.float32)
    bond_num = np.zeros([inter_total_atom_num],dtype=np.int32)
    for i in range(inter_total_atom_num):
        atom_vectors[i] = atom_encoding(element_inter[i])
    bond_vectors, neighbor_indices = bond_construction(element_inter, connectivity, distances, features, CATEGORY_NUM, TOTAL_CATEGORY_NUM, NEIGHBOR_CATEGORY_NUM)
    for i in range(len(bond_vectors)):
        bond_num[i] = bond_vectors[i].shape[0]
#     print(atom_vectors, '**', bond_vectors,  '**', neighbor_indices,  '**', name)
    return atom_vectors, bond_vectors, neighbor_indices