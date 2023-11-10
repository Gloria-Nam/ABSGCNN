import dgcnn.dg_dot as dot
import dgcnn.dg_surface as surface
import dgcnn.dg_inter as inter
import numpy as np
from dgcnn.dg_encoding import *

""" This is where input graph structures are mathematically converted to 
    vector format."""

def data_preparation(X, root_dir, structures_formula, layer_parameters, cut_rds, nbr_type):
    data_num = len(X)
    cut_rds_dot, cut_rds_surface, cut_rds_inter = cut_rds[0], cut_rds[1], cut_rds[2]
    nbr_type_dot, nbr_type_surface, nbr_type_inter = nbr_type[0], nbr_type[1], nbr_type[2]
    # dot part
    atom_number_dot = 1
    atom_number_nbr_dot = 0
    bond_number_dot = 0

    for i in range(data_num):
        atom_vectors_dot, atom_vectors_nbr_dot, bond_vectors_dot, neighbor_dot_index = dot.poscar_to_graph(X[i], root_dir, structures_formula, layer_parameters, cut_rds_dot, nbr_type_dot)
        l = atom_number_dot
        n = atom_vectors_nbr_dot.shape[0]
        m = 0
        if atom_number_nbr_dot < n:
            atom_number_nbr_dot = n
        for j in range(l):
            m = bond_vectors_dot[j].shape[0]
            if bond_number_dot < m:
                bond_number_dot = m            
#     print("dot part : (atom,  bond) : ", atom_number_dot, bond_number_dot)
    BOND_CATEGORY_NUM_dot = np.array(bond_vectors_dot).shape[-1]
    atoms_dot = np.zeros([data_num, atom_number_dot, CATEGORY_NUM], dtype=np.int32)
    atoms_nbr_dot = np.zeros([data_num, atom_number_nbr_dot, CATEGORY_NUM], dtype=np.int32)
    bonds_dot = np.zeros([data_num, atom_number_dot, bond_number_dot, BOND_CATEGORY_NUM_dot], dtype=np.float32)
    bonds_dot_index1 = np.full([data_num, atom_number_dot, bond_number_dot], -1, dtype=np.int32)
    bonds_dot_index2 = np.full([data_num, atom_number_dot, bond_number_dot], -1, dtype=np.int32)

    atom_num_nbr_dot = np.zeros([data_num], dtype=np.int32)

    for i in range(data_num):
        atom_vectors_dot, atom_vectors_nbr_dot, bond_vectors_dot, neighbor_dot_index = dot.poscar_to_graph(X[i], root_dir, structures_formula, layer_parameters, cut_rds_dot, nbr_type_dot)
        l = atom_number_dot
        n = atom_vectors_nbr_dot.shape[0]
        atoms_dot[i] = atom_vectors_dot
        atoms_nbr_dot[i][:n] = atom_vectors_nbr_dot
        atom_num_nbr_dot[i] = n

        for j in range(l):
            m = bond_vectors_dot[j].shape[0]
            for k in range(m):
                bonds_dot[i][j][k] = bond_vectors_dot[j][k]
                bonds_dot_index1[i][j][k] = neighbor_dot_index[j][k][0] 
                bonds_dot_index2[i][j][k] = neighbor_dot_index[j][k][1] 
    
    # inter part
    atom_number_inter = 0
    bond_number_inter = 0

    for i in range(data_num):
        atom_vectors_inter, bond_vectors_inter, neighbor_inter_index = inter.poscar_to_graph(X[i], root_dir, structures_formula, layer_parameters, cut_rds_inter, nbr_type_inter)
        n = atom_vectors_inter.shape[0]
        m = 0
        if atom_number_inter < n:
            atom_number_inter = n
        for j in range(n):

            m = bond_vectors_inter[j].shape[0]
            if bond_number_inter < m:
                bond_number_inter = m
#     print("inter part: (atom,  bond) : ", atom_number_inter, bond_number_inter)
    BOND_CATEGORY_NUM_inter = np.array(bond_vectors_inter[0],dtype=object).shape[-1]
    atoms_inter = np.zeros([data_num, atom_number_inter, CATEGORY_NUM], dtype=np.int32)
    bonds_inter = np.zeros([data_num, atom_number_inter, bond_number_inter, BOND_CATEGORY_NUM_inter], dtype=np.float32)
    bonds_inter_index1 = np.full([data_num, atom_number_inter, bond_number_inter], -1, dtype=np.int32)
    bonds_inter_index2 = np.full([data_num, atom_number_inter, bond_number_inter], -1, dtype=np.int32)
    atom_num_inter = np.zeros([data_num], dtype=np.int32)
    for i in range(data_num):
        atom_vectors_inter, bond_vectors_inter, neighbor_inter_index = inter.poscar_to_graph(X[i], root_dir, structures_formula, layer_parameters, cut_rds_inter, nbr_type_inter)
        n = atom_vectors_inter.shape[0]
        atoms_inter[i][:n] = atom_vectors_inter
        atom_num_inter[i] = n

        for j in range(n):
            m = bond_vectors_inter[j].shape[0]
            for k in range(m):
                bonds_inter[i][j][k] = bond_vectors_inter[j][k]
                bonds_inter_index1[i][j][k] = neighbor_inter_index[j][k][0] 
                bonds_inter_index2[i][j][k] = neighbor_inter_index[j][k][1] 
#                 for l in range(BOND_CATEGORY_NUM_inter):
#                     bonds_inter[i][j][k][l] = bond_vectors_inter[j][k][l]
                
    # surface part
    atom_number_surface = 0
    bond_number_surface = 0
    
    for i in range(data_num):
        atom_vectors_surface, bond_vectors_surface, neighbor_surface_index = surface.poscar_to_graph(X[i], root_dir, structures_formula, layer_parameters, cut_rds_surface, nbr_type_surface)
        n = atom_vectors_surface.shape[0]
        m = 0
        if atom_number_surface < n:
            atom_number_surface = n
        for j in range(n):
    
            m = bond_vectors_surface[j].shape[0]
    
            if bond_number_surface < m:
                bond_number_surface = m
#     print("surface part: (atom, bond) : ", atom_number_surface, bond_number_surface)
    BOND_CATEGORY_NUM_surface = np.array(bond_vectors_surface[0]).shape[-1]
    atoms_surface = np.zeros([data_num, atom_number_surface, CATEGORY_NUM], dtype=np.int32)
    bonds_surface = np.zeros([data_num, atom_number_surface, bond_number_surface, BOND_CATEGORY_NUM_surface], dtype=np.float32)
    bonds_surface_index1 = np.full([data_num, atom_number_surface, bond_number_surface], -1, dtype=np.int32)
    bonds_surface_index2 = np.full([data_num, atom_number_surface, bond_number_surface], -1, dtype=np.int32)
    atom_num_surface = np.zeros(data_num, dtype=np.int32)
    
    for i in range(data_num):
        atom_vectors_surface, bond_vectors_surface, neighbor_surface_index = surface.poscar_to_graph(X[i], root_dir, structures_formula, layer_parameters, cut_rds_surface, nbr_type_surface)
        n = atom_vectors_surface.shape[0]
        atoms_surface[i][:n] = atom_vectors_surface
        atom_num_surface[i] = n
        for j in range(n):
            m = bond_vectors_surface[j].shape[0]
            for k in range(m):
                try:
                    bonds_surface[i][j][k] = bond_vectors_surface[j][k]
                    bonds_surface_index1[i][j][k] = neighbor_surface_index[j][k][0] 
                    bonds_surface_index2[i][j][k] = neighbor_surface_index[j][k][1] 
                except:
                    print(X[i])
    return atom_number_dot, atoms_dot, atoms_nbr_dot, bonds_dot, bonds_dot_index1, bonds_dot_index2, atom_num_nbr_dot, bond_number_dot, \
           atom_number_inter, atoms_inter, bonds_inter, bonds_inter_index1, bonds_inter_index2, atom_num_inter, bond_number_inter, \
           atom_number_surface, atoms_surface, bonds_surface, bonds_surface_index1, bonds_surface_index2, atom_num_surface, bond_number_surface