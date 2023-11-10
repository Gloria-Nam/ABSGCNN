import sgcnn.sg_surface as surface
import sgcnn.sg_inter as inter
import numpy as np
from sgcnn.sg_encoding import *

""" This is where input graph structures are mathematically converted to 
    vector format."""

def data_preparation(X, root_dir, structures_formula, layer_parameters, cut_rds, nbr_type):
    data_num = len(X)
    cut_rds_surface, cut_rds_inter = cut_rds[0], cut_rds[1]
    nbr_type_surface, nbr_type_inter = nbr_type[0], nbr_type[1]
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
#     print("inter layer : (atom,  bond) : ", atom_number_inter, bond_number_inter)

    atoms_inter = np.zeros([data_num, atom_number_inter, CATEGORY_NUM], dtype=np.int32)
    bonds_inter = np.zeros([data_num, atom_number_inter, bond_number_inter, BOND_CATEGORY_NUM], dtype=np.float32)

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
#     print("surface layer: (atom, bond) : ", atom_number_surface, bond_number_surface)
    
    atoms_surface = np.zeros([data_num, atom_number_surface, CATEGORY_NUM], dtype=np.int32)
    bonds_surface = np.zeros([data_num, atom_number_surface, bond_number_surface, BOND_CATEGORY_NUM], dtype=np.float32)
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
    return atom_number_inter, atoms_inter, bonds_inter, bonds_inter_index1, bonds_inter_index2, atom_num_inter, bond_number_inter, \
           atom_number_surface, atoms_surface, bonds_surface, bonds_surface_index1, bonds_surface_index2, atom_num_surface, bond_number_surface