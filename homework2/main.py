import numpy as np
import trimesh
import math
import scipy
import scipy.sparse.linalg as la
from scipy.sparse.linalg import spsolve, eigsh
from tqdm import tqdm
import time

def load_obj(filename):
    tm = trimesh.load_mesh(filename)
    return tm

def cal_angle_cot(v1, v2, v_angle):
    vec1 = -v1 + v_angle
    vec2 = -v2 + v_angle
    cos_angle = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cot_angle = 1 / math.tan(np.arccos(cos_angle))
    return cot_angle
    
def cal_area(tm, idx):
    edges_idx = np.where(tm.edges == idx)[0]
    edges = tm.edges[edges_idx]
    # calculate the vertices which are the ends of the edge
    current_vertex_end = (np.where(edges == idx)[1] == 1)
    # change the order and make the idx the beginning of the edge
    edges[current_vertex_end] = edges[current_vertex_end][:, [1, 0]]
    edges_vec = tm.vertices[edges[:, 0]] - tm.vertices[edges[:, 1]]
    vecs_1, vecs_2 = edges_vec[0::2], edges_vec[1::2]
    areas = np.linalg.norm(np.cross(vecs_1, vecs_2), axis=1) / 2.
    sum_areas = np.sum(areas) / 3.
    return sum_areas
    
    
        
def cal_laplacian_weight(tm):
    n = tm.vertices.shape[0]
    '''calculate the non-uniform weight matrix for vertices'''
    '''
    for each vertices:
          angle_vertex1
            / \ 
           /   \ 
          /     \      
      idx ——————— neighbor
          \     /
           \   /
            \ /
        angle_vertex2  

    '''
    M = np.zeros((n, 1)) # calculate each vertice's triangle area sum
    C = np.zeros((n, n)) # calculate each vertice's cot weight
    for idx in tqdm(range(n), desc="Calculation Progress"):
        vertice_neighbor = tm.vertex_neighbors[idx]
        sum_angles = 0
        for neighbor in vertice_neighbor:
            v_n_n = tm.vertex_neighbors[neighbor]
            common_vertices = np.intersect1d(v_n_n, vertice_neighbor)
            sum_cot_angle = 0
            #print(len(common_vertices))
            for i in range(len(common_vertices)):
                angle_vertex = common_vertices[i]
                cot_angle = cal_angle_cot(tm.vertices[idx], tm.vertices[neighbor], tm.vertices[angle_vertex])
                sum_cot_angle += cot_angle
            C[idx, neighbor] = sum_cot_angle
            sum_angles += sum_cot_angle
            #print(sum_angles)
            
        C[idx, idx] = - sum_angles
        sum_areas = cal_area(tm, idx)
        #print(sum_areas)
        M[idx] = sum_areas * 200000 # the areais too small, so augment the scale of the ratio
    
    L = C / M
    L = scipy.sparse.csc_matrix(L)
    return L

def smooth(tm, iterations, method= 'explicit', lam = 1e-2):
    if method == 'explicit':
        L = cal_laplacian_weight(tm)
        for iteration in iterations:
            vertices = tm.vertices.copy()
            for i in tqdm(range(iteration), desc="Smoothing Progress(iterations = {})".format(iteration)):
                vertices += lam * L @ vertices
            new_tm = tm.copy()
            new_tm.vertices = vertices
            new_tm.export("smooth_non_normal_{}_{}.obj".format(method, iteration))
    elif method == 'implicit':
        L = cal_laplacian_weight(tm)
        I = np.identity(len(tm.vertices))
        A = scipy.sparse.csc_matrix(I - lam * L)
        for iteration in iterations:
            vertices = tm.vertices.copy()
            for i in tqdm(range(iteration), desc="Smoothing Progress(iterations = {})".format(iteration)):
                vertices = spsolve(A, vertices)
            new_tm = tm.copy()
            new_tm.vertices = vertices
            new_tm.export("smooth_non_normal_{}_{}.obj".format(method, iteration))
    
tm = load_obj('smoothing.obj')
print(tm.vertices.shape)
iterations = [5, 10 ,30, 50, 80]
smooth(tm, iterations,'implicit')