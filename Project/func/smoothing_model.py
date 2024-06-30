import numpy as np
import trimesh
import math
import scipy
import scipy.sparse.linalg as la
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

class MeshSmoothing:
    def __init__(self, filename):
        self.tm = self.load_obj(filename)

    def load_obj(self, filename):
        tm = trimesh.load_mesh(filename)
        return tm

    def cal_angle_cot(self, v1, v2, v_angle):
        vec1 = -v1 + v_angle
        vec2 = -v2 + v_angle
        cos_angle = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cot_angle = 1 / math.tan(np.arccos(cos_angle))
        return cot_angle

    def cal_area(self, idx):
        edges_idx = np.where(self.tm.edges == idx)[0]
        edges = self.tm.edges[edges_idx]
        current_vertex_end = (np.where(edges == idx)[1] == 1)
        edges[current_vertex_end] = edges[current_vertex_end][:, [1, 0]]
        edges_vec = self.tm.vertices[edges[:, 0]] - self.tm.vertices[edges[:, 1]]
        vecs_1, vecs_2 = edges_vec[0::2], edges_vec[1::2]
        areas = np.linalg.norm(np.cross(vecs_1, vecs_2), axis=1) / 2.
        sum_areas = np.sum(areas) / 3.
        return sum_areas

    def cal_laplacian_weight(self):
        n = self.tm.vertices.shape[0]
        M = np.zeros((n, 1))  # calculate each vertice's triangle area sum
        C = np.zeros((n, n))  # calculate each vertice's cot weight
        for idx in tqdm(range(n), desc="Calculation Progress"):
            vertice_neighbor = self.tm.vertex_neighbors[idx]
            sum_angles = 0
            for neighbor in vertice_neighbor:
                v_n_n = self.tm.vertex_neighbors[neighbor]
                common_vertices = np.intersect1d(v_n_n, vertice_neighbor)
                sum_cot_angle = 0
                for i in range(len(common_vertices)):
                    angle_vertex = common_vertices[i]
                    cot_angle = self.cal_angle_cot(self.tm.vertices[idx], self.tm.vertices[neighbor], self.tm.vertices[angle_vertex])
                    sum_cot_angle += cot_angle
                C[idx, neighbor] = sum_cot_angle
                sum_angles += sum_cot_angle
            C[idx, idx] = -sum_angles
            sum_areas = self.cal_area(idx)
            M[idx] = sum_areas * 200000  # the area is too small, so augment the scale of the ratio
        L = C / M
        L = scipy.sparse.csc_matrix(L)
        return L

    def smooth(self, iteration, output, method='explicit', lam=1e-2):
        print(method)
        if method == 'explicit':
            L = self.cal_laplacian_weight()
            
            vertices = self.tm.vertices.copy()
            for i in tqdm(range(iteration), desc="Smoothing Progress(iterations = {})".format(iteration)):
                vertices += lam * L @ vertices
            new_tm = self.tm.copy()
            new_tm.vertices = vertices
            new_tm.export(output)
        elif method == 'implicit':
            L = self.cal_laplacian_weight()
            I = np.identity(len(self.tm.vertices))
            A = scipy.sparse.csc_matrix(I - lam * L)
            vertices = self.tm.vertices.copy()
            for i in tqdm(range(iteration), desc="Smoothing Progress(iterations = {})".format(iteration)):
                vertices = spsolve(A, vertices)
            new_tm = self.tm.copy()
            new_tm.vertices = vertices
            new_tm.export(output)
