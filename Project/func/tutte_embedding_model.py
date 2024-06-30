import trimesh
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from collections import defaultdict

class TutteEmbedding:
    def __init__(self, filename):
        self.mesh = self.load_mesh(filename)
        self.v_num = len(self.mesh.vertices)
        print(self.mesh.edges_sorted.shape)
        self.boundary_vertices = None
        self.boundary_num = 0
        self.uvs = None
        self.A = None
        self.bu = None
        self.bv = None
        
    def load_mesh(self, filename):
        return trimesh.load_mesh(filename)
    
    def compute_area(self):
        """Compute the total area of the mesh."""
        area_sum = 0
        for face in self.mesh.faces:
            v_0, v_1, v_2 = self.mesh.vertices[face]
            e_1 = v_1 - v_0
            e_2 = v_2 - v_0
            area_sum += np.linalg.norm(np.cross(e_1, e_2)) / 2.0
            
        return area_sum
    
    def place_boundary_vertices(self):
        self.boundary_vertices = self.get_boundary_vertices()
        self.boundary_num = len(self.boundary_vertices)
        print(self.boundary_num)
        delta_angle = 2 * np.pi / self.boundary_num
        area_sum = self.compute_area()
        area_1_radium = np.sqrt(area_sum / np.pi)
        self.uvs = np.zeros((self.v_num, 2))
        
        for i, v_idx in enumerate(self.boundary_vertices):
            self.uvs[v_idx, 0] = area_1_radium * np.cos(i * delta_angle)
            self.uvs[v_idx, 1] = area_1_radium * np.sin(-i * delta_angle)
            
    def construct_matrix(self):
        self.A = lil_matrix((self.v_num, self.v_num))
        self.bu = np.zeros(self.v_num)
        self.bv = np.zeros(self.v_num)
        
        for v_idx in range(self.v_num):
            if v_idx in self.boundary_vertices:
                self.A[v_idx, v_idx] = 1
                self.bu[v_idx] = self.uvs[v_idx, 0]
                self.bv[v_idx] = self.uvs[v_idx, 1]
            else:
                neighbors = self.mesh.vertex_neighbors[v_idx]
                degree = len(neighbors)
                for neighbor in neighbors:
                    self.A[v_idx, neighbor] = -1
                self.A[v_idx, v_idx] = degree
                
        self.A = self.A.tocsr()   
        
    def solve_linear_system(self):
        xu = spsolve(self.A, self.bu)
        xv = spsolve(self.A, self.bv)
        
        for i in range(self.v_num):
            self.mesh.vertices[i] = [xu[i], xv[i], 0] 
            
    def write_mesh(self, output_filename):
        """Write the parameterized mesh to an OBJ file."""
        self.mesh.export(output_filename)
        
    def parameterize(self, output_filename="output_tutte.obj"):
        """Perform the complete Tutte parameterization process."""
        self.place_boundary_vertices()
        if self.boundary_num == 0:
            print("Error: No boundary vertices found.")
            return
        self.construct_matrix()
        self.solve_linear_system()
        self.write_mesh(output_filename)
    
    def get_boundary_vertices(self):
        # Set of all edges and of boundary edges (those that appear only once).
        edge_set = set()
        boundary_edges = set()

        # Iterate over all edges, as tuples in the form (i, j) (sorted with i < j to remove ambiguities).
        for e in map(tuple, self.mesh.edges_sorted):
            if e not in edge_set:
                edge_set.add(e)
                boundary_edges.add(e)
            elif e in boundary_edges:
                boundary_edges.remove(e)
            else:
                raise RuntimeError(f"The mesh is not a manifold: edge {e} appears more than twice.")

        # Extract boundary vertices from boundary edges
        boundary_vertices_set = set()
        for v1, v2 in boundary_edges:
            boundary_vertices_set.add(v1)
            boundary_vertices_set.add(v2)

        return list(boundary_vertices_set)
    
    def color_mesh(self):
        """Color the mesh vertices randomly."""
        vertex_colors = np.random.randint(0, 255, (len(self.mesh.vertices), 4), dtype=np.uint8)
        vertex_colors[:, 3] = 255  # 设置透明度为255（不透明）
        self.mesh.visual.vertex_colors = vertex_colors