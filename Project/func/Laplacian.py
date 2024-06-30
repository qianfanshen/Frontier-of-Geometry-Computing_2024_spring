import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg
from scipy.sparse import block_diag
from utils.util import *
from collections import defaultdict


class LaplacianDeformation:
    def __init__(self, vertices, faces , weights):
        self.vertices = vertices
        self.faces = faces
        self.weights = weights
        self.L = None
        self.vertex_neighbors = self._compute_vertex_neighbors()
        self.shared_edges = self._compute_shared_edges()

    def _compute_vertex_neighbors(self):
        neighbors = defaultdict(set)
        for face in self.faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        neighbors[face[i]].add(face[j])
        return {k: list(v) for k, v in neighbors.items()}

    def _compute_shared_edges(self):
        edge_faces = defaultdict(list)
        for face in self.faces:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted((face[i], face[j])))
                    edge_faces[edge].append(face)
        return edge_faces
        
    def get_vertex_neighbors(self, vertex_index):
        neighbors = set()
        for face in self.faces:
            if vertex_index in face:
                neighbors.update(face)
        neighbors.discard(vertex_index)
        return list(neighbors)
    
    def LaplacianMatrixCotangent(self, anchorsIdx):
        n = self.vertices.shape[0]  # 顶点数量
        k = len(anchorsIdx)  # 锚点数量
        I = []
        J = []
        V = []

        for i in range(n):
            neighbors = self.vertex_neighbors[i]
            z = len(neighbors)
            I.extend([i] * (z + 1))  # 使用 extend 替换 + 操作
            J.extend(neighbors + [i])  # 使用 extend 替换 + 操作
            weights = []
            for neighbor in neighbors:
                edge = tuple(sorted((i, neighbor)))
                faces = self.shared_edges[edge]
                if len(faces) != 2:
                    continue
                cotangents = []
                for face in faces:
                    other_vertices = [v for v in face if v not in edge]
                    if len(other_vertices) != 1:
                        continue
                    opp_vertex_index = other_vertices[0]
                    if opp_vertex_index >= n:
                        print(f"Warning: Opposite vertex index {opp_vertex_index} is out of bounds")
                        continue
                    P = self.vertices[opp_vertex_index]
                    u = self.vertices[i] - P
                    v = self.vertices[neighbor] - P
                    cotangents.append(np.dot(u, v) / np.linalg.norm(np.cross(u, v)))
                if cotangents:
                    weights.append(-np.sum(cotangents) / len(cotangents))
            if weights:
                V.extend(weights + [-np.sum(weights)])  # 使用 extend 替换 + 操作

        for i in range(k):
            I.append(n + i)
            J.append(anchorsIdx[i])
            V.append(self.weights[i])

        self.L = sparse.coo_matrix((V, (I, J)), shape=(n + k, n)).tocsr()

    def solveLaplacianMesh(self, anchors, anchorsIdx):
        n = self.vertices.shape[0]
        k = len(anchorsIdx)
        self.LaplacianMatrixCotangent(anchorsIdx)
        delta = self.L.dot(self.vertices)

        for i in range(k):
            delta[n + i, :] = self.weights[i] * anchors[i, :]

        # 使用 scipy.sparse.block_diag 创建块对角矩阵
        l = block_diag([self.L, self.L, self.L])
        d = np.hstack((delta[:, 0], delta[:, 1], delta[:, 2]))
        
        # 检查维度
        print("Matrix l shape:", l.shape)
        print("Vector d shape:", d.shape)
        
        # 使用共轭梯度法求解
        ans = lsqr(l, d)
        
        self.vertices = ans[0].reshape(-1, 3)
        
        
if __name__ == '__main__':
    obj_filename = './deformation.obj'
    output_filename = './output.obj'
    parser = OBJParser(obj_filename)
    vertices = parser.get_vertices()
    faces = parser.get_faces()
    
    # 假设一些锚点及其位置（示例数据）
    anchors = np.array([
        [-19.3447, -20.9921, -47.3788],
        [-15.5531, -10.2361, -46.6849]
    ])
    anchorsIdx = [0, 1]  # 对应锚点的顶点索引
    weights = [1, 1]  # 锚点的权重

    deformation = LaplacianDeformation(vertices, faces, weights)
    deformation.solveLaplacianMesh(anchors, anchorsIdx)

    saver = OBJSaver(output_filename, deformation.vertices, faces)