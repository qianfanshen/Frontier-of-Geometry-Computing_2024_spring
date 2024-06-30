'''
rewrite of https://github.com/leventt/arap/blob/master/inCaseMayaSucks.py
'''
import math
import itertools
import numpy as np
import trimesh
from scipy.sparse import csgraph
from scipy.sparse.linalg import spsolve
from utils.util import *
from tqdm import trange

class ARAP:
    def __init__(self, mesh):
        self.verts = np.array(mesh.vertices)
        self.tris = np.array(mesh.faces)

        self.n = len(self.verts)
        self.vertsPrime = np.array(self.verts) # 变形后的位置
        # 每个顶点所在的三角形的列表
        self.vertsToTris = [[j for j, tri in enumerate(self.tris) if i in tri] for i in range(self.n)]
        # 转换顶点位置格式->asmatrix
        self.vertsPrime = np.asmatrix(self.vertsPrime)
        # 构造邻接矩阵
        self.neighbourMatrix = np.zeros((self.n, self.n))
        self.neighbourMatrix[
            tuple(zip(
                *itertools.chain(
                    *map(
                        lambda tri: itertools.permutations(tri, 2),
                        self.tris
                    )
                )
            ))
        ] = 1
        # 初始化局部刚性变换矩阵
        self.cellRotations = np.zeros((self.n, 3, 3))
        # 初始化权重矩阵
        self.weightMatrix = np.zeros((self.n, self.n), dtype=np.float64)
        self.weightSum = np.zeros((self.n, self.n), dtype=np.float64)
        # 计算每个顶点的邻居并分配权重
        for vertID in range(self.n):
            neighbours = self.neighboursOf(vertID)
            for neighbourID in neighbours:
                self.assignWeightForPair(vertID, neighbourID)
                
    def neighboursOf(self, vertID):
        return np.where(self.neighbourMatrix[vertID] == 1)[0]

    def assignWeightForPair(self, i, j):
        if self.weightMatrix[j, i] == 0:
            weightIJ = self.weightForPair(i, j)
        else:
            weightIJ = self.weightMatrix[j, i]
        self.weightSum[i, i] += weightIJ * 0.5
        self.weightSum[j, j] += weightIJ * 0.5
        self.weightMatrix[i, j] = weightIJ

    # 计算cot权重
    def weightForPair(self, i, j):
        localTris = []
        for triID in self.vertsToTris[i]:
            tri = self.tris[triID]
            if i in tri and j in tri:
                localTris.append(tri)

        vertexI = self.verts[i]
        vertexJ = self.verts[j]

        cotThetaSum = 0
        for tri in localTris:
            otherVertID = list(set(tri) - set([i, j]))[0]
            otherVertex = self.verts[otherVertID]

            vA = vertexI - otherVertex
            vB = vertexJ - otherVertex
            cosTheta = vA.dot(vB) / (np.linalg.norm(vA) * np.linalg.norm(vB))
            theta = math.acos(cosTheta)

            cotThetaSum += math.cos(theta) / math.sin(theta)

        return cotThetaSum * 0.5

    def update(self, fixedIDs, handleIDs, deformationMatrices):
        deformationMatrices = list(map(np.matrix, deformationMatrices))
        # 变形后的顶点位置（元组）
        self.deformationVerts = []
        for i in range(self.n):
            if i in handleIDs:
                deformedVector = np.append(self.verts[i], 1)
                deformedVector = deformedVector.dot(deformationMatrices[handleIDs.index(i)])
                deformedVector = np.delete(deformedVector, 3).flatten()
                deformedVector = np.squeeze(np.asarray(deformedVector))
                self.deformationVerts.append((i, deformedVector))
            else:
                self.deformationVerts.append((i, self.verts[i]))

        # 扩张拉普拉斯矩阵
        deformationVertsNum = len(self.deformationVerts)
        self.laplacianMatrix = np.zeros([self.n + deformationVertsNum] * 2, dtype=np.float64)
        self.laplacianMatrix[:self.n, :self.n] = csgraph.laplacian(self.weightMatrix)
        for i in range(deformationVertsNum):
            vertID = self.deformationVerts[i][0]
            ni = i + self.n
            self.laplacianMatrix[ni, vertID] = 1
            self.laplacianMatrix[vertID, ni] = 1

        # 计算每个顶点的差分矩阵
        self.PiArray = []
        for i in range(self.n):
            vertI = self.verts[i]
            neighbourIDs = self.neighboursOf(i)
            neighboursNum = len(neighbourIDs)

            Pi = np.zeros((3, neighboursNum))

            for ni in range(neighboursNum):
                nID = neighbourIDs[ni]
                vertJ = self.verts[nID]
                Pi[:, ni] = (vertI - vertJ)
            self.PiArray.append(Pi)

    # 计算每个顶点的局部刚性变换（旋转矩阵）
    def calculateCellRotations(self):
        for vertID in range(self.n):
            rotation = self.calculateRotationMatrixForCell(vertID)
            self.cellRotations[vertID] = rotation

    # 用旋转矩阵更新顶点位置
    def applyCellRotations(self):
        '''
        \sum w_ij / 2 * (R_i + R_j) * (v_i - v_j)
        '''
        for i in range(self.n):
            self.bArray[i] = np.zeros((1, 3))
            neighbours = self.neighboursOf(i)
            for j in neighbours:
                wij = self.weightMatrix[i, j] / 2.0
                rij = self.cellRotations[i] + self.cellRotations[j]
                pij = self.verts[i] - self.verts[j]
                self.bArray[i] += (wij * rij.dot(pij))

        self.vertsPrime = np.linalg.solve(self.laplacianMatrix, self.bArray)[:self.n]
        
        
    # 计算给定顶点的旋转矩阵
    def calculateRotationMatrixForCell(self, vertID):
        covarianceMatrix = self.calculateConvarianceMatrixForCell(vertID)

        U, s, VTranspose = np.linalg.svd(covarianceMatrix)

        rotation = VTranspose.T.dot(U.T)
        if np.linalg.det(rotation) <= 0:
            U[:0] *= -1
            rotation = VTranspose.T.dot(U.T)
        return rotation

    def calculateConvarianceMatrixForCell(self, vertID):
        vertIPrime = self.vertsPrime[vertID]

        neighbourIDs = self.neighboursOf(vertID)
        neighboursNum = len(neighbourIDs)

        Di = np.zeros((neighboursNum, neighboursNum))

        Pi = self.PiArray[vertID]
        PiPrime = np.zeros((3, neighboursNum))

        for ni in range(neighboursNum):
            nID = neighbourIDs[ni]

            Di[ni, ni] = self.weightMatrix[vertID, nID]

            vertJPrime = self.vertsPrime[nID]
            PiPrime[:, ni] = (vertIPrime - vertJPrime)

        PiPrime = PiPrime.T

        return Pi.dot(Di).dot(PiPrime)

    def apply(self, iterations):
        deformationVertsNum = len(self.deformationVerts)

        self.bArray = np.zeros((self.n + deformationVertsNum, 3))
        for i in range(deformationVertsNum):
            self.bArray[self.n + i] = self.deformationVerts[i][1]

        for t in trange(iterations):
            self.calculateCellRotations()
            self.applyCellRotations()


def deform_mesh(input_mesh, fixed_vertices, handle_vertices, deformation_matrices, iterations):
    arap = ARAP(input_mesh)
    arap.update(fixed_vertices, handle_vertices, deformation_matrices)
    arap.apply(iterations)

    deformed_mesh = input_mesh.copy()
    deformed_mesh.vertices = np.asarray(arap.vertsPrime)
    return deformed_mesh


def deformation(file_path, iterations, handle_points, handle_idx, fixed_idx, delta_pos, output_path, rotation_angle):
    input_mesh = trimesh.load(file_path)
    handle_vertices = np.asarray(handle_idx).reshape((len(handle_idx), )).tolist()
    fixed_ver = np.asarray(fixed_idx).reshape((len(fixed_idx), )).tolist()
    print("-------------")
    handles = list(set(handle_vertices) - set(fixed_ver))
    fixed_vertices = list(set(input_mesh.vertices) - set(handles))
    print("++++++++++++++++++++++")
    handle_cord = np.asarray(handle_points.points)
    new_positions = handle_cord + delta_pos
    
    deformation_matrices = []
    for i in range(len(handle_vertices)):
        # initial_pos = input_mesh.vertices[i]
        # new_pos = new_positions[i]
        deformation_matrix = calculate_deformation_matrix(delta_pos, rotation_angle)
        deformation_matrices.append(deformation_matrix)

    print(deformation_matrices)
    deformed_mesh = deform_mesh(input_mesh, fixed_vertices, handle_vertices, deformation_matrices, iterations)
    deformed_mesh.export(output_path)
    print("Done")
    return deform_mesh

# def main():
#     # 加载OBJ模型
#     input_mesh = trimesh.load('/Users/shenqianfan/Desktop/大二下/几何计算前沿/Project/deformation.obj')

#     # 变形参数
#     fixed_vertices = []  # 固定顶点ID列表
#     handle_vertices = [i for i in range(200)]  # 要移动的顶点ID列表
#     fixed_vertices = list(set(input_mesh.vertices) - set(handle_vertices))
    
#     # 试验的变换矩阵
#     dm = [[ 0.79149242, 0.00816574,  0.61112641,  4.18521004],
#             [-0.26070382,  0.90864017,  0.32603009, -3.32117836],
#             [-0.55275879, -0.41747399,  0.72032679, -3.89183947],
#             [ 0.        ,  0.        ,  0.        ,  1.        ]]
#     deformation_matrices = [dm for _ in range(200)]  # 变形矩阵列表
#     iterations = 10  # 迭代次数
    
#     # 实际上利用新的位置
#     new_positions = np.array([
#         [0.5, 0.5, 0.5],
#         [1.5, 0.5, 0.5],
#         [0.5, 1.5, 0.5],
#         [1.5, 1.5, 0.5],
#         [1.0, 1.0, 1.5]
#     ])
#     # 和旋转角度
#     rotation_angle = 0 
    
#     deformation_matrices = []

#     for i in handle_vertices:
#         initial_pos = input_mesh.vertices[i]
#         new_pos = new_positions[i]
#         deformation_matrix = calculate_deformation_matrix(initial_pos, new_pos, rotation_angle)
#         deformation_matrices.append(deformation_matrix)
        
#     # 进行变形
#     deformed_mesh = deform_mesh(input_mesh, fixed_vertices, handle_vertices, deformation_matrices, iterations)

#     # 保存变形后的OBJ模型
#     deformed_mesh.export('deformed_model.obj')

# if __name__ == '__main__':
#     main()
