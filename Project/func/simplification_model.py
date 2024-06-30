import numpy as np
from utils.util import *

class generate_model:
    def __init__(self, filename):
        self.model_filepath = filename
        parser = OBJParser(self.model_filepath)
        self.vertices = parser.get_vertices()
        self.faces = parser.get_faces()
        self.vertices_num = self.vertices.shape[0]
        self.faces_num = self.faces.shape[0]
        edge_1 = self.faces[:, [0, 1]] 
        edge_2 = self.faces[:, [1, 2]] 
        edge_3 = self.faces[:, [2, 0]]  
        all_edges = np.vstack([edge_1, edge_2, edge_3])
        sorted_edges = np.sort(all_edges, axis=1)
        _, unique_indices = np.unique(sorted_edges, axis=0, return_index=True)
        unique_edges = sorted_edges[unique_indices]
        self.edges = unique_edges
        
        self.cal_plane_equations()
        
        self.cal_Q_matrices()
        
    def cal_plane_equations(self):
        self.plane_equas = []
        
        for i in range(self.faces_num):
            p_1, p_2, p_3 = self.vertices[self.faces[i, 0], :], self.vertices[self.faces[i, 1], :], self.vertices[self.faces[i, 2], :]
            v_1 = p_2 - p_1
            v_2 = p_3 - p_1
            normal = np.cross(v_1, v_2)
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p_1)
            self.plane_equas.append(np.append(normal, d))
        
        self.plane_equas = np.array(self.plane_equas)
        
    def cal_Q_matrices(self):
        self.Q_matrices = []
        for i in range(self.vertices_num):
            face_set_index = np.where(self.faces == i)[0]
            Q_temp = np.zeros((4, 4))
            for j in face_set_index:
                p = self.plane_equas[j, :].reshape(1, -1)
                Q_temp += np.matmul(p.T, p)
            self.Q_matrices.append(Q_temp)
        self.Q_matrices = np.array(self.Q_matrices)   
        
            
class simplification(generate_model):
    def __init__(self, filename, threshold, ratio):
        super().__init__(filename)
        print('import model from: {}'.format(filename))
        self.t = threshold
        self.r = ratio
        
    def generate_valid_pairs(self):
        self.dist_pairs = []
        for i in range(self.vertices_num):
            current_p = self.vertices[i, :]
            current_p_dists = np.linalg.norm(self.vertices - current_p, axis=1)
            valid_pairs_loc = np.where(current_p_dists <= self.t)[0].reshape(1, -1)
            current_valid_pairs = np.hstack([i * np.ones((valid_pairs_loc.shape[0], 1)), valid_pairs_loc])
            if i == 0:
                self.dist_pairs = current_valid_pairs
            else:
                self.dist_pairs = np.vstack([self.dist_pairs, current_valid_pairs])

        self.dist_pairs = np.array(self.dist_pairs, dtype=int)
        # Remove self-loops
        self.dist_pairs = self.dist_pairs[self.dist_pairs[:, 0] != self.dist_pairs[:, 1]]

        if self.dist_pairs.size > 0:
            self.valid_pairs = np.vstack([self.edges, self.dist_pairs])
        else:
            self.valid_pairs = self.edges

        # Remove self-loops again
        self.valid_pairs = self.valid_pairs[self.valid_pairs[:, 0] != self.valid_pairs[:, 1]]

        # Remove duplicate pairs
        self.valid_pairs = np.unique(np.sort(self.valid_pairs, axis=1), axis=0)
        
    def calculate_optimal_contraction_pairs_and_cost(self):
        self.v_optimal = []
        self.cost = []
        valid_pairs_num = self.valid_pairs.shape[0]
        for i in range(valid_pairs_num):
            current_valid_pair = self.valid_pairs[i, :]
            v_1_loc = current_valid_pair[0]
            v_2_loc = current_valid_pair[1]
            Q_1 = self.Q_matrices[v_1_loc]
            Q_2 = self.Q_matrices[v_2_loc]
            Q = Q_1 + Q_2
            Q_new = np.concatenate([Q[:3, :], np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
            
            if np.linalg.det(Q_new) >= 1e-10:
                current_v_opt = np.linalg.solve(Q_new, np.array([0, 0, 0, 1]).reshape(4, 1))
                current_cost = np.matmul(np.matmul(current_v_opt.T, Q), current_v_opt)
                current_v_opt = current_v_opt.reshape(4)[:3]
                
            else:
                v_1 = np.append(self.points[v_1_loc, :], 1).reshape(4, 1)
                v_2 = np.append(self.points[v_2_loc, :], 1).reshape(4, 1)
                v_mid = (v_1 + v_2) / 2
                delta_v_1 = np.matmul(np.matmul(v_1.T, Q), v_1)
                delta_v_2 = np.matmul(np.matmul(v_2.T, Q), v_2)
                delta_v_mid = np.matmul(np.matmul(v_mid.T, Q), v_mid)
                current_cost = np.min([delta_v_1, delta_v_2, delta_v_mid])
                min_delta_loc = np.argmin([delta_v_1, delta_v_2, delta_v_mid])
                current_v_opt = np.concatenate([v_1, v_2, v_mid], axis=1)[:, min_delta_loc].reshape(4)
                current_v_opt = current_v_opt[:3]
                
            self.v_optimal.append(current_v_opt)
            self.cost.append(current_cost)
            
        self.v_optimal = np.array(self.v_optimal)
        self.cost = np.array(self.cost).reshape(-1)
        
        cost_argsort = np.argsort(self.cost)
        self.valid_pairs = self.valid_pairs[cost_argsort, :]
        self.v_optimal = self.v_optimal[cost_argsort, :]
        self.cost = self.cost[cost_argsort]

        self.new_point = self.v_optimal[0, :]
        self.new_valid_pair = self.valid_pairs[0, :]
        
    def iteratively_remove_pairs(self):
        self.new_point_num = 0
        self.status_faces = np.zeros(self.faces_num)
        self.status_vertices = np.zeros(self.vertices_num)
        while (self.vertices_num - self.new_point_num) >= self.r * self.vertices_num:
            current_valid_pair = self.new_valid_pair
            v_1_loc = current_valid_pair[0]
            v_2_loc = current_valid_pair[1]
            
            # 更新顶点
            self.vertices[v_1_loc, :] = self.new_point
            self.vertices[v_2_loc, :] = self.new_point
            
            #标记原来的v2为删除
            self.status_vertices[v_2_loc] = -1 
            
            #标记面状态
            v_1_in_faces_loc = np.where(self.faces == v_1_loc)
            v_2_in_faces_loc = np.where(self.faces == v_2_loc)
            v_1_2_in_one_face_loc = []
            
            for item in v_2_in_faces_loc[0]:
                if np.where(v_1_in_faces_loc[0] == item)[0].size > 0:
                    v_1_2_in_one_face_loc.append(item)
            v_1_2_in_one_face_loc = np.array(v_1_2_in_one_face_loc)

            if v_1_2_in_one_face_loc.size >= 1:
                self.status_faces[v_1_2_in_one_face_loc] = -1
                
            # 更新面
            self.faces[v_2_in_faces_loc] = v_1_loc
            
            # 更新平面方程参数
            v_1_2_in_faces_loc = np.unique(np.append(v_1_in_faces_loc[0], v_2_in_faces_loc[0]))
            self.update_plane_equation_parameters(v_1_2_in_faces_loc)
            
            # 更新Q矩阵
            self.update_Q(current_valid_pair, v_1_loc)

            # 更新有效顶点对、最优顶点和成本
            self.update_valid_pairs_v_optimal_and_cost(v_1_loc)

            # 重新计算最优收缩顶点对和成本
            self.update_optimal_contraction_pairs_and_cost(v_1_loc)

            if self.new_point_num % 100 == 0:
                print(f'Simplification: {100 * (self.vertices_num - self.new_point_num) / self.vertices_num:.2f}%')
                print(f'Remaining: {self.vertices_num - self.new_point_num} points\n')

            self.new_point_num += 1
            
        print(f'Simplification: {100 * (self.vertices_num - self.new_point_num) / self.vertices_num:.2f}%')
        print(f'Remaining: {self.vertices_num - self.new_point_num} points')
        print('End\n')
        
    def calculate_plane_equation_for_one_face(self, p1, p2, p3):
        """
        计算平面方程 ax + by + cz + d = 0
        输入:
            p1, p2, p3: numpy.array, 形状为 (3, ) 或 (1, 3) 或 (3, )
                        面上的三个点 (x, y, z)
        输出:
            numpy.array: (a, b, c, d), 形状为 (1, 4)
        """
        p1 = np.array(p1).reshape(3)
        p2 = np.array(p2).reshape(3)
        p3 = np.array(p3).reshape(3)

        # 计算法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)  # 单位化法向量

        # 计算 d
        d = -np.dot(normal, p1)

        return np.array([normal[0], normal[1], normal[2], d]).reshape(1, 4)
    
    def update_plane_equation_parameters(self, need_updating_loc):
        """
        更新需要更新的面平面方程参数
        输入:
            need_updating_loc: numpy.array, 形状为 (n, ), 需要更新的 self.plane_equas 的位置
        """
        for i in need_updating_loc:
            if self.status_faces[i] == -1:
                self.plane_equas[i, :] = np.array([0, 0, 0, 0]).reshape(1, 4)
            else:
                point_1 = self.vertices[self.faces[i, 0], :]
                point_2 = self.vertices[self.faces[i, 1], :]
                point_3 = self.vertices[self.faces[i, 2], :]
                self.plane_equas[i, :] = self.calculate_plane_equation_for_one_face(point_1, point_2, point_3)
                
    def update_Q(self, replace_locs, target_loc):
        """
        更新指定顶点的Q矩阵
        输入:
            replace_locs: numpy.array, 形状为 (2, ), 需要更新的顶点位置
            target_loc: int, 新位置的顶点索引
        """
        face_set_index = np.where(self.faces == target_loc)[0]
        Q_temp = np.zeros((4, 4))
    
        for j in face_set_index:
            p = self.plane_equas[j, :]
            p = p.reshape(1, len(p))
            Q_temp += np.matmul(p.T, p)
    
        for i in replace_locs:
            self.Q_matrices[i] = Q_temp
            
    def update_valid_pairs_v_optimal_and_cost(self, target_loc):
        # 获取包含当前有效顶点对的位置
        v_1_loc_in_valid_pairs = np.where(self.valid_pairs == self.new_valid_pair[0])
        v_2_loc_in_valid_pairs = np.where(self.valid_pairs == self.new_valid_pair[1])
        # 替换包含当前有效对的顶点索引为新的目标顶点索引
        self.valid_pairs[v_1_loc_in_valid_pairs] = target_loc
        self.valid_pairs[v_2_loc_in_valid_pairs] = target_loc
        
        # 找到需要删除的顶点对位置
        delete_locs = []
        for item in v_1_loc_in_valid_pairs[0]:
            if np.where(v_2_loc_in_valid_pairs[0] == item)[0].size > 0:
                delete_locs.append(item)
        delete_locs = np.array(delete_locs)
    
        # 删除自环（即顶点对中的两个顶点相同的情况）
        find_same = self.valid_pairs[:, 1] - self.valid_pairs[:, 0]
        find_same_loc = np.where(find_same == 0)[0]
        if find_same_loc.size >= 1:
            delete_locs = np.append(delete_locs, find_same_loc)
    
        # 删除重复的顶点对、最优顶点和成本
        self.valid_pairs = np.delete(self.valid_pairs, delete_locs, axis=0)
        self.v_optimal = np.delete(self.v_optimal, delete_locs, axis=0)
        self.cost = np.delete(self.cost, delete_locs, axis=0)
    
        # 保持有效对的唯一性
        unique_valid_pairs_trans, unique_valid_pairs_loc = np.unique(self.valid_pairs[:, 0] * (10**10) + self.valid_pairs[:, 1], return_index=True)
        self.valid_pairs = self.valid_pairs[unique_valid_pairs_loc, :]
        self.v_optimal = self.v_optimal[unique_valid_pairs_loc, :]
        self.cost = self.cost[unique_valid_pairs_loc]

    def update_optimal_contraction_pairs_and_cost(self, target_loc):
        """
        更新有效顶点对、最优收缩点和收缩成本
        输入:
            target_loc: int, 更新后顶点的位置
        """
        v_target_loc_in_valid_pairs = np.where(self.valid_pairs == target_loc)[0]
        for i in v_target_loc_in_valid_pairs:
            current_valid_pair = self.valid_pairs[i, :]
            v_1_loc = current_valid_pair[0]
            v_2_loc = current_valid_pair[1]

            # 获取Q矩阵
            Q_1 = self.Q_matrices[v_1_loc]
            Q_2 = self.Q_matrices[v_2_loc]
            Q = Q_1 + Q_2
            
            Q_new = np.concatenate([Q[:3, :], np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
            
            if np.linalg.det(Q_new) >= 1e-10:
                current_v_opt = np.linalg.solve(Q_new, np.array([0, 0, 0, 1]).reshape(4, 1))
                current_cost = np.matmul(np.matmul(current_v_opt.T, Q), current_v_opt)
                current_v_opt = current_v_opt.reshape(4)[:3]
                
            else:
                v_1 = np.append(self.points[v_1_loc, :], 1).reshape(4, 1)
                v_2 = np.append(self.points[v_2_loc, :], 1).reshape(4, 1)
                v_mid = (v_1 + v_2) / 2
                delta_v_1 = np.matmul(np.matmul(v_1.T, Q), v_1)
                delta_v_2 = np.matmul(np.matmul(v_2.T, Q), v_2)
                delta_v_mid = np.matmul(np.matmul(v_mid.T, Q), v_mid)
                current_cost = np.min([delta_v_1, delta_v_2, delta_v_mid])
                min_delta_loc = np.argmin([delta_v_1, delta_v_2, delta_v_mid])
                current_v_opt = np.concatenate([v_1, v_2, v_mid], axis=1)[:, min_delta_loc].reshape(4)
                current_v_opt = current_v_opt[:3]
                
            # 更新v_optimal和cost
            self.v_optimal[i, :] = current_v_opt
            self.cost[i] = current_cost
        
        # 根据成本排序
        cost_argsort = np.argsort(self.cost)
        self.valid_pairs = self.valid_pairs[cost_argsort, :]
        self.v_optimal = self.v_optimal[cost_argsort, :]
        self.cost = self.cost[cost_argsort]

        # 更新new_point和new_valid_pair
        self.new_point = self.v_optimal[0, :]
        self.new_valid_pair = self.valid_pairs[0, :]   
        
    def generate_new_3d_model(self):
        point_serial_number=np.arange(self.vertices.shape[0])
        points_to_delete_locs=np.where(self.status_vertices==-1)[0]
        self.points=np.delete(self.vertices, points_to_delete_locs, axis=0)
        point_serial_number=np.delete(point_serial_number, points_to_delete_locs)
        point_serial_number_after_del=np.arange(self.points.shape[0])
        
        faces_to_delete_locs=np.where(self.status_faces==-1)[0]
        self.faces=np.delete(self.faces, faces_to_delete_locs, axis=0)
        
        for i in point_serial_number_after_del:
            point_loc_in_face=np.where(self.faces==point_serial_number[i])
            self.faces[point_loc_in_face]=i
        
        self.number_of_points=self.points.shape[0]
        self.number_of_faces=self.faces.shape[0]
    
    def output(self, output_filepath):
        with open(output_filepath, 'w') as file_obj:
            file_obj.write('# '+str(self.number_of_points)+' vertices, '+str(self.number_of_faces)+' faces\n')
            for i in range(self.number_of_points):
                file_obj.write('v '+str(self.points[i,0])+' '+str(self.points[i,1])+' '+str(self.points[i,2])+'\n')
            for i in range(self.number_of_faces):
                file_obj.write('f '+str(self.faces[i,0] + 1)+' '+str(self.faces[i,1] + 1)+' '+str(self.faces[i,2] + 1)+'\n')
        print('Output simplified model: '+str(output_filepath))
            