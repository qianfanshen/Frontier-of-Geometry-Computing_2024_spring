import numpy as np
import os
import pickle

class OBJParser:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        self.parse(filename)
        
    def parse(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    self.parse_vertex(line)
                elif line.startswith('f '):
                    self.parse_face(line)
        print("faces: {}, vertices: {}".format(len(self.faces), len(self.vertices)))
                    
    def parse_vertex(self, line):
        parts = line.strip().split()
        vertex  = list(map(float, parts[1:4]))
        self.vertices.append(vertex)
        
    def parse_face(self, line):
        parts = line.strip().split()
        face = list(map(lambda x: int(x.split('/')[0]) - 1, parts[1:]))  # 处理面部顶点索引
        self.faces.append(face)

    def get_vertices(self):
        return np.array(self.vertices)

    def get_faces(self):
        return np.array(self.faces)
    
class OBJSaver:
    def __init__(self, filename, vertices, faces):
        self.faces = faces
        self.vertices = vertices
        self.save_obj(filename)
        
    def save_obj(self, filename):
        with open(filename, 'w') as f:
            for v in self.vertices:
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            for face in self.faces:
                if len(face) == 4:
                    f.write("f {} {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1, face[3] + 1))  # 保存四个顶点
                elif len(face) == 3:
                    f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))  # 保存三个顶点
        print("faces: {}, vertices: {}".format(len(self.faces), len(self.vertices)))
        print("saved mesh to {}".format(filename))

def load_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            file = pickle.load(f)
        return file
    else:
        print("{} not exist".format(filename))
        return None

def create_translation_matrix(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])
    
def create_rotation_matrix(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta,  cos_theta, 0, 0],
        [0,         0,         1, 0],
        [0,         0,         0, 1]
    ])
    
def calculate_deformation_matrix(delta_pos, rotation_angle=0):
    dx, dy, dz = delta_pos
    translation_matrix = create_translation_matrix(dx, dy, dz)
    rotation_matrix = create_rotation_matrix(rotation_angle)
    deformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return deformation_matrix