import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import OpenGL.GL as gl
import OpenGL.GLUT as glut

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        print(self.vertices.shape)
        print(self.faces.shape)

    @staticmethod
    def load_from_obj(file_path):
        vertices = []
        faces = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    faces.append([int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]])
        return Mesh(np.array(vertices), np.array(faces))
    
    def save_to_obj(self, file_path):
        with open(file_path, 'w') as file:
            for vertex in self.vertices:
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face in self.faces:
                face_plus_1 = [i + 1 for i in face]
                file.write(f"f {face_plus_1[0]} {face_plus_1[1]} {face_plus_1[2]}\n")

    def laplacian_deformation(self, handles, new_positions):
        n = len(self.vertices)
        L = self.build_laplacian_matrix()

        # Create constraints
        constraints = scipy.sparse.lil_matrix((len(handles) * 3, n * 3))
        for i, handle in enumerate(handles):
            constraints[3 * i, 3 * handle] = 1
            constraints[3 * i + 1, 3 * handle + 1] = 1
            constraints[3 * i + 2, 3 * handle + 2] = 1

        # Solve for new vertex positions
        b = np.zeros(len(handles) * 3)
        for i, handle in enumerate(handles):
            b[3 * i] = new_positions[i, 0] - self.vertices[handle, 0]
            b[3 * i + 1] = new_positions[i, 1] - self.vertices[handle, 1]
            b[3 * i + 2] = new_positions[i, 2] - self.vertices[handle, 2]

        A = scipy.sparse.vstack([L, constraints])
        B = np.hstack([np.zeros(n * 3), b])

        # Ensure the dimensions of A and B match
        assert A.shape[0] == B.shape[0], f"A shape: {A.shape}, B shape: {B.shape}"

        new_vertices_flat = scipy.sparse.linalg.lsqr(A, B)[0]
        new_vertices = new_vertices_flat.reshape((n, 3))
        self.vertices += new_vertices

    def build_laplacian_matrix(self):
        n = len(self.vertices)
        I = []
        J = []
        V = []
        for i in range(n):
            neighbors = self.get_neighbors(i)
            for j in neighbors:
                I.append(i)
                J.append(j)
                V.append(1.0 / len(neighbors))
            I.append(i)
            J.append(i)
            V.append(-1)
        # Convert to 3n x 3n matrix
        I = np.repeat(I, 3)
        J = np.repeat(J, 3)
        V = np.tile(V, 3)
        I[1::3] += n
        I[2::3] += 2 * n
        J[1::3] += n
        J[2::3] += 2 * n
        L = scipy.sparse.coo_matrix((V, (I, J)), shape=(n * 3, n * 3))
        return L.tocsr()

    def get_neighbors(self, vertex_index):
        neighbors = set()
        for face in self.faces:
            if vertex_index in face:
                neighbors.update(face)
        neighbors.discard(vertex_index)
        return list(neighbors)

def main():
    # Load and initialize the mesh
    input_file_path = './deformation.obj'
    output_file_path = './output.obj'
    mesh = Mesh.load_from_obj(input_file_path)

    # Set handles and new positions (example)
    handles = [0, 1, 2]  # indices of the vertices to be moved
    new_positions = np.array([
        [-20.0, -22.0, -50.0],  # new position for vertex 0
        [-21.5, -23, -52.0],  # new position for vertex 1
        [-19.0, -21.0, -49.0]   # new position for vertex 2
    ])

    # Perform Laplacian deformation
    mesh.laplacian_deformation(handles, new_positions)

    # Save the deformed mesh to an output file
    mesh.save_to_obj(output_file_path)
    print(f'Deformed mesh saved to {output_file_path}')

if __name__ == "__main__":
    main()
    