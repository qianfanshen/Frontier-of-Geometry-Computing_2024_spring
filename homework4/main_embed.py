import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from skimage import measure
from tqdm import tqdm
import rff

# Dataloader
def load_data(path):
    pointcloud = np.load(os.path.join(path, 'pointcloud.npz'))
    sdf = np.load(os.path.join(path, 'sdf.npz'))
    
    points = pointcloud['points']
    normals = pointcloud['normals']
    sdf_points = sdf['points']
    sdf_grads = sdf['grad']
    sdf_values = sdf['sdf']
    
    return points, normals, sdf_points, sdf_grads, sdf_values

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256, dtype=torch.float64),
            activation_fn(),
            nn.Linear(256, 512, dtype=torch.float64),
            activation_fn(),
            nn.Linear(512, 512, dtype=torch.float64),
            activation_fn(),
            nn.Linear(512, 512, dtype=torch.float64),
            activation_fn(),
            nn.Linear(512, 256, dtype=torch.float64),
            activation_fn(),
            nn.Linear(256, output_dim, dtype=torch.float64)
        )
    
    def forward(self, x):
        return self.fc(x)
    
# Loss Function
def sdf_loss(pred_sdf, true_sdf, gradient, true_normal):
    sdf_loss = torch.mean((pred_sdf - true_sdf) ** 2)
    grad_loss = torch.mean((gradient - true_normal) ** 2)
    return sdf_loss + grad_loss

def extract_mesh(model, device, resolution=128, level=0.05, batch_size=10000):
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    z = np.linspace(-0.5, 0.5, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    query_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
    
    query_points_tensor = torch.tensor(query_points, dtype=torch.float64).to(device)
    encoding = rff.layers.PositionalEncoding(sigma=0.5, m=32)
     
    pred_sdf = []
    with torch.no_grad():
        for i in range(0, len(query_points_tensor), batch_size):
            batch_query_points = query_points_tensor[i:i + batch_size]
            encoded_query_points = encoding(batch_query_points)
            batch_pred_sdf = model(encoded_query_points)
            pred_sdf.append(batch_pred_sdf.cpu().numpy())

    pred_sdf = np.concatenate(pred_sdf).reshape((resolution, resolution, resolution))

    
    vertices, faces, normals, _ = measure.marching_cubes(pred_sdf, level=level)
    
    return vertices, faces, normals

def save_mesh(vertices, faces, filepath):
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points, normals, sdf_points, sdf_grads, sdf_values = load_data(args.data_path)
    print(np.isnan(sdf_points).any(), np.isnan(sdf_grads).any(), np.isnan(sdf_values).any())
    sdf_points_tensor = torch.tensor(sdf_points, dtype=torch.float64).to(device)
    sdf_values_tensor = torch.tensor(sdf_values, dtype=torch.float64).to(device)
    sdf_grad_tensor = torch.tensor(sdf_grads, dtype=torch.float64).to(device)

    dataset = TensorDataset(sdf_points_tensor, sdf_values_tensor, sdf_grad_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MLP(input_dim=192, hidden_dim=256, output_dim=1, activation_fn=args.activation_fn).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for sdf_points_batch, sdf_values_batch, sdf_grad_batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                
                optimizer.zero_grad()
                
                sdf_points_batch = sdf_points_batch.to(device).requires_grad_(True)
                sdf_values_batch = sdf_values_batch.to(device)
                sdf_grad_batch = sdf_grad_batch.to(device)
                
                encoding = rff.layers.PositionalEncoding(sigma=0.5, m=32)
                Xp = encoding(sdf_points_batch)
                
                # predict SDF and Gradient
                pred_sdf = model(Xp)
                pred_grad = torch.autograd.grad(outputs=pred_sdf, inputs=sdf_points_batch, 
                                                grad_outputs=torch.ones(pred_sdf.size(), dtype=torch.float64).to(device), 
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                
                # calculate loss
                loss = sdf_loss(pred_sdf, sdf_values_batch, pred_grad, sdf_grad_batch)
                epoch_loss += loss.item()
                
                # backpropagation and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item())
                
        vertices, faces, normals = extract_mesh(model, device)
        save_mesh(vertices, faces, f'{args.output_path}_epoch_{epoch+1}.obj')

        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}')
        scheduler.step(epoch_loss / len(dataloader))
        

    vertices, faces, normals = extract_mesh(model, device)
    save_mesh(vertices, faces, f'{args.output_path}_final.obj')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MLP model for SDF prediction and mesh extraction')
    parser.add_argument('--data_path', type=str, default='./data/1a04e3eab45ca15dd86060f189eb133', help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate for the optimizer')
    parser.add_argument('--output_path', type=str, default='./4/embed', help='Path to save the output mesh files')
    parser.add_argument('--activation_fn', type=str, choices=['swish', 'elu', 'gelu', 'lrelu'], default='swish', help='Activation function to use (swish, elu, gelu)')
    args = parser.parse_args()
    if args.activation_fn == 'swish':
        activation_fn = Swish
    elif args.activation_fn == 'elu':
        activation_fn = nn.ELU
    elif args.activation_fn == 'gelu':
        activation_fn = nn.GELU
    elif args.activation_fn == 'lrelu':
        activation_fn = nn.LeakyReLU(0.2)
    args.activation_fn = activation_fn
    main(args)
