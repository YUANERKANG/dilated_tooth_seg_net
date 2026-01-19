import torch
import trimesh


class MoveToOriginTransform(object):
    def __init__(self):
        pass

    def __call__(self, samples, mesh):
        """
        Center the point cloud and shift the original mesh.
        
        Args:
            samples: (N, 3) numpy array of sampled points
            mesh: trimesh object
        
        Returns:
            centered samples, shifted mesh
        """
        # Calculate center from samples
        center = samples.mean(axis=0)
        
        # Center the samples
        samples = samples - center
        
        # Shift the mesh vertices
        mesh.vertices = mesh.vertices - center
        
        return samples, mesh


class PreTransform(object):
    def __init__(self, classes=17):
        self.classes = classes

    def __call__(self, data):
        """
        Construct 24D feature tensor from sampled points and mesh data.
        Uses vectorized operations for normalization.
        
        Args:
            data: tuple of (samples, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels)
        
        Returns:
            pos, x (24D features), labels
        """
        samples, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
        
        # Get number of samples
        s = samples.shape[0]
        
        # Initialize feature tensor (N, 24)
        x = torch.zeros(s, 24).float()
        
        # Fill in features:
        # [0:3]   - Triangle vertex 1
        # [3:6]   - Triangle vertex 2
        # [6:9]   - Triangle vertex 3
        # [9:12]  - Sample position (replaces face center)
        # [12:15] - Vertex normal 1
        # [15:18] - Vertex normal 2
        # [18:21] - Vertex normal 3
        # [21:24] - Face normal
        x[:, :3] = mesh_triangles[:, 0]
        x[:, 3:6] = mesh_triangles[:, 1]
        x[:, 6:9] = mesh_triangles[:, 2]
        x[:, 9:12] = samples  # Use sampled positions instead of face centers
        x[:, 12:15] = mesh_vertices_normals[:, 0]
        x[:, 15:18] = mesh_vertices_normals[:, 1]
        x[:, 18:21] = mesh_vertices_normals[:, 2]
        x[:, 21:] = mesh_face_normals
        
        # Vectorized normalization
        # Calculate statistics for triangle vertices (from all triangles)
        all_triangle_vertices = mesh_triangles.reshape(-1, 3)  # (N*3, 3)
        means = all_triangle_vertices.mean(dim=0)  # (3,)
        stds = all_triangle_vertices.std(dim=0)    # (3,)
        
        # Calculate statistics for samples (positions)
        maxs = samples.max(dim=0)[0]  # (3,)
        mins = samples.min(dim=0)[0]  # (3,)
        
        # Calculate statistics for vertex normals
        all_vertex_normals = mesh_vertices_normals.reshape(-1, 3)  # (N*3, 3)
        nmeans = all_vertex_normals.mean(dim=0)  # (3,)
        nstds = all_vertex_normals.std(dim=0)    # (3,)
        
        # Calculate statistics for face normals
        nmeans_f = mesh_face_normals.mean(dim=0)  # (3,)
        nstds_f = mesh_face_normals.std(dim=0)    # (3,)
        
        # Vectorized normalization for all three coordinate dimensions
        for i in range(3):
            # Normalize triangle vertices (Z-Score)
            x[:, i] = (x[:, i] - means[i]) / stds[i]        # vertex 1
            x[:, i + 3] = (x[:, i + 3] - means[i]) / stds[i]  # vertex 2
            x[:, i + 6] = (x[:, i + 6] - means[i]) / stds[i]  # vertex 3
            
            # Normalize sample positions (Min-Max)
            x[:, i + 9] = (x[:, i + 9] - mins[i]) / (maxs[i] - mins[i])
            
            # Normalize vertex normals (Z-Score)
            x[:, i + 12] = (x[:, i + 12] - nmeans[i]) / nstds[i]  # normal 1
            x[:, i + 15] = (x[:, i + 15] - nmeans[i]) / nstds[i]  # normal 2
            x[:, i + 18] = (x[:, i + 18] - nmeans[i]) / nstds[i]  # normal 3
            
            # Normalize face normal (Z-Score)
            x[:, i + 21] = (x[:, i + 21] - nmeans_f[i]) / nstds_f[i]
        
        # Use samples as position
        pos = x[:, 9:12]
        
        return pos, x, labels
