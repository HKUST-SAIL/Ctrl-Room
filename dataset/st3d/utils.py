import numpy as np
import trimesh
import open3d

from typing import List, Tuple

def create_spatial_quad_polygen(quad_vertices: List, 
                                normal: np.array, 
                                camera_center: np.array):
    """create a quad polygen for spatial mesh
    """
    if camera_center is None:
        camera_center = np.array([0, 0, 0])
    quad_vertices = (quad_vertices - camera_center)
    quad_triangles = []
    triangle = np.array([[0, 2, 1], [2, 0, 3]])
    quad_triangles.append(triangle)

    quad_triangles = np.concatenate(quad_triangles, axis=0)

    mesh = trimesh.Trimesh(vertices=quad_vertices,
                           faces=quad_triangles,
                           vertex_normals=np.tile(normal, (4, 1)),
                           process=False)

    centroid = np.mean(quad_vertices, axis=0)
    # print(f'centroid: {centroid}')
    normal_point = centroid + np.array(normal) * 0.5
    # print(f'normal_point: {normal_point}')

    pcd_o3d = open3d.geometry.PointCloud()
    pcd_o3d.points = open3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd_o3d.points.append(normal_point)
    pcd_o3d.points.append(centroid)

    # pcd = trimesh.PointCloud(np.asarray(pcd.points))
    return mesh, pcd_o3d