import argparse
import os
import sys

import numpy as np
import open3d as o3d

USE_NORMALS = True

def load_and_show_ply2(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    mesh.compute_vertex_normals()
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])

def is_triangle_mesh(filepath):
    if filepath.endswith(".obj"):
        return True

    # Check the contents of the PLY file for the 'element face' line
    try:
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                if 'element face' in line:
                    return True
    except Exception as e:
        print(f"Error reading file: {e}")
    return False

# if we want to render videos down the line, this might be nice
# https://www.open3d.org/docs/0.12.0/tutorial/visualization/customized_visualization.html#change-field-of-view
def load_and_show_ply(
    filepath,
    *,
    flip_triangles=False,
    color_normals=True,
    window_width=1600,
    window_height=1200,
):
    if is_triangle_mesh(filepath):
        mesh = o3d.io.read_triangle_mesh(filepath)
        if flip_triangles:
            triangles = np.asarray(mesh.triangles)
            # flip triangles
            triangles = triangles[:, [0, 2, 1]]
            mesh = o3d.geometry.TriangleMesh(vertices=mesh.vertices, triangles=o3d.utility.Vector3iVector(triangles))

        mesh.compute_vertex_normals()
        print("read triangle")
    else:
        mesh = o3d.io.read_point_cloud(filepath)
        print("read point cloud")
    # Visualize the geometry (whether it's a point cloud or triangle mesh)
    # o3d.visualization.draw_geometries([mesh])

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()

    # Create a window with the filename as the title
    filename = os.path.basename(filepath)
    # vis.create_window(window_name=filepath)
    vis.create_window(window_name=filepath, width=window_width, height=window_height)

    if color_normals and mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        colors = (normals + 1) / 2  # Normalize to [0, 1]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Add the geometry to the visualizer
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.set_constant_z_near(0.001)
    ctr.set_constant_z_far(1000)

    center = mesh.get_center()
    
    ctr.set_front(center[:, None])
    ctr.set_up(np.array([0,1,0])[:, None])
        
    renderoption = vis.get_render_option()
    renderoption.light_on = False
        
    # Run the visualizer
    vis.run()

    # Close the visualizer window
    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(description="Quick viewer for meshes or point clouds.")
    parser.add_argument("path", help="Path to the mesh/point cloud file to visualize.")
    parser.add_argument(
        "--flip-triangles",
        action="store_true",
        help="Flip triangle winding before rendering (useful if normals look inverted).",
    )
    parser.add_argument(
        "--no-color-normals",
        action="store_true",
        help="Disable coloring vertices by their normals.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Viewer window width in pixels (default: 1600).",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1200,
        help="Viewer window height in pixels (default: 1200).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("press h for controls (;")

    load_and_show_ply(
        args.path,
        flip_triangles=args.flip_triangles,
        color_normals=not args.no_color_normals,
        window_width=args.window_width,
        window_height=args.window_height,
    )