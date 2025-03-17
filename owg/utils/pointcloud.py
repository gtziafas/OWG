import numpy as np
import open3d as o3d
import trimesh
import copy
import os

def to_o3d(points, colors=None, normals=None):
    x = o3d.geometry.PointCloud()
    x.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        x.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        x.normals = o3d.utility.Vector3dVector(normals)
    return x


def o3d_viewer(geometries, title="display", world_frame=False, background=[0.75, 0.75, 0.75]):
    if world_frame:
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[-0.0,-0.0,-0.0]))
    
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name=title)
    for geometry in geometries:
        viewer.add_geometry(copy.deepcopy(geometry))
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = False
    opt.background_color = np.asarray(background)
    viewer.run()
    viewer.destroy_window()


def trimesh_to_o3d(tri_mesh):
    # Extract vertices and faces from trimesh mesh
    vertices = np.array(tri_mesh.vertices)
    faces = np.array(tri_mesh.faces)

    # Create Open3D mesh
    open3d_mesh = o3d.geometry.TriangleMesh()

    # Set vertices and faces
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    return open3d_mesh


def o3d_geometry_to_trimesh(geometry):
    """
    Convert an Open3D TriangleMesh or PointCloud to Trimesh.
    """
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        vertices = np.asarray(geometry.vertices)
        faces = np.asarray(geometry.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        vertices = np.asarray(geometry.points)
        # Point clouds are treated as point meshes (no faces)
        return trimesh.points.PointCloud(vertices)
    else:
        raise ValueError("Unsupported geometry type: {}".format(type(geometry)))


# gripper_mesh = create_robotiq_mesg('owg_robot/assets/robotiq_2f_140/robotiq_arg2f_140.obj')
def create_robotiq_mesh(path):
    mesh = trimesh.load(path)
    mesh = trimesh_to_o3d(mesh).paint_uniform_color([0,0,0])

    theta = np.pi / 2
    R = np.array([
                    [np.cos(theta), 0, np.sin(theta), -0.0],
                    [0, 1, 0, 0.0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1],
    ])
    mesh.transform(R)
    mesh.translate([-0.2, 0, 0])
    return mesh


def get_camera_matrix(lookat, front, up):
    """
    Build a view matrix from lookat, front, up vectors.
    """
    # Normalize front and up
    front = front / np.linalg.norm(front)
    up = up / np.linalg.norm(up)
    right = np.cross(front, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, front)

    # Create rotation matrix
    rot = np.stack((right, up, -front), axis=1)
    trans = -rot.T @ lookat
    view = np.eye(4)
    view[:3, :3] = rot.T
    view[:3, 3] = trans
    return view


def render_o3d_image(geometry_list, lookat, front, up, zoom, width=224, height=224, background_color=(1, 1, 1)):
    """
    Render a list of Open3D geometries to an image using Visualizer and return it as a numpy array.

    Args:
        geometry_list (list): List of Open3D geometry objects.
        lookat (list or np.ndarray): Camera lookat target [x, y, z].
        front (list or np.ndarray): Camera front direction [x, y, z].
        up (list or np.ndarray): Camera up direction [x, y, z].
        zoom (float): Camera zoom factor.
        width (int): Width of the output image.
        height (int): Height of the output image.
        background_color (tuple): Background color as RGB tuple, values in [0, 1].

    Returns:
        np.ndarray: Rendered image as (height, width, 3) numpy array in RGB format.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray(background_color)

    for geometry in geometry_list:
        vis.add_geometry(geometry)

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    # Capture the screen as a numpy array
    image = vis.capture_screen_float_buffer(do_render=True)
    image_np = (np.asarray(image) * 255).astype(np.uint8)  # Convert to uint8

    vis.destroy_window()

    return image_np

# def render_o3d_image(geometry_list, lookat, front, up, zoom, width=224, height=224, background_color=(1, 1, 1)):
#     # Use offscreen rendering with EGL
#     render = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     render.scene.set_background(background_color)
    
#     # Add geometries to scene
#     for idx, geometry in enumerate(geometry_list):
#         render.scene.add_geometry(f"geometry_{idx}", geometry, o3d.visualization.rendering.Material())
    
#     # Set up the camera
#     camera = o3d.camera.PinholeCameraParameters()
#     # You'll need to convert your lookat/front/up/zoom parameters to a view matrix
#     # This is a simplified version - you may need to adapt this
#     camera_pos = np.array(lookat) - np.array(front) * (1.0/zoom)
#     render.setup_camera(60.0, camera_pos, lookat, up)
    
#     # Render the image
#     img = render.render_to_image()
#     # Convert to numpy array
#     img_np = np.asarray(img)
    
#     return img_np