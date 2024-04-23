import torch
import os
import sys
import timm
import tempfile
import trimesh
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from zoedepth.utils.misc import get_image_from_url, colorize
from zoedepth.utils.geometry import depth_to_points, create_triangles

script_path = Path(__file__).parent


def create3D(img_path):
    # For locally saved model
    path = os.path.join(script_path, "__models__", "models_zoedepth")
    model = torch.hub.load(path, "ZoeD_N", source="local", pretrained=True, trust_repo=True, use_pretrained_midas=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    zoe = model.to(DEVICE)
    img = Image.open(img_path).convert("RGB")
    get_mesh(zoe, img)


def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def predict_depth(model, image):
    depth = model.infer_pil(image)
    #        colored_depth = colorize(depth)
    return depth


def get_mesh(model, image, keep_edges=False):
    depth = predict_depth(model, image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)

    # Create a trimesh mesh from the points
    # Each pixel is connected to its 4 neighbors
    # colors are the RGB values of the image

    verts = pts3d.reshape(-1, 3)
    image = np.array(image)

    if keep_edges:
        triangles = create_triangles(image.shape[0], image.shape[1])
    else:
        triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))

    colors = image.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

    # Save as 3d object
    object_files_folder = "__ObjectFiles__"
    if object_files_folder not in os.listdir(script_path):
        os.mkdir(object_files_folder)

    time_now = str(datetime.datetime.now().strftime("%H:%M:%S")).replace(":", "_")
    cwd = os.getcwd()
    dir_name = f"2d_to_3d_{time_now}"
    os.mkdir(os.path.join(script_path, object_files_folder, dir_name))

    glb_path = os.path.join(script_path, object_files_folder, dir_name, f"{time_now}.glb")
    stl_path = os.path.join(script_path, object_files_folder, dir_name, f"{time_now}.stl")
    obj_path = os.path.join(script_path, object_files_folder, dir_name, f"{time_now}.obj")

    mesh.export(glb_path)
    mesh.export(stl_path)
    mesh.export(obj_path)

    return glb_path
