"""
This script is for 3d model file(glTF/glB) reading using trimesh.
"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
import scipy
import pandas
import os
import json
import copy
import struct

model_path = 'D://Microsoft Download/Formal Files/data file/3d_model/ParkerSolarProbe/PSP.glb'
mesh = trimesh.load(model_path, force='mesh')
all_psp_vertices = copy.deepcopy(mesh.vertices)
all_psp_faces = copy.deepcopy(mesh.faces)
fig_1 = plt.figure()
ax_3d = fig_1.add_subplot(111, projection='3d')
for i in range(len(all_psp_faces[:, 0])):
    ax_3d.plot([all_psp_vertices[all_psp_faces[i, 0], 0], all_psp_vertices[all_psp_faces[i, 1], 0]],
               [all_psp_vertices[all_psp_faces[i, 0], 1], all_psp_vertices[all_psp_faces[i, 1], 1]],
               [all_psp_vertices[all_psp_faces[i, 0], 2], all_psp_vertices[all_psp_faces[i, 1], 2]],
               c='red')
    ax_3d.plot([all_psp_vertices[all_psp_faces[i, 0], 0], all_psp_vertices[all_psp_faces[i, 2], 0]],
               [all_psp_vertices[all_psp_faces[i, 0], 1], all_psp_vertices[all_psp_faces[i, 2], 1]],
               [all_psp_vertices[all_psp_faces[i, 0], 2], all_psp_vertices[all_psp_faces[i, 2], 2]],
               c='red')
    ax_3d.plot([all_psp_vertices[all_psp_faces[i, 2], 0], all_psp_vertices[all_psp_faces[i, 1], 0]],
               [all_psp_vertices[all_psp_faces[i, 2], 1], all_psp_vertices[all_psp_faces[i, 1], 1]],
               [all_psp_vertices[all_psp_faces[i, 2], 2], all_psp_vertices[all_psp_faces[i, 1], 2]],
               c='red')
plt.show()

