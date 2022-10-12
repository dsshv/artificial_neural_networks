import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# плот внутреннего выреза
def display_inliers(cloud, ind):
    # выбор точек внутреннего выреза по индексу
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    # Создание координат на фрейме для плота
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #Границы поинт клауда
    max_bound = cloud.get_max_bound()
    min_bound = cloud.get_min_bound()
    #Выровнять клауд по оси
    axisbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    # плот данных
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, mesh_frame, axisbox], width=1280, height=1024)
#плот поинт клауда
def display_cloud (pcd):
    # Создание координат на фрейме для плота
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    #Границы поинт клауда
    max_bound = pcd.get_max_bound()
    min_bound = pcd.get_min_bound()
    # Выровнять клауд по оси
    axisbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # плот данных
    width = 1280
    height = 1024
    o3d.visualization.draw_geometries([pcd, mesh_frame, axisbox], width=width, height=height)




#Считать поинт клауд
pcd = o3d.io.read_point_cloud("7.ply")
#Вывод исходного клауда
print("Input Info:", pcd)
print("Step 1 - Visualize Input Point Cloud")
display_cloud(pcd)

#Получить все цвета точек в клауде
save_colors = o3d.utility.Vector3dVector(pcd.colors)


#DBSCAN
#Кластеризация, eps - расстояние между соседями, min- минимум точек для объединения в класс
labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=150, print_progress=True))
#получить количество классов
max_label = labels.max()
print(f"Point Cloud has {max_label + 1} clusters")
#назначить цвета для классов
colors = plt.get_cmap("tab20c")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
#записать новые цвета в исходных клауд
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
display_cloud(pcd)

#Восстановить исходные цвета
pcd.colors = save_colors

#Сделать разрез клауда, оставив самые большие классы
pcd_clear, ind = pcd.remove_radius_outlier(nb_points=1000, radius=0.04)



#сегментировать разрез по классам
plane_model, inliers = pcd_clear.segment_plane(distance_threshold=0.0025, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Calcualtion of Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
display_inliers(pcd_clear, inliers)
