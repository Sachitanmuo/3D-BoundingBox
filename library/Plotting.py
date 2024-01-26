import cv2
import numpy as np
from enum import Enum
import itertools

from .File import *
from .Math import *

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)
    point = np.dot(cam_to_img, point)
    #point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)
    return point



# take in 3d points and plot them on image as red circles
def plot_3d_pts(img, pts, center, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        if relative:
            pt = [i + center[j] for j,i in enumerate(pt)] # more pythonic

        point = project_3d_pt(pt, cam_to_img)

        color = cv_colors.RED.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)



def plot_3d_box(img, cam_to_img, ry, dimension, center):

    #plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)
    #plot_bev(corners, center)
    # 繪製Bird's Eye View
    # to see the corners on image as red circles
    plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)
    print(f"center:{center}")
    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    #TODO put into loop
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)

    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.GREEN.value, 1)

    front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]

    cv2.line(img, front_mark[0], front_mark[3], cv_colors.BLUE.value, 1)
    cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)

def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)



def plot_bev(bev_img, center, ry, dimensions, scale=1, image_size=(500, 1000)):
    # 假設corners是一個包含8個角點的列表，每個角點是[x, y, z]格式
    # 並且只取底部的四個角點（通常是列表的前四個元素）
    R = rotation_matrix(ry)
    corners = create_corners(dimensions, location=center, R=R)
    # 提取底部四個角點的X和Y座標
    corner_indices = [2, 3, 6, 7]
    base_corners = [corners[i] for i in corner_indices]
    base_corners[1], base_corners[2], base_corners[3] = base_corners[2], base_corners[3], base_corners[1]
    for item in base_corners:
        print(item)
    x_coords = [corner[0]  for corner in base_corners]
    z_coords = [corner[2]  for corner in base_corners]
    print(f"x_coords = {x_coords}")
    print(f"z_coords = {z_coords}")

    x_offset = image_size[0] // 2
    z_offset = image_size[1]
    origin = (x_offset, z_offset)
    # 將座標縮放並轉換為圖像座標系（原點在底部中間）
    x_coords = [int((x) * scale + x_offset) for x in x_coords]
    z_coords = [int((-z) * scale + z_offset) for z in z_coords]
    print(f"plot_x_coords = {x_coords}")
    print(f"plot_z_coords = {z_coords}")
    # 創建一個空白圖像
    #bev_img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8) + 255

    # 繪製同心圓
    for radius in range(1, 11, ):
        scaled_radius = int(radius*10*scale)
        cv2.circle(bev_img, origin, scaled_radius, (200, 200, 200), 2)
        cv2.putText(bev_img, f"{radius*10}m", (origin[0] + scaled_radius + 5, origin[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # 繪製邊界框的底部
    for i in range(4):
        start_point = (x_coords[i], z_coords[i])
        end_point = (x_coords[(i + 1) % 4], z_coords[(i + 1) % 4])
        cv2.line(bev_img, start_point, end_point, (255, 0, 0), 2)
