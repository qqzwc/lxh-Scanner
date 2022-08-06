
import cv2
import numpy as np

def order_points(pts): # 将顶点按 [左上，右上，右下，左下] 的顺序排列
    rect = np.zeros((4,2), dtype = "float32")

    s = np.sum(pts, axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(img, pts): # 根据给定四对顶点计算目标矩形框并进行透视变换
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 取平均宽高
    widthA = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    widthB = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    # avgWidth = max(int(widthA), int(widthB))
    avgWidth = int((widthA + widthB) / 2)

    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    # avgHeight = max(int(heightA), int(heightB))
    avgHeight = int((heightA + heightB) / 2)

    print(avgWidth, avgHeight)
    dst = np.array([
        [0,0],
        [avgWidth - 1, 0],
        [avgWidth -1, avgHeight -1],
        [0, avgHeight -1]], dtype = "float32")
        
    print(dst)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (avgWidth, avgHeight))
    
    return warped