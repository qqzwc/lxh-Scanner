# coding=utf-8
import cv2
import imutils
import imgEnhance
import numpy as np
from py_utils import four_point_transform
 

class DrawPoints(object):
    def __init__(self, image, image_height, color,
                 marker_type=cv2.MARKER_CROSS,
                 marker_size=20,
                 thickness=1):
        """
        Initialization of class DrawPoints

        Parameters
        ----------
        image: ndarray
            source image. shape is [height, width, channels]
        color: tuple
            a tuple containing uint8 integers, designating B, G, R values,
            separately
        marker_type: int
            marker type, between [0, 6]
        marker_size: int
            marker size, >=1
        thickness: int
            line thickness, >=1
        """

        self.img_height = image_height
        self.ratio = image.shape[0] / img_height
        image = imutils.resize(image, height=image_height)
        self.original_image = image
        self.image_for_show = image.copy()
        self.color = color
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.thickness = thickness
        self.pts = []

    def append(self, x, y):
        """
        add a point to points list

        Parameters
        ----------
        x, y: int, int
            coordinate of a point
        """
        self.pts.append([x, y])

    def pop(self):
        """
        pop a point from points list
        """
        pt = ()
        if self.pts:
            pt = self.pts.pop()
        return pt

    def reset_image(self):
        """
        reset image_for_show using original image
        """
        self.image_for_show = self.original_image.copy()

    def draw(self):
        """
        draw points on image_for_show
        """
        for pt in self.pts:
            cv2.drawMarker(self.image_for_show, pt, color=self.color,
                           markerType=self.marker_type,
                           markerSize=self.marker_size,
                           thickness=self.thickness)

    def get_rectangle(self):
        return np.array(self.pts) * self.ratio

def onmouse_pick_points(event, x, y, flags, draw_pts):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('add: x = %d, y = %d' % (x, y))
        draw_pts.append(x, y)
        draw_pts.draw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        pt = draw_pts.pop()
        if pt:
            print('delete: x = %d, y = %d' % (pt[0], pt[1]))
            draw_pts.reset_image()
            draw_pts.draw()



    
if __name__ == '__main__':
    img_path = "./test_imgs/test1.jpg"
    img = cv2.imread(img_path)
    img_height = 600
    draw_pts = DrawPoints(img, img_height, (0, 255, 0))
    
    cv2.namedWindow('Pick points', 0)
    cv2.setMouseCallback('Pick points', onmouse_pick_points, draw_pts)
    while True:
        cv2.imshow('Pick points', draw_pts.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            exit(0)
        elif key == 32: # space
            userRect = draw_pts.get_rectangle()
            assert len(userRect) == 4, f"You shoud draw 4 points rather than {len(userRect)} points"
            break

    warpedImg = four_point_transform(img, userRect)

    # enhancer = imgEnhance.Enhancer()
    # enhancedImg = enhancer.sharp(warpedImg, 3.0)

    # cv2.imshow("Origin img", imutils.resize(img, height=img_height))
    cv2.imshow("Warped img", imutils.resize(warpedImg, height=img_height))
    # cv2.imshow("Sharped img", imutils.resize(enhancedImg, height=img_height))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
