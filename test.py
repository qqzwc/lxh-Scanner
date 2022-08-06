# -*- coding: utf-8 -*-

import cv2
import numpy as np

WIN_NAME = 'pick_points'


class DrawPoints(object):
    def __init__(self, image, color,
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
        self.pts.append((x, y))

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
    image = np.zeros((256, 256, 3), np.uint8)
    draw_pts = DrawPoints(image, (0, 255, 0))
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, draw_pts)
    while True:
        cv2.imshow(WIN_NAME, draw_pts.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()
