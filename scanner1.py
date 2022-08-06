# coding=utf-8
import cv2
import imutils
import imgEnhance
from py_utils import four_point_transform

def get_rectangle(img, img_height):
    ratio = img.shape[0] / img_height
    img = imutils.resize(img, height=img_height)
    # cv2.imshow('Origin', img)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray', grayimg)
    gaussimg = cv2.GaussianBlur(grayimg, (5, 5), 0)
    # cv2.imshow('Gaussian', gaussimg)
    edgedimg = cv2.Canny(gaussimg, 100, 200)
    # cv2.imshow("edge", edgedimg)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(
        edgedimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,
                      reverse=True)[:5]  # 找出面积最大的前五个轮廓

    # cv2.drawContours(img,contours,-1,(0,0,255),3,lineType=cv2.LINE_AA)
    # cv2.imshow('Contours', img)
    autoRect = None 
    for c in contours:
        peri = cv2.arcLength(c, True)  # Calculating contour circumference
        approxRect = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(len(approxRect))
        if len(approxRect) == 4:
            autoRect = approxRect
            break

    assert autoRect is not None, "Can't find a rectangle!"

    cv2.drawContours(img, [autoRect], -1,
                     (255, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.imshow('Rect', img)
    return autoRect.reshape(4, 2) * ratio





if __name__ == "__main__":
    img_height = 600
    img_path = "./test_imgs/test1.jpg"
    # img_path = "C:/Users/leexi/Desktop/hhh.png"
    img = cv2.imread(img_path)
    assert img is not None, "Can't open {img_path}"
    autoRect = get_rectangle(img, img_height)
    warpedImg = four_point_transform(img, autoRect)

    # cv2.imshow("warped", warped)
    # enhancer = imgEnhance.Enhancer()
    # enhancedImg = enhancer.gamma(warped, 1.63)

    # cv2.imshow("Origin img", imutils.resize(img, height=img_height))
    cv2.imshow("Warped img", imutils.resize(warpedImg, height=img_height))
    # cv2.imshow("gamma", imutils.resize(enhancedImg, height=img_height))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
