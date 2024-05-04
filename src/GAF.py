import os
import argparse
import numpy as np
import cv2
import io
from PIL import Image

def rotateImage(img, orientation):
    #orientationの値に応じて画像を回転させる
    if orientation == 1:
        pass
    elif orientation == 2:
        #左右反転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        #180度回転
        img_rotate = img.transpose(Image.ROTATE_180)
    elif orientation == 4:
        #上下反転
        img_rotate = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        #左右反転して90度回転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
    elif orientation == 6:
        #270度回転
        img_rotate = img.transpose(Image.ROTATE_270)
    elif orientation == 7:
        #左右反転して270度回転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
    elif orientation == 8:
        #90度回転
        img_rotate = img.transpose(Image.ROTATE_90)
    else:
        pass

    return img_rotate

def resize (image, size):
    h, w = image.shape[:2]
    width = size
    height = round(h * (width / w))
    image = cv2.resize(image, dsize=(width, height))
    return image

def main(image):
    image = cv2.imread(str(image))
    originImage = image

    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


    # モデルを読み込む
    weights = "../models/yunet3.onnx"
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    weights = "../models/face_recognizer_fast.onnx"
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    _, width, _ = image.shape
    size = width

    while True:
        image = resize(originImage, size)
        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))
        # 顔を検出する
        hoge, faces = face_detector.detect(image)
        if not(faces is None):
            break
        size = size - 100
        if size < 300:
            break
        print(size)


    # 検出された顔を切り抜く
    aligned_faces = []
    if faces is not None:
        for face in faces:
            aligned_face = face_recognizer.alignCrop(image, face)
            aligned_faces.append(aligned_face)

    # 画像を表示、保存する
    for i, aligned_face in enumerate(aligned_faces):
        cv2.imwrite(os.path.join("../output/crop/", "face{:03}.jpg".format(i + 1)), aligned_face)

if __name__ == '__main__':
    main()

