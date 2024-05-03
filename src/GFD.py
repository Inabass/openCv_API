import os
import sys
import argparse
import numpy as np
import cv2

def main():
    # 画像を開く
    image = cv2.imread("../output/crop/face001.jpg")
    if image is None:
        exit()

    # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # モデルを読み込む
    weights = "../models/face_recognizer_fast.onnx"
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # 特徴を抽出する
    face_feature = face_recognizer.feature(image)

    # 特徴を保存する
    # basename = os.path.splitext(os.path.basename(args.image))[0]
    dictionary = os.path.join("../output/npy", "newface")
    np.save(dictionary, face_feature)

    if face_feature is None:
        return False
    
    return True


if __name__ == '__main__':
    main()
