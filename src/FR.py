import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
import io

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

# 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
def match(recognizer, feature1, dictionary):
    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score > COSINE_THRESHOLD:
            return True, (user_id, score * 100)
    return False, ("", 0.0)

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
    # 特徴を読み込む
    dictionary = []
    files = glob.glob(os.path.join("../output/npy/", "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feature))

    # モデルを読み込む
    weights = "../models/yunet3.onnx"
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    weights = "../models/face_recognizer_fast.onnx"
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    while True:
        # 画像を読み込む
        image = cv2.imread(str(image))
        originImage = image
    
        h, w = image.shape[:2]
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

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        result, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            # 顔を切り抜き特徴を抽出する
            aligned_face = face_recognizer.alignCrop(image, face)
            feature = face_recognizer.feature(aligned_face)

            # 辞書とマッチングする
            result, user = match(face_recognizer, feature, dictionary)

            # 顔のバウンディングボックスを描画する
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # 認識の結果を描画する
            id, score = user if result else ("unknown", 0.0)
            text = "{0} ({1:.2f})".format(id, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

        # 画像を表示する
        cv2.imwrite(os.path.join("../output/result/", "result.jpg".format(1)), image)

if __name__ == '__main__':
    main()