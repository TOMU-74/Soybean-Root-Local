#動作確認
#print("hello")
#カレントディレクトリの確認
#print(os.getcwd())

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os

#まとめて実行
images = glob.glob('./input/*.JPG')
#個別に実行
#images = glob.glob('./input/-002-1_cal.JPG')
#結果が真円になっているか確認する用
#images = glob.glob('./output/-002-1_norm.JPG')

for fname in images:
    #ファイルの名前付けのための変数
    fbase = os.path.basename(fname.replace('_cal', ''))
    ftitle, fext = os.path.splitext(fbase)
    #画像の読み込み
    image = cv2.imread(fname)
    #画像の調整
    #トリミング[top : bottom, left : right]
    image_T =image[0 : 2040, 0 : 2040]
    #グレースケールに変換する
    gray_image = cv2.cvtColor(image_T, cv2.COLOR_BGR2GRAY) 
    #2値化 (画像, 値以下を, 値にする, type)
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    #楕円の輪郭抽出
    contours,hierarchy =  cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    #楕円フィッティング((中心x, 中心y),(楕円縦方向の長さ, 楕円横方向の長さ),傾き角度)
    ellipse = cv2.fitEllipse(max_contour)
    cx = ellipse[0][0]
    cy = ellipse[0][1]
    center = (cx + 2000, cy + 2000)
    h = ellipse[1][0]
    w = ellipse[1][1]
    deg = ellipse[2]
    #結果の表示
    #print('elipse : ', ellipse)
    #print(cx, cy, h, w, deg)

    #楕円を真円に変換
    afin_matrix1 = np.float32([[1, 0, 2000], [0, 1, 2000]])
    afin_matrix2 = cv2.getRotationMatrix2D(center = center, angle=deg, scale=1)
    afin_matrix3 = np.float32([[(w/h), 0, 0], [0, 1, 0]])
    afin_matrix4 = cv2.getRotationMatrix2D(center = center, angle=-deg, scale=1)
    afin_matrix5 = np.float32([[1, 0, -2000], [0, 1, -2000]])
    image_A = cv2.warpAffine(image, afin_matrix1, (4160 * 2, 4160 * 2))
    image_A = cv2.warpAffine(image_A, afin_matrix2, (4160 * 2, 4160 * 2))
    image_A = cv2.warpAffine(image_A, afin_matrix3, (4160 * 2, 4160 * 2))
    image_A = cv2.warpAffine(image_A, afin_matrix4, (4160 * 2, 4160 * 2))
    image_A = cv2.warpAffine(image_A, afin_matrix5, (4160, 4160))
    cv2.imwrite('./output/' + ftitle + '_norm' + fext, image_A)
    print(ftitle)

    
#画像の確認
#plt.imshow(image_A)
#plt.show()