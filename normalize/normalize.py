#動作確認
#print("hello")
#カレントディレクトリの確認
#print(os.getcwd())

#---ライブラリのインポート---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os

#---入力画像の指定---
#まとめて実行
images = glob.glob('./input/*.JPG')
#個別に実行
#images = glob.glob('./input/JMC126+3_cal.JPG')
#images = glob.glob('./input/WMC191+1_cal.JPG')
#結果の確認する用
#images = glob.glob('./ok_out/*.JPG')

#============メイン部分=================
for fname in images:
    #---処理する画像の読み込み---
    #ファイルの名前付けのための変数
    fbase = os.path.basename(fname.replace('_cal', ''))
    ftitle, fext = os.path.splitext(fbase)
    #画像の読み込み
    image = cv2.imread(fname)

    #---画像サイズを4160x4160にする---
    # 新しい黒い画像（4160x4160）を生成
    new_image = np.zeros((4160, 4160, 3), dtype=np.uint8)
    # 元の画像を新しい画像の中央に配置
    y_offset = (4160 - image.shape[0]) // 2
    x_offset = (4160 - image.shape[1]) // 2
    new_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
    image = new_image

    #---楕円フィッティングのための前処理---
    # トリミング[top : bottom, left : right]
    image_T =image[0 : 1560, 2600 : 4160]
    #グレースケールに変換する
    gray_image = cv2.cvtColor(image_T, cv2.COLOR_BGR2GRAY) 
    #2値化 (画像, 値以下なら黒, 以上を白, type)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    #楕円の輪郭抽出
    contours,hierarchy =  cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
    max_contour = max(contours, key=cv2.contourArea)

    #---楕円フィッティング---
    #elipse:((中心x, 中心y),(楕円縦方向の長さ, 楕円横方向の長さ),傾き角度)
    ellipse = cv2.fitEllipse(max_contour)
    cx = ellipse[0][0]
    cy = ellipse[0][1]
    h = ellipse[1][0]
    w = ellipse[1][1]
    deg = ellipse[2]

    #=============デバッグ用部分=================
    #---取得した楕円の確認---
    #結果の表示
    #print('elipse : ', ellipse)
    #print(cx, cy, h, w, deg)

    # 検出した楕円を画像に描画
    #output_image_T = image_T.copy()
    #cv2.ellipse(output_image_T, ellipse, (0, 255, 0), 2)  # 緑色で楕円を描画

    # 描画された画像を保存
    #cv2.imwrite('./check_out/' + ftitle + '_ellipse' + fext, output_image_T)
    #print(ftitle, ' : Ellipse drawn on image and saved.')
    #=========================================

    #---傾き補正のため画像を変換---
    #アフィン変換の行列を作成
    #画像を中心へ平行移動
    #中心の6240を基準とし、cx,cyを引いて楕円の中心へ。この時x方向に2600引いているのは、トリミングした分を考慮するため。
    center = (6240 - 2600 -cx, 6240 - cy)
    afin_matrix1 = np.float32([[1, 0, center[0]], [0, 1, center[1]]]) 
    #画像を中心を軸に楕円の傾き分回転
    afin_matrix2 = cv2.getRotationMatrix2D(center = (6240, 6240), angle=deg, scale=1)
    #楕円を真円にするため、楕円の横方向の長さwと縦方向の長さhの比w/hの倍率で拡大縮小
    afin_matrix3 = np.float32([[(w/h), 0, 0], [0, 1, 0]])
    #画像を再び水平に戻す
    afin_matrix4 = cv2.getRotationMatrix2D(center = (6240, 6240), angle=-deg, scale=1)
    #画像を元の位置に戻す
    afin_matrix5 = np.float32([[1, 0, -center[0]], [0, 1, -center[1]]])

    #アフィン変換の実行
    # 4160*3の大きな画像に対して変換を行い、変換時に画像が切れないようにする
    image_A = cv2.warpAffine(image, afin_matrix1, (4160 * 3, 4160 * 3))
    image_A = cv2.warpAffine(image_A, afin_matrix2, (4160 * 3, 4160 * 3))
    image_A = cv2.warpAffine(image_A, afin_matrix3, (4160 * 3, 4160 * 3))
    image_A = cv2.warpAffine(image_A, afin_matrix4, (4160 * 3, 4160 * 3))
    image_A = cv2.warpAffine(image_A, afin_matrix5, (4160, 4160))
    
    #---結果の確認と保存---
    #トリミングして楕円が真円になっているか確認
    image_check_T =image_A[0 : 1560, 2600 : 4160]
    gray_image_check = cv2.cvtColor(image_check_T, cv2.COLOR_BGR2GRAY) 
    _, binary_image_check = cv2.threshold(gray_image_check, 150, 255, cv2.THRESH_BINARY)
    contours_check,hierarchy_check =  cv2.findContours(binary_image_check,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #輪郭が取れなかった場合の処理
    if not contours_check:
        print(ftitle, ' : NG (no contour)')
        cv2.imwrite('./ng_out/' + ftitle + '_null' + fext, image)
        continue
    #輪郭が取れた場合の処理
    max_contour_check = max(contours_check, key=cv2.contourArea)
    ellipse_check = cv2.fitEllipse(max_contour_check)
    h_check = ellipse_check[1][0]
    w_check = ellipse_check[1][1]
    # 長編径と短径の差が1以下ならOK, それ以上ならNGとして保存
    if(abs(h_check - w_check) > 1):
        print(ftitle, ' : NG')
        cv2.imwrite('./ng_out/' + ftitle + '_norm' + fext, image_A)
    else:
        print(ftitle, ' : OK')
        cv2.imwrite('./ok_out/' + ftitle + '_norm' + fext, image_A)

'''
# =============デバッグ用部分=================
# ---2値化画像の確認---
for fname in images:
    #ファイルの名前付けのための変数
    fbase = os.path.basename(fname.replace('_cal', ''))
    ftitle, fext = os.path.splitext(fbase)
    #画像の読み込み
    image = cv2.imread(fname)
    #画像の調整
    #グレースケールに変換する
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #2値化 (画像, 値以下を, 値にする, type)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./check_out/' + ftitle + '_binary' + fext, binary_image)
    print(ftitle, ' : Binary image saved.')
#=========================================
'''