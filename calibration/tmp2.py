import numpy as np
import cv2
import glob
import os


#===========キャリブレーション用の対応点の設定================
# 精度の閾値条件（？）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# チェスボードの3D座標を設定
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)

# 3Dの点がobjpoints, 2Dの点がimgpoints
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


#===========チェスボードの格子点の画像座標を取得し、objpoint,imgpointsに格納================
#キャリブレーション用の画像の読み込み
train_images = glob.glob('train/*.JPG')
i=0

#画像枚数でループ
for fname in train_images:
    #画像読み込み
    img = cv2.imread(fname)
    #グレースケール
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #チェスボードの角を検出
    ret, corners = cv2.findChessboardCorners(gray, (7,10), None)
    #もし、チェスボードの角が検出されたら
    if ret == True:
        #3Dの点を追加
        objpoints.append(objp)
        #精度を上げるための処理
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #2Dの点を追加
        imgpoints.append(corners2)
        # 検出結果を描画
        img = cv2.drawChessboardCorners(img, (7,10), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        # 画像を保存
        cv2.imwrite("cal{0}.jpg".format(i),img)
        i = i+1
        print("Yes point: {}".format(fname))
    else:
        print("No point: {}".format(fname))
cv2.destroyAllWindows()

#===========キャリブレーションの実行================
#キャリブレーションの実行
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#再投影誤差の検証
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
#0.0405874093573348

print("total error: ", mean_error/len(objpoints))

#===========歪み補正================
#補正画像の読み込み
images2 = glob.glob('./input/*.JPG')

for fname in images2:
    #ファイルの名前付けのための変数
    fbase = os.path.basename(fname)
    ftitle, fext = os.path.splitext(fbase)
    #画像の読み込み
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))    
    # 歪み補正
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    #余計な部分をトリミング
    dst = dst[y:y+h, x:x+w]
    #修正後の画像を保存
    cv2.imwrite('./output/' + ftitle + '_cal' + fext,dst)
