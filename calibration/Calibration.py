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
    #gray = cv2.resize(gray_b, (2080,2080))
    #ret1, gray = cv2.threshold(img, 100,255, cv2.THRESH_BINARY)
    #cv2.imshow('img',gray)
    #cv2.waitKey(0)
    
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
        #画像を保存
        cv2.imwrite("cal{0}.jpg".format(i),img)
        i = i+1
        print("Yes point: {}".format(fname))
    else:
        print("No point: {}".format(fname))
cv2.destroyAllWindows()

#再投影誤差の検証
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
#0.0405874093573348

print("total error: ", mean_error/len(objpoints))


'''
#===========キャリブレーションの実行================
#キャリブレーションの実行
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("objpoints:{}".format(objpoints))
print("imgpoints:{}".format(imgpoints))
print("gray.shape[::-1]:{}".format(gray.shape[::-1]))
print("ret:{}".format(ret))
print("mtx:{}".format(mtx))
print("dist:{}".format(dist))
print("rvecs:{}".format(rvecs))
print("tvecs:{}".format(tvecs))

#キャリブレーション結果の保存
#np.save('np_save', objpoints=objpoints, imgpoints=imgpoints, grayshape=gray.shape[::-1],ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
np.savez('np_save1', objpoints, imgpoints, gray.shape[::-1])
np.savez('np_save2', ret, mtx, dist)
np.savez('np_save3', rvecs, tvecs)




#===========キャリブレーション結果の読み込み================
#保管したデータを取り出す。
np_save1 = np.load('np_save1.npz')
np_save2 = np.load('np_save2.npz')
np_save3 = np.load('np_save3.npz')

"""
#print(np_save1.files)
#print(np_save2.files)
#print(np_save3.files)
print(np_save1['arr_0'])
print(np_save1['arr_1'])
print(np_save1['arr_2'])
print(np_save2['arr_0'])
print(np_save2['arr_1'])
print(np_save2['arr_2'])
print(np_save3['arr_0'])
print(np_save3['arr_1'])
"""
#それぞれのデータを変数に格納
#3Dの点
objpoints = np_save1['arr_0']
#2Dの点
imgpoints = np_save1['arr_1']
#mtx: カメラ行列、dist: 歪み係数, rvecs: 回転ベクトル, tvecs: 平行移動ベクトル
ret, mtx, dist, rvecs, tvecs = np_save2['arr_0'], np_save2['arr_1'], np_save2['arr_2'], np_save3['arr_0'], np_save3['arr_1']


"""
print("ret:{}".format(ret))
print("mtx:{}".format(mtx))
print("dist:{}".format(dist))
print("rvecs:{}".format(rvecs))
print("tvecs:{}".format(tvecs))
"""

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




'''
#1,2,5,7,9