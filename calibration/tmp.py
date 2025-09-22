import numpy as np
import cv2
import glob
import os

#===========キャリブレーション用の対応点の設定================
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)

#===========トレーニングデータでカメラパラメータを求める================
objpoints_train = [] # 3D points in real world space (train)
imgpoints_train = [] # 2D points in image plane (train)

# キャリブレーション用の画像の読み込み
train_images = glob.glob('train/*.JPG')

for fname in train_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,10), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints_train.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints_train.append(corners2)
        print("Train image found: {}".format(fname))
    else:
        print("Train image not found: {}".format(fname))

# カメラキャリブレーションの実行
train_ret, train_mtx, train_dist, train_rvecs, train_tvecs = cv2.calibrateCamera(objpoints_train, imgpoints_train, gray.shape[::-1], None, None)

print("\n--- Calibration Results (from train data) ---")
print("Camera Matrix (mtx):\n{}".format(train_mtx))
print("Distortion Coefficients (dist):\n{}".format(train_dist))

#===========テストデータで再投影誤差を検証する================
objpoints_test = [] # 3D points in real world space (test)
imgpoints_test = [] # 2D points in image plane (test)
image_paths = []

test_images = glob.glob('test/*.JPG')

for fname in test_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,10), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints_test.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints_test.append(corners2)
        image_paths.append(fname)
        print("Test image found: {}".format(fname))
    else:
        print("Test image not found: {}".format(fname))


#テストデータを使って再投影誤差を計算

test_ret, test_mtx, test_dist, test_rvecs, test_tvecs = cv2.calibrateCamera(objpoints_train, imgpoints_train, gray.shape[::-1], None, None)
mean_error = 0
for i in range(len(objpoints_test)):
    # ここで、trainで求めたパラメータ(mtx, dist, rvecs, tvecs)を使用ｓ
    imgpoints2, _ = cv2.projectPoints(objpoints_test[i], test_rvecs[i], test_tvecs[i], train_mtx, train_dist)
    error = cv2.norm(imgpoints_test[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error


    # === 再投影誤差の可視化 ===
    img_original = cv2.imread(image_paths[i])
    
    # 検出された元の点（緑の円）を描画
    for corner in np.squeeze(imgpoints_test[i]):
        cv2.circle(img_original, tuple(corner.astype(int)), 5, (0, 255, 0), -1) # 緑色
    
    # 再投影された点（赤の円）を描画
    for projected_point in np.squeeze(imgpoints2):
        cv2.circle(img_original, tuple(projected_point.astype(int)), 5, (0, 0, 255), -1) # 赤色
        
    # 再投影された点から元の点への線を描画
    for j in range(len(imgpoints_test[i])):
        p1 = tuple(np.squeeze(imgpoints_test[i][j]).astype(int))
        p2 = tuple(np.squeeze(imgpoints2[j]).astype(int))
        cv2.line(img_original, p1, p2, (255, 0, 0), 1) # 青色
    
    # 可視化画像を保存
    fbase = os.path.basename(image_paths[i])
    ftitle, fext = os.path.splitext(fbase)
    output_path = './output/' + ftitle + '_reproj' + fext
    cv2.imwrite(output_path, img_original)
    print(f"Reprojection visualization saved: {output_path}")

print("\n--- Reprojection Error on Test Data ---")
print("Total reprojection error: {}".format(mean_error / len(objpoints_test)))

#===========歪み補正================
# この部分はテストデータでもtrainデータでも使用可能
# 例えば、テストデータの歪み補正を行う場合:
images_to_undistort = glob.glob('test/*.JPG')

for fname in images_to_undistort:
    fbase = os.path.basename(fname)
    ftitle, fext = os.path.splitext(fbase)
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(train_mtx, train_dist, (w,h), 1, (w,h))    
    dst = cv2.undistort(img, train_mtx, train_dist, None, newcameramtx)
    x, y, w_roi, h_roi = roi
    dst = dst[y:y+h_roi, x:x+w_roi]
    
    cv2.imwrite('./output/' + ftitle + '_cal' + fext, dst)
    print("Undistorted image saved: " + ftitle + "_cal" + fext)