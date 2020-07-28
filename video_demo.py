import sys
import glob
import os
import cv2
import math
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

def dir_check(_dir): #해당 디렉토리 내의 모든 avi파일 주소 목록
    file_list = glob.glob(_dir)
    for file_check in file_list:
        if os.path.isdir(file_check):
            dir_check("{}/*".format(file_check))
        elif file_check.endswith(".avi"):
            avi_list.append(file_check)

def get_keydata(name, frameNo, mykey, rate, angle, direct): 
    key_list = f"{name}, {frameNo}"  
    if len(keypoints) == 0:
        for value in range(15):
            key_list = f"{key_list}, 0, 0" 
        key_list = f"{key_list}, 0, 0, 0 \n"   
    else:
        mykey = keypoints[0, 0:14, 0:2]  
        mykey = mykey.reshape(-1)  
        for value in mykey:
            key_list = f"{key_list}, {value}" 
        key_list = f"{key_list}, {rate:5.2f}, {angle:5.2f}, {direct:4d} \n" 
    return key_list

def get_angle(keypoints):
    angle = 90
    if len(keypoints) != 0:
        x1,x2,x3 = keypoints[0][8][0], keypoints[0][11][0], keypoints[0][1][0] 
        y1,y2,y3 = keypoints[0][8][1], keypoints[0][11][1], keypoints[0][1][1] 
        
        x4 = (x1+x2)/2
        y4 = (y1+y2)/2
        x = x4 - x3 
        y = y4 - y3
        angle = math.degrees(math.atan2(y,x))
        if angle >= 90:
            angle = 180-angle
        elif angle >= -90:
            if angle < 0 : 
                angle = -angle
        else:
            angle = 180 + angle
    return angle


def get_rate(keypoints):
    rate = 0
    if len(keypoints) != 0:
        x = [keypoints[0][2][0], keypoints[0][5][0], keypoints[0][8][0], keypoints[0][11][0]]
        y = [keypoints[0][2][1], keypoints[0][5][1], keypoints[0][8][1], keypoints[0][11][1]]    
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        w = xmax - xmin
        h = ymax - ymin
        rate = w/h
    return rate

def get_direction(keypoints):
    direct = 0
    if len(keypoints) != 0:
        y1,y2 = keypoints[0][8][1], keypoints[0][1][1]   
        if y2 >= y1:
            direct = -1
        else:
            direct = 1
    return direct

#video_path = "./examples/fall_test/*" #avi파일 찾을 위치
video_path = "./examples/media/*" #avi파일 찾을 위치
avi_list = []
dir_check(video_path)

with open('./fallen_data.csv', 'w') as f:
    data = 'fName, frameNo, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, '
    data += 'x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13, rate, angle, direct\n'
    f.write(data)
    for video_fullpath in avi_list:
        output_name = f"{os.path.dirname(video_fullpath).split(os.path.sep)[-1]}_{os.path.basename(video_fullpath)}"    
        estimator = BodyPoseEstimator(pretrained=True)       
        videoclip = cv2.VideoCapture(video_fullpath)
        count = 0
        while videoclip.isOpened():
            flag, frame = videoclip.read()            
            if not flag:
                break
            count +=1
            try :
                keypoints = estimator(frame)
                frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
                frame = draw_keypoints(frame, keypoints, radius=3, alpha=0.8)                                               
                angle = get_angle(keypoints)
                rate  = get_rate(keypoints)
                direct  = get_direction(keypoints)
                data  = get_keydata(output_name, count, keypoints, angle, rate, direct)                
                f.write(data)
                print(f"{output_name} No={count:4d}   rate={rate:7.2f}   angle={angle:7.2f}   direct={direct:3d}")                
                '''
                cv2.imshow('Video Demo', frame)
                if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                    break
                '''
            except KeyboardInterrupt:
                videoclip.release() 
                cv2.destroyAllWindows()
                f.close()
                exit(1)
            except:
                print("'에러 발생")
                     
        videoclip.release()        
    cv2.destroyAllWindows()
f.close()
