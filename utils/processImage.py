import cv2
import os

if __name__ =="__main__":
    rbgfile_path = "/media/jinghuan/西塔/3D FUrniture reconstruction shape with TextURE/train/image"
    maskfile_path = "/media/jinghuan/西塔/3D FUrniture reconstruction shape with TextURE/train/mask"

    maskfile_name = os.listdir(maskfile_path)
    for i in maskfile_name:
        maskImg = cv2.imread(os.path.join(maskfile_path, i))
        rbgImg = cv2.imread(os.path.join(rbgfile_path, i.replace(".png", ".jpg")))
        r, b, g = maskImg.T
        idx =((255 == r) & (255 == b) & (255 == g)).T
        maskImg[idx] = rbgImg[idx]
        cv2.imshow("aa", maskImg)
        cv2.waitKey(0)
        break
        pass 

    
    pass