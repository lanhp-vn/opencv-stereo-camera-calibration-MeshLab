import cv2


cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Left usb below
cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Right usb above

num = 0


while cap_left.isOpened():

    succes1, img_left = cap_left.read() # Left
    succes2, img_right = cap_right.read() # Right


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        # cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img_left)
        # cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img_right)
        # cv2.imwrite('unCalibLeft.png', img_left)
        # cv2.imwrite('unCalibRight.png', img_right)
        cv2.imwrite('left_image.png', img_left)
        cv2.imwrite('right_image.png', img_right)
        print("images saved!")
        num += 1

    cv2.imshow('Img left',img_left)
    cv2.imshow('Img right',img_right)