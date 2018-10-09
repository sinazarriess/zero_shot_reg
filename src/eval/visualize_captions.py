import cv2 as cv



img = cv.imread("/mnt/Data/zero_shot_reg/coco-caption/images/train2014/COCO_train2014_000000340703.jpg")
img = cv.rectangle(img, (7,15), (326, 165), (0,255,0), 4)
#img = cv.rectangle(img, (250,70), (330,150), (0,255,0), 4)
cv.imshow('test', img)
cv.waitKey(0)


img = cv.imread("/mnt/Data/zero_shot_reg/coco-caption/images/train2014/COCO_train2014_000000313786.jpg")
img = cv.rectangle(img, (138, 40), (550, 155), (0,255,0), 4)
cv.imshow('test', img)
cv.waitKey(0)