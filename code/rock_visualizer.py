import cv2


def threshold(img):
    low = (140, 140, 140)
    high = (5, 70, 70)
    under_thresh = (img[:, :, 0] < low[0]) & (img[:, :, 1] < low[1]) & (img[:, :, 2] < low[2])
    above_thresh = (img[:, :, 0] > high[0]) & (img[:, :, 1] > high[1]) & (img[:, :, 2] > high[2])
    img[under_thresh] = 0
    img[above_thresh] = 0


for filename in ('rock.png', 'rock2.png', 'rock3.png'):
    img = cv2.imread(filename)
    threshold(img)
    cv2.imshow(filename, img)

while True:
    cv2.waitKey(0)
