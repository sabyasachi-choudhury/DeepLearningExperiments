import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

# image = cv2.cvtColor(cv2.imread(r"C:\Users\Sabyasachi\PycharmProjects\deep_learning_class\train_breeds\0a77d498ff491945347bb895d8ae4008.jpg"), cv2.COLOR_RGB2GRAY)
# image = cv2.imread(r"C:\Users\Sabyasachi\PycharmProjects\deep_learning_class\train_breeds\0a77d498ff491945347bb895d8ae4008.jpg")
# cv2.imshow("1", image)
# cv2.waitKey(0)

# def to_channel(k: np.ndarray):
#     output = np.zeros(shape=[*k.shape, 3])
#     for i in range(k.shape[0]):
#         for j in range(k.shape[1]):
#             output[i][j] = [k[i][j]] * 3
#     return output
#
# def run_convolution(image: np.ndarray, kernel: np.ndarray):
#     output = np.zeros(shape=[image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1])
#     for x in range(output.shape[0]):
#         for y in range(output.shape[1]):
#             output[x][y] = np.sum(kernel * image[x: x + kernel.shape[0], y: y + kernel.shape[1]])
#     # output = np.where(output < 0, 0, output)
#     # output = np.where(output > 100, 255, output)
#     m = np.max(output)
#     output *= (255/m)
#     # output = output.astype(dtype=np.uint8)
#     return output
#
# gaussian = np.array([
#     [1, 3, 5, 3, 1],
#     [3, 5, 7, 5, 3],
#     [5, 7, 9, 7, 5],
#     [3, 5, 7, 5, 3],
#     [1, 3, 5, 3, 1]
# ])
# blur = run_convolution(image, gaussian)
#
# sobel_x = np.array([
#     [1, 0, -1],
#     [2, 0, -2],
#     [1, 0, -1]
# ])
# sobel_y = np.array([
#     [1, 2, 1],
#     [0, 0, 0],
#     [-1, -2, -1]
# ])
#
# x_conv, y_conv = run_convolution(image, sobel_x), run_convolution(image, sobel_y)
# hypotenuse = np.sqrt(x_conv**2 + y_conv**2)
# hypotenuse *= 255/np.max(hypotenuse)
# hypotenuse = hypotenuse.astype(dtype=np.uint8)
# angles = np.rad2deg(np.arctan2(y_conv, x_conv))

# cv2.imshow("image", image)

# edges = cv2.Canny(image, 250, 255)
# plt.subplot(121)
# plt.imshow(image, cmap='gray')
# plt.title("og_image")
# plt.xticks([])
# plt.yticks([])
#
# plt.subplot(122)
# plt.imshow(edges, cmap='gray')
# plt.title("canny_image")
# plt.xticks([])
# plt.yticks([])
#
# plt.show()
image = cv2.cvtColor(cv2.imread(r"C:\Users\Sabyasachi\PycharmProjects\deep_learning_class\train_breeds\0a77d498ff491945347bb895d8ae4008.jpg"), cv2.COLOR_RGB2GRAY)
image = cv2.GaussianBlur(image, ksize=[3, 3], sigmaX=10, sigmaY=10)

# pixel grouping
def pix_group(img, num):
    src = img.copy()
    interval = int(255/num)
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            src[x][y] = int(src[x][y]/interval) * interval
            print(src[x][y])
    return src

for i in range(5):
    dilate = cv2.dilate(image, kernel=np.ones([3, 3], dtype=np.uint8), iterations=2)
    dilate = cv2.erode(image, kernel=np.ones([3, 3], dtype=np.uint8), iterations=2)
group = pix_group(dilate, 3)
canny = cv2.Canny(group, 200, 250)
# cv2.imshow("a", dilate)
# cv2.imshow("i", image)
cv2.imshow("c", canny)
# cv2.imshow("g", group)
# blurred = cv2.GaussianBlur(image, ksize=[21, 21], sigmaX=0)
# mask = np.zeros([333, 500], dtype=np.uint8)
# mask = cv2.circle(mask, (250, 162), 75, 1, thickness=-1)
# answer = image * mask
# cv2.imshow("a", mask)
# cv2.imshow("og", image)
# cv2.imshow("blurred", blurred)
# cv2.imshow("answer", answer)

contours, h = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# image = cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
rectangle_details = [cv2.boundingRect(arr) for arr in contours]
for [x, y, w, h] in rectangle_details:
    if x + w > 100 and y + h > 80:
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)
# cv2.merge()
cv2.imshow("i", image)
cv2.waitKey(0)

# cv2.imwrite("image_processing/answer.jpg", answer)

# cv2.imshow("x", x_conv)
# cv2.imshow("y", y_conv)
# cv2.imshow("h", hypotenuse)
# cv2.imshow("a", y_conv)
# cv2.imshow("ang", hypotenuse)
# cv2.waitKey(0)