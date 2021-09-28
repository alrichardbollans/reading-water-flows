import numpy as np
import matplotlib.pyplot as plt
import cv2

from process_video_data import IMAGE_FOLDER


def find_edges(img_file, output_file):
    vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    h_filter = [[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]]

    img = cv2.imread(img_file)

    n, m, d = img.shape

    edges_img = img.copy()

    for row in range(3, n - 2):
        for col in range(3, m - 2):
            local_pixels = img[row - 1:row + 2, col - 1:col + 2, 0]
            v_transformed_pixels = vertical_filter * local_pixels
            v_score = v_transformed_pixels.sum() / 4

            h_transformed_pixels = h_filter * local_pixels
            h_score = h_transformed_pixels.sum() / 4

            edge_score = (v_score ** 2 + h_score ** 2) ** 0.5

            edges_img[row, col] = [edge_score] * 3

    # edges_img = edges_img / edges_img.max()

    cv2.imwrite(output_file, edges_img)


def find_colours(input_file, output_file):
    # TODO: find heighest (cluster of) red points
    img = cv2.imread(input_file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0, 50, 150), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 50, 150), (180, 255, 255))

    mask = mask1 + mask2
    red_indices = mask > 0

    red = np.zeros_like(img, np.uint8)

    red[red_indices] = img[red_indices]
    cv2.imshow('img', img)
    cv2.imshow('mask', mask1)
    cv2.imshow('mask2', mask2)
    cv2.imshow('red', red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # find_edges(IMAGE_FOLDER + '/garden.MOV/frame0.jpg', 'output_file.jpg')
    find_colours(IMAGE_FOLDER + '/garden.MOV/frame0.jpg', 'output_file.jpg')
