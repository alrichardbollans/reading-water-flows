# This file provides methods for taking images from video data
import glob
import os

import natsort
import cv2
import numpy as np

VIDEO_FOLDER = "test videos"
IMAGE_FOLDER = "test images"
IMG_TYPE = "jpg"

# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
COLOR_CHANNELS = 3
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, COLOR_CHANNELS)

NUMBER_REAL_FRAMES = 15
NUMBER_FRAMES_TO_PREDICT = 15

INPUT_IMG_SEQ_SHAPE = (NUMBER_REAL_FRAMES,) + IMG_SHAPE
OUTPUT_IMG_SEQ_SHAPE = (NUMBER_FRAMES_TO_PREDICT,) + IMG_SHAPE


class Video:
    def __init__(self, filename):
        self.filename = filename
        self.frames = self.save_images()
        # self.save_training_tensors()

    def save_images(self):
        if not os.path.exists(VIDEO_FOLDER):
            raise FileNotFoundError(VIDEO_FOLDER)
        frame_folder = get_video_img_folder(self.filename)
        if not os.path.exists(frame_folder):
            print(frame_folder)
            os.mkdir(frame_folder)

        vidcap = cv2.VideoCapture(VIDEO_FOLDER + "/" + self.filename)
        single_imgs = []
        success, image = vidcap.read()

        while success:
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            single_imgs.append(image)
            success, image = vidcap.read()
            print('Read a new frame: ', success)

        count = 0
        for i in range(0, len(single_imgs) - 1):
            img1 = single_imgs[i]

            cv2.imwrite(get_image_file(self.filename, count), img1)  # save frame as JPEG file
            count += 1

        return single_imgs


def get_video_img_folder(video_filename):
    return IMAGE_FOLDER + "/" + video_filename


def get_image_file(video_filename, frame_number):
    return get_video_img_folder(video_filename) + "/frame%d" % frame_number + "." + IMG_TYPE


def stitch_imgs_together(img1, img2, output_file):
    output_img = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(output_file, output_img)  # save frame as JPEG file


if __name__ == '__main__':
    VIDEO_FOLDER = "test videos"
    IMAGE_FOLDER = "test images"
    test_vid = Video("garden.MOV")
