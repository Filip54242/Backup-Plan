import os

import cv2


def video_to_images(video_path, save_path, frame_skip=1, image_count=0):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    image_name_count = image_count
    frame_count = 0
    while success:
        if frame_count % frame_skip == 0:
            cv2.imwrite(save_path + "frame%d.jpg" % image_name_count, image)
            image_name_count += 1
        success, image = vidcap.read()
        print('Read a new frame')
        frame_count += 1
    return image_name_count


def get_all_files_from_path(path, extension):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if extension in file:
                files.append(os.path.join(r, file))
    return files


def video_to_image():
    video_path = '/media/filip/HDD1/Videos/'
    save_path = '/media/filip/HDD1/Images/'
    image_count = 0

    for video in get_all_files_from_path(video_path, 'mp4'):
        image_count = video_to_images(video, save_path, image_count=image_count)