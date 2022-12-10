import os

import cv2
import pandas as pd
import numpy as np
from calculate_disp import get_disparity_map


def get_file_list(folder_path):
    file_list = os.listdir(folder_path)
    file_list.sort()
    file_list = [os.path.join(folder_path, i) for i in file_list]

    return file_list


def generate_kitti_dataset():
    dataset_root = "/root/data/kitti/training"

    left_rgb_folder = os.path.join(dataset_root, "colored_0")
    right_rgb_folder = os.path.join(dataset_root, "colored_1")
    disp_folder = os.path.join(dataset_root, "disp_noc")

    left_rgb_list = get_file_list(left_rgb_folder)
    right_rgb_list = get_file_list(right_rgb_folder)
    disp_list = get_file_list(disp_folder)

    result = list(zip(left_rgb_list, right_rgb_list, disp_list))
    result = pd.DataFrame(result)

    result.to_csv("kitti2012_list.csv", index=False, header=False)


def generate_outdoor_dataset_all():
    dataset_root = "/root/Real-time-self-adaptive-deep-stereo/data/result"

    image_paths = get_file_list(dataset_root)
    left_rgb_list = [i for i in image_paths if "left" in i]
    right_rgb_list = [i for i in image_paths if "right" in i]
    disp_list = [i for i in image_paths if "disp" in i]
    result = list(zip(left_rgb_list, right_rgb_list, disp_list))
    result = pd.DataFrame(result)

    result.to_csv("outdoor_list_all.csv", index=False, header=False)


def generate_outdoor_dataset():
    dataset_root = "/root/Real-time-self-adaptive-deep-stereo/data/result"

    image_paths = get_file_list(dataset_root)
    timestamp = "1657431262191"
    left_rgb_list = [sorted([i for i in image_paths if "left" in i and timestamp in i])[0]] * 400
    right_rgb_list = [sorted([i for i in image_paths if "right" in i and timestamp in i])[0]] * 400
    disp_list = [sorted([i for i in image_paths if "disp" in i and timestamp in i])[0]] * 400
    result = list(zip(left_rgb_list, right_rgb_list, disp_list))
    result = pd.DataFrame(result)

    result.to_csv("outdoor_list.csv", index=False, header=False)


def generate_outdoor_video_dataset():
    dataset_root = "data/madnet_images/rgb_1657791278049"

    image_paths = get_file_list(dataset_root)
    left_rgb_list = sorted([i for i in image_paths if "left" in i], key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))[50:]
    right_rgb_list = sorted([i for i in image_paths if "right" in i], key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))[50:]
    disp_list = sorted([i for i in image_paths if "disp" in i], key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))[50:]

    image_paths = """data/result/left_NOH-AL10_1657429288494.png
data/result/left_NOH-AL10_1657429289537.png
data/result/left_NOH-AL10_1657429291016.png
data/result/left_NOH-AL10_1657429292940.png
data/result/left_NOH-AL10_1657429294396.png
data/result/left_NOH-AL10_1657429301154.png
data/result/left_NOH-AL10_1657429302233.png
data/result/left_NOH-AL10_1657429303230.png""".split()

    left_rgb_list.extend(image_paths)
    right_rgb_list.extend([i.replace("left", "right") for i in image_paths])
    disp_list.extend([i.replace("left", "disp") for i in image_paths])

    result = list(zip(left_rgb_list, right_rgb_list, disp_list))
    result = pd.DataFrame(result)

    result.to_csv("outdoor_list_rgb_1657789702514.csv", index=False, header=False)


def generate_all_outdoor_video_dataset():
    image_folder_to_demo_images = {
        # r"C:\Users\Bangwen\Dataset\2020719-2\madnet_train_videos\rgb_1658214109268": None,
        # "data/221207-1/madnet_train_videos/images/221207-1/rgb_1670390247883": None,
        # "data/221207-1/madnet_train_videos/images/221207-1/rgb_1670390285265": None,
        # "data/221207-1/madnet_train_videos/images/221207-1/rgb_1670390920222": None,
        # "data/221207-1/madnet_train_videos/images/221207-1/rgb_1670391055015": None,
        # "data/221207-1/madnet_train_videos/images/221207-1/rgb_1670391077073": None,
        # "data/221205-2/madnet_train_videos/rgb_1670224157259": None,
        # "data/221205-2/madnet_train_videos/rgb_1670224216719": None,
        # "data/221205-2/madnet_train_videos/rgb_1670224267591": None,
        # "data/221205-2/madnet_train_videos/rgb_1670224320893": None,
        # "data/221205-2-all/KFC": None
        # "data/221208-3/madnet_train_videos/rgb_1670482225865": None,
        # "data/221208-3/madnet_train_videos/rgb_1670482344154": None
        "images/221210-1/madnet_train_videos/rgb_5m": None,
        "images/221210-1/madnet_train_videos/rgb_3m": None,
        "images/221210-1/madnet_train_videos/rgb_1m": None,
        "images/221210-1/madnet_train_videos/rgb_0.5m": None,
    }

    for dataset_root, demo_image_paths in image_folder_to_demo_images.items():
        image_paths = get_file_list(dataset_root)
        left_rgb_list = sorted([i for i in image_paths if "left" in i], key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))
        right_rgb_list = sorted([i for i in image_paths if "right" in i], key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))
        disp_list = sorted([i for i in image_paths if "disp" in i], key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))
        if len(disp_list) != len(left_rgb_list):
            for left_image_path, right_image_path in zip(left_rgb_list, right_rgb_list):
                left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
                right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
                disp, _ = get_disparity_map(left_image, right_image)

                cv2.imwrite(left_image_path.replace("left", "disp"), np.tile(disp[:, :, np.newaxis], 3))
            disp_list = sorted([i for i in image_paths if "disp" in i], key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))

        demo_image_paths = demo_image_paths.split() * 20 if demo_image_paths is not None else []

        left_rgb_list.extend(demo_image_paths)
        right_rgb_list.extend([i.replace("left", "right") for i in demo_image_paths])
        disp_list.extend([i.replace("left", "disp") for i in demo_image_paths])

        left_rgb_list, right_rgb_list = right_rgb_list, left_rgb_list  # for pixel4
        result = list(zip(left_rgb_list, right_rgb_list, disp_list))
        result = pd.DataFrame(result)

        result.to_csv(f"outdoor_list_{os.path.split(dataset_root)[1]}.csv", index=False, header=False)


if __name__ == "__main__":
    # generate_outdoor_dataset_all()
    # generate_outdoor_dataset()
    # generate_kitti_dataset()
    # generate_outdoor_video_dataset()
    generate_all_outdoor_video_dataset()
