import torch
import clip
from PIL import Image
import time
import argparse
import os
import shutil
import numpy as np

image_extensions = ['.jpg', '.png', '.bmp', '.tif', '.tiff', '.jpeg']
device = "cpu"


def copy_image_to_result(image_path, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    shutil.copy(image_path, result_folder)


def move_image_to_result(image_path, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    shutil.move(image_path, result_folder)


def img_pair_txt(label_list, input_image):
    image = preprocess(input_image).unsqueeze(0).to(device)
    text = clip.tokenize(label_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs


def output_txt(probs):
    for idx in range(len(probs[0])):
        ida = max(probs[0])
        if ida < threshold:
            return -1
        if probs[0][idx] == ida:
            return idx


def get_image_paths(folder_path):
    image_paths = []
    for root, directories, files in os.walk(folder_path):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in image_extensions:
                image_path = os.path.join(root, file_name)
                image_paths.append(image_path)
    return sorted(image_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, help='image_folder root path')
    parser.add_argument('--threshold', type=float, required=False, default=0.7, help='threshold')
    parser.add_argument('--result_folder', type=str, required=True, default="RESULTS", help='result_folder path')

    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    args = parser.parse_args()
    threshold = args.threshold if args.threshold else 0.5

    label_list_new = [
        "indoor",
        "outdoor",
    ]


    image_paths = get_image_paths(args.image_folder)
    print("---------------First Level---------------")
    for image_path in image_paths:
        input_image_new = Image.open(image_path)
        probs_new = img_pair_txt(label_list_new, input_image_new)
        predicted_id = np.argmax(probs_new)

        scene = label_list_new[predicted_id]
        result_image_path = os.path.join(args.result_folder, scene)
        copy_image_to_result(image_path, result_image_path)
        print(f"Image_path: {image_path}, Predict_Class: {scene}.")


    label_list_new = [
        "school",
        "road",
    ]

    image_paths = get_image_paths(args.result_folder)
    print("---------------Second Level---------------")
    for image_path in image_paths:
        input_image_new = Image.open(image_path)
        probs_new = img_pair_txt(label_list_new, input_image_new)
        output = output_txt(probs_new)
        if output == -1:
            print(f"Image_path: {image_path}, No suitable label.")
            predicted_classname = 'Unlabeled'
        else:
            scene = label_mapping[label_list_new[output]]
            predicted_classname = scene
            print(f"Image_path: {image_path}, Predict_Class: {predicted_classname}.")
        result_image_path = os.path.join(args.result_folder, image_path.split("/")[-2], predicted_classname)
        move_image_to_result(image_path, result_image_path)


    print("************ End. ************")
