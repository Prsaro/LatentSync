# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from latentsync.utils.image_processor import ImageProcessor
import torch
from einops import rearrange
import os
import tqdm
import subprocess
from multiprocessing import Process
import shutil

import os
import cv2
import sys
import subprocess

import numpy as np
# import dlib
import math

from insightface.app import FaceAnalysis

paths = []

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("/zjs/model/shape_predictor_68_face_landmarks.dat")

detector_if = FaceAnalysis(name='antelopev2' ,allowed_modules=['detection', 'landmark_2d_106'])
detector_if.prepare(ctx_id=0, det_size=(640, 640))

MAP_LIST = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17, 43, 48, 49, 51, 50, 102, 103, 104, 105, 101, 72, 73, 74, 86, 78, 79, 80, 85, 84, 35, 41, 42, 39, 37, 36, 89, 95, 96, 93, 91, 90, 52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54]

eyed = 0.1125 #0.15

def get_landmarks106_insightface(image):
    faces = detector_if.get(image)
    landmarks = None
    for face in faces:
        if face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106.astype(int)
            break
        else:
            landmarks = None
            print("face not detected")
    return landmarks

# def get_landmarks_dlib(image):
#     landmarks_array = np.zeros((68, 2))
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     if len(faces) == 0:
#         return None
#     face = faces[0]
#     landmarks = predictor(gray, face)
#     for n in range(0, 68):
#         landmarks_array[n] = (landmarks.part(n).x, landmarks.part(n).y)
#     landmarks_array = landmarks_array.astype(np.float32)
#     return landmarks_array

def align_and_crop_face(image_path, pred_g, desired_face_width=256, desired_face_height=256, use_dlib_landmark=False):
    # 检测关键点
    if isinstance(image_path, str):
        image=cv2.imread(image_path)
    else:
        image = image_path
    keypoints = get_landmarks106_insightface(image)
    if keypoints is None:
        print("get landmarks failed, img_path is {}".format(image_path))
        keypoints = pred_g
    else:
        pred_g = keypoints
    if not use_dlib_landmark:
        keypoints = keypoints[MAP_LIST].astype(np.float32)
        left_eye_center = np.mean(keypoints[36:42], axis=0)
        right_eye_center = np.mean(keypoints[42:48], axis=0)
        left_head = np.mean(keypoints[0:3], axis=0)
        right_head = np.mean(keypoints[14:17], axis=0)
        nose = np.mean(keypoints[27:31], axis=0)
        left_center = 0.125*left_eye_center + 0.375*left_head + 0.5 * nose
        right_center = 0.125*right_eye_center + 0.375*right_head + 0.5 * nose
        
        face_center = (left_center + right_center) / 2
        # 计算眼睛连线的角度，并增加60度
        # 计算右眼相对于左眼的角度
        angle = math.atan2(right_center[1] - left_center[1], 
                        right_center[0] - left_center[0])
        triangle_height = math.sqrt((right_center[0] - left_center[0])**2 +
                                    (right_center[1] - left_center[1])**2) * math.sqrt(3) / 2

        # 确定第三个点的位置
        third_point = [face_center[0] - triangle_height * math.sin(angle),
                        face_center[1] + triangle_height * math.cos(angle)]
        # 定义仿射变换的源点和目标点
        src_points = np.array([left_center, right_center, third_point], dtype='float32')
        dst_points = np.array([[desired_face_width * (0.50-eyed), desired_face_height * 0.45],
                            [desired_face_width * (0.50+eyed), desired_face_height * 0.45],
                            [desired_face_width * 0.50, desired_face_height * (math.sqrt(3) * eyed  + 0.45)]], dtype='float32')
        # 获取仿射变换矩阵
        M = cv2.getAffineTransform(src_points, dst_points)
        M_extended = np.vstack([M, [0, 0, 1]])
        M_inv_extended = np.linalg.inv(M_extended)
        M_inv = M_inv_extended[:2, :]
        # 应用仿射变换
        aligned = cv2.warpAffine(image, M, (desired_face_width, desired_face_height))
        landmarks_transformed = cv2.transform(keypoints.reshape(1, -1, 2), M)[0]
    else:
        left_center = np.mean(keypoints[33:43], axis=0)
        right_center = np.mean(keypoints[87:97], axis=0)
        # 计算眼睛中心连线的中点
        # eye_center = (left_eye_center + right_eye_center) / 2
        face_center = (left_center + right_center) / 2

        # 计算眼睛连线的角度，并增加60度
        # 计算右眼相对于左眼的角度
        angle = math.atan2(right_center[1] - left_center[1], 
                        right_center[0] - left_center[0])
        triangle_height = math.sqrt((right_center[0] - left_center[0])**2 +
                                    (right_center[1] - left_center[1])**2) * math.sqrt(3) / 2

        # 确定第三个点的位置
        third_point = [face_center[0] - triangle_height * math.sin(angle),
                    face_center[1] + triangle_height * math.cos(angle)]

        # 定义仿射变换的源点和目标点
        src_points = np.array([left_center, right_center, third_point], dtype='float32')
        dst_points = np.array([[desired_face_width * (0.50-eyed), desired_face_height * 0.45],
                            [desired_face_width * (0.50+eyed), desired_face_height * 0.45],
                            [desired_face_width * 0.50, desired_face_height * (math.sqrt(3) * eyed  + 0.45)]], dtype='float32')
        # 获取仿射变换矩阵
        M = cv2.getAffineTransform(src_points, dst_points)
        # 应用仿射变换
        aligned = cv2.warpAffine(image, M, (desired_face_width, desired_face_height))
        landmarks_transformed = cv2.transform(keypoints.reshape(1, -1, 2), M)[0]
        landmarks_transformed = landmarks_transformed[MAP_LIST]
    return aligned, src_points, keypoints, landmarks_transformed

# 不需要更改
def gather_video_paths(input_dir, output_dir):
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append((video_input, video_output))
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_video_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


class FaceDetector:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, "fix_mask", device)
        self.device = device

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            # face, box, affine_matrix
            # frame, _, _ = self.image_processor.affine_transform(frame)
            frame, _, _, _ = align_and_crop_face(frame, None, 256, 256, use_dlib_landmark=False)
            # frame_tensor = torch.from_numpy(frame).to(self.device)
            # results.append(frame_tensor)
            results.append(frame)
        # results = torch.stack(results)
        results = np.stack(results) # (102, 256, 256, 3)

        # results = rearrange(results, "f c h w -> f h w c")
        return results

    def close(self):
        self.image_processor.close()


def combine_video_audio(video_frames, video_input_path, video_output_path, process_temp_dir):
    video_name = os.path.basename(video_input_path)[:-4]
    # audio_temp = os.path.join(process_temp_dir, f"{video_name}_temp.wav")
    # origin audio
    audio_temp = video_input_path[:-4] + ".wav"
    video_temp = os.path.join(process_temp_dir, f"{video_name}_temp.mp4")

    write_video(video_temp, video_frames, fps=25)

    # command = f"ffmpeg -y -loglevel error -i {video_input_path} -q:a 0 -map a {audio_temp}"
    # subprocess.run(command, shell=True)

    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
    command = f"ffmpeg -y -loglevel error -i {video_temp} -i {audio_temp} -c:v libx264 -c:a aac -map 0:v -map 1:a -q:v 0 -q:a 0 {video_output_path}"
    subprocess.run(command, shell=True)

    # os.remove(audio_temp)
    os.remove(video_temp)


def func(paths, process_temp_dir, device_id, resolution):
    os.makedirs(process_temp_dir, exist_ok=True)
    face_detector = FaceDetector(resolution, f"cuda:{device_id}")

    for video_input, video_output in paths:
        if os.path.isfile(video_output):
            continue
        try:
            video_frames = face_detector.affine_transform_video(video_input)
        except Exception as e:  # Handle the exception of face not detcted
            print(f"Exception: {e} - {video_input}")
            continue

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        combine_video_audio(video_frames, video_input, video_output, process_temp_dir)
        print(f"Saved: {video_output}")

    face_detector.close()


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def gotu_affine_transform_multi_gpus(input_dir, output_dir, temp_dir, resolution, num_workers):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_video_paths(input_dir, output_dir)
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No GPUs found")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    split_paths = list(split(paths, num_workers * num_devices))

    processes = []

    for i in range(num_devices):
        for j in range(num_workers):
            process_index = i * num_workers + j
            process = Process(
                target=func, args=(split_paths[process_index], os.path.join(temp_dir, f"process_{i}"), i, resolution)
            )
            process.start()
            processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/avatars/resampled/train"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/avatars/affine_transformed/train"
    temp_dir = "temp"
    resolution = 256
    num_workers = 10  # How many processes per device

    affine_transform_multi_gpus(input_dir, output_dir, temp_dir, resolution, num_workers)
