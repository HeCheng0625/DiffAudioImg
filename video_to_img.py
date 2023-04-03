import cv2
import os
import shutil
from tqdm import tqdm

def get_frame_from_video(video_name, save_dict, interval):
    video_capture = cv2.VideoCapture(video_name)
    i = 1
    while True:
        success, frame = video_capture.read()
        # print(success)
        if not success:
            break
        if i % interval == 0:
            save_name = os.path.join(save_dict, str(i//interval-1), video_name.split(".mp4")[0].split("/")[-1] + "_" + str(i//interval-1) + ".jpg")
            cv2.imwrite(save_name, frame)
        i += 1

video_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/vggsound/video"
save_dict = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/vggsound/img_spilt"
# save_set = set(os.listdir(save_dict))
# print(len(save_set))

video_list = os.listdir(video_path)
video_list.sort()

for video_name in tqdm(video_list[100000:]):
    video_name = os.path.join(video_path, video_name)
    # if video_name.split(".mp4")[0].split("/")[-1] + "_" + "4" + ".jpg" in save_set:
    #     continue
    get_frame_from_video(video_name, save_dict, 10)