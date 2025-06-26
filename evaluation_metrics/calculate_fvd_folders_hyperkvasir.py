import numpy as np
import torch
from tqdm import tqdm

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    # if the last two dims are different, resize to the same size
    if videos1.shape[-1] != videos2.shape[-1] or videos1.shape[-2] != videos2.shape[-2]:
        print("resize the videos1 and videos2 to the same size")
        from torchvision import transforms

        # resize to the same size
        transform = transforms.Compose([
            transforms.Resize((videos2.shape[-2], videos2.shape[-1])),
        ])

        videos1_ = []
        for i in range(videos1.shape[0]):
            video = videos1[i]
            video_ = torch.stack([transform(video[j]) for j in range(video.shape[0])])
            videos1_.append(video_)

        videos1 = torch.stack(videos1_)

    assert videos1.shape == videos2.shape, "Expect the same shape of videos1 and videos2, but got {} and {}".format(videos1.shape, videos2.shape)

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))
    
    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)): # 10 to n_timestamps
            # print("clip_timestamp: ", clip_timestamp)
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]
            # N_videos, 3, clip_timestamp, H, W.
            # The clip_timestamp is ::10, ::11, ::12, ..., ::n_timestamps

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    result = {
        "value": fvd_results,
    }

    return result

# test code / using example
import os
import cv2
def main(class_name, folder1, folder2, oversample):
    from PIL import Image
    import torchvision.transforms.functional as TVF
    import torch
    import torchvision.transforms as T
    # read images from each subfolder, and convert to tensor, then concatenate to a video
    # NUMBER_OF_VIDEOS = 127
    videos1 = os.listdir(folder1)
    videos2 = os.listdir(folder2)
    if oversample:
        print("Oversample the videos when the number of videos in the folder1 is less than the number of videos in the folder2.")
        if len(videos1) < len(videos2):
            videos1 = videos1 * (len(videos2) // len(videos1)) + videos1[:len(videos2) % len(videos1)]
    NUMBER_OF_VIDEOS = min(len(videos1), len(videos2))
    print("The number of videos in folder1 is {}, folder2 is {}. The set number of videos is {}".format(len(videos1), len(videos2), NUMBER_OF_VIDEOS))
    print("We will use {} videos for FVD calculation as this is the minimum number of videos.".format(NUMBER_OF_VIDEOS))
    videos1 = videos1[:NUMBER_OF_VIDEOS]
    videos2 = videos2[:NUMBER_OF_VIDEOS]
    # read frames
    def read_frames(folder):
        # the "folder" can be a video folder with multiple frames, can also be a .mp4 file
        if folder.endswith(".mp4"):
            # read video
            frames = []
            cap = cv2.VideoCapture(folder)
            for i in range(16):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(frame)
            cap.release()
        else:
            frames = []
            img_list = os.listdir(folder)
            if len(img_list) < 16:
                FRAME_INTERPOLATION = True
            else:
                FRAME_INTERPOLATION = False
            # sort according to the frame number
            img_list = sorted(img_list, key=lambda x: int(x.split(".")[0]))
            for frame in img_list:
                frame_path = os.path.join(folder, frame)
                # print("frame_path: ", frame_path)
                frame = Image.open(frame_path)
                frames.append(frame)
            # convert to tensor
            # frame interpolation to 16
            if FRAME_INTERPOLATION:
                frames_ = []
                for i in range(len(frames) - 1):
                    frames_.append(frames[i])
                    for j in range(1, 16 // (len(frames) - 1)):
                        alpha = j / (16 // (len(frames) - 1))
                        interpolated_frame = Image.blend(frames[i], frames[i + 1], alpha)
                        frames_.append(interpolated_frame)
                frames_.append(frames[-1])
                frames_.append(frames[-1])
                frames = frames_
        frames = [TVF.to_tensor(frame) for frame in frames]
        # concatenate to a video
        try:
            video = torch.stack(frames)
        except:
            print("Error in reading frames from folder: ", folder)
            return None
        return video
    
    videos1_ = []
    for video in tqdm(videos1):
        video_path = os.path.join(folder1, video)
        ret = read_frames(video_path)
        videos1_.append(ret) if ret is not None else None
    # videos1 = [read_frames(os.path.join(folder1, video)) for video in videos1]
    videos2 = [read_frames(os.path.join(folder2, video)) for video in videos2]
    # stack 
    videos1 = torch.stack(videos1_)
    videos2 = torch.stack(videos2)
    # no grad for the two videos
    videos1 = videos1.detach()
    videos2 = videos2.detach()
    device = torch.device("cuda")

    result_stylegan = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True)
    print("[fvd-styleganv]", result_stylegan["value"])

    # save the results to a txt file
    if class_name not in folder1:
        fold = "/".join(folder1.split("/")[:-1])
    else:
        fold = folder1.split(class_name)[0]
    with open(os.path.join(fold, "fvd_results.txt"), "a") as f:
        f.write("class: {}\n".format(class_name))
        # f.write("fvd-videogpt: {}\n".format(result["value"]))
        f.write("fvd-styleganv: {}\n".format(result_stylegan["value"]))
        f.write("\n")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True, help="The folder path of the first set of videos.")
    parser.add_argument("--oversample", action="store_true", help="Whether to oversample the videos when the number of videos in the folder1 is less than the number of videos in the folder2.")
    parser.add_argument("--folder2", type=str, default="/data_hdd/xyliu/medical_videos/hyperkvasir/frames_short_clip_128", help="The folder path of the second set of videos.")
    args = parser.parse_args()
    mapping = {0: 'barretts', 1: 'cancer', 2: 'esophagitis', 3: 'gastric-antral-vascular-ectasia', 4: 'gastric-banding-perforated', 5: 'polyps', 6: 'ulcer', 7: 'varices'}
    class_names = []
    class_ids = []
    for i in range(8):
        class_names.append(mapping[i])
        class_ids.append(i)
    print(class_names)
    print(class_ids)
    # input()
    for i in range(8):
        print("class: ", class_names[i])
        if os.path.exists(os.path.join(args.folder1, class_names[i])):
            folder1 = os.path.join(args.folder1, class_names[i])
        elif os.path.exists(os.path.join(args.folder1, str(class_ids[i]))):
            folder1 = os.path.join(args.folder1, str(class_ids[i]))
        else:
            print("The folder1 does not exist.")
            raise ValueError
        folder2 = os.path.join(args.folder2, class_names[i])
        main(class_names[i], folder1, folder2, args.oversample)
