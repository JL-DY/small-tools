import cv2
import tkinter as tk
from tkinter import filedialog
import os

def choose_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi *.mkv")])
    return file_path

def find_character_index(input_string, character):
    try:
        index = input_string.index(character)
        return index
    except ValueError:
        return "未找到字符."
    
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件！")
        return
    
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_name = os.path.basename(video_path)
    index = find_character_index(video_name, '.')
    video_name = video_name[:index]

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"{output_folder}/{video_name}_{frame_count}.png" #f"{}"一种字符串格式化的用法
        if frame_count%video_fps==0:
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"视频{video_name}中共有 {frame_count} 帧，已保存到 {output_folder} 文件夹中。")


if __name__ == "__main__":
    video_file = choose_video_file()
    if not video_file:
        print("未选择视频文件。")

    output_folder = "E:/Project-python/small_tools/extract_frame/"
    extract_frames(video_file, output_folder)
