# 🎯 YOLOv8 Object Tracker with Custom Box Colors

This script performs object tracking on input videos using [YOLOv8](https://github.com/ultralytics/ultralytics) and allows you to specify bounding box colors. It supports threading for I/O, system resource monitoring, and preserves original video audio using FFmpeg.

---

## 🚀 Features

- ✅ **YOLOv8 Object Detection and Tracking**
- 🎨 **Customizable Bounding Box Color**
- ⚙️ **Threaded Frame Reading/Writing**
- 📊 **Real-Time CPU/GPU/RAM Monitoring**
- 🧠 **GPU Acceleration (CUDA supported)**
- 🔈 **Preserve Audio using FFmpeg**
- 💾 **Automatic Output Directory Creation**
- 🔄 **Progress Bar with `tqdm`**

---

## 🛠️ Setup

### Clone the Repository

```bash
git clone https://github.com/sayann70/ObjectTracking-Python && cd ObjectTracking-Python
```

### Install Dependencies

```bash
pip install -r requirements.txt
```
- Ensure **FFmpeg** is installed and accessible from the command line.

- If you want to use GPU power to process the video, ensure NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) is installed properly.

- Cuda can be a pain in the 🍑HOLE if you have a 30 or 40 series card, here is a [FIX](https://www.reddit.com/r/StableDiffusion/comments/13n16r7/cuda_not_available_fix_for_anybody_that_is/)

---

## 🧪 Usage

Run the script and enter the input video path when prompted:

```bash
python3 ot.py                      
```

## For custom RGB colours (Default is Green)
🔵 Blue
```bash
python3 ot.py --box_color 0,0,255  
```
 🔴 Red
```bash
python3 ot.py --box_color 255,0,0  
```
🟢 Green
```bash
python3 ot.py --box_color 0,255,0  
```

## 📂 Output

- Videos are saved in the `output/` directory.
- Format: `<original_name>_processed.mp4`

---

## ✅ Example Result


[Here](https://drive.google.com/file/d/1kV9-v5E5T7AiDEnNQWlmznmK0GhN4JMc/view)
---

 Made with ❤️ by Sayan and Shitij
