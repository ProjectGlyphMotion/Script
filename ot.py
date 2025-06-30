import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import torch
import os
import sys
import subprocess
import threading
import time
import queue # For threaded frame reading

# Attempt to import psutil for CPU/Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ [WARNING] psutil library not found. RAM usage limiting and CPU/Memory utilization display will not be available.")

# Attempt to import GPUtil for GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except (ImportError, Exception):
    GPUTIL_AVAILABLE = False


# --- FFMPEG Check ---
FFMPEG_AVAILABLE = False

def check_ffmpeg():
    global FFMPEG_AVAILABLE
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("[INFO] ffmpeg found and seems to be working.")
        FFMPEG_AVAILABLE = True
    except Exception:
        print("⚠️ [WARNING] ffmpeg command not found or not working. Audio processing will be skipped.")
        FFMPEG_AVAILABLE = False
    return FFMPEG_AVAILABLE

# --- Configuration & Setup ---
DEFAULT_MODEL_PATH = "yolov8m.pt"
DEFAULT_OUTPUT_VIDEO_PATH_MARKER = "auto"
DEFAULT_ALLOWED_CLASSES = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "bird", "cat", "dog",
    "train", "boat", "bench", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "chair", "couch", "potted plant", "bed", "dining table",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "refrigerator", "book", "clock", "vase", "scissors"
]
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
TEMP_VIDEO_BASENAME = "temp_video_processed_silent.mp4"
OUTPUT_SUBDIRECTORY = "output"
FRAME_QUEUE_SIZE = 30
UTILIZATION_UPDATE_INTERVAL = 1.0
MAX_RAM_USAGE_PERCENT = 85.0

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ User Configuration +++
ENABLE_PREVIEW_IN_SCRIPT = False
USE_GPU_IN_SCRIPT = True
TARGET_PROCESSING_WIDTH = None
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_system_utilization(device_to_use):
    util_stats = {}
    if PSUTIL_AVAILABLE:
        util_stats['cpu'] = psutil.cpu_percent()
        mem_info = psutil.virtual_memory()
        util_stats['mem_used_gb'] = mem_info.used / (1024**3)
        util_stats['mem_total_gb'] = mem_info.total / (1024**3)
        util_stats['mem'] = mem_info.percent
    if GPUTIL_AVAILABLE and device_to_use == "cuda":
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                util_stats['gpu_load'] = gpu.load * 100
                util_stats['gpu_mem'] = gpu.memoryUtil * 100
        except Exception: pass
    return util_stats

def format_utilization_string(stats):
    parts = []
    if 'cpu' in stats: parts.append(f"CPU:{stats['cpu']:.1f}%")
    if 'mem' in stats:
        parts.append(f"Mem:{stats['mem']:.1f}% ({stats.get('mem_used_gb',0):.1f}/{stats.get('mem_total_gb',0):.1f}GB)")
    if 'gpu_load' in stats: parts.append(f"GPU-L:{stats['gpu_load']:.1f}%")
    if 'gpu_mem' in stats: parts.append(f"GPU-M:{stats['gpu_mem']:.1f}%")
    return " | ".join(parts) if parts else "Stats N/A"


def frame_reader_thread_func(cap, frame_input_queue, stop_event, target_width=None, rotate_for_portrait=False):
    print("[INFO] Frame reader thread started.")
    count = 0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    intermediate_width = original_width
    intermediate_height = original_height

    if rotate_for_portrait:
        print(f"[INFO] Frame reader: Rotating frames 90 degrees clockwise to correct portrait orientation.")
        intermediate_width = original_height
        intermediate_height = original_width

    processing_width = intermediate_width
    processing_height = intermediate_height
    do_resize = False

    if target_width and target_width > 0 and target_width < intermediate_width:
        aspect_ratio = intermediate_height / intermediate_width
        processing_width = target_width
        processing_height = int(processing_width * aspect_ratio)
        if processing_height % 2 != 0: processing_height +=1
        do_resize = True
        print(f"[INFO] Frame reader: Resizing frames from {intermediate_width}x{intermediate_height} to {processing_width}x{processing_height} (after potential rotation).")
    else:
        print(f"[INFO] Frame reader: Processing at intermediate resolution {intermediate_width}x{intermediate_height} (after potential rotation, if any).")


    while not stop_event.is_set() and cap.isOpened():
        if PSUTIL_AVAILABLE:
            current_ram_usage = psutil.virtual_memory().percent
            ram_check_loops = 0
            while current_ram_usage > MAX_RAM_USAGE_PERCENT and not stop_event.is_set():
                if ram_check_loops % 5 == 0:
                    print(f"⚠️ [WARNING] [FRAME_READER] High RAM usage: {current_ram_usage:.1f}%. Pausing frame reading for 1s...")
                time.sleep(1.0)
                current_ram_usage = psutil.virtual_memory().percent
                ram_check_loops +=1
                if stop_event.is_set():
                    print("[INFO] [FRAME_READER] Stop event received during RAM pause.")
                    if not frame_input_queue.full(): frame_input_queue.put((False, None, None, None))
                    return

        if frame_input_queue.qsize() < FRAME_QUEUE_SIZE:
            ret, frame = cap.read()
            if not ret:
                print(f"[INFO] Frame reader: End of video or cannot read frame after {count} frames.")
                frame_input_queue.put((False, None, None, None))
                break

            if rotate_for_portrait:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if do_resize:
                try:
                    processed_frame = cv2.resize(frame, (processing_width, processing_height), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    print(f"❌ [ERROR] [FRAME_READER] Failed to resize frame: {e}")
                    processed_frame = frame
            else:
                processed_frame = frame

            frame_input_queue.put((True, processed_frame, processing_width, processing_height))
            count += 1
        else:
            time.sleep(0.005)

    if not frame_input_queue.full():
        frame_input_queue.put((False, None, None, None))
    print(f"[INFO] Frame reader thread finished after reading {count} frames.")


def frame_writer_thread_func(video_writer, frame_output_queue, stop_event):
    print("[INFO] Frame writer thread started.")
    count = 0
    while not stop_event.is_set():
        try:
            ret, frame_to_write = frame_output_queue.get(timeout=0.1)
            if not ret:
                print(f"[INFO] Frame writer: End signal received after writing {count} frames.")
                break
            if frame_to_write is not None:
                video_writer.write(frame_to_write)
                count +=1
            frame_output_queue.task_done()
        except queue.Empty:
            if stop_event.is_set() and frame_output_queue.empty():
                print("[INFO] Frame writer: Stop event set and queue empty.")
                break
            continue
        except Exception as e:
            print(f"❌ [ERROR] [FRAME_WRITER] Error writing frame: {e}")
            break

    while not frame_output_queue.empty():
        try:
            ret, frame_to_write = frame_output_queue.get_nowait()
            if ret and frame_to_write is not None:
                video_writer.write(frame_to_write)
                count +=1
            frame_output_queue.task_done()
        except queue.Empty: break
    print(f"[INFO] Frame writer thread finished. Total frames written: {count}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Object Tracking: YOLOv8, Threaded I/O, GPU/CPU, Progress, ffmpeg Audio, Utilization")
    parser.add_argument("--model",type=str,default=DEFAULT_MODEL_PATH,help="Path to YOLOv8 model.")
    parser.add_argument("--output_video",type=str,default=DEFAULT_OUTPUT_VIDEO_PATH_MARKER, help="Path for final video. Default: 'auto' (derived from input name, saved in 'output/' subdir).")
    parser.add_argument("--allowed_classes",nargs="+",default=DEFAULT_ALLOWED_CLASSES,help=f"Classes to track.")
    parser.add_argument("--confidence_threshold",type=float,default=DEFAULT_CONFIDENCE_THRESHOLD,help=f"Min confidence.")
    parser.add_argument("--box_color", type=str, default="0,255,0", help="Color for bounding box as R,G,B (default: green)")
    return parser.parse_args()

def process_audio_ffmpeg(video_source_path, temp_silent_video_abs_path, final_output_video_path, target_fps):
    if not FFMPEG_AVAILABLE:
        print("⚠️ [WARNING] [AUDIO_THREAD] ffmpeg not available. Skipping audio merge.")
        if os.path.exists(temp_silent_video_abs_path) and not os.path.exists(final_output_video_path):
            try: os.rename(temp_silent_video_abs_path, final_output_video_path); print(f"✅ [SUCCESS] Silent video saved as '{final_output_video_path}'.")
            except OSError as e: print(f"❌ [ERROR] Could not rename temp file: {e}.")
        return False

    if not os.path.exists(temp_silent_video_abs_path):
        print(f"❌ [ERROR] [AUDIO_THREAD] Temporary silent video '{temp_silent_video_abs_path}' not found. Cannot merge audio.")
        return False

    print(f"\n[AUDIO_THREAD] Adding audio using ffmpeg (direct video stream copy mode).")
    if os.path.exists(final_output_video_path): print(f"⚠️ [WARNING] Output file {final_output_video_path} exists. Overwriting.")

    ffmpeg_command_base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
                           "-i", temp_silent_video_abs_path, "-i", video_source_path]

    video_codec_params = ["-c:v", "copy"]

    ffmpeg_command = ffmpeg_command_base + video_codec_params + ["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_output_video_path]

    try:
        print(f"[AUDIO_THREAD] Executing: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
        print(f"\n✅ [SUCCESS] [AUDIO_THREAD] ffmpeg successfully processed. Output: '{final_output_video_path}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [ERROR] [AUDIO_THREAD] ffmpeg failed (code {e.returncode}).")
    except Exception as e_ffmpeg: print(f"❌ [ERROR] [AUDIO_THREAD] ffmpeg error: {e_ffmpeg}")

    return False

def main():
    start_time_total = time.time()

    temp_video_file_abs_path = os.path.abspath(TEMP_VIDEO_BASENAME)

    # --- PyTorch CPU Threading ---
    try:
        cpu_cores = os.cpu_count()
        if cpu_cores:
            num_threads_to_set = max(1, cpu_cores // 2)
            torch.set_num_threads(num_threads_to_set)
            print(f"[INFO] Suggested {num_threads_to_set} threads for PyTorch CPU operations.")
        else:
            print("[INFO] Could not determine CPU core count for PyTorch thread setting.")
    except Exception as e:
        print(f"⚠️ [WARNING] Could not set PyTorch CPU threads: {e}")


    print("--- FFMPEG Check ---"); check_ffmpeg(); print("--------------------")
    if PSUTIL_AVAILABLE: print("[DEBUG] Priming psutil.cpu_percent()..."); psutil.cpu_percent()
    if GPUTIL_AVAILABLE:
        print("[DEBUG] Attempting to prime GPUtil.getGPUs()...")
        try: GPUtil.getGPUs(); print("[DEBUG] GPUtil.getGPUs() primed.")
        except Exception as e: print(f"[DEBUG] Error GPUtil priming: {e}")

    args = parse_arguments()
    try:
        box_color = tuple(int(c) for c in args.box_color.split(','))[::-1]
        if len(box_color) != 3 or not all(0 <= c <= 255 for c in box_color):
            raise ValueError
    except:
        print("❌ [ERROR] Invalid --box_color format. Use R,G,B with values 0-255. Defaulting to green.")
        box_color = (0, 255, 0)
        box_color = (0, 255, 0)

    input_video_path_interactive = input("➡️ Please enter the path to the input video file: ").strip()
    if input_video_path_interactive.startswith("'") and input_video_path_interactive.endswith("'"):
        input_video_path_interactive = input_video_path_interactive[1:-1]
    if input_video_path_interactive.startswith('"') and input_video_path_interactive.endswith('"'):
        input_video_path_interactive = input_video_path_interactive[1:-1]
    if not input_video_path_interactive:
        print("❌ [ERROR] No input video path provided. Exiting."); return
    if not os.path.exists(input_video_path_interactive) or not os.path.isfile(input_video_path_interactive):
        print(f"❌ [ERROR] Input video file not found or is not a file: {input_video_path_interactive}"); return

    current_input_video = input_video_path_interactive

    show_preview = ENABLE_PREVIEW_IN_SCRIPT
    use_gpu = USE_GPU_IN_SCRIPT
    target_processing_width = TARGET_PROCESSING_WIDTH

    os.makedirs(OUTPUT_SUBDIRECTORY, exist_ok=True)

    final_output_video_path = args.output_video
    if args.output_video == DEFAULT_OUTPUT_VIDEO_PATH_MARKER:
        input_basename_only = os.path.basename(current_input_video)
        name, ext = os.path.splitext(input_basename_only)
        base_name = f"{name}_processed{ext}"
        final_output_video_path = os.path.join(OUTPUT_SUBDIRECTORY, base_name)
    else:
        if not os.path.isabs(args.output_video):
            final_output_video_path = os.path.join(OUTPUT_SUBDIRECTORY, os.path.basename(args.output_video))

    print(f"[INFO] Final output video will be saved as: {final_output_video_path}")

    print("[INFO] ffmpeg will use direct video stream copy (original quality from processing).")

    device_to_use = "cpu"
    if use_gpu and torch.cuda.is_available(): device_to_use = "cuda"; print("✅ [SUCCESS] CUDA GPU available. Using GPU.")
    elif use_gpu: print("⚠️ [WARNING] CUDA GPU not found. Falling back to CPU.")
    else: print("ℹ️ [INFO] Using CPU.")

    print(f"[INFO] Loading model: {args.model} on {device_to_use}")
    try: model = YOLO(args.model); model.to(device_to_use); print(f"✅ [SUCCESS] Model loaded.")
    except Exception as e: print(f"❌ [ERROR] Failed to load model: {e}"); return

    print(f"[INFO] Tracking classes: {args.allowed_classes}")

    video_source_path = current_input_video
    video_capture_source = current_input_video

    cap = cv2.VideoCapture(video_capture_source)
    if not cap.isOpened(): print(f"❌ [ERROR] Failed to open input video file: {video_capture_source}"); return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rotate_for_portrait = False

    if original_width / original_height > 1.2 and original_height / original_width > 0.5:
            rotate_for_portrait = True
            print(f"[INFO] Input video dimensions {original_width}x{original_height} suggest a portrait video read as landscape. Frames will be rotated.")
    else:
            print(f"[INFO] Input video dimensions {original_width}x{original_height} appear to be correct. No rotation assumed.")


    output_width = original_width
    output_height = original_height

    if rotate_for_portrait:
        output_width = original_height
        output_height = original_width
        print(f"[INFO] Output video dimensions will be {output_width}x{output_height} (after rotation).")

    effective_target_width_for_reader = None

    if target_processing_width and target_processing_width > 0:
        if target_processing_width < output_width:
            effective_target_width_for_reader = target_processing_width
            aspect_ratio = output_height / output_width
            output_width = effective_target_width_for_reader
            output_height = int(output_width * aspect_ratio)
            if output_height % 2 != 0: output_height +=1
            print(f"[INFO] Target processing width set to {output_width}. Final output video will be resized to {output_width}x{output_height}")
        else:
            print(f"[INFO] Target processing width ({target_processing_width}) is not less than the effective video width ({output_width}). Processing at effective resolution.")


    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps==0.0 or fps is None: fps=30.0; print(f"⚠️ [WARNING] FPS not readable, defaulting to {fps} FPS.")
    print(f"[INFO] Input video (FPS for output writer): {fps:.2f} FPS")

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    print(f"[INFO] Using FourCC: MP4V for temporary video writer.")
    temp_out_silent_video_writer=cv2.VideoWriter(temp_video_file_abs_path, fourcc,float(fps),(output_width,output_height))
    if not temp_out_silent_video_writer.isOpened():
        print(f"❌ [ERROR] Failed to open temp video writer with MP4V: {temp_video_file_abs_path}"); cap.release(); return
    print(f"[INFO] Temp silent video will be {output_width}x{output_height} at {temp_video_file_abs_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stop_event=threading.Event()
    frame_input_queue=queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    reader_thread=threading.Thread(target=frame_reader_thread_func,args=(cap,frame_input_queue,stop_event, effective_target_width_for_reader, rotate_for_portrait))
    reader_thread.daemon = True
    reader_thread.start()

    frame_output_queue=queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    writer_thread=threading.Thread(target=frame_writer_thread_func,args=(temp_out_silent_video_writer,frame_output_queue,stop_event))
    writer_thread.daemon = True
    writer_thread.start()

    pbar_desc=f"Processing Frames ({device_to_use.upper()})"
    if total_frames > 0:
        pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        pbar = tqdm(total=total_frames, desc=pbar_desc, unit="frame", bar_format=pbar_format, mininterval=UTILIZATION_UPDATE_INTERVAL)
    else:
        pbar_format = "{l_bar}{bar}| {n_fmt} frames [{elapsed}, {rate_fmt}{postfix}]"
        pbar = tqdm(desc=pbar_desc, unit="frame", bar_format=pbar_format, mininterval=UTILIZATION_UPDATE_INTERVAL)

    print(f"🚀 Starting video frame processing on {device_to_use.upper()}...")
    frame_count=0; processing_loop_active=True; last_util_update_time=time.time()

    try:
        while processing_loop_active:
            try:
                ret, frame_to_process, current_proc_w, current_proc_h = frame_input_queue.get(timeout=1.0)
                if not ret: processing_loop_active=False; break
                frame_input_queue.task_done()
            except queue.Empty:
                if not reader_thread.is_alive() and frame_input_queue.empty(): processing_loop_active=False; break
                continue

            frame_count += 1
            results = model.track(frame_to_process, persist=True, verbose=False, conf=args.confidence_threshold)
            annotated_frame = frame_to_process.copy()

            if results and results[0].boxes:
                for box in results[0].boxes:
                    if box.id is None: continue
                    cls_id=int(box.cls[0]); class_name=model.names[cls_id]; track_id=int(box.id[0])
                    if class_name in args.allowed_classes:
                        x1,y1,x2,y2=map(int,box.xyxy[0]); conf=box.conf[0]
                        label=f"ID:{track_id} {class_name} {conf:.2f}"
                        cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),box_color,2)
                        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                        cv2.rectangle(annotated_frame,(x1,y1-th-10),(x1+tw,y1-5),box_color,-1)
                        cv2.putText(annotated_frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

            frame_output_queue.put((True, annotated_frame))

            current_time = time.time()
            if current_time - last_util_update_time >= UTILIZATION_UPDATE_INTERVAL:
                util_stats = get_system_utilization(device_to_use)
                pbar.set_postfix_str(format_utilization_string(util_stats), refresh=True)
                last_util_update_time = current_time
            if show_preview:
                cv2.imshow("Object Tracking", annotated_frame)
                if cv2.waitKey(1)&0xFF==ord("q"): processing_loop_active=False; break
            pbar.update(1)
            del results

    except KeyboardInterrupt: print("\n[INFO] KeyboardInterrupt. Stopping..."); processing_loop_active = False
    finally:
        print("[INFO] Main loop finished. Signaling I/O threads to stop...")
        if stop_event: stop_event.set()

        if reader_thread is not None and reader_thread.is_alive():
            print("[INFO] Waiting for frame reader thread..."); reader_thread.join(timeout=5.0)
            if reader_thread.is_alive(): print("⚠️ [WARNING] Frame reader didn't finish in time.")

        if frame_output_queue is not None:
             frame_output_queue.put((False, None))

        if writer_thread is not None and writer_thread.is_alive():
            print("[INFO] Waiting for frame writer thread..."); writer_thread.join(timeout=10.0)
            if writer_thread.is_alive(): print("⚠️ [WARNING] Frame writer didn't finish in time.")

        pbar.close();
        if cap.isOpened(): cap.release()
        if temp_out_silent_video_writer.isOpened(): temp_out_silent_video_writer.release()
        if show_preview: cv2.destroyAllWindows()

    print(f"✅ [SUCCESS] Video frame processing complete. Silent video: '{temp_video_file_abs_path}'.")
    print(f"[INFO] Total frames processed: {frame_count}")

    audio_processing_was_successful = False
    if FFMPEG_AVAILABLE:
        print("[INFO] Starting audio processing thread...")
        audio_processing_was_successful = process_audio_ffmpeg(video_source_path, temp_video_file_abs_path, final_output_video_path, float(fps))
    else:
        if os.path.exists(temp_video_file_abs_path):
            try:
                if os.path.exists(final_output_video_path): os.remove(final_output_video_path)
                os.rename(temp_video_file_abs_path, final_output_video_path); print(f"✅ [SUCCESS] Silent video saved: '{final_output_video_path}'.")
                audio_processing_was_successful = True
            except OSError as e: print(f"❌ [ERROR] Could not rename temp file: {e}.")

    if os.path.exists(temp_video_file_abs_path):
        if audio_processing_was_successful and os.path.abspath(temp_video_file_abs_path) != os.path.abspath(final_output_video_path):
            try:
                os.remove(temp_video_file_abs_path)
                print(f"[CLEANUP] Temp silent video '{temp_video_file_abs_path}' deleted.")
            except OSError as e:
                print(f"⚠️ [WARNING] Error deleting temporary file '{temp_video_file_abs_path}': {e}")
        elif not audio_processing_was_successful and os.path.abspath(temp_video_file_abs_path) == os.path.abspath(final_output_video_path):
            print(f"[INFO] Audio processing failed/skipped. Output is the processed silent video: '{final_output_video_path}'.")
        elif not audio_processing_was_successful and os.path.exists(temp_video_file_abs_path):
             print(f"[INFO] Audio processing failed/skipped. Temporary file '{temp_video_file_abs_path}' is kept.")

    end_time_total = time.time()
    total_processing_time = end_time_total - start_time_total
    minutes = int(total_processing_time // 60)
    seconds = int(total_processing_time % 60)
    print(f"\n[DONE] Total script execution time: {minutes} minute(s) and {seconds} second(s).")
    print(f"[INFO] Final output video is at: {final_output_video_path}")

if __name__ == "__main__":
    main()
