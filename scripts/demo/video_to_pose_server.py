#!/usr/bin/env python3
import os
import cv2
import joblib
import numpy as np
import time

import tensorflow as tf
import tensorflow_hub as hub


import asyncio
from aiohttp import web
import cv2
import aiohttp
import numpy as np
import threading
from scipy.spatial.transform import Rotation as sRot

import time
import torch
from collections import deque
from datetime import datetime
from torchvision import transforms as T
import time
from ultralytics import YOLO
import scipy.interpolate as interpolate

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

det_model = YOLO("yolov8s.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
STANDING_POSE = np.array([[[-0.1443, -0.9426, -0.2548],
         [-0.2070, -0.8571, -0.2571],
         [-0.0800, -0.8503, -0.2675],
         [-0.1555, -1.0663, -0.3057],
         [-0.2639, -0.5003, -0.2846],
         [-0.0345, -0.4931, -0.3108],
         [-0.1587, -1.2094, -0.2755],
         [-0.2534, -0.1022, -0.3361],
         [-0.0699, -0.1012, -0.3517],
         [-0.1548, -1.2679, -0.2675],
         [-0.2959, -0.0627, -0.2105],
         [-0.0213, -0.0424, -0.2277],
         [-0.1408, -1.4894, -0.2892],
         [-0.2271, -1.3865, -0.2622],
         [-0.0715, -1.3832, -0.2977],
         [-0.1428, -1.5753, -0.2303],
         [-0.3643, -1.3792, -0.2646],
         [ 0.0509, -1.3730, -0.3271],
         [-0.3861, -1.1423, -0.3032],
         [ 0.0634, -1.1300, -0.3714],
         [-0.4086, -0.9130, -0.2000],
         [ 0.1203, -0.8943, -0.3002],
         [-0.4000, -0.8282, -0.1817],
         [ 0.1207, -0.8087, -0.2787]]]).repeat(5, axis = 0)

def fps_20_to_30(mdm_jts):
    jts = []
    N = mdm_jts.shape[0]
    for i in range(24):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x)
        f_y = interpolate.interp1d(x, int_y)
        f_z = interpolate.interp1d(x, int_z)
        
        new_x = f_x(np.linspace(0, N-1, int(N * 1.5)))
        new_y = f_y(np.linspace(0, N-1, int(N * 1.5)))
        new_z = f_z(np.linspace(0, N-1, int(N * 1.5)))
        jts.append(np.stack([new_x, new_y, new_z], axis = 1))
    jts = np.stack(jts, axis = 1)
    return jts


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path

def start_pose_estimate():
    global pose_mat, trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, fps
    offset = np.zeros((5, 1))

    from scipy.spatial.transform import Rotation as sRot
    global_transform = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix()
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()

    prev_box = None
    t_s = time.time()
    print('### Run Model...')
    
    # model = tf.saved_model.load(download_model('metrabs_mob3l_y4'))
    model = hub.load('https://bit.ly/metrabs_s') # or _l

    skeleton = 'smpl_24'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
    # viz = poseviz.PoseViz(joint_names, joint_edges)
    print("==================================> Metrabs model loaded <==================================")
    
    with torch.no_grad():
        while True:
            if not frame is None:
                # pred = model.detect_poses(frame, skeleton=skeleton, default_fov_degrees=55, detector_threshold=0.5, num_aug=5)
                pred = model.estimate_poses(frame, tf.constant(bbox, dtype=tf.float32), skeleton=skeleton, default_fov_degrees=55, num_aug=1)
                
                dt = time.time() - t_s
                fps = 1/dt
                
                # camera = poseviz.Camera.from_fov(55, frame.shape[:2])
                # viz.update(frame, pred['boxes'], pred['poses3d'], camera)
                pred_j3d = pred['poses3d'].numpy()
                num_ppl = min(pred_j3d.shape[0], 5)
                
                j3d_curr = pred_j3d[:num_ppl]/1000
                if num_ppl < 5:
                    j3d[num_ppl:, 0, 0] = np.arange(5 - num_ppl) + 1
                    
                j2d =  pred['poses2d'].numpy()
                t_s = time.time()
                
                if reset_offset:
                    offset[:num_ppl] = - offset_height - j3d_curr[:num_ppl, [0], 1]
                    reset_offset = False
                
                j3d_curr[:offset.shape[0], ..., 1] += offset[:num_ppl]
                
                j3d = j3d.copy() # Trying to handle race condition
                j3d[:num_ppl] = j3d_curr
            
def get_max_iou_box(det_output, prev_bbox, thrd=0.9):
    max_score = 0
    max_bbox = None
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        # if float(score) < thrd:
        #     continue
        # area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        iou = calc_iou(prev_bbox, bbox)
        iou_score = float(score) * iou
        if float(iou_score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = iou_score
    
    if max_bbox is None:
        max_bbox = prev_bbox

    return max_bbox

def calc_iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    box2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def get_one_box(det_output, thrd=0.9):
    max_area = 0
    max_bbox = None

    if det_output['boxes'].shape[0] == 0 or thrd < 1e-5:
        return None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    if max_bbox is None:
        return get_one_box(det_output, thrd=thrd - 0.1)

    return max_bbox

def commandline_input():
    global pose_mat, trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, fps
    
    while True:
        command = input('Type a message to send to the server: ')
        if command == 'exit':
            print('Exiting!')
            raise SystemExit(0)
        elif command.startswith("r"):
            splits = command.split(":")
            if len(splits) > 1:
                offset_height = float(splits[-1])
            reset_offset = True
        elif command.startswith("fps"):
            print(fps)
        else:
            print("Unkonw command")

def frames_from_webcam():
    global frame, images_acc, recording, j2d, bbox
    cap = cv2.VideoCapture(-1)
    prev_box = None
    
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame_orig = cap.read()
        if not ret:
            continue
        # x1, y1, x2, y2 = bbox
        detec_threshold = 0.6
        
        frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB) # send to the detector & model 
        yolo_output = det_model.predict(source=frame, show=False, classes=[0], verbose=False)
        
        if len(yolo_output[0].boxes) > 0:
            yolo_out_xyxy = yolo_output[0].boxes.xyxy.cpu().numpy()
            
            bbox = np.stack([yolo_out_xyxy[:, 0], yolo_out_xyxy[:, 1], (yolo_out_xyxy[:, 2] - yolo_out_xyxy[:, 0]), (yolo_out_xyxy[:, 3] - yolo_out_xyxy[:, 1])], axis = 1)
            
            for i in range(len(yolo_out_xyxy)):
                x1, y1, x2, y2 = yolo_out_xyxy[i]
                frame_orig = cv2.rectangle(frame_orig, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)
            
        if not j2d is None:
            for pt in j2d.reshape(-1, 2):
                x, y = pt
                frame_orig = cv2.circle(frame_orig, (int(x), int(y)), 3, (255, 136, 132), 3)
                
        if recording:
            images_acc.append(frame_orig.copy())
            
        cv2.imshow('frame', frame_orig)
        
        if cv2.waitKey(1) == ord('q'):
            break
        # yield frame

async def pose_getter(request):
    # query env configurations
    global pose_mat, trans, dt, j3d, superfast
    curr_paths = {}
    if superfast:
        json_resp = {
            "j3d": j3d.tolist(),
            "dt": dt,
        }

    else:
        json_resp = {
            "pose_mat": pose_mat.tolist(),
            "trans": trans.tolist(),
            "dt": dt,
        }
        
    return web.json_response(json_resp)

# async def commad_interface(request):
    

async def websocket_handler(request):
    print('Websocket connection starting')
    global pose_mat, trans, dt, sim_talker
    sim_talker = aiohttp.web.WebSocketResponse()

    await sim_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in sim_talker:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == "get_pose":
                await sim_talker.send_json({
                    "pose_mat": pose_mat.tolist(),
                    "trans": trans.tolist(),
                    "dt": dt,
                })

    print('Websocket connection closed')
    return sim_talker

def write_frames_to_video(frames, out_file_name = "output.mp4", frame_rate = 30, add_text = None, text_color = (255, 255, 255)):
    print(f"######################## Writing number of frames {len(frames)} ########################")
    if len(frames) == 0:
        return 
    y_shape, x_shape, _ = frames[0].shape
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'FMP4'), frame_rate, (x_shape, y_shape))
    transform_dtype = False
    transform_256 = False

    if frames[0].dtype != np.uint8:
        transform_dtype = True
    if np.max(frames[0]) < 1:
        transform_256 = True

    for i in range(len(frames)):
        curr_frame = frames[i]

        if transform_256:
            curr_frame = curr_frame * 256
        if transform_dtype:
            curr_frame = curr_frame.astype(np.uint8)
        if not add_text is None:
            cv2.putText(curr_frame, add_text , (0,  20), 3, 1, text_color)

        out.write(curr_frame)
    out.release()

async def talk_websocket_handler(request):
    print('Websocket connection starting')
    global reset_offset, trans, offset_height, recording, images_acc
    ws_talker = aiohttp.web.WebSocketResponse()

    await ws_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in ws_talker:
        #       print(msg)
        if msg.type == aiohttp.WSMsgType.TEXT:
            print("\n" + msg.data)
            if msg.data.startswith("r"):
                splits = msg.data.split(":")
                if len(splits) > 1:
                    offset_height = float(splits[-1])
                reset_offset = True
            elif msg.data.startswith("s"):
                recording = True
                print(f"----------------> recording: {recording}")
                # if recording:
                    # pass
                # if recording and not sim_talker is None:
                    # await sim_talker.send_json({"action": "start_record"})
            elif msg.data.startswith("e"):
                recording = False
                print(f"----------------> recording: {recording}")

            elif msg.data.startswith("w"):
                curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                out_file_name = f"output/hybrik_{curr_date_time}.mp4"
                print(f"----------------> writing video: {out_file_name}")
                write_frames_to_video(images_acc, out_file_name = out_file_name)
                images_acc = deque(maxlen = 24000)
            elif msg.data.startswith("get_pose"):
                await sim_talker.send_json({
                    "j3d": j3d.tolist(),
                    "dt": dt,
                })

            await ws_talker.send_str("Done!")

    print('Websocket connection closed')
    return ws_talker


bbox, pose_mat, j3d, j2d, trans, dt, ws_talkers, reset_offset, offset_height, images_acc, recording, sim_talker, num_ppl, fps= np.zeros([5, 4]), np.zeros([24, 3, 3]), np.zeros([5, 24, 3]), None, np.zeros([3]), 1 / 10, [], True, 0.92, deque(maxlen = 24000), False, None, 0, 0
frame = None
superfast = True
# main()
app = web.Application(client_max_size=1024**2)
app.router.add_route('GET', '/ws', websocket_handler)
app.router.add_route('GET', '/ws_talk', talk_websocket_handler)
app.router.add_route('GET', '/get_pose', pose_getter)
threading.Thread(target=frames_from_webcam, daemon=True).start()
threading.Thread(target=start_pose_estimate, daemon=True).start()
threading.Thread(target=commandline_input, daemon=True).start()
print("=================================================================")
print("r: reset offset (use r:0.91), s: start recording, e: end recording, w: write video")
print("=================================================================")
web.run_app(app, port=8080)