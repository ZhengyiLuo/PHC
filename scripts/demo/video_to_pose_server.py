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
import poseviz
import time
from zen_tracker import run, parse_opt

# det_transform = T.Compose([T.ToTensor()])
# det_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
# det_model.classes = [0]
# det_model.cuda()
# det_model.eval()

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

async def main():
    global pose_mat, trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, tracking_res, images_acc
    offset = np.zeros((5, 1))
    
    ## debug 

    from scipy.spatial.transform import Rotation as sRot
    global_transform = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix()
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()

    t_s = time.time()
    print('### Run Model...')
    
    # model = tf.saved_model.load(download_model('metrabs_mob3l_y4'))
    # model = tf.saved_model.load(download_model('metrabs_eff2s_y4'))
    model = hub.load('https://bit.ly/metrabs_s') # or _s

    skeleton = 'smpl_24'
    # viz = poseviz.PoseViz(joint_names, joint_edges)
    print("==================================> Metrabs model loaded <==================================")
    
    with torch.no_grad():
        while True:
            if 'img' in tracking_res and "detections" in tracking_res:
                
                tracking_boxes = tracking_res['detections']
                frame = tracking_res['img']
                bbox = tracking_boxes[:, :4]
                
                # pred = model.detect_poses(frame, skeleton=skeleton, default_fov_degrees=55, detector_threshold=0.5, num_aug=5)
                pred = model.estimate_poses(frame, tf.constant(bbox, dtype=tf.float32), skeleton=skeleton, default_fov_degrees=55, num_aug=1)
                
                dt = time.time() - t_s
                print(f'\r {1/dt:.2f} fps', end='')
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
                    
                tracking_res['j2d'] = j2d
                
             
def frames_from_webcam_lite():
    global frame, images_acc, recording, j2d, bbox, tracking_res
    cap = cv2.VideoCapture(-1)
    
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame_orig = cap.read()
        tracking_res['img'] = frame_orig
        
        if recording:
            images_acc.append(frame_orig.copy())
        # if cv2.waitKey(1) == ord('q'):
            # break




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
                tracking_res['recording'] =  True
                print(f"----------------> recording: {recording}")
                # if recording:
                    # pass
                if recording and not sim_talker is None:
                    await sim_talker.send_json({"action": "start_record"})
            elif msg.data.startswith("e"):
                recording = False
                tracking_res['recording'] =  False
                print(f"----------------> recording: {recording}")
                if not recording and not sim_talker is None:
                    await sim_talker.send_json({"action": "end_record"})

            elif msg.data.startswith("w"):
                curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                out_file_name = f"output/hybrik_{curr_date_time}.mp4"
                print(f"----------------> writing video: {out_file_name}")
                write_frames_to_video(tracking_res['images_acc'], out_file_name = out_file_name)
                tracking_res['images_acc'] = deque(maxlen = 24000)
            elif msg.data.startswith("get_pose"):
                await sim_talker.send_json({
                    "j3d": j3d.tolist(),
                    "dt": dt,
                })

            await ws_talker.send_str("Done!")

    print('Websocket connection closed')
    return ws_talker


def start_pose_estimate():
    loop = asyncio.new_event_loop()  # <-- create new loop in this thread here
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        
    opt = parse_opt() # Tracker
    
    bbox, pose_mat, j3d, j2d, trans, dt, ws_talkers, reset_offset, offset_height, images_acc, recording, sim_talker, num_ppl = np.zeros([5, 4]), np.zeros([24, 3, 3]), np.zeros([5, 24, 3]), None, np.zeros([3]), 1 / 10, [], True, 0.92, deque(maxlen = 24000), False, None, 0
    j3d[:, 0, 0] = np.arange(5)
    tracking_res = {}
    frame = None
    superfast = True
    
    tracking_res['recording'] = recording
    tracking_res['images_acc'] = deque(maxlen = 24000)
    # main()
    app = web.Application(client_max_size=1024**2)
    app.router.add_route('GET', '/ws', websocket_handler)
    app.router.add_route('GET', '/ws_talk', talk_websocket_handler)
    app.router.add_route('GET', '/get_pose', pose_getter)
    # threading.Thread(target=frames_from_webcam_lite, daemon=True).start()
    threading.Thread(target=tracking_from_tracker, daemon=True).start()
    threading.Thread(target=start_pose_estimate, daemon=True).start()
    # tracking_from_tracker()
    
    
    print("=================================================================")
    print("r: reset offset (use r:0.91), s: start recording, e: end recording, w: write video")
    print("=================================================================")
    web.run_app(app, port=8080)
