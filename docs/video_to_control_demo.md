## Running the video to control demo. 



Install dependency:
```
conda create -n metrabs python=3.9
conda activate metrabs
pip install tensorflow asyncio aiohttp  opencv-python scipy joblib charset-normalizer tensorflow-hub pandas ultralytics
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```