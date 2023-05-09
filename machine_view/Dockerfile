FROM tensorflow/tensorflow:2.12.0-gpu-jupyter
WORKDIR /tf/.
RUN apt install protobuf-compiler -y
RUN pip3 install pillow ipython tflite-model-maker opencv-python-headless tensorflow-object-detection-api

RUN git clone https://github.com/tensorflow/models.git
RUN cd models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
