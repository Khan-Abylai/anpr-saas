FROM nvcr.io/nvidia/tritonserver:22.12-py3

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata sudo libgl1

RUN sudo pip3 install opencv-python
RUN sudo pip3 install pillow
RUN sudo pip3 install fastapi python-multipart pydantic_settings gunicorn uvicorn
RUN sudo pip3 install torchvision tritonclient[all] pydantic_settings
