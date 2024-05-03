FROM ubuntu:latest

ENV PYTHON_VERSION 3.12
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

# タイムゾーン
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# apt
RUN apt update
RUN apt install -y libopencv-dev

# install python and pip
RUN apt install -y python3 python3-pip
# RUN pip install --upgrade pip

# set working directory and copy files
WORKDIR /usr/app/src

# install opencv
RUN pip install opencv-python==4.7.0.72 --break-system-packages
RUN pip install opencv-contrib-python --break-system-packages
RUN pip install numpy --break-system-packages

# install FastAPI
RUN pip install fastapi --break-system-packages
RUN pip install uvicorn[standard] --break-system-packages
RUN pip install python-multipart --break-system-packages
RUN pip install Pillow --break-system-packages