FROM tensorflow/tensorflow:latest-gpu as gpu

WORKDIR /home/app_user/app

RUN pip install --upgrade pip
RUN pip install matplotlib
RUN pip install scipy
RUN pip install pyyaml
RUN pip install click
RUN pip install wandb

COPY src src
COPY data data
COPY conf conf
COPY train.py train.py