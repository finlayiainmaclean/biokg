FROM mambaorg/micromamba:1.5-jammy-cuda-11.8.0 as build-stage

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y gnupg2 && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get install -y wget sudo rsync git tmux nano curl libopenmpi-dev ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml requirements.txt ./

ENV MAMBA_NO_LOW_SPEED_LIMIT=1
RUN --mount=type=cache,target=/opt/micromamba/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=pip,target=/root/.pip/pip.conf \
    micromamba install --name base --file environment.yaml && \
    micromamba clean --all --yes

RUN --mount=type=secret,id=pip,target=/root/.pip/pip.conf \
    micromamba run -n base pip install -r requirements.txt && \
    micromamba run -n base pip install -e biokg

ENV PATH="/opt/conda/bin:${PATH}"
RUN echo 'alias jupyter_notebook="jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"' >> ~/.bashrc

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]