FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Python and essential tools
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git curl wget unzip nano vim htop \
    openjdk-17-jre-headless zsh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Oh My Zsh for better developer experience
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Data‑science + GPU H2O
RUN pip install --no-cache-dir \
    jupyterlab notebook numpy pandas matplotlib seaborn scikit-learn \
    tensorflow torch \
    h2o[xgboost] mlflow fastapi streamlit

WORKDIR /workspace
RUN mkdir -p notebooks data models code venvs

# Jupyter config
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'\nc.NotebookApp.open_browser = False\nc.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

ENV H2O_XGB_BACKEND=external-gpu \
    NVIDIA_VISIBLE_DEVICES=all

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
