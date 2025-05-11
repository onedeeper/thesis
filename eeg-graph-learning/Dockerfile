# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Install Miniconda
#hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile && \
    echo "conda activate base" >> ~/.bashrc

# Make RUN commands use the Miniconda/Conda environment
ENV PATH /opt/conda/bin:$PATH

# Copy the environment file and create the Conda environment
COPY environment.linux.yml /tmp/environment.linux.yml
RUN conda env create -f /tmp/environment.linux.yml && \
    conda clean -all -f -y

# Make the new Conda environment the default for subsequent shell commands
SHELL ["conda", "run", "-n", "eeg-graph-learning-linux", "/bin/bash", "-c"]

# Set the working directory in the container
WORKDIR /app

# Copy the application code (respecting .dockerignore)
# This provides a baseline in the image. For development, we'll use volume mounts.
COPY . /app/

# Install IPython kernel (as in clean.sh, ensuring it's in the correct env)
RUN python -m ipykernel install --user --name=eeg-graph-learning-linux --display-name="Python (eeg-graph-learning-linux)"

# Expose any necessary ports (e.g., for Jupyter if you plan to use it from outside)
# EXPOSE 8888

# Default command drops into a bash shell in the activated environment
CMD ["/bin/bash"]