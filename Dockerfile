# Use the base image
FROM pytorchlightning/pytorch_lightning:latest

# Set the working directory in the container
WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir \
    datasets==2.20.0 \
    librosa==0.10.2.post1 \
    transformers==4.43.2 \
    torchaudio==2.2.1 \
    accelerate==0.33.0 \
    beartype==0.1.1 \
    einops==0.8.0 \
    lion-pytorch==0.2.2 \
    wandb==0.17.5 \
    tensorboard==2.17.0
