# Stable Diffusion from scratch (excluding the weights(I am gpu poor)

![PyTorch](https://img.shields.io/badge/PyTorch-DD4B39?style=for-the-badge&logo=pytorch&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI/ML-000000?style=for-the-badge&logo=ai&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

## Overview

This project involves the creation of a stable diffusion model from scratch using PyTorch, leveraging the DDPM (Denoising Diffusion Probabilistic Models) scheduler for efficient and effective image generation. The model is designed to transform noise into coherent images, showcasing the power of diffusion models in generating high-quality, realistic images.

## DDPM Scheduler Implementation

The DDPM scheduler is a key component of this project, responsible for guiding the diffusion process. It operates by iteratively updating the sample being diffused, using a timestep to predict the updated version of the sample. The scheduler is designed to be model, system, and framework independent, allowing for rapid experimentation and cleaner abstractions in the code. This design principle separates the model prediction from the sample update, facilitating the trade-off between speed and generation quality [0](https://huggingface.co/docs/diffusers/v0.13.0/en/api/schedulers/overview).

## Stable Diffusion Model Architecture

The architecture of the stable diffusion model is based on a modified U-Net structure, which is a popular choice for image processing tasks due to its ability to capture fine-grained details while maintaining context. The model architecture includes a series of convolutional layers that gradually increase in size, followed by corresponding layers that decrease in size. This structure allows the model to learn hierarchical representations of the input data, enabling it to generate high-quality images from noise.

The diffusion process is split into two main components: the forward diffusion process and the reverse diffusion process. The forward diffusion process transforms an image into noise, while the reverse diffusion process aims to reconstruct the original image from the noise. The model predicts the entire noise to be removed in a given timestep, ensuring that the diffusion process is guided towards the target image [1](https://medium.com/@kemalpiro/step-by-step-visual-introduction-to-diffusion-models-235942d2f15c).

## Key Features

- **DDPM Scheduler**: Utilizes the DDPM scheduler for efficient diffusion process management.
- **Stable Diffusion Model**: Implements a modified U-Net architecture for high-quality image generation.
- **PyTorch Implementation**: Leverages the power of PyTorch for fast and efficient model training and inference.

## Conclusion

This project demonstrates the potential of diffusion models in generating high-quality images from noise, showcasing the capabilities of PyTorch and the DDPM scheduler in creating advanced AI/ML models. The stable diffusion model, with its modified U-Net architecture, provides a robust framework for image generation tasks, offering a glimpse into the future of generative models.


-get the weights from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
