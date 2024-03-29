# Stable Diffusion from scratch (excluding the weights(I am gpu poor)

![PyTorch](https://img.shields.io/badge/PyTorch-DD4B39?style=for-the-badge&logo=pytorch&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI/ML-000000?style=for-the-badge&logo=ai&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)


## Stable Diffusion Model Architecture

The architecture of the stable diffusion model is based on a modified U-Net structure, which is a popular choice for image processing tasks due to its ability to capture fine-grained details while maintaining context. The model architecture includes a series of convolutional layers that gradually increase in size, followed by corresponding layers that decrease in size. This structure allows the model to learn hierarchical representations of the input data, enabling it to generate high-quality images from noise.

I got the: - 
- **encoder** (reduce image size while increasing channels.)
- **decoder**(...self explanatory)
- **ddpm**( methods to set inference timesteps, set the strength of noise to be added, and to perform a diffusion step.)
- **clip**(combines token and positional embeddings, class includes self-attention and feedforward layers within each transformer layer,  class orchestrates the embedding and transformer layers, forming the CLIP             model)
- **diffusion**(superhero of image processing, )
- **model loader** (....self explanatory)
- **model converter** (bunch of model names converted to a more matchable dictionary)

  
## Key Features

- **DDPM Scheduler**: Utilizes the DDPM scheduler for efficient diffusion process management.
- **Stable Diffusion Model**: Implements a modified U-Net architecture for high-quality image generation.
- **PyTorch Implementation**: Leverages the power of PyTorch for fast and efficient model training and inference.

## Conclusion

This project demonstrates the potential of diffusion models in generating high-quality images from noise, showcasing the capabilities of PyTorch and the DDPM scheduler in creating advanced AI/ML models. The stable diffusion model, with its modified U-Net architecture, provides a robust framework for image generation tasks, offering a glimpse into the future of generative models.


-get the v1-5-pruned-emaonly.ckpt pre trained weight from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
