# import torch 
# from torch import nn
# from torch.nn import functional as F
# from decoder import VAE_attention, VAE_residual


# class VAE_encoder(nn.Sequential):
#     def __inti__(self):
#         super().__init__(
#             ###feature badhaoo size ghataoo charo dishaoo me fehlaoo#### 
#             #batch_size, channel, height, width -> batch_size, 128 (features hai bhai), height, width
#             nn.Conv2d(3, 128, kernel_size=3, padding=1),
#             ##par bhai ye residual block kyu hai??
#             # batch_size, 128 (features hai bhai), height, width -> batch_size, 128, height, width
#             VAE_residual(128,128),#ek input, ek output
            
#             # batch_size, 128, height, width -> batch_size, 128, height/, width/
#             VAE_residual(128,128),
            
#             # batch_size, 128, height/, width/ -> batch_size, 128, height/2, width/2
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            
#             # batch_size, 128, height/2, width/2 -> batch_size, 256(output badh gaya), height/2, width/2
#             VAE_residual(128,256),
            
#             # batch_size, 256, height/2, width/2 -> batch_size, 256, height/2, width/2
#             VAE_residual(256,256),
            
#             # batch_size, 256, height/2, width/2 -> batch_size, 256, height/2, width/2(image har bar 2 se devide ho ho ke choti hoti jaa rahi hai )
#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
#             # batch_size, 256, height/4, width/4 -> batch_size, 512(output badh gaya), height/4, width/4(image choti hogyi)
#             VAE_residual(256,512),
            
#             VAE_residual(512,512),
            
#             # batch_size, 256, height/4, width/4 -> batch_size, 512(output badh gaya), height/8, width/8(image aur choti hogyi)
#             nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
#             VAE_residual(256,512),
            
#             VAE_residual(512,512),
            
#             VAE_residual(512,512),
            
            
#             VAE_attention(512),
            
#             VAE_residual(512,512),

#             nn.GroupNorm(32, 512),

#             nn.SiLu(),
            
#             nn.Conv2d(512, 8, kernel_size=3, padding = 1),#8 features hai bhai decrease kardia(bottleneck layer)
#             nn.Conv2d(8, 8, kernel_size=1, padding = 0)#each kernel over each pixel
#             )
        
        
#     def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
#         #x = batch_size, channel, height, width 
#         #noise = batch_size, output channel, height/8, width/8
#         for module in self:
#             if getattr(module, 'stride', None) == (2,2):#manually padding add kar rahe hai kyuki upar nahi kia, symetric nahi rehta
#                 #padding = (left, right, top, bottom)
#                 x = F.pad(x, (0,1,0,1))
#             x = module(x)
#         #batch, 8, height/8, width/8 --> two tensors of shape (batch, 4, height/8, width/8)
#         mean, logVar = torch.chunk(x, 2, dim=1)
#         logVar = torch.clamp(logVar, -30, 20)#logVar ki value -30 se 20 ke beech me hogi
        
#         var = logVar.exp()
#         stdev = var.sqrt()# dono hi tensor se ched chad nahi karenge
        
#         # Z= N(0,1) koi distribution hai usse N(given mean, vairance kaise niklegi?)
#         # isko manlo X = mean+stdev * Z
#         x = mean+stdev*noise
        
#         #scale karo output by a constant(# Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1)
#         x *= 0.18215   
             
#         return x
        
import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x

