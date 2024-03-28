import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width)

        residue = x 

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        x = x.transpose(-1, -2)
        
        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width) 
        x += residue

        # (Batch_Size, Features, Height, Width)
        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)

        residue = x

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = F.silu(x)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = F.silu(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.GroupNorm(32, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.SiLU(), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x

# import torch
# from torch import nn
# from torch.nn import functional as F
# from attention import self_attention

# class VAE_attention(nn.Module):
#     def __init__(self, channels:int):
#         super().__init__()
#         self.groupNorm = nn.GroupNorm(32, channels)
#         self.attention = self_attention(1, channels)

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         #x: batch_size, channels, height, width
#         residue = x
        
#         x = self.groupNorm(x)
#         n, c, h, w = x.shape 
#         x = x.view(n, c, h*w) #batch_size, featrures, height, width -> batch_size, featrures, height*width
#         x = x.transpose(-1,-2) #batch_size, featrures, height*width -> batch_size, height*width, featrures
        
#         x = self.attention(x)
        
#         x = x.transpose(-1,-2) #batch_size, height*width, featrures -> batch_size, featrures, height*width
        
#         x = x.view(n, c, h, w) #batch_size, featrures, height*width -> batch_size, featrures, height, width
        
#         x+= residue
        
#         return x
        

# class VAE_residual(nn.Module):
    
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.GroupNorm1 = nn.GroupNorm(32, in_channels)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
#         self.GroupNorm2 = nn.GroupNorm(32, out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
#         if in_channels == out_channels:
#             self.residual_layer = nn.Identity()
#         else:
#             self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding = 0)
            
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         #x: batch_size, in_channels, height, width
        
#         residue = x
#         x = self.GroupNorm1(x)
#         x = F.silu(x)
#         x = self.conv1(x)
        
#         x = self.GroupNorm2(x)
#         x = F.silu(x)
#         x = self.conv2(x)
        
#         return x + self.residual_layer(residue)#dimensions match karne ko
    
    
# class VAE_decoder(nn.Sequential):
    
#     def __init__(self):
#         super().__init__(
#             nn.Conv2d(4, 4, kernel_size=1, padding = 0),
#             nn.Conv2d(4, 512, kernel_size=3, padding = 1),
            
#             VAE_residual(512,512),
            
#             VAE_attention(512),
            
#             VAE_residual(512,512),
#             VAE_residual(512,512),
#             VAE_residual(512,512),                
#             VAE_residual(512,512), #batch_size, 512, height/8, width/8
            
#             nn.Upsample(scale_factor=2), #batch_size, 512, height/4, width/4(literally upscaled this mf)
            
#             nn.Conv2d(512, 512, kernel_size=3, padding = 1),
#             VAE_residual(512,512),
#             VAE_residual(512,512),
#             VAE_residual(512,512),

#             nn.Upsample(scale_factor=2), #batch_size, 512, height/2, width/2(literally upscaled this mf)(again)
            
#             nn.Conv2d(512, 512, kernel_size=3, padding = 1),
            
#             VAE_residual(512,256),
#             VAE_residual(256,256),
#             VAE_residual(256,256),
            
#             nn.Upsample(scale_factor=2), #batch_size, 256, height, width(original size)
            
#             nn.Conv2d(256, 256, kernel_size=3, padding = 1),
            
#             VAE_residual(256,128),
#             VAE_residual(128,128),
#             VAE_residual(128,128),
            
#             nn.GroupNorm(32, 128),
#             nn.SiLU(),
            
#             nn.Conv2d(128, 3, kernel_size=3, padding = 1)# batch_size, 128, height, width -> batch_size, 3, height, width                
#         )
        
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         #x: batch_size, 4, height/8, width/8(ye input aya tha bhai)
#         x /= 0.18215 #nullify scaling factor
        
#         for module in self:
#             x = module(x)
            
#         return x #batch_size, 3, height, width
