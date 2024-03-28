import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
        # (Batch, 4, Height / 8, Width / 8)
        return output


# import torch
# from torch import nn
# from torch.nn import functional as F
# from attention import self_attention, cross_attention


# class TimeEmbedding(nn.Module):
#     def __init__(self, n_embd:int):
#         super().__init__()
#         self.linear = nn.Linear(320,4* n_embd)
#         self.linear2 = nn.Linear(4*n_embd, 4*n_embd)
        
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         #x: batch_size, 320
#         x = self.linear(x)
#         x = F.SiLU(x)
#         x = self.linear2(x)
        
#         return x




# class UNET_residual(nn.Module):
#     def __init__(self, in_channels:int, out_channels:int, n_time = 1280):
#         super().__init__()
#         self.GroupNorm_features = nn.GroupNorm(32, in_channels)
#         self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.Linear = nn.Linear(n_time, out_channels)
        
#         self.GroupNorm_merged = nn.GroupNorm(32, out_channels)
#         self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
#         if in_channels == out_channels:
#             self.residual_layer = nn.Identity()
#         else:
#             self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding = 0)

#     def forward(self, feature, time):
#         #feature: batch, in_channels, height, width
#         #time: batch, 1280
        
#         residue = feature
#         feature = self.GroupNorm_features(feature)
        
#         feature = F.silu(feature)
#         feature = self.conv_features(feature)
#         time = self.linear_time(time)
        
#         merged = feature + time.unsqueeze(-1).unsqueeze(-1) 
#         merged = self.GroupNorm_merged(merged)
#         merged = F.silu(merged)
#         merged = self.conv_merged(merged)
        
#         return merged + self.residual_layer(residue)
    
    
    
    
    
# class UNET_attention(nn.Module):
#     def __init__(self, n_head: int, n_embd: int, d_context: 768):
#         super().__init__()
#         channels = n_head * n_embd
        
#         self.groupnorms = nn.GroupNorm(32, channels, eps = 1e-6)
#         self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
#         self.layer_norm1 = nn.LayerNorm(channels)
#         self.attention1 = self_attention(n_head, n_embd, in_proj_bias=False)
#         self.layer_norm2 = nn.LayerNorm(channels)
#         self.attention2 = cross_attention(n_head, n_embd, d_context, in_proj_bias=False)    
#         self.layer_norm3 = nn.LayerNorm(channels)
#         self.Linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
#         self.Linear_geglu_2 = nn.Linear(4 * channels, channels)
        
#         self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
#     def forward(self, x, context):
#         #x: batch, features, height, width
#         #context: batch, seq_len, dim
#         residue_long = x
        
#         x = self.groupnorms(x)
#         x = self.conv_input(x)
        
#         n,c,h,w = x.shape
#         x = x.view((n,n,h*w)) # batch, features, height, width -> batch, features, height*width
#         x = x.transpose(-1,-2) # batch, features, height*width -> batch, height*width, features
        
#         #normalisation+self attention with skip connection
#         residue_short = x
        
#         x = self.layer_norm1(x)
#         x = self.attention1(x)
#         x += residue_short
        
#         residue_short = x
        
#         #normalization + cross attention with skip connection
#         x = self.layer_norm2(x)
        
#         #cross attention
#         self.attention2(x, context)
#         x += residue_short
#         residue_short = x
        
#         #normalization + feedforward with geglu and skip connection(geglu combination hai gelu and linear ka)
#         x = self.layer_norm3(x)
        
#         x, gate = self.Linear_geglu_1(x).chunk(2, dim = -1)
#         x = x * F.gelu(gate)
        
#         x = self.Linear_geglu_2(x)
#         x += residue_short
        
#         x = x.transpose(-1,-2) # batch, height*width, features -> batch, features, height*width
#         x = x.view((n,c,h,w)) # batch, features, height*width -> batch, features, height, width
        
#         # if x.shape != residue_long.shape:
#         #     x = F.interpolate(x, size = residue_long.shape[2:], mode = 'bilinear')
        
#         return self.conv_output(x) + residue_long
        

        
        








# class SwitchSequential(nn.Sequential):
#     def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
#         for layer in self:
#             if isinstance(layer, UNET_attention):
#                 x = layer(x, context)
#             elif isinstance(layer, UNET_residual):
#                 x = layer(x, time)
#         return x



# class Upsample(nn.Module):
#     def __init__(self, channels:int):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)   
    
#     def forward(self, x):
#         #batchSize, features, height, width -> batchSize, features, height*2, width*2
#         x = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.conv(x)
#         return x




# class UNET(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #based on a repository, hai to hai hume ni pata kyu hai
#         self.encoder = nn.Module([
#             #pehle 512 ki thi
#             #batch, 4, height/8, width/8(512/8 hua = 64)
#             SwitchSequential(nn.Conv2(4,320, Kernel_size=3, padding=1)),
            
#             SwitchSequential(UNET_residual(320,320), UNET_attention(8, 40)),
#             SwitchSequential(UNET_residual(320,320), UNET_attention(8, 40)),#ye bani 64x64
            
#             #batch, 320, height/16 , width/16(aur chota hogya hai)
#             SwitchSequential(nn.Conv2(320,320, Kernel_size=3, stride=2, padding=1)),
            
#             SwitchSequential(UNET_residual(320,640), UNET_attention(8, 80)),
#             SwitchSequential(UNET_residual(640,640), UNET_attention(8, 80)),#ye bani 32x32
            
#             #batch, 640, height/16 , width/16 -> batch, 640, height/32 , width/32
#             SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
#             SwitchSequential(UNET_residual(640,1280), UNET_attention(8, 160)),#8=head size, 160=embedding size
#             SwitchSequential(UNET_residual(1280,1280), UNET_attention(8, 80)),#ye bani 16x16
            
#             #batch, 1280, height/32 , width/32 -> batch, 1280, height/64 , width/64
#             SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
#             SwitchSequential(UNET_residual(1280,1280)),
#             SwitchSequential(UNET_residual(1280,1280)),# batch, 1280, height/64, width/64  -> batch, 1280, height/64, width/64
#             ])        
        
#         self.bottleneck = SwitchSequential([
#             UNET_residual(1280,1280),
#             UNET_attention(8, 160),
#             UNET_residual(1280,1280),
#         ])
        
#         self.decoder = nn.ModuleList([
#             SwitchSequential(UNET_residual(2560,1280)), #skip connection ko dhyan me rakhte hue, bottleneck ke bhar 1280 hi  niklega par skip connection se 2560 ho jaega
#             SwitchSequential(UNET_residual(2560,1280)),
            
#             SwitchSequential(UNET_residual(2560,1280), Upsample(1280)),
            
#             SwitchSequential(UNET_residual(2560,1280), UNET_attention(8, 160)),
            
#             SwitchSequential(UNET_residual(2560,1280), Upsample(1280)),
            
#             SwitchSequential(UNET_residual(2560,1280), UNET_attention(8, 160)),
#             SwitchSequential(UNET_residual(2560,1280), UNET_attention(8, 160)),
            
#             SwitchSequential(UNET_residual(1920,1280), UNET_attention(8, 160), Upsample(1280)), 
            
#             SwitchSequential(UNET_residual(1920,640), UNET_attention(8, 80)),
#             SwitchSequential(UNET_residual(1280,640), UNET_attention(8, 80)),
            
#             SwitchSequential(UNET_residual(960,640), UNET_attention(8, 160), Upsample(640)),
            
#             SwitchSequential(UNET_residual(960,320), UNET_attention(8, 40)),
#             SwitchSequential(UNET_residual(640,320), UNET_attention(8, 80)),
#             SwitchSequential(UNET_residual(640,320), UNET_attention(8, 40)),

#              ])


# class UNET_Output(nn.Module):
#     def __init__(self, in_channels:int, out_channels:int):
#         super().__init__()
#         self.groupnorm = nn.GroupNorm(32, in_channels)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
#     def forward(self, x):
#         #x: batch, 320, height/8, width/8
#         x = self.groupnorm(x)
#         x = F.silu(x)
#         x = self.conv(x)
#         return x




# ##################tamij se padhna hai firse ##################################
# class Diffusion(nn.Module): 
#     def __init__(self):
#         super().__init__()
#         self.time_embedding = TimeEmbedding(320)#(size of time embedding)becasue we need to give time at which it was noisified
#         self.unet = UNET()
#         self.final = UNET_Output(320,4)
        
#     def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
#         #batch size, 4(output of the encoder), height/8, width/8
#         #context = batch size,seq_len, dim
#         #time = 1,320
        
#         time = self.time_embedding(time)#(1,320) -> (1,1280)
        
#         #batch size, 4, height/8, width/8 -> batch size, 320, height/8, width/8
#         output = self.unet(latent, context, time)
        
#         #batch size, 320, height/8, width/8 -> batch size, 4, height/8, width/8
#         output = self.final(output)        
#         return output
    
