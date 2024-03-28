import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        residue = x
        
        ### SELF ATTENTION ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output

# import torch 
# from torch import nn
# from torch.nn import functional as F
# from attention import self_attention


# class CLIPEmbedding(nn.Module):
#     def __init__(self, n_vocab:int, n_embd:int, n_tokens:int):
#         super().__init__()
#         self.token_embedding = nn.Embedding(n_vocab, n_embd)
#         self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
        
#     def forward(self, tokens):
#         #batch, seq_len -> batch, seq_len, dim
#         x = self.token_embedding(tokens)
        
#         x += self.position_embedding
#         return x
        

# class CLIPlayer(nn.Module):
#     def __init__(self, n_head:int, n_embd:int):
#         super().__init__()
        
#         self.layer_norm = nn.LayerNorm(n_embd)
#         self.attention = self_attention(n_head, n_embd)
#         self.layer_norm_2 = nn.LayerNorm(n_embd)
#         self.Linear1 = nn.Linear(n_embd, 4*n_embd)
#         self.Linear2 = nn.Linear(4*n_embd, n_embd)
        
#     def forward(self, x:torch.Tensor)->torch.Tensor:
#         #batch, seq_len, dim
#         residue = x
        
#         x = self.layer_norm(x)
        
#         x = self.attention(x, causal_mask = True)
#         x += residue#connection
        
#         #feedforward
#         residue = x
        
#         x = self.layer_norm_2(x)
#         x = self.Linear1(x)
#         x = x* torch.sigmoid(1.702*x) #quick gelu function (man kia to kardia)
#         x = self.Linear2(x)
#         x += residue
        
#         return x
        




# class CLIP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embeddings = CLIPEmbedding(49408, 768, 77) #vocab size, embedding size, seq_len(abhi ke lie)
        
        
#         self.layers = nn.ModuleList([
#             CLIPlayer(12, 768) for i in range (12)#no. of heads of attention, embedding size
#         ])
        
        
#         self.layernorm = nn.LayerNorm(768)
    
#     def forward(self, token: torch.LongTensor) -> torch.FloatTensor:
#         tokens = tokens.type(torch.long)
        
#         #batch, seq_len -> batch, seq_len, dim
#         state = self.embeddings(tokens)
        
#         for layer in self.layers:
#             state = layer(state)    
        
#         #batch, seq_len, dim -> batch, seq_len, dim
#         output = self.layernorm(state)
#         return output
        
    
    