import torch.nn as nn

embedding_dim = 16 # 16차원의 벡터로 변환
embed_layer = nn.Embedding(len(strToidx), embedding_dim) # 16차원의 임베딩을 생성하는 임베딩 층 생성

input_embeddings = embed_layer(torch.tensor(input_ids) ) # 
input_embeddings = input_embeddings.unsqueeze(0)