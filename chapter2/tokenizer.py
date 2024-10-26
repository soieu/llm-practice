import torch.nn as nn
import torch

# 띄어쓰기 단위로 분리
input_text = "나는 세상에서 제일 똑똑하고 예쁘다"
input_text_list = input_text.split(" ")
print(f"input_text_list: {input_text_list}")
# input_text_list: ['나는', '세상에서', '제일', '똑똑하고', '예쁘다']

# 토큰 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리
str_to_idx = {word: idx for idx, word in enumerate(input_text_list)}
print(f"strToidx: {str_to_idx}")
# strToidx: {'나는': 0, '세상에서': 1, '제일': 2, '똑똑하고': 3, '예쁘다': 4}

idx_to_str = {idx: word for idx, word in enumerate(input_text_list)}
print(f"idxTostr: {idx_to_str}")
# idxTostr: {0: '나는', 1: '세상에서', 2: '제일', 3: '똑똑하고', 4: '예쁘다'}

# 토큰 -> 토큰 아이디
input_ids = [str_to_idx[word] for word in input_text_list]
print(f"input_ids: {input_ids}")
# input_ids: [0, 1, 2, 3, 4]



# 토큰 아이디 -> 벡터
embedding_dim = 16  # 각 단어를 16차원의 벡터로 변환하도록 임베딩 차원 설정
embed_layer = nn.Embedding(len(str_to_idx), embedding_dim)  # 단어 사전 크기와 임베딩 차원을 지정하여 임베딩 레이어 생성

# input_ids 리스트를 텐서로 변환하여 임베딩 레이어에 입력
input_embeddings = embed_layer(torch.tensor(input_ids))
# 각 단어의 인덱스가 16차원 벡터로 변환됨. 결과는 (문장 길이, 임베딩 차원) 형태인 텐서가 됨

input_embeddings = input_embeddings.unsqueeze(0)  # 첫 번째 축에 배치 차원을 추가하여 (1, 문장 길이, 임베딩 차원) 형태로 변경

print(input_embeddings.shape)  # 결과 텐서의 형태 확인
# 결과 출력: torch.Size([1, 5, 16])


# 절대적 위치 인코딩
embeding_dim = 16
max_postion = 12
embed_layer = nn.Embedding(len(str_to_idx), embedding_dim)
position_embed_layer = nn.Embedding(max_postion, embedding_dim)

position_ids = torch.arrange(len(input_ids), dtype=torch.long).unsqueeze(0)
position_encodings = position_embed_layer(position_ids)
token_embeddings = embed_layer(torch.tensor(input_ids))
token_embeddings = token_embeddings.unsqueeze(0)
input_embeddings = token_embeddings + position_encodings
input_embeddings.shape
