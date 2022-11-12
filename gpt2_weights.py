from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import pickle
import torch 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
attention_weight_matrix = []
mlp_weight_matrix = []
for name, param in model.named_parameters():
    print(name, param.shape)

for name, param in model.named_parameters():
    print(name, param.shape)
    if name.__contains__("attn") and name.__contains__("weight") and not name.__contains__("proj"):
        attention_weight_matrix.append(param.detach().numpy())
    elif name.__contains__("mlp") and name.__contains__("weight") and not name.__contains__("proj"):
        mlp_weight_matrix.append(param.detach().numpy())

print(len(attention_weight_matrix))
print(len(mlp_weight_matrix))
mlp_weight_matrix_part1 = mlp_weight_matrix[0:6]
mlp_weight_matrix_part2 = mlp_weight_matrix[6:12]
print(len(mlp_weight_matrix_part1))
print(len(mlp_weight_matrix_part2))

with open("attention_weights", 'wb') as f:
    pickle.dump(attention_weight_matrix, f)

with open("mlp_weights_part1", 'wb') as f:
    pickle.dump(mlp_weight_matrix_part1, f)

with open("mlp_weights_part2", 'wb') as f:
    pickle.dump(mlp_weight_matrix_part2, f)

#this is how you load it again
with open("attention_weights", 'rb') as f:
    attention_weight_matrix = pickle.load( f)

with open("mlp_weights_part1", 'rb') as f:
    mlp_weight_matrix_part1 = pickle.load( f)

with open("mlp_weights_part2", 'rb') as f:
    mlp_weight_matrix_part2 = pickle.load( f)