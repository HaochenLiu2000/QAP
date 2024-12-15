import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
import random
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
import random
dataset_name='medqa'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
llm="flant5-3b"


if llm=="flant5-3b":
    token_dim=2048
elif llm=="flant5-11b":
    token_dim=2048*2
elif llm=="llama2-7bchat":
    token_dim=2048*2
elif llm=="llama2-13bchat":
    token_dim=5120

if dataset_name=='riddle':
    option_num=5
elif dataset_name=='obqa' or dataset_name=='medqa':
    option_num=4

if dataset_name=='obqa' or dataset_name=='riddle':
    entity_embeddings = np.load('../data_preprocessed/cpnet/tzw.ent.npy')
    entity_embeddings = torch.FloatTensor(entity_embeddings).to(device)
    entities=[]
    with open('../data_preprocessed/cpnet/concept.txt', 'r', encoding='utf-8') as file:
        for line in file:
            entities.append(line.strip())
    relations = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
    ]
    relation_num=17
elif dataset_name=='medqa':
    entity_embeddings = np.load('../data_preprocessed_biomed/ddb/ent_emb.npy')
    entity_embeddings = torch.FloatTensor(entity_embeddings).to(device)
    entities=[]
    with open('../data_preprocessed_biomed/ddb/vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            entities.append(line.strip())
    relations = [
    'belongs_to_the_category_of',
    'is_a_category',
    'may_cause',
    'is_a_subtype_of',
    'is_a_risk_factor_of',
    'is_associated_with',
    'may_contraindicate',
    'interacts_with',
    'belongs_to_the_drug_family_of',
    'belongs_to_drug_super-family',
    'is_a_vector_for',
    'may_be_allelic_with',
    'see_also',
    'is_an_ingradient_of',
    'may_treat'
    ]
    relation_num=15






def load_and_preprocess_data_with_options(file_path, istest=False):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        i=0
        for line in tqdm(f):
            item = json.loads(line)
            
            question = item['question']['stem']
            choices = item['question']['choices']
            options_text = ' '.join([f"{choice['label']}. {choice['text']}\n" for choice in choices])
            #input_text = f"Choose the most suitable option as a continuation of the following statement: {question} () Options: {options_text} Your answer: "
            input_text = question
            options = {choice['label']: choice['text'] for choice in choices}
            for choice in choices:
                if dataset_name=='csqa' and istest:
                    if choice['label']=='A':
                        answer = f"{choice['label']}: {choice['text']}"
                else:
                    if choice['label']==item['answerKey']:
                        answer = f"{choice['label']}: {choice['text']}"
            
            data.append({'id': i, 'input_text': input_text, 'answer': answer, 'options': options, 'option_text': options_text})
            i+=1
    return data



def load_graph_data(file_path):
    with open(file_path, 'rb') as f:
        if dataset_name=='riddle' or dataset_name=='obqa':
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel,_ = pickle.load(f)
        elif dataset_name=='medqa':
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel = pickle.load(f)
    
    edge_index_converted = list(map(list, zip(*(iter(edge_index),) * option_num)))
    edge_type_converted = list(map(list, zip(*(iter(edge_type),) * option_num)))
    
    graphs_data = []
    for i in tqdm(range(len(concept_ids))):
        concept_ids_m = concept_ids[i].to(device)
        node_type_ids_m = node_type_ids[i].to(device)
        edge_index_m = edge_index_converted[i//option_num][i%option_num].to(device)
        edge_type_m = edge_type_converted[i//option_num][i%option_num].to(device)
        
        
        concept_ids_m=concept_ids_m[1:]-1
        nonzero_indices = torch.nonzero(concept_ids_m, as_tuple=True)[0]
        if nonzero_indices.numel() == 0:
            concept_ids_m = torch.tensor([], dtype=concept_ids_m.dtype).to(device)
        else:
            last_nonzero_index = nonzero_indices[-1]
            concept_ids_m = concept_ids_m[:last_nonzero_index + 1]
        
        node_type_ids_m=node_type_ids_m[1:]
        
        mask = ~torch.any(edge_index_m == 0, dim=0)
        edge_index_m = edge_index_m[:, mask]-1
        edge_type_m = edge_type_m[mask]
        edge_type_m = torch.where(edge_type_m < relation_num+2, edge_type_m - 2, edge_type_m - 4)
        
        
        graph_info = {
            'concept_ids': concept_ids_m,
            #'node_type_ids': node_type_ids_m,
            'edge_index': edge_index_m,
            'edge_type': edge_type_m
        }
        graphs_data.append(graph_info)
    return graphs_data


if dataset_name=='obqa':
    train_data_path = '../data_preprocessed/obqa/statement/train.statement.jsonl'
    train_data = load_and_preprocess_data_with_options(train_data_path,istest=False)
    train_graph_path = '../data_preprocessed/obqa/graph/train.graph.adj.pk-nodenum200.loaded_cache'
    train_graph = load_graph_data(train_graph_path)
elif dataset_name=='riddle':
    train_data_path = '../data_preprocessed/riddle/statement/train.statement.jsonl'
    train_data = load_and_preprocess_data_with_options(train_data_path,istest=False)
    train_graph_path = '../data_preprocessed/riddle/graph/train.graph.adj.pk-nodenum200.loaded_cache'
    train_graph = load_graph_data(train_graph_path)
elif dataset_name=='medqa':
    train_data_path = '../data_preprocessed_biomed/medqa_usmle/statement/train.statement.jsonl'
    train_data = load_and_preprocess_data_with_options(train_data_path,istest=False)
    train_graph_path = '../data_preprocessed_biomed/medqa_usmle/graph/train.graph.adj.pk.loaded_cache'
    train_graph = load_graph_data(train_graph_path)


if dataset_name=='obqa':
    dev_data_path = '../data_preprocessed/obqa/statement/dev.statement.jsonl'
    dev_data = load_and_preprocess_data_with_options(dev_data_path)
    dev_graph_path = '../data_preprocessed/obqa/graph/dev.graph.adj.pk-nodenum200.loaded_cache'
    dev_graph = load_graph_data(dev_graph_path)
elif dataset_name=='riddle':
    dev_data_path = '../data_preprocessed/riddle/statement/dev.statement.jsonl'
    dev_data = load_and_preprocess_data_with_options(dev_data_path)
    dev_graph_path = '../data_preprocessed/riddle/graph/dev.graph.adj.pk-nodenum200.loaded_cache'
    dev_graph = load_graph_data(dev_graph_path)
elif dataset_name=='medqa':
    dev_data_path = '../data_preprocessed_biomed/medqa_usmle/statement/dev.statement.jsonl'
    dev_data = load_and_preprocess_data_with_options(dev_data_path)
    dev_graph_path = '../data_preprocessed_biomed/medqa_usmle/graph/dev.graph.adj.pk.loaded_cache'
    dev_graph = load_graph_data(dev_graph_path)



if dataset_name=='obqa':
    test_data_path = '../data_preprocessed/obqa/statement/test.statement.jsonl'
    test_data = load_and_preprocess_data_with_options(test_data_path,istest=False)
    test_graph_path = '../data_preprocessed/obqa/graph/test.graph.adj.pk-nodenum200.loaded_cache'
    test_graph = load_graph_data(test_graph_path)
elif dataset_name=='riddle':
    test_data_path = '../data_preprocessed/riddle/statement/test.statement.jsonl'
    test_data = load_and_preprocess_data_with_options(test_data_path,istest=False)
    test_graph_path = '../data_preprocessed/riddle/graph/test.graph.adj.pk-nodenum200.loaded_cache'
    test_graph = load_graph_data(test_graph_path)
elif dataset_name=='medqa':
    test_data_path = '../data_preprocessed_biomed/medqa_usmle/statement/test.statement.jsonl'
    test_data = load_and_preprocess_data_with_options(test_data_path,istest=False)
    test_graph_path = '../data_preprocessed_biomed/medqa_usmle/graph/test.graph.adj.pk.loaded_cache'
    test_graph = load_graph_data(test_graph_path)

class QuestionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        question = self.data[idx]
        return {
            "id": question["id"],
            "input_text": question["input_text"],
            "answer": question["answer"],
            "options": question["options"],
            "option_text": question["option_text"]
        }

train_dataset = QuestionDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

dev_dataset = QuestionDataset(dev_data)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True)


test_dataset = QuestionDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


def set_graph(q, graph):
    
    i=q["id"]
    i_g = [x for ii in i for x in range(option_num*ii, option_num*ii+option_num)]
    subgraphs=[graph[ii] for ii in i_g]
    concept_ids=[]
    edge_index=[]
    edge_type=[]
    graph_data_list=[]
    for j in range(len(i_g)):
        concept_ids=subgraphs[j]["concept_ids"]
        #node_type_ids=subgraphs[j]["node_type_ids"]
        
        edge_index=subgraphs[j]["edge_index"]
        edge_type=subgraphs[j]["edge_type"]
        
        
        graph_data_list.append(Data(x=concept_ids,edge_index=edge_index,edge_attr=edge_type,concept_ids=concept_ids))
        
    subgraphs=Batch.from_data_list(graph_data_list).to(device)
    
    batch=subgraphs.batch
    #batch_num_nodes = scatter_sum(torch.ones(batch.shape).to(device), batch)       
    #head_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(device),batch_num_nodes[:-1]]), 0).long()
    #tail_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(device),batch_num_nodes[:-1]]), 0).long() + 1
    
    return {'subgraphs':subgraphs, 'batch':batch}




class CustomGraphTransformerLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(CustomGraphTransformerLayer, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_k = out_channels // heads  

        
        self.lin_query = Linear(in_channels, out_channels).to(device)
        self.lin_key = Linear(in_channels, out_channels).to(device)
        self.lin_value = Linear(in_channels, out_channels).to(device)
        self.lin_sentence = Linear(token_dim, out_channels).to(device)

        
        self.lin_out = Linear(out_channels, out_channels).to(device)

        self.sqdk=torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)).to(device)
        
        
        self.residual_lin = Linear(in_channels, out_channels).to(device) if in_channels != out_channels else None


    def map_batch_to_group(self, batch):
        
        return batch // option_num

    def forward(self, x, edge_index, sentence_embeddings, batch):
        
        H, C = self.heads, self.d_k
        
        group_indices = self.map_batch_to_group(batch)
        
        
        Q = self.lin_query(x).view(-1, H, C)
        K = self.lin_key(x).view(-1, H, C)
        V = self.lin_value(x).view(-1, H, C)
        sentence = sentence_embeddings[group_indices] 
        Q_sentence = self.lin_sentence(sentence).view(-1, H, C)
        K_sentence = self.lin_sentence(sentence).view(-1, H, C)
        row, col = edge_index
        Q_i = Q[row]#[E,H,C]
        K_j = K[col]
        V_j = V[col]
        Q_sentence_i = Q_sentence[row]
        K_sentence_j = K_sentence[col]
        
        alpha_qk = (Q_i * K_j).sum(dim=-1) / self.sqdk#[E,H]
        alpha_qs_k = (Q_sentence_i * K_j).sum(dim=-1) / self.sqdk
        alpha_q_ks = (Q_i * K_sentence_j).sum(dim=-1) / self.sqdk


        alpha = (alpha_qk + alpha_qs_k + alpha_q_ks) / 3.0
        
        alpha = F.leaky_relu(alpha, negative_slope=0.2)#【E，H】
        
        alpha = F.softmax(alpha, dim=0)
        
        
        out = torch.zeros_like(V)#[N,H,C]
        out.index_add_(0, row, V_j * alpha.view(-1, H, 1))
        
        out = self.lin_out(out.view(-1, H * C))

        if self.residual_lin is not None:
            x = self.residual_lin(x)
        out = out + x 
        
        return out



class GraphTransformer(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, heads=1):
        super(GraphTransformer, self).__init__()
        self.layers = torch.nn.ModuleList([
            CustomGraphTransformerLayer(in_channels if i == 0 else out_channels, out_channels, heads)
            for i in range(num_layers)
        ])

        self.ffn_layer1=Linear(out_channels, out_channels).to(device)
        self.ffn_layer2=Linear(token_dim, out_channels).to(device)
        self.ffn_layer3=Linear(out_channels, token_dim).to(device)
        self.output_layer = Linear(option_num * out_channels, out_channels).to(device)
        
    def attention1(self, x, group_indices, text_embedding):
        
        Q=x
        K=V=text_embedding
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        expanded_indices = group_indices.view(1, -1, 1).expand(1, -1, scores.shape[2])
        selected = scores.gather(0, expanded_indices)
        scores=selected.squeeze(0)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        expanded_indices = group_indices.view(1, -1, 1).expand(1, -1, attention_output.shape[2])
        selected = attention_output.gather(0, expanded_indices)
        attention_output=selected.squeeze(0)
        
        return attention_output
    
    def attention_multihead(self, x, group_indices, text_embedding):
        B, T, _ = text_embedding.shape
        N, _ = x.shape

        
        Q = x.unsqueeze(0).expand(B, -1, -1)  
        K = V = text_embedding 
        
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  
        
        
        expanded_indices = group_indices.view(1, -1, 1).expand(B, -1, scores.shape[2])  
        selected_scores = scores.gather(1, expanded_indices) 
        
        attention_weights = F.softmax(selected_scores, dim=-1)  
        
        attention_output = torch.matmul(attention_weights, V) 
        
        expanded_indices = group_indices.view(1, -1, 1).expand(B, N, attention_output.shape[-1])  
        selected_output = attention_output.gather(1, expanded_indices) 

        concat_output = selected_output.permute(1, 0, 2).contiguous().view(N, -1) 
        
        final_output = self.output_layer(concat_output) 

        return final_output

    
    def map_batch_to_group(self, batch):
        
        return batch // option_num
    
    def forward(self, concept_ids, edge_index, edge_type, sentence_embeddings, token_embedding, batch):
        
        group_indices = self.map_batch_to_group(batch)
        
        x=entity_embeddings[concept_ids]
            
        
        for layer in self.layers:
            
            x = layer(x, edge_index, sentence_embeddings, batch)
        
        x=self.ffn_layer1(x)
        token_embedding=self.ffn_layer2(token_embedding)
        
        x = self.attention_multihead(x, group_indices, token_embedding)
        
        x = global_max_pool(x, batch)
        
        soft_prompt = self.ffn_layer3(x)
        
        soft_prompt=soft_prompt.view(int(int(batch[-1]+1)/option_num), option_num, soft_prompt.size(-1))
        
        return soft_prompt


in_features = entity_embeddings.size(1)#1024
out_features = 2048*6
num_layers=3
heads=4
decoder = GraphTransformer(num_layers=num_layers, in_channels=in_features, out_channels=out_features, heads=heads)


if llm=="flant5-3b":
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    train_note="flan-t5-3b.txt"
    model_path='model/decoder_flan-t5-3b.pth'
elif llm=="flant5-11b":
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
    train_note="flan-t5-11b.txt"
    model_path='model/decoder_flan-t5-11b.pth'
elif llm=="llama2-7bchat":
    tokenizer = LlamaTokenizer.from_pretrained("./../../llama2-7bchat")
    model = LlamaForCausalLM.from_pretrained("./../../llama2-7bchat")
    train_note="llama2-7bchat.txt"
    model_path='model/decoder_llama2-7b.pth'
elif llm=="llama2-13bchat":
    tokenizer = LlamaTokenizer.from_pretrained("./../../llama2-13bchat")
    model = LlamaForCausalLM.from_pretrained("./../../llama2-13bchat")
    train_note="llama2-13bchat.txt"
    model_path='model/decoder_llama2-13b.pth'
model.to(device2)

for param in model.parameters():
    param.requires_grad = False



def generate_soft_prompt(concept_ids, edge_index, edge_type, sentence_embeddings, token_embedding, batch):
    
    soft_prompt = decoder(concept_ids, edge_index, edge_type, sentence_embeddings, token_embedding, batch)
    
    
    
    
    
    return soft_prompt

def combine1(input_text, options):

    
    attention_text=[]
    for value in options.values():
        attention_text.append(input_text[0]+value[0])
        
    input = tokenizer(attention_text, return_tensors="pt", padding=True, truncation=True)
    input_mask = input.attention_mask.to(device2)
    input_ids = input.input_ids.to(device2)
    
    
    token_embedding = model.get_input_embeddings()(input_ids).to(device)
    
    
    return token_embedding

def combine2(concept_ids, edge_index, edge_type, soft_prompt, input_text, option_text, batch, answer):

    soft_prompt=soft_prompt.to(device2)
    batch_edge=batch[edge_index[0]]
    group_edge= batch_edge // option_num
    
    qh_s=[]
    for i in range(len(input_text)):
        qh_s.append(f'''Question: {input_text[i]}
            {option_text[i]} Answer:''')
    
    qh_input = tokenizer(qh_s, return_tensors="pt", padding=True, truncation=True)
    qh_input_mask = qh_input.attention_mask.to(device2)
    qh_input_ids = qh_input.input_ids.to(device2)
    qh_input_embedding = model.get_input_embeddings()(qh_input_ids)
    
    
    
    prefix = torch.ones((len(input_text), option_num), dtype=torch.bool).to(device2)
    input_mask = torch.cat((prefix, qh_input_mask), dim=1)
    
    input_emb=torch.cat([soft_prompt, qh_input_embedding],dim=1)
    
    
    
    
    return input_emb, input_mask

def answer_question(input_text, options, option_text, graph, answer):
    tokenizer.pad_token_id=0
    input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_mask = input.attention_mask.to(device2)
    input_ids = input.input_ids.to(device2)
    decoder_start_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]]*len(input_text)).to(device2)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids,output_hidden_states=True)

    if llm=="llama2-7bchat" or llm=="llama2-13bchat":
        last_hidden_states = outputs.hidden_states[-1]
    elif llm=="flant5-3b" or llm=="flant5-11b":
        last_hidden_states = outputs.encoder_last_hidden_state
    sentence_embeddings = (last_hidden_states * input_mask.unsqueeze(-1)).sum(dim=1) / input_mask.sum(dim=1).unsqueeze(-1)
    sentence_embeddings=sentence_embeddings.to(device)
    
    
    
    
    token_embedding = combine1(input_text, options)
    
    concept_ids=graph['subgraphs'].concept_ids
    
    edge_index=graph['subgraphs'].edge_index
    edge_type=graph['subgraphs'].edge_attr
    batch=graph['batch']
    
            
    
    soft_prompt = generate_soft_prompt(concept_ids, edge_index, edge_type, sentence_embeddings, token_embedding, batch)
    
    
    
    input_emb, input_mask = combine2(concept_ids, edge_index, edge_type, soft_prompt, input_text, option_text, batch, answer)
    
    
    
    if llm=="llama2-7bchat" or llm=="llama2-13bchat":
        label_text = answer
        labels = tokenizer(label_text, return_tensors="pt", padding=True, truncation=True)
        labels_ids = labels.input_ids.to(device2)
        labels_mask=labels.attention_mask.to(device2)
        input_emb = torch.cat([input_emb, model.get_input_embeddings()(labels_ids)], dim=1)
        labels_ids = torch.cat([torch.full_like(input_mask, -100), labels_ids], dim=1)
        input_mask = torch.cat([input_mask, labels_mask], dim=1)
        outputs = model(inputs_embeds=input_emb, attention_mask=input_mask, labels=labels_ids)
        loss = outputs.loss/len(input_text)
    elif llm=="flant5-3b" or llm=="flant5-11b":
        label_text = answer
        labels = tokenizer(label_text, return_tensors="pt", padding=True, truncation=True)
        labels_ids = labels.input_ids.to(device2)
        outputs = model(inputs_embeds=input_emb, attention_mask=input_mask, labels=labels_ids)
        loss = outputs.loss/len(input_text)
    
    return loss


def answer_question2(input_text, options, option_text, graph, answer):
    
    tokenizer.pad_token_id=0
    input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_mask = input.attention_mask.to(device2)
    input_ids = input.input_ids.to(device2)
    decoder_start_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]]*len(input_text)).to(device2)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids,output_hidden_states=True)

    if llm=="llama2-7bchat" or llm=="llama2-13bchat":
        last_hidden_states = outputs.hidden_states[-1]
    elif llm=="flant5-3b" or llm=="flant5-11b":
        last_hidden_states = outputs.encoder_last_hidden_state

    sentence_embeddings = (last_hidden_states * input_mask.unsqueeze(-1)).sum(dim=1) / input_mask.sum(dim=1).unsqueeze(-1)
    sentence_embeddings=sentence_embeddings.to(device)
    
    
    token_embedding = combine1(input_text, options)
    
    concept_ids=graph['subgraphs'].concept_ids
    edge_index=graph['subgraphs'].edge_index
    edge_type=graph['subgraphs'].edge_attr
    batch=graph['batch']
    
            
    
    soft_prompt = generate_soft_prompt(concept_ids, edge_index, edge_type, sentence_embeddings, token_embedding, batch)
    
    
    input_emb, input_mask = combine2(concept_ids, edge_index, edge_type, soft_prompt, input_text, option_text, batch, answer)
    
    
    outputs = model(inputs_embeds=input_emb, attention_mask=input_mask,
                        decoder_input_ids=decoder_input_ids, 
                        past_key_values=None, 
                        use_cache=True)
    if option_num==5:
        if llm=="flant5-3b" or llm=="flant5-11b":
            logits = outputs.logits[:, :, [71, 272, 205, 309, 262]]
        elif llm=="llama2-7bchat" or llm=="llama2-13bchat":
            logits = outputs.logits[:, [-1], [319,350,315,360,382]]
        logits = logits.view(logits.size(-1))
        max_values, max_indices = torch.max(logits,dim=0)
    elif option_num==4:
        if llm=="flant5-3b" or llm=="flant5-11b":
            logits = outputs.logits[:, :, [71, 272, 205, 309]]
        elif llm=="llama2-7bchat" or llm=="llama2-13bchat":
            logits = outputs.logits[:, [-1], [319,350,315,360]]
        logits = logits.view(logits.size(-1))
        max_values, max_indices = torch.max(logits,dim=0)
    
    t=max_indices.item()
    if t==0:
        generated_text="A"
    elif t==1:
        generated_text="B"
    elif t==2:
        generated_text="C"
    elif t==3:
        generated_text="D"
    elif t==4:
        generated_text="E"
    
    return generated_text




num_epochs=100


optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-6)
optimizer.zero_grad()

train_note="train_note.txt"
acc_test=0
for epoch in range(num_epochs):
    
    total_train_loss = 0
    total_dev_loss = 0
    total_test_loss = 0
    idx=0
    for q in tqdm(train_dataloader):
        graph=set_graph(q,train_graph)
        id=q['id']
        input_text = q['input_text']
        options = q['options']
        answer = q['answer']
        option_text = q['option_text']
        batch=graph['batch']
        if int(int(batch[-1]+1)/option_num)==0:
            continue
        loss = answer_question(input_text, options, option_text, graph, answer)
        total_train_loss += loss.item()
    
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        #torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        idx+=1
    
    idx2=0   
    correct2=0 
    for q in tqdm(dev_dataloader):
        graph=set_graph(q,dev_graph)
        id=q['id']
        input_text = q['input_text']
        options = q['options']
        answer = q['answer']
        option_text = q['option_text']
        batch=graph['batch']
        if int(int(batch[-1]+1)/option_num)==0:
            continue
        with torch.no_grad():
            generated_text = answer_question2(input_text, options, option_text, graph, answer)
        if len(generated_text)>0 and answer[0][0].lower() == generated_text[0].lower():
            correct2+=1
        idx2+=1
    
    
    idx3=0   
    correct3=0 
    for q in tqdm(test_dataloader):
        graph=set_graph(q,test_graph)
        id=q['id']
        input_text = q['input_text']
        options = q['options']
        answer = q['answer']
        option_text = q['option_text']
        batch=graph['batch']
        if int(int(batch[-1]+1)/option_num)==0:
            continue
        with torch.no_grad():
            generated_text = answer_question2(input_text, options, option_text, graph, answer)
        if len(generated_text)>0 and answer[0][0].lower() == generated_text[0].lower():
            correct3+=1
        idx3+=1
        
        
        
        
    train_loss=total_train_loss/len(train_dataloader)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss_train: {train_loss}, acc_dev: {correct2}/1272, acc_test: {correct3}/1273")
    if correct2>=acc_test:
        torch.save(decoder.state_dict(), 'model/model.pth')
        acc_test=correct2
    with open(train_note, 'a') as f:
        f.write(f"Epoch {epoch + 1}/{num_epochs}, Loss_train: {train_loss}, acc_dev: {correct2}/1272, acc_test: {correct3}/1273")
        f.write('\n')