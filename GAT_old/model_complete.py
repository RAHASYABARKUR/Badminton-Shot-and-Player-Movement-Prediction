import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

PAD = 0
# make adjacency matrix according to encode length which number of row is encode length times 2
def initialize_adjacency_matrix(batch_size, encode_length, shot_type):
    adjacency_matrix = torch.zeros((13, encode_length * 2, encode_length * 2), dtype=int).to(shot_type.device)
    complete_adjacency_matrix = torch.ones((1, (encode_length) * 2, (encode_length) * 2), dtype=int).to(shot_type.device) - torch.eye((encode_length) * 2).to(shot_type.device)
    
    adjacency_matrix = torch.cat((adjacency_matrix, complete_adjacency_matrix), dim=0)
    
    for row in range(encode_length * 2):
        node_index = int(row / 2)
        if row % 2 == 0: # black node
            if node_index % 2 == 0: # even black node
                if (node_index + 1) * 2 <= encode_length * 2 - 1:
                    adjacency_matrix[11][row][(node_index + 1) * 2] = 1
                    adjacency_matrix[11][(node_index + 1) * 2][row] = 1
                    adjacency_matrix[13][row][(node_index + 1) * 2] = 0
                    adjacency_matrix[13][(node_index + 1) * 2][row] = 0
            if node_index % 2 == 1: # odd black node
                if (node_index + 1) * 2 <= encode_length * 2 - 1:
                    adjacency_matrix[12][row][(node_index + 1) * 2] = 1
                    adjacency_matrix[12][(node_index + 1) * 2][row] = 1
                    adjacency_matrix[13][row][(node_index + 1) * 2] = 0
                    adjacency_matrix[13][(node_index + 1) * 2][row] = 0
        if row % 2 == 1: # white node
            if node_index % 2 == 0: # even white node
                if (node_index + 1) * 2 + 1 <= encode_length * 2 - 1:
                    adjacency_matrix[12][row][(node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[12][(node_index + 1) * 2 + 1][row] = 1
                    adjacency_matrix[13][row][(node_index + 1) * 2 + 1] = 0
                    adjacency_matrix[13][(node_index + 1) * 2 + 1][row] = 0
            if node_index % 2 == 1: # odd white node                
                if (node_index + 1) * 2 + 1 <= encode_length * 2 - 1:
                    adjacency_matrix[11][row][(node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[11][(node_index + 1) * 2 + 1][row] = 1
                    adjacency_matrix[13][row][(node_index + 1) * 2 + 1] = 0
                    adjacency_matrix[13][(node_index + 1) * 2 + 1][row] = 0
    
    # for row in range(2, encode_length * 2, 2):
    #     adjacency_matrix[11][row-2][row-1] = 1
    #     adjacency_matrix[11][row-1][row-2] = 1
    #     adjacency_matrix[14][row-2][row-1] = 0
    #     adjacency_matrix[14][row-1][row-2] = 0
    
    adjacency_matrix = torch.tile(adjacency_matrix, (batch_size, 1, 1, 1))
    for batch in range(batch_size):
        for step in range(len(shot_type[batch])):
            if step % 2 == 0:
                adjacency_matrix[batch][shot_type[batch][step]][step * 2][(step + 1) * 2 + 1] = 1
                adjacency_matrix[batch][shot_type[batch][step]][(step + 1) * 2 + 1][step * 2] = 1
                adjacency_matrix[batch][13][step * 2][(step + 1) * 2 + 1] = 0
                adjacency_matrix[batch][13][(step + 1) * 2 + 1][step * 2] = 0
            if step % 2 == 1:
                adjacency_matrix[batch][shot_type[batch][step]][(step * 2) + 1][(step + 1) * 2] = 1
                adjacency_matrix[batch][shot_type[batch][step]][(step + 1) * 2][(step * 2) + 1] = 1
                adjacency_matrix[batch][13][(step * 2) + 1][(step + 1) * 2] = 0
                adjacency_matrix[batch][13][(step + 1) * 2][(step * 2) + 1] = 0

    return adjacency_matrix

def update_adjacency_matrix(batch_size, step, adjacency_matrix, shot_type_predict=False):
    new_adjacency_matrix = torch.zeros((batch_size, 13, step  * 2, step * 2), dtype=int).to(adjacency_matrix.device)
    complete_adjacency_matrix = torch.ones((batch_size, 1, step * 2, step * 2), dtype=int).to(adjacency_matrix.device) - torch.eye(step * 2).to(adjacency_matrix.device)
    new_adjacency_matrix = torch.cat((new_adjacency_matrix, complete_adjacency_matrix), dim=1)

    new_adjacency_matrix[:, :, :-2, :-2] = adjacency_matrix
    adjacency_matrix = new_adjacency_matrix.clone()
        
    for row in range(step * 2):
        node_index = int(row / 2)
        if row % 2 == 0: # black node
            if node_index % 2 == 0: # even black node
                if (node_index + 1) * 2 <= step * 2 - 1:
                    adjacency_matrix[:, 11, row, (node_index + 1) * 2] = 1
                    adjacency_matrix[:, 11, (node_index + 1) * 2, row] = 1
                    adjacency_matrix[:, 13, row, (node_index + 1) * 2] = 0
                    adjacency_matrix[:, 13, (node_index + 1) * 2, row] = 0
            if node_index % 2 == 1: # odd black node
                if (node_index + 1) * 2 <= step * 2 - 1:
                    adjacency_matrix[:, 12, row, (node_index + 1) * 2] = 1
                    adjacency_matrix[:, 12, (node_index + 1) * 2, row] = 1
                    adjacency_matrix[:, 13, row, (node_index + 1) * 2] = 0
                    adjacency_matrix[:, 13, (node_index + 1) * 2, row] = 0
        if row % 2 == 1: # white node
            if node_index % 2 == 0: # even white node
                if (node_index + 1) * 2 + 1 <= step * 2 - 1:
                    adjacency_matrix[:, 12, row, (node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[:, 12, (node_index + 1) * 2 + 1, row] = 1
                    adjacency_matrix[:, 13, row, (node_index + 1) * 2 + 1] = 0
                    adjacency_matrix[:, 13, (node_index + 1) * 2 + 1, row] = 0
            if node_index % 2 == 1: # odd white node                
                if (node_index + 1) * 2 + 1 <= step * 2 - 1:
                    adjacency_matrix[:, 11, row, (node_index + 1) * 2 + 1] = 1
                    adjacency_matrix[:, 11, (node_index + 1) * 2 + 1, row] = 1
                    adjacency_matrix[:, 13, row, (node_index + 1) * 2 + 1] = 0
                    adjacency_matrix[:, 13, (node_index + 1) * 2 + 1, row] = 0

    # for row in range(2, step * 2, 2):
    #     adjacency_matrix[:, 11, row-2, row-1] = 1
    #     adjacency_matrix[:, 11, row-1, row-2] = 1
    #     adjacency_matrix[:, 14, row-2, row-1] = 0
    #     adjacency_matrix[:, 14, row-1, row-2] = 0

    if shot_type_predict:        
        if step % 2 == 0:
            adjacency_matrix[:, :, (step - 1) * 2, :] = 0
            adjacency_matrix[:, :, :, (step - 1) * 2] = 0
        if step % 2 == 1: 
            adjacency_matrix[:, :, (step - 1) * 2 + 1, :] = 0
            adjacency_matrix[:, :, :, (step - 1) * 2 + 1] = 0
   
    return adjacency_matrix

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features

        self.attention = nn.Parameter(torch.Tensor(num_heads, out_features, 2 * in_features))
        self.linear = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.attention, gain=nn.init.calculate_gain('relu'))

    def forward(self, node_features, adj_matrix):
        h = self.linear(node_features)
        h = h.view(h.size(0), h.size(1), self.num_heads, self.out_features)
        h = h.permute(0, 2, 1, 3)

        src = h.unsqueeze(3)
        tgt = h.unsqueeze(2)
        scores = torch.cat([src, tgt], dim=-1)
        scores = torch.einsum("bhndh,hdo->bhndo", scores, self.attention)
        scores = self.leaky_relu(scores)

        mask = adj_matrix.unsqueeze(1).to(dtype=torch.bool)
        scores = scores.masked_fill(~mask, float('-inf'))

        attention = self.softmax(scores)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h)
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous()
        h_prime = h_prime.view(h_prime.size(0), h_prime.size(1), -1)

        return h_prime


class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, num_layers, dropout=0.1):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_features, hidden_features, num_heads, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(num_heads * hidden_features, hidden_features, num_heads, dropout))
        self.layers.append(GATLayer(num_heads * hidden_features, out_features, num_heads, dropout))

    def forward(self, node_features, adj_matrix):
        for layer in self.layers[:-1]:
            node_features = layer(node_features, adj_matrix)
            node_features = F.relu(node_features)
        node_features = self.layers[-1](node_features, adj_matrix)
        return node_features

class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()
        player_num = args['player_num']
        player_dim = args['player_dim']

        type_num = args['type_num']

        location_dim = args['location_dim']

        hidden_size = args['hidden_size']

        self.player_num = args['player_num']
        self.type_num = args['type_num']

        num_layer = args['num_layer']

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.coordination_transform = nn.Linear(2, location_dim)

        self.model_input_linear = nn.Linear(player_dim + location_dim, hidden_size)

        self.rGCN = relational_GCN(hidden_size, type_num, args['num_basis'], num_layer, device)

        self.predict_shot_type = nn.Linear(hidden_size * 2, type_num)
        self.predict_xy = nn.Linear(hidden_size*2, 10)

    def forward(self, player, step, encode_node_embedding, adjacency_matrix, 
                player_A_x, player_A_y, player_B_x, player_B_y, 
                shot_type=None, train=False, first=False):
        batch_size = player.size(0)

        prev_player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        prev_player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()

        player_embedding = self.player_embedding(player)

        prev_coordination_sequence = torch.stack((prev_player_A_coordination, prev_player_B_coordination), dim=2).view(player.size(0), -1, 2)
        prev_coordination_transform = self.coordination_transform(prev_coordination_sequence)
        prev_coordination_transform = F.relu(prev_coordination_transform)

        rally_information = torch.cat((prev_coordination_transform, player_embedding), dim=-1)
        initial_embedding = self.model_input_linear(rally_information)

        if not first:
            player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
            player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()
            coordination_sequence = torch.stack((player_A_coordination, player_B_coordination), dim=2).view(player.size(0), -1, 2)
            coordination_transform = self.coordination_transform(coordination_sequence)
            coordination_transform = F.relu(coordination_transform)
            model_input = torch.cat((coordination_transform, player_embedding), dim=-1)
            model_input = self.model_input_linear(model_input)
            model_input = torch.cat((encode_node_embedding, model_input), dim=1)
            tmp_embedding = self.rGCN(model_input, adjacency_matrix)
            passed_node_embedding = torch.cat((encode_node_embedding[:, :-2, :], tmp_embedding[:, -4:, :]), dim=1)
        else:
            passed_node_embedding = encode_node_embedding.clone()
     
        batch_size = player.size(0)
        tmp_adjacency_matrix = update_adjacency_matrix(batch_size, step, adjacency_matrix, shot_type_predict=True)
        model_input = torch.cat((passed_node_embedding, initial_embedding), dim=1)
        # ===============================================================================================================
        if step % 2 == 0:            
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, :, torch.arange(tmp_adjacency_matrix.size(3))!=(step-1)*2]
        if step % 2 == 1:
            tmp_model_input = model_input[:, torch.arange(model_input.size(1))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, torch.arange(tmp_adjacency_matrix.size(2))!=(step-1)*2+1]
            tmp_adjacency_matrix = tmp_adjacency_matrix[:, :, :, torch.arange(tmp_adjacency_matrix.size(3))!=(step-1)*2+1]

        tmp_embedding = self.rGCN(tmp_model_input, tmp_adjacency_matrix)
        padding_full_graph_node_embedding = torch.zeros((tmp_embedding.size(0), tmp_embedding.size(1)+1, tmp_embedding.size(2))).to(player.device)

        if step % 2 == 0:
            padding_full_graph_node_embedding[:, :-2, :] = tmp_embedding[:, :-1, :]
            padding_full_graph_node_embedding[:, -1, :] = tmp_embedding[:, -1, :]  
        if step % 2 == 1: 
            padding_full_graph_node_embedding[:, :-1, :] = tmp_embedding

        shot_type_predict = torch.cat((passed_node_embedding[:, :-2, :], padding_full_graph_node_embedding[:, -4:, :]), dim=1)
        
        if step % 2 == 0:
            black_node = shot_type_predict[:, (step - 1) * 2 + 1, :]
            white_node = shot_type_predict[:, (step - 2) * 2, :]
        if step % 2 == 1:
            black_node = shot_type_predict[:, (step - 2) * 2 + 1, :]
            white_node = shot_type_predict[:, (step - 1) * 2, :]            

        type_predict_node = torch.cat((black_node, white_node), dim=-1) 
        predict_shot_type_logit = self.predict_shot_type(type_predict_node)

        adjacency_matrix = update_adjacency_matrix(batch_size, step, adjacency_matrix)
        if train:
            for batch in range(batch_size):
                if step % 2 == 0:
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 2) * 2][(step - 1) * 2 + 1] = 1
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 1) * 2 + 1][(step - 2) * 2] = 1
                    adjacency_matrix[batch][13][(step - 2) * 2][(step - 1) * 2 + 1] = 0
                    adjacency_matrix[batch][13][(step - 1) * 2 + 1][(step - 2) * 2] = 0
                if step % 2 == 1:
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 1) * 2][(step - 2) * 2 + 1] = 1
                    adjacency_matrix[batch][shot_type[batch][0]][(step - 2) * 2 + 1][(step - 1) * 2] = 1
                    adjacency_matrix[batch][13][(step - 1) * 2][(step - 2) * 2 + 1] = 0
                    adjacency_matrix[batch][13][(step - 2) * 2 + 1][(step - 1) * 2] = 0
        else:
            weights = predict_shot_type_logit[0, 1:]
            weights = F.softmax(weights, dim=0)
            predict_shot_type = torch.multinomial(weights, 1).unsqueeze(0) + 1

            for batch in range(batch_size):
                if step % 2 == 0:
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 2) * 2][(step - 1) * 2 + 1] = 1
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 1) * 2 + 1][(step - 2) * 2] = 1
                    adjacency_matrix[batch][13][(step - 2) * 2][(step - 1) * 2 + 1] = 0
                    adjacency_matrix[batch][13][(step - 1) * 2 + 1][(step - 2) * 2] = 0
                if step % 2 == 1:
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 1) * 2][(step - 2) * 2 + 1] = 1
                    adjacency_matrix[batch][predict_shot_type[batch][0]][(step - 2) * 2 + 1][(step - 1) * 2] = 1
                    adjacency_matrix[batch][13][(step - 1) * 2][(step - 2) * 2 + 1] = 0
                    adjacency_matrix[batch][13][(step - 2) * 2 + 1][(step - 1) * 2] = 0


            
        tmp_embedding = self.rGCN(model_input, adjacency_matrix)
        node_embedding = torch.cat((model_input[:, :-2, :], tmp_embedding[:, -2:, :]), dim=1)

        last_two_node = node_embedding[:, -2:, :].view(batch_size, -1)
        predict_xy = self.predict_xy(last_two_node)             
        predict_xy = predict_xy.view(batch_size, 2, 5)
        
        return predict_xy, predict_shot_type_logit, adjacency_matrix, passed_node_embedding

class Encoder(nn.Module):
    def __init__(self, args, device):
        super(Encoder, self).__init__()
        player_num = args['player_num']
        player_dim = args['player_dim']

        type_num = args['type_num']

        location_dim = args['location_dim']

        hidden_size = args['hidden_size']

        num_layer = args['num_layer']

        self.player_num = player_num

        self.player_embedding = nn.Embedding(player_num, player_dim)
        self.coordination_transform = nn.Linear(2, location_dim)
        
        self.model_input_linear = nn.Linear(player_dim + location_dim , hidden_size)

        self.gat = GAT(args['hidden_size'], args['hidden_size'], args['hidden_size'], 4, 2, dropout=0.1)

    def forward(self, player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y, encode_length):
        # get the initial(encode) adjacency matrix
        batch_size = player.size(0)
        adjacency_matrix = initialize_adjacency_matrix(batch_size, encode_length, shot_type)
        
        player_A_coordination = torch.cat((player_A_x.unsqueeze(2), player_A_y.unsqueeze(2)), dim=2).float()
        player_B_coordination = torch.cat((player_B_x.unsqueeze(2), player_B_y.unsqueeze(2)), dim=2).float()

        # interleave the player and opponent location
        coordination_sequence = torch.stack((player_A_coordination, player_B_coordination), dim=2).view(player.size(0), -1, 2)
        coordination_transform = self.coordination_transform(coordination_sequence)
        coordination_transform = F.relu(coordination_transform)

        player = player.repeat([1, encode_length])
        player_embedding = self.player_embedding(player)

        rally_information = torch.cat((coordination_transform, player_embedding), dim=-1)
        
        model_input = self.model_input_linear(rally_information)

        # fixed node embedding in decoder
        node_embedding = self.gat( model_input, adjacency_matrix)
        
        return node_embedding, adjacency_matrix