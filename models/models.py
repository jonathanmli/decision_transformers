import torch.nn as nn
import torch
# import transformers
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

# neural network output odds for different actions, approximates policy
class LogitsNet(nn.Module):

    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


class DecisionTransformer(nn.Module):

    def __init__(self, embed_dim, max_timestep, state_dim, action_dim, act_f, n_layer, n_head):
        super().__init__()
        # Initializing a GPT2 configuration
        # see https: // huggingface.co / transformers / model_doc / gpt2.html  # gpt2config
        config = GPT2Config(vocab_size=1, n_embd=embed_dim, activation_function=act_f, n_layer=n_layer, n_head=n_head)

        # Initializing a model from the configuration
        self.GPT2 = GPT2Model(config)
        self.embed_t = nn.Embedding(max_timestep, embed_dim) #y not 1 dim for timesteps (which are ints)?
        # because want 1-hot to allow more flexibility
        self.embed_s = nn.Linear(state_dim, embed_dim)
        self.embed_R = nn.Linear(1, embed_dim)
        self.embed_a = nn.Linear(action_dim, embed_dim)
        self.pred_a = nn.Linear(embed_dim, action_dim)

        self.normalize_embed = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.max_length = max_timestep
        pass

    def forward(self, rtg, states, actions, timesteps, mask=None):
        '''
        Takes as input tensors of states with dimension Z * K * X, where Z is the batch size, K the traj length, and X
        the dim of states

        Returns predicted actions of dim Z * K * X, where X is the dim of actions
        '''

        # convert timesteps to 1-hot? no, nn.embed does that automatically
#         print('rtg', rtg)
#         print('ts', timesteps.reshape((-1, 1)).float())
#         print('hello world')
        # need to compute and stack embeddings for each timestep
        # convert timesteps to right dimensions and convert to floats

        # if len of actions smaller, add blank actions at the end
        # this is for prediction
        # timesteps.reshape((-1, 1)).float()

        #         print('s', states)

        batch_size = states.shape[0]

        # compute embeddings for tokens
        # embeddings should be of dimension Z * K * Y, where Y is embedded dimension
        pos_emb = self.embed_t(timesteps)
        s_emb = self.embed_s(states) + pos_emb
        a_emb = self.embed_a(actions) + pos_emb
        R_emb = self.embed_R(rtg) + pos_emb

        s_emb = self.normalize_embed(s_emb)
        a_emb = self.normalize_embed(a_emb)
        R_emb = self.normalize_embed(R_emb)
#         print('semd', s_emb)
#         print('remd', R_emb)
#         print('tst', torch.stack((R_emb, s_emb, a_emb), 1).shape)

        # interleave tokens as (R_1, s_1, a_1, ..., R_K, s_K, a_K)
        # should return tensor of dim Z * 3K * Y
        # I think that padding the last a_K here might be helpful
        input_embeds = torch.stack((R_emb, s_emb, a_emb), 2).reshape((batch_size, -1, self.embed_dim))

#         print('ie', input_embeds.shape)
        # print('mask', mask)
        # attention_mask = torch.stack((mask, mask, mask), 1).reshape((-1))
        # print('a mask', attention_mask)
        # , attention_mask = attention_mask

        # use transformer to get hidden states from output dic
        # this should be a tensor of size Z * 3K * Y
        hidden_states = self.GPT2(inputs_embeds=input_embeds)['last_hidden_state']
#         print('hs', hidden_states)
#         print('hs s', hidden_states.shape)

        # select hidden states for action prediction tokens
        # (ie. get every third element of the hidden state starting from second element)
        a_hidden = hidden_states[:, 1::3]

        # predict action
        return self.pred_a(a_hidden)
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        return self.forward(returns_to_go, states, actions, timesteps)
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            returns_to_go, states, actions, timesteps, mask=attention_mask, **kwargs)

        return action_preds[0,-1]