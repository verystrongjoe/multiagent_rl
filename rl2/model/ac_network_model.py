import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module, is_seq=False, is_lstm=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.is_lstm = is_lstm
        self.is_seq = is_seq

    def forward(self, x, hidden_states=None):

        if len(x.size()) <= 2:
            return self.module(x)

        s = None
        t = None
        n = None

        if len(x.size()) == 3:
            t, n = x.size(0), x.size(1)
        elif len(x.size()) == 4:
            s, t, n = x.size(0), x.size(1), x.size(2)

        # we assumed that hx, cx shaped t*n, shape of hx/cx
        if hidden_states is not None:
            (hx, cx) = hidden_states  # (t * n * hx), (t * n * cx)
            if len(x.size()) == 3:
                hx = hx.contiguous().view(t * n, hx.size()[1])
                cx = cx.contiguous().view(t * n, cx.size()[1])
            elif len(x.size()) == 4:  # sequence
                # hx = hx.contiguous().view(3, hx.size()[1])
                # cx = cx.contiguous().view(3, cx.size()[1])
                hx = hx.contiguous().view(1, 6, 64)
                cx = cx.contiguous().view(1, 6, 64)

        elif self.is_lstm:
            hx = torch.zeros((1, 6, 64), requires_grad=False)
            cx = torch.zeros((1, 6, 64), requires_grad=False)

        x_reshape = None
        if len(x.size()) == 3:
            if self.is_lstm and self.is_seq is False:
                x_reshape = x.contiguous().view(t,  n, x.size(2))
            else:
                x_reshape = x.contiguous().view(t * n, x.size(2))
        elif len(x.size()) == 4:
            if self.is_lstm:
                x_reshape = x.contiguous().view(s,  t * n, x.size(3))
            else:
                x_reshape = x.contiguous().view(s * t * n, x.size(3))

        # forward, one step
        # if hidden_states is not None or self.is_lstm:
        if self.is_lstm is True:
            if self.is_seq is True:
                hx, cx = self.module(x_reshape.data, (hx.cuda(), cx.cuda())) # (t * n , hx) ,  ( t * n, cx)
                return hx, cx
            else:
                output, (hx, cx) = self.module(x_reshape.data, (hx.cuda(), cx.cuda())) # (t * n , hx) ,  ( t * n, cx)
                # output = output.contiguous().view(s, t, n, output.size()[2])
                return output, (hx,cx)

        else:  # sequence
            # merge batch and seq dimensions
            if len(x.size()) == 3:
                y = self.module(x_reshape)
                y = y.contiguous().view(t, n, y.size()[1])
            elif len(x.size()) == 4:
                y = self.module(x_reshape.contiguous().view(s, t * n, x.size(3))) # batch * agent
                y = y.contiguous().view(s, t, n, y.size()[2])

            return y


class ActorNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int) : Number of dimensions in input  (agents, observation)
            out_dim (int)   : Number of dimensions in output
            hidden_dim (int) : Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ActorNetwork, self).__init__()

        self.nonlin = F.relu
        self.dense1 = TimeDistributed(nn.Linear(input_dim, 64))

        self.lstm_cell = TimeDistributed(nn.LSTMCell(64, 64, 1), is_lstm=True, is_seq=True)  # input_size, hidden_size, num_layers
        self.lstm_sequence = TimeDistributed(nn.LSTM(64, 64), is_lstm=True)  # input_size, hidden_size

        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstm = nn.LSTM(64, 32, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.dense2 = TimeDistributed(nn.Linear(64, out_dim))
        self.dense3 = TimeDistributed(nn.Linear(64, input_dim))

        self.cx = torch.zeros((3, 64), requires_grad=False)
        self.hx = torch.zeros((3, 64), requires_grad=False)

        self.cx2 = torch.zeros((1, 6, 64), requires_grad=False)
        self.hx2 = torch.zeros((1, 6, 64), requires_grad=False)


    def reset_lstm_hidden_state(self, done=True):
        if done == True:
            self.cx = torch.zeros((3, 64), requires_grad=False)
            self.hx = torch.zeros((3, 64), requires_grad=False)

            self.cx2 = torch.zeros((1, 6, 64), requires_grad=False)
            self.hx2 = torch.zeros((1, 6, 64), requires_grad=False)

        else:
            self.hx = torch.zeros(self.hx.data, requires_grad=False)
            self.cx = torch.zeros(self.cx.data, requires_grad=False)

            self.cx2 = torch.zeros(self.cx2.data, requires_grad=False)
            self.hx2 = torch.zeros(self.hx2.data, requires_grad=False)


    def forward(self, obs, hidden_states=None):
        """
        Inputs:
            obs (PyTorch Matrix): Batch of observations
            hidden_states :
                None
                    this is forward phase, it uses lstm cell. step
                not None :
                    this is training phase, it uses lstm. sequence
        Outputs:
            out (PyTorch Matrix): policy, next_state
        """
        hid = F.relu(self.dense1(obs))

        if hidden_states == None and len(obs.size()) == 3:
            hx, cx = self.lstm_cell(hid, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
            hid = hx
            hid.unsqueeze_(0)
        elif hidden_states == None and len(obs.size()) == 4:
            output, (hx,cx) = self.lstm_sequence(hid, (self.hx2, self.cx2))
            hid = output
        elif hidden_states is not None and len(obs.size()) == 4:
            output, (hx, cx) = self.lstm_sequence(hid, hidden_states)
            hid = output

        hid, _ = self.bilstm(hid, None)
        hid = F.relu(hid)
        policy = self.dense2(hid)
        policy = nn.Softmax(dim=-1)(policy)
        next_state = self.dense3(hid)
        return policy, next_state, (hx.squeeze_(), cx.squeeze_())


class CriticNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int): Number of dimensions in input  (agents, observation)
            out_dim (int): Number of dimensions in output

            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CriticNetwork, self).__init__()

        self.nonlin = F.relu
        self.dense1 = TimeDistributed(nn.Linear(input_dim, 64))
        self.lstm_sequence = TimeDistributed(nn.LSTM(64, 64), is_lstm=True)  # input_size, hidden_size

        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.lstm = nn.LSTM(64, 64, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.dense2 = nn.Linear(64, out_dim)
        self.dense3 = nn.Linear(64, out_dim)

    def forward(self, obs, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Q-function
            out (PyTorch Matrix): reward
        """
        #obs = obs.contiguous().view(obs.size()[0], obs.size()[1]*obs.size()[2], obs.size()[3])
        action = action.contiguous().view(obs.size()[0], obs.size()[1], obs.size()[2], 5)

        obs_act = torch.cat((obs, action), dim=-1)
        hid = F.relu(self.dense1(obs_act))
        output, (hx, cx) = self.lstm_sequence(hid)
        hid = output
        hid, _ = self.lstm(hid, None)

        hid = hid.contiguous().view(obs.size()[0], obs.size()[1], obs.size()[2], 64)

        hid = F.relu(hid[:, :, -1, :])
        Q = self.dense2(hid)
        r = self.dense3(hid)
        return Q, r #, (hx, cx)
