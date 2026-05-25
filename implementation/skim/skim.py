import torch
import torch.nn as nn
from espnet2.enh.layers.dprnn import SingleRNN, merge_feature, split_feature
from espnet2.enh.layers.tcn import choose_norm

class MemLSTM(nn.Module):

    def __init__(self, hidden_size, dropout=0.0, bidirectional=False, mem_type='hc', norm_type='cLN'):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type
        assert mem_type in ['hc', 'h', 'c', 'id'], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"
        if mem_type in ['hc', 'h']:
            self.h_net = SingleRNN('LSTM', input_size=self.input_size, hidden_size=self.hidden_size, dropout=dropout, bidirectional=bidirectional)
            self.h_norm = choose_norm(norm_type=norm_type, channel_size=self.input_size, shape='BTD')
        if mem_type in ['hc', 'c']:
            self.c_net = SingleRNN('LSTM', input_size=self.input_size, hidden_size=self.hidden_size, dropout=dropout, bidirectional=bidirectional)
            self.c_norm = choose_norm(norm_type=norm_type, channel_size=self.input_size, shape='BTD')

    def extra_repr(self) -> str:
        return f'Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}'

    def forward(self, hc, S):
        if self.mem_type == 'id':
            ret_val = hc
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)
            if self.mem_type == 'hc':
                h = h + self.h_norm(self.h_net(h)[0])
                c = c + self.c_norm(self.c_net(c)[0])
            elif self.mem_type == 'h':
                h = h + self.h_norm(self.h_net(h)[0])
                c = torch.zeros_like(c)
            elif self.mem_type == 'c':
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c)[0])
            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)
        if not self.bidirectional:
            causal_ret_val = []
            for x in ret_val:
                x = x.transpose(1, 0).contiguous().view(B, S, d * H)
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                x_ = x_.view(B * S, d, H).transpose(1, 0).contiguous()
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)
        return ret_val

    def forward_one_step(self, hc, state):
        if self.mem_type == 'id':
            pass
        else:
            h, c = hc
            d, B, H = h.shape
            h = h.transpose(1, 0).contiguous().view(B, 1, d * H)
            c = c.transpose(1, 0).contiguous().view(B, 1, d * H)
            if self.mem_type == 'hc':
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            elif self.mem_type == 'h':
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c = torch.zeros_like(c)
            elif self.mem_type == 'c':
                h = torch.zeros_like(h)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            h = h.transpose(1, 0).contiguous()
            c = c.transpose(1, 0).contiguous()
            hc = (h, c)
        return (hc, state)

class SegLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0, bidirectional=False, norm_type='cLN'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm(norm_type=norm_type, channel_size=input_size, shape='BTD')

    def forward(self, input, hc):
        B, T, H = input.shape
        if hc is None:
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
            c = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
        else:
            h, c = hc
        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(input.shape)
        output = input + self.norm(output)
        return (output, (h, c))

class SkiM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout=0.0, num_blocks=2, segment_size=20, bidirectional=True, mem_type='hc', norm_type='gLN', seg_overlap=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap
        assert mem_type in ['hc', 'h', 'c', 'id', None], f"only support 'hc', 'h', 'c', 'id', and None, current type: {mem_type}"
        self.seg_lstms = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_lstms.append(SegLSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, bidirectional=bidirectional, norm_type=norm_type))
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_lstms.append(MemLSTM(hidden_size, dropout=dropout, bidirectional=bidirectional, mem_type=mem_type, norm_type=norm_type))
        self.output_fc = nn.Sequential(nn.PReLU(), nn.Conv1d(input_size, output_size, 1))

    def forward(self, input):
        B, T, D = input.shape
        if self.seg_overlap:
            input, rest = split_feature(input.transpose(1, 2), segment_size=self.segment_size)
            input = input.permute(0, 3, 2, 1).contiguous()
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)
        B, S, K, D = input.shape
        assert K == self.segment_size
        output = input.view(B * S, K, D).contiguous()
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S)
                pass
        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)
            output = merge_feature(output, rest)
            output = self.output_fc(output).transpose(1, 2)
        else:
            output = output.view(B, S * K, D)[:, :T, :]
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)
        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size
        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return (input, rest)

    def forward_stream(self, input_frame, states):
        B, _, N = input_frame.shape

        def empty_seg_states():
            shp = (1, B, self.hidden_size)
            return (torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype), torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype))
        B, _, N = input_frame.shape
        if not states:
            states = {'current_step': 0, 'seg_state': [empty_seg_states() for i in range(self.num_blocks)], 'mem_state': [[None, None] for i in range(self.num_blocks - 1)]}
        output = input_frame
        if states['current_step'] and states['current_step'] % self.segment_size == 0:
            tmp_states = [empty_seg_states() for i in range(self.num_blocks)]
            for i in range(self.num_blocks - 1):
                tmp_states[i + 1], states['mem_state'][i] = self.mem_lstms[i].forward_one_step(states['seg_state'][i], states['mem_state'][i])
            states['seg_state'] = tmp_states
        for i in range(self.num_blocks):
            output, states['seg_state'][i] = self.seg_lstms[i](output, states['seg_state'][i])
        states['current_step'] += 1
        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)
        return (output, states)
if __name__ == '__main__':
    torch.manual_seed(111)
    seq_len = 100
    model = SkiM(16, 11, 16, dropout=0.0, num_blocks=4, segment_size=20, bidirectional=False, mem_type='hc', norm_type='cLN', seg_overlap=False)
    model.eval()
    input = torch.randn(3, seq_len, 16)
    seg_output = model(input)
    state = None
    for i in range(seq_len):
        input_frame = input[:, i:i + 1, :]
        output, state = model.forward_stream(input_frame=input_frame, states=state)
        torch.testing.assert_allclose(output, seg_output[:, i:i + 1, :])
    print('streaming ok')
