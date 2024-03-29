import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        shape = logits.shape
        logits = logits.reshape(logits.shape[0], logits.shape[1], -1)  # [N, C, HW]
        logits = logits.transpose(1, 2)   # [N, HW, C]
        logits = logits.reshape(-1, logits.shape[2])    # [NHW, C]
        target = target.reshape(-1, 1)    # [NHW，1]

        logits = -torch.log(logits)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = logits.reshape(shape[0], shape[2], shape[3])

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class model(torch.nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_unit=64, item_num=50000, head_num=1, layer_num=1,
                 max_insert_size=5):
        """
        item_num include total item quantity, padding, EOS token, masked token
        For example, one dataset contains 10000 items
        0-padding 1-10000 item_id 10001-EOS 10002-mask
        EOS is used to determine when sequence insertion stops
        max_insert_size refers to the maximum number of items inserted before each time step
        """
        super(model, self).__init__()
        self.dropout_rate = dropout_rate
        self.hidden_unit = hidden_unit
        self.item_num = item_num
        self.head_num = head_num
        self.max_insert_size = max_insert_size
        self.full_layer = nn.Linear(self.hidden_unit, 3)


        self.layer_num = layer_num
        self.net = nn.ModuleList(
            TransformerEncoderLayer(d_model=self.hidden_unit, nhead=self.head_num, dim_feedforward=self.hidden_unit * 4,
                                    dropout=self.dropout_rate, activation="gelu") for _ in range(self.layer_num))
        self.recommender = TransformerEncoderLayer(d_model=self.hidden_unit, nhead=self.head_num,
                                                   dim_feedforward=self.hidden_unit * 4, dropout=self.dropout_rate, activation="gelu")
        self.insert_net = TransformerEncoderLayer(d_model=self.hidden_unit, nhead=self.head_num,
                                                  dim_feedforward=self.hidden_unit * 4, dropout=self.dropout_rate, activation="gelu")

        self.item_emb = torch.nn.Embedding(self.item_num, self.hidden_unit,
                                           padding_idx=0)
        self.position_emb = torch.nn.Embedding(100, self.hidden_unit)
        self.crossEntropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.crossEntropy2 = CrossEntropyLoss(reduction='none')
        self.dropout = nn.Dropout(self.dropout_rate)

        self.Wco = torch.nn.Parameter(torch.rand(self.hidden_unit, self.hidden_unit))
        self.Uco = torch.nn.Parameter(torch.rand(self.hidden_unit, self.hidden_unit))
        self.Vco = torch.nn.Parameter(torch.rand(self.hidden_unit, 1))
        self.WcoS = torch.nn.Parameter(torch.rand(self.hidden_unit, 2))
        self.WallT = torch.nn.Parameter(torch.rand(self.hidden_unit, 2))

    def seq2tensor(self, seqs):   #add position embedding
        """
        Get item embeddings
        """
        seqs_emb = self.item_emb(seqs)

        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])

        positions = torch.tensor(positions, device=seqs_emb.device).long()

        seqs_emb += self.position_emb(positions)

        return seqs_emb

    def encoder(self, seqs, padding_mask):
        """
        Encoder part
        """    
        seqs_emb = self.seq2tensor(seqs)

        seqs_emb = self.dropout(seqs_emb)

        seqs_emb = torch.transpose(seqs_emb, 0, 1)

        encoder_output = seqs_emb

        for mod in self.net:

            encoder_output = mod(encoder_output, src_key_padding_mask=padding_mask)

        return encoder_output


    def insert(self, encoder_output, insert_seqs,
               insert_padding_mask, sim_output, sim_seqs, hs, candicate_items, zero_mark):
        """
        At each time step, regardless of whether the model predicts the need for insertion, the insert model needs to be
        substituted
        """
        insert_seqs_emb = self.item_emb(insert_seqs)  # (batch_size,seqs_len,max_insert_size-1,emb_size)

        encoder_output = torch.transpose(encoder_output, 0, 1)  # (batch_size,seqs_len,emb_size)

        encoder_output = encoder_output.unsqueeze(2)  # (batch_size,seqs_len,1,emb_size)

        insert_seqs_emb = torch.cat([encoder_output, insert_seqs_emb], dim=2)  # (batch,seqs,max_insert_size,emb_size)

        insert_seqs_emb = torch.reshape(insert_seqs_emb, (
            insert_seqs_emb.shape[0] * insert_seqs_emb.shape[1], insert_seqs_emb.shape[2], insert_seqs_emb.shape[3]))  # batch_size,seqs_len,max_insert,emb_size）

        positions = np.tile(np.array(range(insert_seqs_emb.shape[1])), [insert_seqs_emb.shape[0], 1])  # (batch_size*seqs_len,max_insert_size)

        position_emb = self.position_emb(torch.tensor(positions, device=insert_seqs_emb.device).long())  # (batch_size*seqs_len,max_insert_size,emb_size)

        insert_seqs_emb += position_emb

        insert_seqs_emb = self.dropout(insert_seqs_emb)

        insert_seqs_emb = torch.transpose(insert_seqs_emb, 0, 1)  # (max_insert_size,batch_size*seqs_len,emb_size)

        src_mask = (1 - torch.tril(
            torch.ones(insert_seqs_emb.shape[0], insert_seqs_emb.shape[0], device=insert_seqs_emb.device))).bool()  # previous time steps are not visible to subsequent time steps

        insert_padding_mask[:, 0] = False  # if the whole sequence is padding, nan error will occur

        insert_seqs_output = self.insert_net(insert_seqs_emb, src_key_padding_mask=insert_padding_mask, src_mask=src_mask)  # (max_insert_size,batch_size*seqs_len,emb_size)

        insert_seqs_output2 = insert_seqs_output.clone()

        shape = sim_output.shape     #(sim_num, batch_size, seqs_len, emb_size)
        shape1 = insert_seqs_output.shape
        insert_seqs_output = torch.reshape(insert_seqs_output, (shape1[0], shape[1], shape[2], shape[3]))
        insert_seqs_output = insert_seqs_output.permute(1, 0, 2, 3)
        insert_seqs_output = torch.reshape(insert_seqs_output, (shape[1], shape1[0]*shape[2], shape[3]))

        h_tj = insert_seqs_output.clone()

        insert_seqs_output = torch.unsqueeze(insert_seqs_output, 2)
        insert_seqs_output = insert_seqs_output.repeat(1, 1, shape[0], 1)

        hs = torch.reshape(hs, (shape[1], shape[0], shape[3]))
        hs = torch.unsqueeze(hs, 1)
        hs = hs.repeat(1, shape1[0]*shape[2], 1, 1)

        insert_seqs_output = torch.matmul(insert_seqs_output, self.Wco)
        hs = torch.matmul(hs, self.Uco)
        es = torch.matmul(torch.tanh(insert_seqs_output+hs), self.Vco)
        es = torch.squeeze(es)

        zero_mark = torch.unsqueeze(zero_mark, 1)
        zero_mark = zero_mark.repeat(1, shape1[0]*shape[2], 1)
        zero_mark = (1-zero_mark) * -1000000
        es = es + zero_mark
        es = torch.exp(es)
        a = es/(es.sum(-1, keepdims=True)+(es.sum(-1, keepdims=True) == 0).float())

        a = torch.unsqueeze(a, -1)
        a_hs = a * hs
        a_hs = a_hs.sum(2)

        c = torch.matmul(a_hs, self.WcoS)
        h = torch.matmul(h_tj, self.WallT)
        p_co_all = torch.softmax((c+h), -1)

        # E^t * hn
        unnormalized_hn = torch.matmul(insert_seqs_output2, self.item_emb.weight.T)
        unnormalized_hn = torch.reshape(unnormalized_hn, (shape1[0], shape[1], shape[2], self.item_num))
        P_all = torch.softmax(unnormalized_hn, -1)

        # cadicate_items -> [batch, item_num]
        candicate_items = torch.unsqueeze(candicate_items, 0)
        candicate_items = torch.unsqueeze(candicate_items, 2)      # [1, batch, 1, item_num]

        candicate_items = candicate_items.repeat(shape1[0], 1, shape[2], 1)    # [max_insert, batch, seqs_len, item_num]
        exp_unnormalized_hn = torch.exp(unnormalized_hn) * candicate_items   # [max_insert, batch, seqs_len, item_num]
        sum_exp_unnormalized_hn = exp_unnormalized_hn.sum(-1)     # [max_insert, batch, seqs_len]
        sum_exp_unnormalized_hn = torch.unsqueeze(sum_exp_unnormalized_hn, -1)   # [max_insert, batch, seqs_len, 1]
        P_col = exp_unnormalized_hn / (sum_exp_unnormalized_hn + (sum_exp_unnormalized_hn == 0).float())    # [max_insert, batch, seqs_len, item_num]

        p_co_all = torch.reshape(p_co_all, (shape1[0], shape[1], shape[2], 2))   # [max_insert, batch, seq_len, 2]
        p_co_all = torch.reshape(p_co_all, (shape1[0], shape[1]*shape[2], 2))     # [max_insert, batch * seq_len, 2]

        P_all = torch.reshape(P_all, (shape1[0], shape[1]*shape[2], self.item_num))
        P_col = torch.reshape(P_col, (shape1[0], shape[1]*shape[2], self.item_num))  # [max_insert, batch * seq_len, item_num]

        insert_net_output = p_co_all[:, :, 0:1]*P_col + p_co_all[:, :, 1:2]*P_all

        return insert_net_output

    def corrector_loss(self, full_layer_output, insert_net_output, l1_ground_truth, l2_ground_truth, padding_mask):
        """
        l1_ground_truth corresponds to the actual operation of each time step (keep, delete or insert)
        That is 0 for keep, 1 for delete, 2 for insert
        l2_ground_truth corresponds to the sequence that should be inserted ahead for each time step
        l1_loss refers to the loss of the keep delete insert operation predicted by the model
        l2_loss calculates the loss doing insert operation
        """
        l1_loss_entropy = self.l1_loss(l1_ground_truth, full_layer_output, padding_mask)

        l2_loss_entropy = self.l2_loss(insert_net_output, l1_ground_truth, l2_ground_truth)

        return l1_loss_entropy, l2_loss_entropy

    def l1_loss(self, l1_ground_truth, full_layer_output,
                padding_mask):
        """
        As for the time step of padding, l1_loss equals to 0
        """
        padding_mask = padding_mask.float()  # (batch_size,seqs_len)

        full_layer_output = torch.transpose(full_layer_output, 1, 2)  # (batch_size,3,seqs_len)

        cross_entropy_l1 = self.crossEntropy(full_layer_output, l1_ground_truth)  # (batch_size,seqs_len)

        input_padding = 1 - padding_mask

        cross_entropy_l1 *= input_padding

        return cross_entropy_l1

    def l2_loss(self, insert_net_output, l1_ground_truth, l2_ground_truth):
        """
        Calculating l2_loss needs two types of mask
        The first one is:
        Exclude time steps correspond to keep operation and delete operation
        The second one is:
        Exclude the position that corresponds to EOS token in the input
        For example,
        0 1 2 3 4 -> 1 2 3 4 5  all of five positions calculate the loss
        0 1 2 3 4 -> 1 2 3 4 EOS  all of five positions calculate the loss
        0 1 2 3 eos -> 1 2 3 eos padding   all of the positions calculate the loss except for the position corresponds to EOS, that is, first position, second position, third position, forth position
        Generally, as for sequence 0 1 2 3 4, 0 corresponds to input_seqs, 1 2 3 4 corresponds to input_insert_seqs
        """
        insert_net_output = insert_net_output.permute(0, 3, 1, 2)  # (batch,item_num,seqs,max_insert_size)

        #cross_entropy_l2 = self.crossEntropy(insert_net_output, l2_ground_truth)  # (batch_size,seqs_len,max_insert_size)

        cross_entropy_l2 = self.crossEntropy2(insert_net_output, l2_ground_truth)

        # first mask

        insert_mask = (l1_ground_truth == 2).float()  # (batch_size,seqs_len)

        insert_mask = insert_mask.unsqueeze(-1)  # (batch_size,seqs_len,1)

        cross_entropy_l2 *= insert_mask

        # second mask

        insert_seq_mask = (l2_ground_truth != 0).float()  # (batch_size,seqs_len,max_insert_size)

        cross_entropy_l2 *= insert_seq_mask

        return cross_entropy_l2

    def recommender_loss(self, recommender_output, rec_loss_mask, seqs):

        recommender_output = torch.transpose(recommender_output, 1, 2)  # (batch_size,item_num,seqs_len)

        cross_entropy_rec = self.crossEntropy(recommender_output,seqs)  # (batch_size,seqs_len)

        cross_entropy_rec *= rec_loss_mask
        
        return cross_entropy_rec

    def corrector_forward(self, input_seqs, input_insert_seqs, sim_seqs, candicate_items, zero_mark):
        """
        Item-wise corrector part
        :param input_seqs: (batch_size,seqs_len)
        :param input_insert_seqs: (batch_size,seqs_len,max_insert_size-1)
        """
        par_list = []
        padding_mask = (input_seqs == 0)

        encoder_output = self.encoder(input_seqs, padding_mask)   # (seqs_len, batch_size, emb_size)
        encoder_output_t = torch.transpose(encoder_output, 0, 1)  # (batch_size, seqs_len, emb_size)

        shape = sim_seqs.shape  #(batch, k, seq_len)

        sim_seqs = torch.reshape(sim_seqs, (shape[0] * shape[1], shape[2]))
        sim_mask = (sim_seqs == 0)   # [batch * k ,seq_len]
        sim_mask2 = sim_mask.permute(1, 0)
        sim_mask2 = 1 - torch.unsqueeze(sim_mask2, -1).float()    #[seq_len, batch * k, 1]
        seq_len = sim_mask2.sum(0)     #[batch * k, 1]

        sim_mask3 = (sim_seqs == 0)   # [batch * k ,seq_len]
        sim_mask3[:, 0] = 0  # for the whole padding sequence
        sim_output = self.encoder(sim_seqs, sim_mask3)  # len * (batch * k) * emb_size
        sim_output2 = sim_mask2 * sim_output    # len * (batch * k) * emb_size
        sim_output2 = sim_output2.sum(0)    # (batch * k) * emb_size
        hs = sim_output2 / (seq_len + (seq_len == 0).float())   # (batch * k) * emb_size

        sim_output = torch.reshape(sim_output, (shape[2], shape[0], shape[1], self.hidden_unit))  # len * batch * k * emb_size
        sim_output = sim_output.permute(2, 1, 3, 0)   # k * batch * emb_size * len

        attention = torch.matmul(encoder_output_t, sim_output)  # k * batch * seqs_len * len
        attn_sim_mask = sim_mask.float()  # (batch * k) * sim_len
        attn_sim_mask = torch.reshape(attn_sim_mask, (shape[0], shape[1], 1, shape[2]))  # batch * k * 1 * sim_len
        attn_sim_mask = attn_sim_mask.permute(1, 0, 2, 3)  # k * batch * 1 * sim_len
        attn_sim_mask = attn_sim_mask.repeat(1, 1, attention.shape[2], 1)  # k * batch * seqs_len * sim_len
        attn_seq_len = (1-attn_sim_mask).sum(-1, keepdims=True)  # k * batch * seqs_len * 1
        attn_sim_mask = attn_sim_mask*-1000000
        attention = attention+attn_sim_mask

        attention = attention - attention.max(-1, keepdims=True)[0]
        attention = torch.exp(attention)
        par_list.append(attention.sum())
        attention_sum = attention.sum(-1, keepdims=True)  # k * batch * seqs_len * 1
        attention = attention / (attention_sum+(attn_seq_len == 0).float())  # k * batch * seqs_len * sim_len

        sim_output = torch.transpose(sim_output, 2, 3)  # k * batch * len * emb_size

        sim_mask = 1 - sim_mask.float()  # (batch * k) * len
        sim_mask = torch.reshape(sim_mask, (shape[0], shape[1], shape[2], 1))  # batch * k * len * 1
        sim_mask = sim_mask.permute(1, 0, 2, 3)    # k * batch  * len * 1

        sim_output = sim_output * sim_mask

        x = torch.matmul(attention, sim_output)   # k * batch * seqs_len *  emb_size
        x = x.sum(0)   # batch * seqs_len *  emb_size
        seq_len = torch.reshape(seq_len, (shape[0], shape[1], 1, 1))  # batch*k*1*1
        K = (seq_len > 0).sum(1)  # batch*1*1
        x = x / (K+(K == 0).float())

        encoder_output_t = encoder_output_t + x

        full_layer_output = self.full_layer(encoder_output_t)  # (batch_size, seqs_len, 3), in addition, 3 refers to keep, delete, insert

        temp_input_seqs = input_seqs.unsqueeze(-1)  # (batch_size,seqs_len,1)

        temp_input_insert_mask = torch.cat([temp_input_seqs, input_insert_seqs],
                                           dim=-1)  # (batch_size,seqs_len,max_insert_size)

        insert_padding_mask = (temp_input_insert_mask == 0)  # (batch_size,seqs_len,max_insert_size)

        insert_padding_mask = torch.reshape(insert_padding_mask, (
            insert_padding_mask.shape[0] * insert_padding_mask.shape[1],
            insert_padding_mask.shape[2]))  # (batch_size,seqs_len,max_insert_size)

        insert_net_output = self.insert(encoder_output, input_insert_seqs, insert_padding_mask, sim_output, sim_seqs, hs, candicate_items, zero_mark)

        insert_net_output = torch.transpose(insert_net_output, 0, 1)  # (batch_size*seqs_len,max_insert_size,item_num)

        insert_net_output = torch.reshape(insert_net_output, (
            temp_input_insert_mask.shape[0], temp_input_insert_mask.shape[1], temp_input_insert_mask.shape[2],
            insert_net_output.shape[2]))  # (batch_size,seqs_len,max_insert_size,item_num)

        return full_layer_output, insert_net_output, padding_mask, par_list

    def forward(self, input_seqs):
        """
        Recommender part
        """
        padding_mask = (input_seqs == 0)

        encoder_output = self.encoder(input_seqs, padding_mask)  # (seqs_len,batch_size,emb_size)

        recommender_output = self.recommender(encoder_output, src_key_padding_mask=padding_mask)  # (seqs_len,batch_size,emb_size)

        recommender_output = torch.matmul(recommender_output, self.item_emb.weight.T)  # seqs_len,batch,item_num

        recommender_output = torch.transpose(recommender_output, 0, 1)  # (batch,seqs_len,item_num)

        return recommender_output

    def corrector_inference(self, input_seqs, sim_seqs, candicate_items, zero_mark):
        """
        Correct the original sequence by the corrector model
        return: sequence after correcting the original sequence
        """

        padding_mask = (input_seqs == 0)

        encoder_output = self.encoder(input_seqs, padding_mask)
        encoder_output_t = torch.transpose(encoder_output, 0, 1)  # (batch_size, seqs_len, emb_size)

        shape = sim_seqs.shape  # (batch, k, seq_len)

        sim_seqs = torch.reshape(sim_seqs, (shape[0] * shape[1], shape[2]))
        sim_mask = (sim_seqs == 0)  # [batch * k ,seq_len]
        sim_mask2 = sim_mask.permute(1, 0)
        sim_mask2 = 1 - torch.unsqueeze(sim_mask2, -1).float()  # [seq_len, batch * k, 1]
        seq_len = sim_mask2.sum(0)  # [batch * k, 1]

        sim_mask3 = (sim_seqs == 0)  # [batch * k ,seq_len]
        sim_mask3[:, 0] = 0  # for the whole padding sequence
        sim_output = self.encoder(sim_seqs, sim_mask3)  # len * (batch * k) * emb_size
        sim_output2 = sim_mask2 * sim_output  # len * (batch * k) * emb_size
        sim_output2 = sim_output2.sum(0)  # (batch * k) * emb_size
        hs = sim_output2 / (seq_len + (seq_len == 0).float())  # (batch * k) * emb_size

        sim_output = torch.reshape(sim_output,
                                   (shape[2], shape[0], shape[1], self.hidden_unit))  # len * batch * k * emb_size
        sim_output = sim_output.permute(2, 1, 3, 0)  # k * batch * emb_size * len

        attention = torch.matmul(encoder_output_t, sim_output)  # k * batch * seqs_len * len
        attn_sim_mask = sim_mask.float()  # (batch * k) * sim_len
        attn_sim_mask = torch.reshape(attn_sim_mask, (shape[0], shape[1], 1, shape[2]))  # batch * k * 1 * sim_len
        attn_sim_mask = attn_sim_mask.permute(1, 0, 2, 3)  # k * batch * 1 * sim_len
        attn_sim_mask = attn_sim_mask.repeat(1, 1, attention.shape[2], 1)  # k * batch * seqs_len * sim_len
        attn_seq_len = (1-attn_sim_mask).sum(-1, keepdims=True)  # k * batch * seqs_len * 1
        attn_sim_mask = attn_sim_mask*-1000000
        attention = attention+attn_sim_mask
        attention = attention - attention.max(-1, keepdims=True)[0]
        attention = torch.exp(attention)
        attention_sum = attention.sum(-1, keepdims=True)  # k * batch * seqs_len * 1
        attention = attention / (attention_sum+(attn_seq_len == 0).float())  # k * batch * seqs_len * sim_len

        sim_output = torch.transpose(sim_output, 2, 3)  # k * batch * len * emb_size

        sim_mask = 1 - sim_mask.float()  # (batch * k) * len
        sim_mask = torch.reshape(sim_mask, (shape[0], shape[1], shape[2], 1))  # batch * k * len * 1
        sim_mask = sim_mask.permute(1, 0, 2, 3)  # k * batch  * len * 1

        sim_output = sim_output * sim_mask

        x = torch.matmul(attention, sim_output)  # k * batch * seqs_len *  emb_size
        x = x.sum(0)   # batch * seqs_len *  emb_size
        seq_len = torch.reshape(seq_len, (shape[0], shape[1], 1, 1))  # batch*k*1*1
        K = (seq_len > 0).sum(1)  # batch*1*1
        x = x / (K+(K == 0).float())

        encoder_output_t = encoder_output_t + x

        full_layer_output = self.full_layer(encoder_output_t)  # (batch_size, seqs_len,,3), in addition, 3 refers to keep, delete, insert

        decisions = full_layer_output.argmax(-1)  # (batch_size, seqs_len)

        encoder_output = torch.transpose(encoder_output, 0, 1)  # (batch_size, seqs_len, emb_size)

        input_insert_embedding = torch.reshape(encoder_output, (1, encoder_output.shape[0] * encoder_output.shape[1], encoder_output.shape[2]))  # (1,batch_size*seqs_len,emb_size)

        for i in range(self.max_insert_size):

            positions = np.tile(np.array(range(input_insert_embedding.shape[0])), [input_insert_embedding.shape[1], 1])

            positions = torch.tensor(positions, device=input_insert_embedding.device).long()

            positions = torch.transpose(positions, 0, 1)  # (1,batch_size*seqs_len,1)

            input_insert_embedding += self.position_emb(positions)

            if i == 0:
                tgt_mask = (1 - torch.tril(torch.ones(i + 1, i + 1, device=input_insert_embedding.device))).bool()

                insert_output = self.insert_net(input_insert_embedding, src_mask=tgt_mask)  # (1, batch_size*seqs_len, emb_size)

                insert_output2 = insert_output.clone()
                shape = sim_output.shape
                shape1 = insert_output.shape
                insert_output = torch.reshape(insert_output, (shape1[0], shape[1], shape[2], shape[3]))
                insert_output = insert_output.permute(1, 0, 2, 3)
                insert_output = torch.reshape(insert_output, (shape[1], shape1[0]*shape[2], shape[3]))

                h_tj = insert_output.clone()

                insert_output = torch.unsqueeze(insert_output, 2)
                insert_output = insert_output.repeat(1, 1, shape[0], 1)

                hs_i = torch.reshape(hs, (shape[1], shape[0], shape[3]))
                hs_i = torch.unsqueeze(hs_i, 1)
                hs_i = hs_i.repeat(1, shape1[0]*shape[2], 1, 1)

                insert_output = torch.matmul(insert_output, self.Wco)
                hs_i = torch.matmul(hs_i, self.Uco)
                es = torch.matmul(torch.tanh(insert_output+hs_i), self.Vco)
                es = torch.squeeze(es)

                zero_mark_i = torch.unsqueeze(zero_mark, 1)
                zero_mark_i = zero_mark_i.repeat(1, shape1[0]*shape[2], 1)
                zero_mark_i = (1-zero_mark_i) * -1000000
                es = es + zero_mark_i
                es = torch.exp(es)
                a = es/(es.sum(-1, keepdims=True)+(es.sum(-1, keepdims=True) == 0).float())

                a = torch.unsqueeze(a, -1)
                a_hs = a * hs_i
                a_hs = a_hs.sum(2)

                c = torch.matmul(a_hs, self.WcoS)
                h = torch.matmul(h_tj, self.WallT)
                p_co_all = torch.softmax((c+h), -1)

                # E^t * hn
                unnormalized_hn = torch.matmul(insert_output2, self.item_emb.weight.T)  # (1, batch_size*seqs_len, item_num)
                unnormalized_hn = torch.reshape(unnormalized_hn, (shape1[0], shape[1], shape[2], self.item_num))   # (1, batch_size, seqs_len,item_num)
                P_all = torch.softmax(unnormalized_hn, -1)  # (1, batch_size, seqs_len,item_num)

                # cadicate_items -> [batch, item_num]
                candicate_items_i = torch.unsqueeze(candicate_items, 0)
                candicate_items_i = torch.unsqueeze(candicate_items_i, 2)      # [1, batch, 1, item_num]

                candicate_items_i = candicate_items_i.repeat(shape1[0], 1, shape[2], 1)    # [1, batch, seqs_len, item_num]
                exp_unnormalized_hn = torch.exp(unnormalized_hn) * candicate_items_i   # [1, batch, seqs_len, item_num]
                sum_exp_unnormalized_hn = exp_unnormalized_hn.sum(-1)     # [1, batch, seqs_len]
                sum_exp_unnormalized_hn = torch.unsqueeze(sum_exp_unnormalized_hn, -1)   # [1, batch, seqs_len, 1]
                P_col = exp_unnormalized_hn / (sum_exp_unnormalized_hn + (sum_exp_unnormalized_hn == 0).float())    # [1, batch, seqs_len, item_num]

                p_co_all = torch.reshape(p_co_all, (shape1[0], shape[1], shape[2], 2))   # [1, batch, seq_len, 2]
                p_co_all = torch.reshape(p_co_all, (shape1[0], shape[1]*shape[2], 2))     # [1, batch * seq_len, 2]

                P_all = torch.reshape(P_all, (shape1[0], shape[1]*shape[2], self.item_num))
                P_col = torch.reshape(P_col, (shape1[0], shape[1]*shape[2], self.item_num))  # [1, batch * seq_len, item_num]

                insert_output = p_co_all[:, :, 0:1]*P_col + p_co_all[:, :, 1:2]*P_all       #  [1, batch * seq_len, item_num]
                insert_output = insert_output[-1, :, :]   #  [batch * seq_len, item_num]

                i_insert_seqs = insert_output.argmax(-1, keepdim=True)   # (batch_size*seqs_len, 1)

                insert_seqs = i_insert_seqs  # (batch_size*seqs_len, 1)

            else:
                tgt_mask = (1 - torch.tril(torch.ones(i + 1, i + 1, device=input_insert_embedding.device))).bool()

                insert_output = self.insert_net(input_insert_embedding, src_mask=tgt_mask)  # ((i+1), batch_size*seqs_len, emb_size)

                insert_output2 = insert_output.clone()
                shape = sim_output.shape
                shape1 = insert_output.shape
                insert_output = torch.reshape(insert_output, (shape1[0], shape[1], shape[2], shape[3]))
                insert_output = insert_output.permute(1, 0, 2, 3)
                insert_output = torch.reshape(insert_output, (shape[1], shape1[0]*shape[2], shape[3]))

                h_tj = insert_output.clone()

                insert_output = torch.unsqueeze(insert_output, 2)
                insert_output = insert_output.repeat(1, 1, shape[0], 1)

                hs_i = torch.reshape(hs, (shape[1], shape[0], shape[3]))
                hs_i = torch.unsqueeze(hs_i, 1)
                hs_i = hs_i.repeat(1, shape1[0]*shape[2], 1, 1)

                insert_output = torch.matmul(insert_output, self.Wco)
                hs_i = torch.matmul(hs_i, self.Uco)
                es = torch.matmul(torch.tanh(insert_output+hs_i), self.Vco)
                es = torch.squeeze(es)

                zero_mark_i = torch.unsqueeze(zero_mark, 1)
                zero_mark_i = zero_mark_i.repeat(1, shape1[0]*shape[2], 1)
                zero_mark_i = (1-zero_mark_i) * -1000000
                es = es + zero_mark_i
                es = torch.exp(es)
                a = es/(es.sum(-1, keepdims=True)+(es.sum(-1, keepdims=True) == 0).float())

                a = torch.unsqueeze(a, -1)
                a_hs = a * hs_i
                a_hs = a_hs.sum(2)

                c = torch.matmul(a_hs, self.WcoS)
                h = torch.matmul(h_tj, self.WallT)
                p_co_all = torch.softmax((c+h), -1)

                # E^t * hn
                unnormalized_hn = torch.matmul(insert_output2, self.item_emb.weight.T)  # ((i+1), batch_size*seqs_len, item_num)
                unnormalized_hn = torch.reshape(unnormalized_hn, (shape1[0], shape[1], shape[2], self.item_num))   # ((i+1), batch_size, seqs_len,item_num)
                P_all = torch.softmax(unnormalized_hn, -1)  # ((i+1), batch_size, seqs_len,item_num)

                # cadicate_items -> [batch, item_num]
                candicate_items_i = torch.unsqueeze(candicate_items, 0)
                candicate_items_i = torch.unsqueeze(candicate_items_i, 2)      # [1, batch, 1, item_num]

                candicate_items_i = candicate_items_i.repeat(shape1[0], 1, shape[2], 1)    # [(i+1), batch, seqs_len, item_num]
                exp_unnormalized_hn = torch.exp(unnormalized_hn) * candicate_items_i   # [(i+1), batch, seqs_len, item_num]
                sum_exp_unnormalized_hn = exp_unnormalized_hn.sum(-1)     # [(i+1), batch, seqs_len]
                sum_exp_unnormalized_hn = torch.unsqueeze(sum_exp_unnormalized_hn, -1)   # [(i+1), batch, seqs_len, 1]
                P_col = exp_unnormalized_hn / (sum_exp_unnormalized_hn + (sum_exp_unnormalized_hn == 0).float())    # [(i+1), batch, seqs_len, item_num]

                p_co_all = torch.reshape(p_co_all, (shape1[0], shape[1], shape[2], 2))   # [(i+1), batch, seq_len, 2]
                p_co_all = torch.reshape(p_co_all, (shape1[0], shape[1]*shape[2], 2))     # [(i+1), batch * seq_len, 2]

                P_all = torch.reshape(P_all, (shape1[0], shape[1]*shape[2], self.item_num))
                P_col = torch.reshape(P_col, (shape1[0], shape[1]*shape[2], self.item_num))  # [(i+1), batch * seq_len, item_num]

                insert_output = p_co_all[:, :, 0:1]*P_col + p_co_all[:,:,1:2]*P_all       #  [(i+1), batch * seq_len, item_num]
                insert_output = insert_output[-1, :, :]   #  [batch * seq_len, item_num]

                i_insert_seqs = insert_output.argmax(-1, keepdim=True)  # (batch_size*seqs_len, 1)

                insert_seqs = torch.cat([insert_seqs, i_insert_seqs], -1)  # (batch_size*seqs_len, (i+1))

            i_insert_seqs_embedding = self.item_emb(i_insert_seqs)  # (batch_size*seqs_len, 1, emb_size)

            i_insert_seqs_embedding = torch.transpose(i_insert_seqs_embedding, 0, 1)  # (1, batch_size*seqs_len, emb_size)

            input_insert_embedding = torch.cat([input_insert_embedding, i_insert_seqs_embedding])  # ((i+1)+1, batch_size*seqs_len, emb_size)

        insert_seqs = torch.reshape(insert_seqs, (decisions.shape[0], decisions.shape[1], -1))  # (batch_size,seqs_len,max_insert_size)

        return decisions, insert_seqs

    def seqs_correction(self, decisions, insert_seqs, input_seqs):
        """
        Through function corrector_inference, we can get the modified prediction for every time step of each sequence,
        this function corrects the original sequence via the result from the function corrector_inference
        """

        modified_seqs = input_seqs.clone()

        decisions[modified_seqs == 0] = 0

        modified_seqs[decisions == 1] = 0

        modified_seqs = modified_seqs.tolist()

        dim_0_index_insert, dim_1_index_insert = torch.where(decisions == 2)

        insert_seqs_corrector = insert_seqs[dim_0_index_insert, dim_1_index_insert].tolist()

        i = 0

        pre_dim_0 = 0

        k = 0

        while i < len(insert_seqs_corrector):

            j = 0

            if pre_dim_0 != dim_0_index_insert[i]:

                k = 0

                pre_dim_0 = dim_0_index_insert[i]

            while j < len(insert_seqs_corrector[i]):

                if insert_seqs_corrector[i][j] == self.item_num - 2:

                    break

                modified_seqs[dim_0_index_insert[i]].insert((dim_1_index_insert[i] + k), insert_seqs_corrector[i][j])

                j += 1

            k += j

            i += 1

        batch_size = len(modified_seqs)

        for i in range(batch_size):

            modified_seqs[i] = list(filter(lambda x: (x != 0 and x != self.item_num - 2 and x != self.item_num - 1), modified_seqs[i]))   # filter padding, EOS, mask token

        return modified_seqs
