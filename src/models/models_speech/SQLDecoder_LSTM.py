#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/25

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.dataset_semQL import  gen_x_batch

from src.utils.beam import Beams, ActionInfo
from src.utils.dataset import Batch
from src.rule import semQL as define_rule
from src.models.common_modules import nn_utils, pointer_net

from src.config import cfg

class SQLDecoder(nn.Module):
    def __init__(self, grammar, LSTM_num_layers):
        super(SQLDecoder, self).__init__()
        self.LSTM_num_layers = LSTM_num_layers
        args = cfg
        self.cuda = cfg.cuda
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.DECODER.column_pointer # if use pointerNet for column select
        self.ifMatch = args.ifMatch
        if args.cuda:
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_tensor = torch.FloatTensor

        input_dim = args.DECODER.action_embed_size + \
                    args.DECODER.att_vec_size  + \
                    args.DECODER.type_embed_size

        
        # decoder cell
        lf_decoder_lstms = [nn.LSTMCell(input_dim if i == 0 else args.DECODER.hidden_size, args.DECODER.hidden_size) for i in range(LSTM_num_layers)]

        self.lf_decoder_lstms = nn.ModuleList(lf_decoder_lstms) # for detail
        

        sketch_decoder_lstms = [nn.LSTMCell(input_dim if i == 0 else args.DECODER.hidden_size, args.DECODER.hidden_size) for i in range(LSTM_num_layers)]
        self.sketch_decoder_lstms = nn.ModuleList(sketch_decoder_lstms) # for sketch
        
        self.decoder_cell_init = nn.Linear(args.TRANSFORMER.hidden_size, args.DECODER.hidden_size) # decoder init

        # operation after decoder cell
        # transfer encoder size for dot_prod_attention (h_t & src_encoding)
        self.att_sketch_linear = nn.Linear(args.TRANSFORMER.hidden_size, args.DECODER.hidden_size, bias = False)
        self.att_lf_linear = nn.Linear(args.TRANSFORMER.hidden_size, args.DECODER.hidden_size, bias = False)
        
        # [h_t, ctx_t] --> att_t  hidden_state + attention_state --> new_attention_state
        self.sketch_att_vec_linear = nn.Linear(args.DECODER.hidden_size + args.DECODER.hidden_size, args.DECODER.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.DECODER.hidden_size + args.DECODER.hidden_size, args.DECODER.att_vec_size, bias=False)

        # gate prob
        self.prob_att = nn.Linear(args.DECODER.att_vec_size, 1)
        

        # function for action output
        self.production_embed = nn.Embedding(len(grammar.prod2id), args.DECODER.action_embed_size)  #weight
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_()) # bias
        self.read_out_act = torch.tanh if args.DECODER.readout == 'non_linear' else nn_utils.identity 
        self.query_vec_to_action_embed = nn.Linear(args.DECODER.att_vec_size, args.DECODER.action_embed_size,
                                                   bias=args.DECODER.readout == 'non_linear')
        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        #self.q_att = nn.Linear(args.hidden_size, args.embed_size)
        #self.att_project = nn.Linear(args.hidden_size + args.type_embed_size, args.hidden_size)
        #self.N_embed = nn.Embedding(len(define_rule.N._init_grammar()), args.DECODER.action_embed_size)


        # type embed
        self.type_embed = nn.Embedding(len(grammar.type2id), args.DECODER.type_embed_size)

        self.column_rnn_input = nn.Linear(args.DECODER.hidden_size, args.DECODER.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(args.DECODER.hidden_size, args.DECODER.action_embed_size, bias=False)
        # 我不是很清楚这部分有什么意义
        self.column_embedding_linear = nn.Linear(args.DECODER.hidden_size, args.DECODER.hidden_size, bias=False)  # TODO: if need for attention?
        self.table_embedding_linear = nn.Linear(args.DECODER.hidden_size, args.DECODER.hidden_size, bias=False)  # TODO: if need for attention?
        self.column_pointer_net = pointer_net.PointerNet(args.DECODER.hidden_size, args.DECODER.col_embed_size, attention_type=args.DECODER.column_att)
        self.table_pointer_net = pointer_net.PointerNet(args.DECODER.hidden_size, args.DECODER.col_embed_size, attention_type=args.DECODER.column_att)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        #nn.init.xavier_normal_(self.N_embed.weight.data)
        # print('Use Column Pointer: ', True if self.use_column_pointer else False)

        # dropout
        self.dropout = nn.Dropout(args.DECODER.dropout)
        self.activation = nn.Sigmoid()


    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(table_embedding.size(0),
                                                                          src_embedding.size(1),
                                                                          table_embedding.size(2))

            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)
        embedding_differ.data.masked_fill_(table_unk_mask.unsqueeze(2).expand(
            table_embedding.size(0),
            table_embedding.size(1),
            embedding_differ.size(2)
        ).bool(), 0)

        return embedding_differ
        

    def get_seperate_embedding(self,schema_outputs,col_len,table_len):
        column_embeddings =[]
        table_embeddings = []
        for embedding,col,table in zip(schema_outputs,col_len,table_len):
            table_embedding = embedding[:table,]
            column_embedding = embedding[table:col+table,] 
            column_embeddings.append(column_embedding)
            table_embeddings.append(table_embedding)
        column_embeddings = nn.utils.rnn.pad_sequence(column_embeddings, batch_first = True)
        table_embeddings = nn.utils.rnn.pad_sequence(table_embeddings, batch_first = True)
        return column_embeddings,table_embeddings

    def forward(self, batch, encoder_outputs, schema_outputs):
        #print('SQLDecoder inputs: ', encoder_outputs.size(), schema_len)
        examples = batch.examples

        # prepare input for decoder (initial state & encoder_outputs)
        # src_encodings = encoder_outputs.clone().detach().requires_grad_()
        src_encodings = self.dropout(encoder_outputs)
        # utterance_encodings_sketch_linear = src_encodings.clone().detach().requires_grad_()
        # utterance_encodings_lf_linear = src_encodings.clone().detach().requires_grad_()
        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings) 
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        # decode initial state
        init_cell = F.adaptive_max_pool1d(encoder_outputs.transpose(1, 2), 1)
        init_cell = init_cell.squeeze(-1)
        init_cell.squeeze_(-1)
        dec_init_vec = self.init_decoder_state(init_cell)
        h_tm1 = dec_init_vec # last hidden_size
        
        zero_action_embed = Variable(self.new_tensor(self.args.DECODER.action_embed_size).zero_())
        zero_type_embed = Variable(self.new_tensor(self.args.DECODER.type_embed_size).zero_())
        
        action_probs = [[] for _ in range(len(batch))]
        sketch_attention_history = list()

        for t in range(batch.max_sketch_num): # prediction step
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.sketch_decoder_lstms[0].input_size).zero_(),
                             requires_grad = False) # 
            else:
                # ------------------- action embed ------------------- #
                a_tm1_embeds = [] 
                for e_id, example in enumerate(examples):

                    if t < len(example.sketch):
                        # get the last action
                        # This is the action embedding
                        action_tm1 = example.sketch[t - 1]
                        if type(action_tm1) in [define_rule.Root0,
                                                define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.Bin,
                                                define_rule.N,
                                                define_rule.Order,
                                                define_rule.Vis,
                                                define_rule.Group]:
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        else:
                            print(action_tm1, 'only for sketch')
                            quit()
                            a_tm1_embed = zero_action_embed
                            pass
                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                
                # --------------------- type embed ---------------- #
                pre_types = [] # type_embed
                for e_id, example in enumerate(examples):
                    if t < len(example.sketch):
                        action_tm = example.sketch[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                # construct inputs for decoder lstm
                inputs = [a_tm1_embeds] # pre_action embed
                inputs.append(att_tm1) # attention vector
                inputs.append(pre_types) # pre_type embed 
                x = torch.cat(inputs, dim=-1)
    
            #src_mask = batch.src_token_mask # mask for speech & schema (speech is after CNN)
            #src_mask = batch.src_speech_mask
            src_mask = None

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_sketch_linear, self.sketch_decoder_lstms,
                                                 self.sketch_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)
            sketch_attention_history.append(att_t)

            # get the Root possibility
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)
            
            for e_id, example in enumerate(examples):
                if t < len(example.sketch):
                    action_t = example.sketch[t]
                    act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        sketch_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0) # [batch,]

        #if schema_embedding is None:
        test_mask = batch.table_dict_mask3
        # column_embedding = schema_outputs[:, -examples[0].col_len:]
        # table_embedding = schema_outputs[:, :-examples[0].col_len]
        column_embedding,table_embedding = self.get_seperate_embedding(schema_outputs,batch.col_len,batch.table_len)
        
        if self.ifMatch:
            src_embedding = encoder_outputs
            schema_differ = nn_utils.embedding_cosine(src_embedding = src_embedding, table_embedding = schema_embedding,
                                                  table_unk_mask = batch.schema_token_mask)

            schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

            schema_embedding = schema_embedding + schema_ctx
        column_embedding = self.column_embedding_linear(column_embedding)
        table_embedding = self.table_embedding_linear(table_embedding)
        # schema_differ = self.embedding_cosine(src_embedding=src_encodings, table_embedding=schema_embedding,
        #                                       table_unk_mask=batch.schema_token_mask)

        # schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        # schema_ctx = F.softmax(schema_ctx,dim=-1)

        # schema_embedding = schema_embedding + schema_ctx
        #batch_schema_dict = batch.schema_sents_word
        batch_table_dict = batch.col_table_dict
        table_enable = np.zeros(shape=(len(examples)))
        action_probs = [[] for _ in examples]

        h_tm1 = dec_init_vec
        for t in range(batch.max_action_num):
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.lf_decoder_lstms[0].input_size).zero_(), requires_grad = False)
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm1 = example.tgt_actions[t - 1]
                        if type(action_tm1) in [define_rule.Root0,
                                                define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.Bin,
                                                define_rule.N,
                                                define_rule.Order,
                                                define_rule.Vis,
                                                define_rule.Group
                                                ]:

                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]

                        else:
                            table_num = example.table_len
                            if isinstance(action_tm1, define_rule.C):
                                #print(schema_embedding[e_id, action_tm1.id_c + table_num].size(),)
                                a_tm1_embed = self.column_rnn_input(column_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.T):
                                a_tm1_embed = self.column_rnn_input(table_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.A):
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            else:
                                print(action_tm1, 'not implement')
                                quit()
                                a_tm1_embed = zero_action_embed
                                pass

                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                
                # tgt t-1 action type
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm = example.tgt_actions[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)
                
                inputs = [a_tm1_embeds]
                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            #src_mask = None
            #src_mask = batch.src_speech_mask
            src_mask = None

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.lf_decoder_lstms,
                                                 self.lf_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)
            
            # @prob: apply_rule
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            # @prob: select column/table, table is not able to seperate due to the unfixed length
            column_weights = self.column_pointer_net(src_encodings = column_embedding, query_vec = att_t.unsqueeze(0),
                                                  src_token_mask = batch.col_token_mask)
            column_weights.data.masked_fill_(batch.col_token_mask.bool(), -float('inf'))
            column_attention_weights = F.softmax(column_weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings = table_embedding, query_vec = att_t.unsqueeze(0),
                                                  src_token_mask = None)
            table_dict = [batch_table_dict[x_id][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            schema_token_mask = batch.schema_token_mask2.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))
            table_dict = [batch_table_dict[x_id][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.softmax(table_weights, dim=-1)
            
            # now get the loss
            for e_id, example in enumerate(examples):
                table_num = example.table_len
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        #table_appear_mask[e_id, action_t.id_c] = 1
                        table_enable[e_id] = action_t.id_c
                        act_prob_t_i = column_attention_weights[e_id, action_t.id_c ]
                        action_probs[e_id].append(act_prob_t_i)
                        #print('C', act_prob_t_i)
                        
                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)

                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                        #print('A', act_prob_t_i)

                    else:
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        #print(action_probs)
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)
        return [sketch_prob_var, lf_prob_var]


    def parse(self, batch, encoder_outputs, schema_outputs, beam_size = 5, match_outputs = None):
        """
            parse one example a time
            inputs: batch, beam_size
        """
        #print('SQLDecoder inputs: ', encoder_outputs.size(), schema_len)
        examples = batch.examples
        #table_appear_mask = batch.table_appear_mask
        # src_encodings = encoder_outputs
        src_encodings = self.dropout(encoder_outputs)
        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)
        # utterance_encodings_sketch_linear = src_encodings
        # utterance_encodings_lf_linear =src_encodings
        init_cell = F.adaptive_max_pool1d(encoder_outputs.transpose(1, 2), 1)
        init_cell.squeeze_(-1)
        dec_init_vec = self.init_decoder_state(init_cell)
        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=True)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < cfg.DECODER.decode_max_time_step:
            hyp_num = len(beams)
            exp_src_enconding = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_sketch_linear = utterance_encodings_sketch_linear.expand(hyp_num,
                                                                                       utterance_encodings_sketch_linear.size(1),
                                                                                       utterance_encodings_sketch_linear.size(2))
            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.sketch_decoder_lstms[0].input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root0,
                                            define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.Bin,
                                            define_rule.N,
                                            define_rule.Order,
                                            define_rule.Vis,
                                            define_rule.Group]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)
            
            #TODO: if src_token_mask is needed
            src_mask = None

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_enconding,
                                             exp_src_encodings_sketch_linear, self.sketch_decoder_lstms,
                                             self.sketch_att_vec_linear,
                                             src_token_mask=src_mask)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            new_hyp_meta = []
            # 生成根节点？
            for hyp_id, hyp in enumerate(beams):
                action_class = hyp.get_availableClass()
                if action_class in [define_rule.Root0,
                                    define_rule.Root1,
                                    define_rule.Root,
                                    define_rule.Sel,
                                    define_rule.Filter,
                                    define_rule.Sup,
                                    define_rule.Bin,
                                    define_rule.N,
                                    define_rule.Order,
                                    define_rule.Vis,
                                    define_rule.Group]:
                    possible_productions = self.grammar.get_production(action_class)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]
                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        #print('hyp: {}, prod: {}, new: {}'.format(hyp.score, prod_score, new_hyp_score))
                        # prev_hyp_id是什么意思
                        meta_entry = {'action_type': action_class, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    raise RuntimeError('No right action class')

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            #print('new_hyp_scores', new_hyp_scores)
            # 排序？ completed_beams的意义是什么？
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))
            #print('top_new_hyp_scores', top_new_hyp_scores)
            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]
                action_type_str = hyp_meta_entry['action_type']
                prod_id = hyp_meta_entry['prod_id']
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']
                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                for i in range(self.LSTM_num_layers):
                    h_t[i] = h_t[i][live_hyp_ids]
                    cell_t[i] = cell_t[i][live_hyp_ids]
                h_tm1 = (h_t, cell_t)
                att_tm1 = att_t[live_hyp_ids]
                beams = new_beams
                t += 1
            else:
                break

        # now get the sketch result
        completed_beams.sort(key=lambda hyp: -hyp.score)
        if len(completed_beams) == 0:
            return [[], []]
        #print('sketch:', [[hyp.actions, hyp.score] for hyp in completed_beams])
        sketch_actions = completed_beams[0].actions
        # sketch_actions = examples.sketch
        # if" ".join([str(x) for x in examples[0].sketch])==" ".join([str(x) for x in sketch_actions]):
        #     print('same')
        padding_sketch = self.padding_sketch(sketch_actions) # sketch -> detail
        
        schema_embedding = schema_outputs
        column_embedding = schema_outputs[:, -examples[0].col_len:]
        table_embedding = schema_outputs[:, :-examples[0].col_len]
        
        if self.ifMatch:
            src_embedding = encoder_outputs
            schema_differ = nn_utils.embedding_cosine(src_embedding = src_embedding, table_embedding = schema_embedding,
                                                  table_unk_mask = batch.schema_token_mask)

            schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

            schema_embedding = schema_embedding + schema_ctx
        column_embedding = self.column_embedding_linear(column_embedding)
        table_embedding = self.table_embedding_linear(table_embedding)
       
        # schema_differ = self.embedding_cosine(src_embedding=src_encodings, table_embedding=schema_embedding,
        #                                       table_unk_mask=batch.schema_token_mask)

        # schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        # schema_ctx = F.softmax(schema_ctx)

        # schema_embedding = schema_embedding + schema_ctx
        #schema_embedding = encoder_outputs[:, -schema_len:, :]
        #batch_schema_dict = batch.schema_sents_word
        batch_table_dict = batch.col_table_dict


        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=False)]
        completed_beams = []
        length = 0
        while len(completed_beams) < beam_size and t < cfg.DECODER.decode_max_time_step:
            hyp_num = len(beams)

            # expand value
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_utterance_encodings_lf_linear = utterance_encodings_lf_linear.expand(hyp_num,
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         1),
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         2))

            exp_column_embedding = column_embedding.expand(hyp_num, column_embedding.size(1),
                                                           column_embedding.size(2))
            
            exp_table_embedding = table_embedding.expand(hyp_num, table_embedding.size(1),
                                                           table_embedding.size(2))



            #table_appear_mask = batch.table_appear_mask
            #table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_enable = np.zeros(shape=(hyp_num))
            for e_id, hyp in enumerate(beams):
                for act in hyp.actions:
                    if type(act) == define_rule.C:
                        #table_appear_mask[e_id][act.id_c] = 1
                        table_enable[e_id] = act.id_c

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.lf_decoder_lstms[0].input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                table_num = examples[0].table_len
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root0,
                                            define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.Bin,
                                            define_rule.N,
                                            define_rule.Order,
                                            define_rule.Vis,
                                            define_rule.Group]:

                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.C):
                        a_tm1_embed = self.column_rnn_input(column_embedding[0, action_tm1.id_c ])
                    elif isinstance(action_tm1, define_rule.T):
                        a_tm1_embed = self.column_rnn_input(table_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.A):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            #src_mask = batch.src_token_mask
            src_mask = None
            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_utterance_encodings_lf_linear, self.lf_decoder_lstms,
                                             self.lf_att_vec_linear,
                                             src_token_mask=src_mask)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)
            #print('apply_rule_log_prob:', apply_rule_log_prob)

            column_weights = self.column_pointer_net(src_encodings=exp_column_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.col_token_mask)
            column_selection_log_prob = F.log_softmax(column_weights, dim=-1)
            # 为什么这里的src_token_mask是none？
            table_weights = self.table_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)
            
            schema_token_mask = batch.schema_token_mask2.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))

            table_dict = [batch_table_dict[0][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.log_softmax(table_weights, dim=-1)
            # table_dict = [batch_table_dict[0][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            # # 不用2会怎样？
            # table_mask = batch.table_dict_mask2(table_dict)
            # weights.data.masked_fill_(table_mask.bool(), -float('inf'))
            
            #print('column_selection_log_prob:', column_selection_log_prob)


            table_num = examples[0].table_len
            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                if type(padding_sketch[t]) == define_rule.A:
                    possible_productions = self.grammar.get_production(define_rule.A)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': define_rule.A, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        # 为什么要加table_num  rg没有加-》因为这里是把col和table放在一起了
                        try:
                            # 这个感觉是选择col最关键的一步
                            col_sel_score = column_selection_log_prob[hyp_id, col_id ] 
                        except:
                            return [examples[0].tts]
                        # col_sel_score = column_selection_log_prob[hyp_id, col_id ]
                        if match_outputs is not None:
                            col_sel_score += match_outputs[0][col_id + table_num][0].log()
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': define_rule.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == define_rule.T:
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {'action_type': define_rule.T, 't_id': t_id,
                                      'score': t_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    prod_id = self.grammar.prod2id[padding_sketch[t].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)
                    meta_entry = {'action_type': type(padding_sketch[t]), 'prod_id': prod_id,
                                  'score': torch.tensor(0.0), 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if 'prod_id' in hyp_meta_entry:
                    prod_id = hyp_meta_entry['prod_id']
                if action_type_str == define_rule.C:
                    col_id = hyp_meta_entry['col_id']
                    action = define_rule.C(col_id)
                elif action_type_str == define_rule.T:
                    t_id = hyp_meta_entry['t_id']
                    action = define_rule.T(t_id)
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)
                if len(new_hyp.actions)>length:
                    length = len(new_hyp.actions)
                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                for i in range(self.LSTM_num_layers):
                    h_t[i] = h_t[i][live_hyp_ids]
                    cell_t[i] = cell_t[i][live_hyp_ids]
                h_tm1 = (h_t, cell_t)
                att_tm1 = att_t[live_hyp_ids]

                beams = new_beams
                t += 1
            else:
                break

        completed_beams.sort(key=lambda hyp: -hyp.score)
        # if len(completed_beams) !=0:
        #     print('completed_beams!=0')
        # pred_actions = completed_beams[0].actions
        # if" ".join([str(x) for x in examples[0].tgt_actions])==" ".join([str(x) for x in pred_actions]):
        #     print('same')
        return [completed_beams, sketch_actions]

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        # Process the first layer
        h_t, c_t = [], []
        h_1, c_1 = decoder[0](x, (h_tm1[0][0],h_tm1[1][0]))
        h_t.append(h_1)
        c_t.append(c_1)
        
        # Process remaining layers
        for i in range(1, self.LSTM_num_layers):
            h_i, c_i = decoder[i](h_t[i-1], (h_tm1[0][i],h_tm1[1][i]))
            h_t.append(h_i)
            c_t.append(c_i)
        # h_t, cell_t = decoder(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t[-1],
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)
        att_t = torch.tanh(attention_func(torch.cat([h_t[-1], ctx_t], 1)))
        att_t = self.dropout(att_t)

        # h_t =  torch.cat(tuple(h_t), 0)
        # c_t =  torch.cat(tuple(c_t), 0)
        if return_att_weight:
            return (h_t, c_t), att_t, alpha_t
        else:
            return (h_t, c_t), att_t

    def init_decoder_state(self, enc_last_cell):
        h_list = []
        c_list = []
        for i in range(self.LSTM_num_layers):
            h_0 = self.decoder_cell_init(enc_last_cell)
            h_0 = torch.tanh(h_0)
            
            h_list.append(h_0)
            c_list.append(Variable(self.new_tensor(h_0.size()).zero_()))
        
        return h_list,c_list 
    def padding_sketch(self, sketch):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.N:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Filter and 'A' in action.production:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Order or type(action) == define_rule.Sup  or type(action) == define_rule.Bin or (type(action) == define_rule.Group and action.id_c == 0):
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))

        return padding_result

