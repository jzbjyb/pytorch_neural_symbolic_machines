"""
Probing attention

Usage:
    attn.py --model=<file> --test-file=<file> --table-file=<file> [options]

Options:
    -h --help                               show this screen.
    --seed=<int>                            seed [default: 0]
    --eval-batch-size=<int>                 batch size for evaluation [default: 2]
"""

from typing import List, Dict
from docopt import docopt
import sys
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from table_bert import TableBertModel, VanillaTableBert
from table.experiments import load_environments
from nsm.parser_module.table_bert_helper import get_table_bert_input_from_context
from nsm import nn_util


def get_cross_attn_sum(attn,  # (B, seq_len, seq_len)
                       split_len,  # (B, )
                       max_len):  # (B, )
    bs = attn.size(0)
    # 2 means attend
    a2bs = []
    b2as = []
    for i in range(bs):
        split_point = split_len[i]
        max_point = max_len[i]
        matrix = attn[i]
        torch.set_printoptions(edgeitems=100)
        # sum over last dim, average over tokens, max over heads (skip special tokens)
        a2b = matrix[:, 1:split_point][:, :, split_point+1:].sum(-1).mean(-1).max().item()
        b2a = matrix[:, split_point+1:max_point][:, :, 1:split_point].sum(-1).mean(-1).max().item()
        a2bs.append(a2b)
        b2as.append(b2a)
    return a2bs, b2as

def main():
    args = docopt(__doc__)
    test_file = args['--test-file']
    table_file = args['--table-file']
    model_path = args['--model']
    batch_size = int(args['--eval-batch-size'])

    # load model
    model = TableBertModel.from_pretrained(model_path)
    if type(model) == VanillaTableBert:
        model.config.column_representation = 'mean_pool_column_name'

    # load env
    test_envs = load_environments(
        [test_file],
        table_file=table_file,
        table_representation_method='canonical',
        bert_tokenizer=model.tokenizer
    )

    context2tables: Dict[int, List] = defaultdict(list)
    table2contexts: Dict[int, List] = defaultdict(list)
    with torch.no_grad():
        batch_iter = nn_util.batch_iter(test_envs, batch_size, shuffle=False)
        for batched_envs in tqdm(batch_iter, total=len(test_envs) // batch_size, file=sys.stdout):
            batched_envs_context = [env.get_context() for env in batched_envs]
            contexts, tables = get_table_bert_input_from_context(
                batched_envs_context, model, is_training=False, content_snapshot_strategy='synthetic_row')
            tensor_dict, _ = model.to_tensor_dict(contexts, tables)
            context_len = tensor_dict['context_token_indices'].max(-1)[0] + 1
            max_len = tensor_dict['attention_mask'].long().sum(-1)
            device = next(model.parameters()).device
            for key in tensor_dict.keys():
                tensor_dict[key] = tensor_dict[key].to(device)
            output = model.bert_all(
                input_ids=tensor_dict['input_ids'],
                token_type_ids=tensor_dict['segment_ids'],
                attention_mask=tensor_dict['attention_mask'],
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True)
            attns = output.attentions
            for i, attn in enumerate(attns):
                c2t, t2c = get_cross_attn_sum(attn, context_len, max_len)
                context2tables[i].extend(c2t)
                table2contexts[i].extend(t2c)
    for i in context2tables:
        print('layer {}, c2t {}, t2c {}'.format(i, np.mean(context2tables[i]), np.mean(table2contexts[i])))


if __name__ == '__main__':
    main()