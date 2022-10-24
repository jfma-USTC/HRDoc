import torch
from torch.nn import functional as F


def align_labels(labels):
    '''
        输入：labels 为list [[N], [N]]
        输出：aligned_labels 为[B, N], mask[B, N]
    '''
    batch_size = len(labels)
    dtype = labels[0].dtype
    device = labels[0].device

    max_len = max([labels_pb.shape[0] for labels_pb in labels])

    aligned_labels = list()
    masks = torch.zeros([batch_size, max_len], dtype=dtype, device=device)
    
    for batch_idx in range(batch_size):
        labels_pb = labels[batch_idx]
        cur_len = labels_pb.shape[0]
        aligned_labels_pi = F.pad(
            labels_pb,
            (0, max_len-cur_len),
            mode='constant',
            value=0
        )
        aligned_labels.append(aligned_labels_pi)

        masks[batch_idx, :cur_len] = 1
    aligned_labels = torch.stack(aligned_labels, dim=0)
    return aligned_labels, masks


def align_feats(feats):
    '''
        输入：feats 为list [[N, C], [N, C]]
        输出：feats 为[B, N, C], mask 为[B, N]
    '''
    batch_size = len(feats)
    dtype = feats[0].dtype
    device = feats[0].device

    max_len = max([feats_pb.shape[0] for feats_pb in feats])

    aligned_feats = list()
    masks = torch.zeros([batch_size, max_len], dtype=dtype, device=device)
    
    for batch_idx in range(batch_size):
        feats_pb = feats[batch_idx]
        cur_len = feats_pb.shape[0]
        aligned_feats_pi = F.pad(
            feats_pb,
            (0, 0, 0, max_len-cur_len),
            mode='constant',
            value=0
        )
        aligned_feats.append(aligned_feats_pi)

        masks[batch_idx, :cur_len] = 1
    aligned_feats = torch.stack(aligned_feats, dim=0)
    return aligned_feats, masks


def align_tokens(tokens_ids, device, dtype):
    batch_size = len(tokens_ids)

    max_length = max([tokens_ids_pi.shape[1] for tokens_ids_pi in tokens_ids])

    aligned_tokens_ids = list()
    masks = torch.zeros([batch_size, max_length], dtype=dtype, device=device)
    for batch_idx, tokens_ids_pb in enumerate(tokens_ids):
        cur_length = tokens_ids_pb.shape[1]
        aligned_tokens_ids_pi = F.pad(
            tokens_ids_pb,
            (0, max_length-cur_length, 0, 0),
            mode='constant',
            value=0
        )
        aligned_tokens_ids.append(aligned_tokens_ids_pi)
        masks[batch_idx, :cur_length] = 1   
    aligned_tokens_ids = torch.cat(aligned_tokens_ids, dim=0)
    return aligned_tokens_ids, masks


def tokenize(tokenizer, token_batch, max_token=50):
    tokens_ids = []
    for sentence in token_batch:
        tokens = tokenizer(''.join(sentence), return_tensors="pt")
        tokens_ids.append(tokens['input_ids'][:, :max_token])
    aligned_tokens_ids, masks = align_tokens(tokens_ids, tokens_ids[0].device, tokens_ids[0].dtype)
    return  aligned_tokens_ids, masks