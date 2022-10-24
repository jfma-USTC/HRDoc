import os
import torch
from ..utils.vocab import TypeVocab, RelationVocab
from transformers import BertTokenizer, BertModel
from .extractor.model import Model as Extractor

device = torch.device('cuda')
valid_batch_size = 1
valid_num_workers = 0
valid_data_path = "pdf_parser/acl_stage2.json"
ly_vocab = TypeVocab()
re_vocab = RelationVocab()
extractor = Extractor().eval()
tokenizer = BertTokenizer.from_pretrained("models/English_base_cased")
bert = BertModel.from_pretrained("models/English_base_cased").eval()
in_dim = 64
encoder_layers = [1, 1]
encoder_dim = 128
scale = 1.0
pool_size = (3,3)
word_dim = 768
embed_dim = 128
feat_dim = 128
lm_state_dim = 128
proj_dim = 128
cover_kernel = 7
base_lr = 0.0005
min_lr = 1e-6
weight_decay = 0
num_epochs = 30
sync_rate = 20
valid_epoch = 1
log_sep = 5
cache_nums = 1000
work_dir = 'strcut_recover/experiment'