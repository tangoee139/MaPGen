import os, json
import torch
from torch import nn
import argparse
import numpy as np
from modules.metrics import compute_scores
from modules.trainer import Trainer
from models.qwen2_mrg import Qwen2MRG
import torch.distributed as dist
from dataset import create_dataset 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
#from transformers import BertTokenizer 
from transformers import AutoTokenizer
import pandas as pd

os.environ['TOKENIZERS_PARALLELISM'] = 'True'

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--ann_path', type=str, default='data/mimic_cxr/mimic_annotation_promptmrg.json', help='the path to the directory containing the data.')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr','mri'], help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')
    parser.add_argument('--sample_n', type=int, default=25, help='the sample number of train_set and test_set if 0 not sample')

    # Model settings 
    parser.add_argument('--load_pretrained', type=str, default=None, help='pretrained path if any')

    # Sample related
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--gen_max_len', type=int, default=150, help='the maximum token length for text generation.')
    parser.add_argument('--gen_min_len', type=int, default=100, help='the minimum token length for text generation.')

    # Trainer settings
    #parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/promptmrg', help='the path to save the models.')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1', help='the metric to be monitored.')

    # Optimization
    parser.add_argument('--init_lr', type=float, default=5e-5, help='.')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='.')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help='.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='the weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed     training')
    parser.add_argument('--device', default='cuda')


    parser.add_argument('--num_label', type=int, default=300, help='only for mri')
    

    #modified by anonymous
    parser.add_argument('--num_latents', type=int, default=64, help='only for mri')
    parser.add_argument('--cls_weight', type=float, default=1, help='Loss weight of classification branch.')
    parser.add_argument('--enable_3seq', type=bool, default=False, help='only for mri')
    parser.add_argument('--condition', type=bool, default=True, help='only for mri')
    parser.add_argument('--mask_prompt', type=bool, default=True, help='only for mri')
    #parser.add_argument('--mask_prompt_ratio', type=float, default=0.3, help='only for mri')
    parser.add_argument('--retain_ratio', type=float, default=0.6, help='only for mri')
    parser.add_argument('--drop_ratio', type=float, default=0.4, help='only for mri')
    parser.add_argument('--replace_ratio', type=float, default=0.0, help='only for mri')
    parser.add_argument('--model_type', type=str, default='/mnt/nvme_share/XXX/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c291d6fce4804a1d39305f388dd32897d1f7acc4/', help='only for mri')
    parser.add_argument('--nproc_per_node', type=int, default=1, help='only for mri')
    parser.add_argument('--total_steps', type=int, default=1, help='only for mri')


    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse_agrs()
    if (not os.path.exists(args.save_dir)) and utils.get_rank()==0:
        os.makedirs(args.save_dir)
    args_file_path = args.save_dir + '/args.json'
    # 保存参数为 JSON 文件
    with open(args_file_path, "w") as file:
        json.dump(vars(args), file, indent=4)
    print(args)
    utils.init_distributed_mode(args) # from blip
    device = torch.device(args.device)

    # fix random seeds
    seed = args.seed + utils.get_rank() # from blip
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # create tokenizer
    if args.dataset_name == 'mri':
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    # tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])
    # tokenizer.add_tokens(['\n','髌','腘','髁','嵴','襞','骺'])



    #### Dataset #### 
    print("Creating dataset...")
    train_dataset, val_dataset, test_dataset = create_dataset('generation_%s'%args.dataset_name, args)
    print('number of training samples: %d'%len(train_dataset))
    print('number of validation samples: %d'%len(val_dataset))
    print('number of testing samples: %d'%len(test_dataset))

    args.total_steps = (len(train_dataset)/args.batch_size/args.nproc_per_node)*args.epochs


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
        samplers = [samplers[0], None, None]
    else:
        samplers = [None, None, None]

    train_dataloader, val_dataloader, test_dataloader = create_loader([train_dataset, val_dataset, test_dataset], samplers, batch_size=[args.batch_size, 16, 16], num_workers=[4,4,4], is_trains=[True, False, False], collate_fns=[None, None, None]) 

    model = Qwen2MRG(args, tokenizer)
    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print("load checkpoint from {}".format(args.load_pretrained))

    # get function handles of loss and metrics
    criterion_cls = nn.CrossEntropyLoss(reduction='none')
    metrics = compute_scores

    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
   

    # build trainer and start to train
    trainer = Trainer(model, criterion_cls, metrics, args, train_dataloader, val_dataloader, test_dataloader, device, utils.is_main_process)
    trainer.train()

if __name__ == '__main__':
    main()
