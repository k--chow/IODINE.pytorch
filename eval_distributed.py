import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import pdb

import h5py
import numpy as np
from PIL import Image
from sacred import Experiment, cli_option
from lib.datasets import ds
from lib.datasets import StaticHdF5Dataset
from lib.model import net
from lib.model import IODINE
from lib.utils import mvn, _softplus_to_std, std_mvn, gmm_loglikelihood
from lib.metrics import adjusted_rand_index
from pathlib import Path
import shutil
from tqdm import tqdm 

@cli_option('-r','--local_rank')
def local_rank_option(args, run):
    run.info['local_rank'] = args

ex = Experiment('EVAL', ingredients=[ds, net], additional_cli_options=[local_rank_option])

torch.set_printoptions(threshold=10000, linewidth=300)

@ex.config
def cfg():
    test = {
            'output_size': [3,64,64],
            'mode': 'test',
            'num_workers': 8,
            'out_dir': '',
            'checkpoint_dir': '',
            'checkpoint': '',
            'experiment_name': 'NAME_HERE',
        }

# @ex.capture ??
def restore_from_checkpoint(test, checkpoint, local_rank):
    state = torch.load(checkpoint)
    
    model = IODINE(batch_size=1)

    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model.load_state_dict(state['model'], strict=True)
    print(f'loaded {checkpoint}')
    return model


@ex.capture
def do_eval(test, local_rank, seed):
    # Fix random seed
    print(f'setting random seed to {seed}')
    # Auto-set by sacred
    # torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = 'cuda:{}'.format(local_rank)
    torch.cuda.set_device(local_rank)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Data
    #te_dataset = StaticHdF5Dataset(d_set=test['mode'])
    te_dataset = torchvision.datasets.CIFAR10('./', download=True, train=False, transform=transform)
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=1, shuffle=True, num_workers=test['num_workers'], drop_last=True)
    checkpoint = Path(test['checkpoint_dir'], test['checkpoint'])
    checkpoint_dir = Path(test['out_dir'], 'weights')
    # TODO: Assert nproc = 1
    out_dir = Path(test['out_dir'], 'results', test['experiment_name'], checkpoint.stem + '.pth' + f'-seed={seed}')
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    print(f'saving results in {out_dir}')
    model = restore_from_checkpoint(test, checkpoint, local_rank)
    model.eval()
    
    if test['use_geco']:
        model_geco = GECO(test['geco_reconstruction_target'], test['geco_ema_alpha'])
    else:
        model_geco = None
        
    checkpoint = checkpoint_dir / test['checkpoint']
    map_location = {'cuda:0': local_rank}
    state = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(state['model'])
    #model_opt.load_state_dict(state['model_opt'])
    step = state['step']
    checkpoint_step = step
   
    
    num_images = 320
    all_ARI = []
    all_pred_masks = []

    total_images = 0
    for i,batch in enumerate(tqdm(te_dataloader)):
        
        if total_images >= num_images:
            break
        imgs = batch[0].to('cuda')
        outs = model(imgs, model_geco, step)
        with torch.no_grad():
            
            p_z = std_mvn(shape=[1 * 4, 64], device=imgs.device)# 64
            
            lamda = outs['lamda']
            loc_z, sp_z = lamda.chunk(2, dim=1)
            loc_z, sp_z = loc_z.contiguous(), sp_z.contiguous()
            q_z = mvn(loc_z, sp_z)

            z = q_z.sample()
            # for NLL 
            x_loc, mask_logits = model.module.image_decoder(z)  #[N*K, C, H, W]
            x_loc = x_loc.view(1, 4, 3, 32, 32) #N*K pos 2 

            # softmax across slots
            mask_logits = mask_logits.view(1, 4, 1, 32, 32)
            mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)

            # NLL [batch_size, 1, H, W]
            log_var = (2 * model.module.gmm_log_scale).to(imgs.device)
            nll, ll_outs = gmm_loglikelihood(imgs, x_loc, log_var, mask_logprobs)
            
            log_pz = torch.sum(p_z.log_prob(z)) # sum also!
            log_qzx = torch.sum(q_z.log_prob(z)) # sum also!
            
            pdb.set_trace()
            """
            true_masks = batch['masks'].to('cuda')
            H = imgs.shape[2]
            W = imgs.shape[3]
            pred_mask_logs = outs['masks']
            pred_masks = pred_mask_logs.exp()

            resized_masks = []
            pred_masks_ = pred_masks.data.cpu().numpy()
            all_pred_masks += [pred_masks_]

            for i in range(pred_masks.shape[1]):
                PIL_mask = Image.fromarray(pred_masks_[0,i,0], mode="F")
                PIL_mask = PIL_mask.resize((192,192))
                resized_masks += [np.array(PIL_mask)[...,None]]
            resized_masks = np.stack(resized_masks)[None]  # [1,K,H,W,C]
            resized_masks = np.transpose(resized_masks, (0,1,4,2,3))

            pred_masks = torch.from_numpy(resized_masks).to(true_masks.device)
            ari, pred_mask_image = adjusted_rand_index(true_masks, pred_masks)

            ari = ari.data.cpu().numpy().reshape(-1)
            all_ARI += [ari]
            """
        
    print('mean ARI: {}, std dev: {}'.format(np.mean(all_ARI), np.std(all_ARI)))
    all_pred_masks = np.stack(all_pred_masks)
    np.save(out_dir / 'masks.npy', all_pred_masks)

@ex.automain
def run(_run, seed):
    do_eval(local_rank=_run.info['local_rank'], seed=seed)
