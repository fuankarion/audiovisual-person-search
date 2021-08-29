import os
import sys

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import core.io_config as io_conf
import core.exp_config as exp_conf
from core.util import set_up_log_and_ws_out
from core.optimization import optimize_tripplet_loss
from core.dataset import TripletLossDatasetAudio


if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(69)
    torch.backends.cudnn.deterministic = True

    cuda_device_number = str(sys.argv[1])
    clip_size = int(sys.argv[2])

    experiment_name = 'tripplet_loss_audio_across_clip_'+str(clip_size)
    io_config = io_conf.audio_reid_input
    opt_config = exp_conf.triplet_loss_optimization_params_audio

    # io config
    log, target_models = set_up_log_and_ws_out(io_config['models_out'],
                                               opt_config,
                                               experiment_name)

    # cuda config
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')

    # backbone
    backbone = opt_config['backbone'](pretrained=True)
    backbone = backbone.to(device)

    # Optimization config
    criterion = opt_config['criterion']
    optimizer = opt_config['optimizer'](backbone.parameters(),
                                        lr=opt_config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])

    audio_train_path = os.path.join(io_config['audio_dir'], 'train')
    audio_val_path = os.path.join(io_config['audio_dir'], 'val')

    d_train = TripletLossDatasetAudio(audio_train_path, io_config['csv_train_full'],
                                      io_config['csv_identities_train'], clip_size, across=True)
    d_val = TripletLossDatasetAudio(audio_val_path, io_config['csv_val_full'],
                                    io_config['csv_identities_val'], clip_size, across=False)

    dl_train = DataLoader(d_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'])
    dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                        shuffle=True, num_workers=opt_config['threads'])

    model = optimize_tripplet_loss(0.1, backbone, dl_train, dl_val, device,
                                   criterion, optimizer, scheduler,
                                   num_epochs=opt_config['epochs'],
                                   models_out=target_models, log=log)
