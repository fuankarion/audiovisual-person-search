import core.losses as ls

import torch.optim as optim
import core.models as mreid


triplet_loss_optimization_params_audio = {
    # Net Arch
    'backbone': mreid.triplet_reid_resnet_18_audio,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': ls.TripletLoss(),
    'learning_rate': 5e-4,
    'epochs': 120,
    'step_size': 50,
    'gamma': 0.1,

    # Hardware Stuff
    'batch_size': 128,
    'threads': 4
}

triplet_loss_optimization_params_video = {
    # Net Arch
    'backbone': mreid.triplet_reid_resnet_18_visual,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': ls.TripletLoss(),
    'learning_rate': 3e-4,
    'epochs': 70,
    'step_size': 30,
    'gamma': 0.1,

    # Hardware Stuff
    'batch_size': 128,
    'threads': 4
}

triplet_loss_optimization_params_audio_video = {
    # Net Arch
    'backbone': mreid.triplet_reid_resnet_18_audio_video,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': ls.TripletLoss(),
    'learning_rate': 5e-5,
    'epochs': 120,
    'step_size': 45,
    'gamma': 0.1,

    # Hardware Stuff
    'batch_size': 128,
    'threads': 4
}
