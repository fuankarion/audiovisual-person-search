

audio_reid_input = {
    # input files
    'csv_train_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_val_augmented.csv',
    'csv_test_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_test_augmented.csv',

    'csv_identities_train': '/home/alcazajl/Dataset/APES/ava-person-search-train-v3.csv',
    'csv_identities_val': '/home/alcazajl/Dataset/APES/ava-person-search-val-v2.csv',

    # Data config
    'audio_dir': '/Dataset/ava_active_speaker/instance_wavs_time/',
    'models_out': '/home/alcazajl/Models/APES/audio/'
}


visual_reid_inputs = {
    # input files
    'csv_train_full': '/media/SSD2/jcleon/AVA/csv/full/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/media/SSD2/jcleon/AVA/csv/full/ava_activespeaker_val_augmented.csv',

    'csv_identities_train': '/home/jcleon/test_reid/ava-person-search-train-v3.csv',
    'csv_identities_val': '/home/jcleon/test_reid/ava-person-search-val-v2.csv',

    # Data config
    'video_dir': '/Dataset/ava_active_speaker/instance_crops_time/',
    'models_out': '/home/alcazajl/Models/APES/video'
}


audio_visual_reid_inputs = {
    # input files
    'csv_train_full': '/media/SSD2/jcleon/AVA/csv/full/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/media/SSD2/jcleon/AVA/csv/full/ava_activespeaker_val_augmented.csv',

    'csv_identities_train': '/home/jcleon/test_reid/ava-person-search-train-v3.csv',
    'csv_identities_val': '/home/jcleon/test_reid/ava-person-search-val-v2.csv',

    # Data config
    'audio_dir': '/Dataset/ava_active_speaker/instance_wavs_time/',
    'video_dir': '/Dataset/ava_active_speaker/instance_crops_time/',
    'models_out': '/home/alcazajl/Models/APES/av'
}
