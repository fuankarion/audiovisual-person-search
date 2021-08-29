import os
from PIL import Image
from scipy.io import wavfile
from core.util import Logger
import numpy as np
import python_speech_features
import csv
import json


def _pil_loader(path, target_size):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.resize(target_size)
            return img.convert('RGB')
    except OSError as e:
        return Image.new('RGB', target_size)


def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list


def _generate_mel_spectrogram(audio_clip, sample_rate):
    mfcc = zip(*python_speech_features.mfcc(audio_clip, sample_rate))
    audio_features = np.stack([np.array(i) for i in mfcc])
    audio_features = np.expand_dims(audio_features, axis=0)
    return audio_features


def _fit_audio_clip(audio_clip, sample_rate, video_clip_lenght):
    target_audio_length = (1.0/27.0)*sample_rate*video_clip_lenght
    pad_required = int((target_audio_length-len(audio_clip))/2)
    if pad_required > 0:
        #print('Y',pad_required,len(audio_clip))
        audio_clip = np.pad(audio_clip, pad_width=(pad_required, pad_required),
                            mode='reflect')
    # TODO why some audio clips are larger?
    if pad_required < 0:
        audio_clip = audio_clip[-1*pad_required:pad_required]

    return audio_clip


def load_av_clip_from_metadata(clip_meta_data, frames_source, audio_source,
                               audio_offset, target_size):

    video_data = load_clip_only_from_metadata(clip_meta_data, frames_source, target_size)
    audio_features = load_audio_only_from_metadata(clip_meta_data, audio_source, audio_offset)
    return video_data, audio_features


def load_clip_only_from_metadata(clip_meta_data, frames_source, target_size):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    entity_id = clip_meta_data[0][0]

    selected_frames = [os.path.join(frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    video_data = [_pil_loader(p, target_size) for p in selected_frames]

    return video_data


def load_audio_only_from_metadata(clip_meta_data, audio_source, audio_offset):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    midle = int(len(clip_meta_data) / 2)
    mid_meta = clip_meta_data[midle]

    if mid_meta[-1] == 0: # person not speaking
        entity_id = clip_meta_data[0][0]
        audio_file = os.path.join(audio_source, entity_id+'.wav')
        sample_rate, audio_data = wavfile.read(audio_file)
        audio_clip = np.zeros((int(sample_rate*(len(clip_meta_data)/25))))
        audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(ts_sequence))
        audio_features = _generate_mel_spectrogram(audio_clip, sample_rate)
        return audio_features

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    audio_file = os.path.join(audio_source, entity_id+'.wav')
    sample_rate, audio_data = wavfile.read(audio_file)

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    #TODO some audio samples seem shorther than they should be
    if len(audio_clip) < sample_rate*(2/25):
        audio_clip = np.zeros((int(sample_rate*(len(clip_meta_data)/25))))

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(ts_sequence))
    audio_features = _generate_mel_spectrogram(audio_clip, sample_rate)

    return audio_features


def load_frame_from_metadata(clip_meta_data, frames_source, target_size,
                             pil_backend=True):
    entity_id = clip_meta_data[0]
    ts = clip_meta_data[1]
    target_frame = os.path.join(frames_source, entity_id, ts+'.jpg')
    frame = _pil_loader(target_frame, target_size)
    return frame
