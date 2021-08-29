import os
import random
import numpy as np

import core.io as io
import core.clip_utils as cu

from torch.utils import data


class CachedAVSourceReId(data.Dataset):
    def __init__(self):
        # Cached data
        self.entity_data = {}
        self.identity_data = {}
        self.entity_to_identity = {}

        # Reproducibilty
        random.seed(42)
        np.random.seed(0)

    def _cache_identity_meta_data(self, csv_file_path, entities_in_set):
        csv_data = io.csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        valid_entities = set()

        for csv_row in csv_data:
            entity_id = csv_row[0]
            video_id = csv_row[1]
            identity = csv_row[2]

            if identity == 'EXTRA_OR_AMBIGUOUS':
                continue

            k = entity_id.rfind('_')
            entity_id = entity_id[:k] + ":" + entity_id[k+1:]
            # this is a tad redundant, but keeps the code simple
            if (video_id, entity_id) not in entities_in_set:
                continue

            if identity not in self.identity_data.keys():
                self.identity_data[identity] = []
            if entity_id not in self.entity_to_identity.keys():
                self.entity_to_identity[entity_id] = identity

            self.identity_data[identity].append((video_id, entity_id))
            valid_entities.add(entity_id)

        return valid_entities

    def _cache_entity_data(self, csv_file_path, include_labels=[0, 1, 2]):
        entity_set = set()

        csv_data = io.csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]
            entity_id = csv_row[-3]
            timestamp = csv_row[1]
            label = int(csv_row[-2])
            if label not in include_labels:
                continue

            # Store minimal entity data
            minimal_entity_data = (entity_id, timestamp, label)
            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}
            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
                entity_set.add((video_id, entity_id))
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

        return entity_set

    def _entity_set_postprocessing(self, entity_set):
        print('Initial', len(entity_set))

        # filter out missing data on disk
        all_disk_data = set(os.listdir(self.video_root))
        for video_id, entity_id in entity_set.copy():
            if entity_id not in all_disk_data:
                entity_set.remove((video_id, entity_id))

            if entity_id not in self.entity_to_identity.keys():
                entity_set.remove((video_id, entity_id))

        print('Pruned not in disk, no identity', len(entity_set))
        return entity_set

    def get_negative_anchor(self, anchor_video, anchor_identity, entity_list,
                            across):
        negative_identity = anchor_identity
        negative_entity_id = None
        negative_video_id = None
        if across:
            while negative_identity == anchor_identity:
                negative_video_id, negative_entity_id = random.choice(entity_list)
                negative_identity = self.entity_to_identity[negative_entity_id]
        else:
            while negative_identity == anchor_identity and negative_video_id != anchor_video:
                negative_video_id, negative_entity_id = random.choice(entity_list)
                negative_identity = self.entity_to_identity[negative_entity_id]

        return negative_video_id, negative_entity_id

    def _identity_set_postprocessing(self, entity_set, valid_entities):
        print('Initial', len(entity_set))

        # filter out missing data
        all_disk_data = set(os.listdir(self.video_root))
        for video_id, entity_id in entity_set.copy():
            # remove if data not in disk
            if entity_id not in all_disk_data:
                entity_set.remove((video_id, entity_id))
                continue

            # remove if identity data is unknown
            if entity_id not in self.entity_to_identity.keys():
                entity_set.remove((video_id, entity_id))
                continue

        print('Pruned not in disk, no identity', len(entity_set))
        return entity_set

    def _identity_set_postprocessing_audio(self, entity_set, valid_entities):
        print('Initial', len(entity_set))

        # filter out missing data
        for video_id, entity_id in entity_set.copy():
            if entity_id not in self.entity_to_identity.keys():
                entity_set.remove((video_id, entity_id))
                continue

        print('Pruned not in disk, no identity', len(entity_set))
        return entity_set


class TripletLossDatasetAudio(CachedAVSourceReId):
    def __init__(self, audio_root, active_speaker_csv, identity_csv, clip_size,
                 across=False):
        super().__init__()
        # Data directories
        self.audio_root = audio_root
        self.half_clip_size = int((clip_size-1)/2)
        self.across = across

        # Gater Meta-data
        entity_set = self._cache_entity_data(active_speaker_csv, include_labels=[1])
        valid_entities = self._cache_identity_meta_data(identity_csv, entity_set)
        entity_set = self._identity_set_postprocessing_audio(entity_set, valid_entities)

        self.entity_list = list(entity_set)
        print('Fileterd entities', len(self.entity_list))

    def __len__(self):
        return int(len(self.entity_list)/1)

    def __getitem__(self, index):
        anchor_video_id, anchor_entity_id = self.entity_list[index]
        anchor_identity = self.entity_to_identity[anchor_entity_id]
        positive_video_id, positive_entity_id = random.choice(self.identity_data[anchor_identity])
        negative_video_id, negative_entity_id = self.get_negative_anchor(anchor_video_id, anchor_entity_id, self.entity_list, self.across)

        # Get meta-data
        anchor_metadata = self.entity_data[anchor_video_id][anchor_entity_id]
        positive_metadata = self.entity_data[positive_video_id][positive_entity_id]
        negative_metadata = self.entity_data[negative_video_id][negative_entity_id]

        anchor_target_index = random.randint(0, len(anchor_metadata)-1)
        positive_target_index = random.randint(0, len(positive_metadata)-1)
        negative_target_index = random.randint(0, len(negative_metadata)-1)

        anchor_clip = cu.generate_clip_meta(anchor_metadata, anchor_target_index, self.half_clip_size)
        positive_clip = cu.generate_clip_meta(positive_metadata, positive_target_index, self.half_clip_size)
        negative_clip = cu.generate_clip_meta(negative_metadata, negative_target_index, self.half_clip_size)

        anchor_audio_offset = float(anchor_metadata[0][1])
        positive_audio_offset = float(positive_metadata[0][1])
        negative_audio_offset = float(negative_metadata[0][1])

        # Load Actual Data
        anchor_data = io.load_audio_only_from_metadata(anchor_clip,
                                 self.audio_root, anchor_audio_offset)
        positive_data = io.load_audio_only_from_metadata(positive_clip,
                                 self.audio_root, positive_audio_offset)
        negative_data = io.load_audio_only_from_metadata(negative_clip,
                                 self.audio_root, negative_audio_offset)

        return np.float32(anchor_data), np.float32(positive_data), np.float32(negative_data)
