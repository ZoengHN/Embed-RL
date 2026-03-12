import os
import json
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Iterator
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class InterleavedSubBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        sub_batch_size: Optional[int] = None,
        num_sub_batches_per_batch: int = 8,
        max_video_sub_batches_per_batch: Optional[int] = 2,
        probabilities: Optional[List[float]] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.max_video_sub_batches_per_batch = max_video_sub_batches_per_batch

        self.global_batch_size = batch_size * num_replicas

        if sub_batch_size is None or sub_batch_size == 0:
            self.sub_batch_size = self.global_batch_size // num_sub_batches_per_batch
            if self.sub_batch_size == 0:
                self.sub_batch_size = self.global_batch_size
        else:
            self.sub_batch_size = sub_batch_size

        if self.global_batch_size % self.sub_batch_size != 0:
            self.sub_batch_size = self.global_batch_size // num_sub_batches_per_batch

        self.num_sub_batches = self.global_batch_size // self.sub_batch_size

        if not hasattr(dataset, 'cum_sizes'):
            raise ValueError("Dataset must have cum_sizes attribute (ConcatDataset)")

        self._build_dataset_indices()

        self.dataset_names = list(self.dataset_to_indices.keys())

        self.video_dataset_indices = []
        self.image_dataset_indices = []
        for i, name in enumerate(self.dataset_names):
            if 'llavahound' in name.lower():
                self.video_dataset_indices.append(i)
            else:
                self.image_dataset_indices.append(i)

        if probabilities is None:
            if hasattr(dataset, 'sub_dataset_weights') and len(dataset.sub_dataset_weights) > 0:
                weighted_sizes = []
                for i, (name, indices) in enumerate(self.dataset_to_indices.items()):
                    weight = dataset.sub_dataset_weights[i]
                    num_samples = len(indices)
                    weighted_sizes.append(weight * num_samples)

                total_weighted = sum(weighted_sizes)
                self.probabilities = [w / total_weighted for w in weighted_sizes]
            else:
                total_samples = sum(len(indices) for indices in self.dataset_to_indices.values())
                self.probabilities = [
                    len(indices) / total_samples
                    for indices in self.dataset_to_indices.values()
                ]
        else:
            if len(probabilities) != len(self.dataset_to_indices):
                raise ValueError(f"Number of probabilities ({len(probabilities)}) must match "
                               f"number of datasets ({len(self.dataset_to_indices)})")
            prob_sum = sum(probabilities)
            self.probabilities = [p / prob_sum for p in probabilities]

    def _build_dataset_indices(self):
        self.dataset_to_indices = {}

        if not hasattr(self.dataset, 'sub_dataset_names'):
            num_datasets = len(self.dataset.cum_sizes)
            sub_names = [f"dataset_{i}" for i in range(num_datasets)]
        else:
            sub_names = self.dataset.sub_dataset_names

        cum_sizes = self.dataset.cum_sizes
        prev_end = 0

        for i, name in enumerate(sub_names):
            start = prev_end
            end = cum_sizes[i]
            prev_end = end

            indices = list(range(start, end))
            self.dataset_to_indices[name] = indices

    def _get_dataset_iterators(self, rng: random.Random) -> Dict[str, Iterator[int]]:
        iterators = {}

        for name, indices in self.dataset_to_indices.items():
            if self.shuffle:
                def make_reshuffling_cycle_iter(idx_list, random_gen):
                    while True:
                        current_indices = idx_list.copy()
                        random_gen.shuffle(current_indices)
                        for idx in current_indices:
                            yield idx

                iterators[name] = make_reshuffling_cycle_iter(indices, rng)
            else:
                def make_cycle_iter(idx_list):
                    while True:
                        for idx in idx_list:
                            yield idx
                iterators[name] = make_cycle_iter(indices)

        return iterators

    def _sample_dataset_sequence(
        self,
        rng: np.random.Generator,
        num_samples: int
    ) -> List[int]:
        num_dataset_choices = (num_samples + self.sub_batch_size - 1) // self.sub_batch_size

        if self.max_video_sub_batches_per_batch is None or len(self.video_dataset_indices) == 0:
            dataset_indices = rng.choice(
                len(self.dataset_names),
                size=num_dataset_choices,
                p=self.probabilities
            )
        else:
            dataset_indices = []

            num_sub_batches_per_global_batch = self.global_batch_size // self.sub_batch_size

            num_global_batches = (num_dataset_choices + num_sub_batches_per_global_batch - 1) // num_sub_batches_per_global_batch

            for _ in range(num_global_batches):
                batch_indices = []
                video_count = 0

                for _ in range(num_sub_batches_per_global_batch):
                    if len(batch_indices) >= num_dataset_choices:
                        break

                    selected_idx = int(rng.choice(len(self.dataset_names), p=self.probabilities))

                    if selected_idx in self.video_dataset_indices:
                        if video_count < self.max_video_sub_batches_per_batch:
                            batch_indices.append(selected_idx)
                            video_count += 1
                        else:
                            if len(self.image_dataset_indices) > 0:
                                image_probs = np.array([
                                    self.probabilities[i] for i in self.image_dataset_indices
                                ])
                                image_probs = image_probs / image_probs.sum()

                                selected_image_idx = rng.choice(
                                    len(self.image_dataset_indices),
                                    p=image_probs
                                )
                                batch_indices.append(self.image_dataset_indices[selected_image_idx])
                            else:
                                batch_indices.append(selected_idx)
                    else:
                        batch_indices.append(selected_idx)

                dataset_indices.extend(batch_indices)

            dataset_indices = np.array(dataset_indices[:num_dataset_choices])

        sequence = []
        for dataset_idx in dataset_indices:
            sequence.extend([int(dataset_idx)] * self.sub_batch_size)

        return sequence[:num_samples]

    def __iter__(self) -> Iterator[List[int]]:
        numpy_rng = np.random.default_rng(self.seed + self.epoch)
        python_rng = random.Random(self.seed + self.epoch)

        dataset_iterators = self._get_dataset_iterators(python_rng)

        dataset_sample_counts = {name: 0 for name in self.dataset_names}
        dataset_full_cycles = {name: 0 for name in self.dataset_names}
        dataset_sizes = {name: len(self.dataset_to_indices[name]) for name in self.dataset_names}

        total_samples = sum(len(indices) for indices in self.dataset_to_indices.values())

        if self.drop_last:
            num_global_batches = total_samples // self.global_batch_size
            samples_per_replica = num_global_batches * self.batch_size
        else:
            num_global_batches = (total_samples + self.global_batch_size - 1) // self.global_batch_size
            samples_per_replica = (total_samples + self.num_replicas - 1) // self.num_replicas

        total_global_samples = num_global_batches * self.global_batch_size
        dataset_sequence = self._sample_dataset_sequence(numpy_rng, total_global_samples)

        global_indices_to_dataset = {}
        for global_idx, dataset_idx in enumerate(dataset_sequence):
            global_indices_to_dataset[global_idx] = dataset_idx

        samples = []

        dataset_counters = {name: 0 for name in self.dataset_names}

        for global_batch_idx in range(num_global_batches):
            global_batch_start = global_batch_idx * self.global_batch_size

            for sub_batch_idx in range(self.num_sub_batches):
                sub_batch_start = global_batch_start + sub_batch_idx * self.sub_batch_size

                if sub_batch_start < len(dataset_sequence):
                    dataset_idx = global_indices_to_dataset[sub_batch_start]
                    dataset_name = self.dataset_names[dataset_idx]

                    samples_per_gpu_per_subbatch = self.sub_batch_size // self.num_replicas

                    for position_in_subbatch in range(self.sub_batch_size):
                        sample_idx = next(dataset_iterators[dataset_name])

                        owner_gpu = position_in_subbatch // samples_per_gpu_per_subbatch

                        if owner_gpu == self.rank:
                            samples.append(sample_idx)

                            dataset_counters[dataset_name] += 1
                            dataset_sample_counts[dataset_name] += 1

                            if dataset_sample_counts[dataset_name] > 0 and \
                               dataset_sample_counts[dataset_name] % dataset_sizes[dataset_name] == 0:
                                dataset_full_cycles[dataset_name] += 1

            if len(samples) >= samples_per_replica:
                break

        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        total_samples = sum(len(indices) for indices in self.dataset_to_indices.values())

        if self.drop_last:
            num_global_batches = total_samples // self.global_batch_size
            samples_per_replica = num_global_batches * self.batch_size
        else:
            samples_per_replica = (total_samples + self.num_replicas - 1) // self.num_replicas

        if self.drop_last:
            return samples_per_replica // self.batch_size
        else:
            return (samples_per_replica + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch