from recbole.sampler.sampler import AbstractSampler, Sampler
import numpy as np
import torch
import random

class PJFSampler(Sampler):
    def __init__(self, phases, datasets, distribution="uniform", alpha=1.0):
        # self.count_max = [0, 0]
        # self.direct = 0
        super().__init__(phases, datasets, distribution=distribution, alpha=alpha)

    def get_pos_used_ids(self, direct):
        """
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        self.direct = direct
        used_item_id = dict()
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid, d, label in zip(
                dataset.inter_feat[self.uid_field].numpy(),
                dataset.inter_feat[self.iid_field].numpy(),
                dataset.inter_feat['direct'].numpy(),
                dataset.inter_feat['label'].numpy(),
            ):
                if d == direct and int(label) == 1:
                    cur[uid].add(iid)
            last = used_item_id[phase] = cur

        for i in used_item_id['train']:
            if len(i) == 0:
                i.add(-1)

        return used_item_id['train']

    def pos_sample_by_user_ids(self, user_ids, item_ids, num):
        key_ids = np.array(user_ids)
        total_num = len(key_ids) * num
        value_ids = np.zeros(total_num, dtype=np.int64)
        key_ids = np.tile(key_ids, num)
        sample_func = np.vectorize(lambda s: random.sample(s, 1)[0])
        value_ids = np.array(sample_func(self.pos_used_ids))
        return torch.tensor(value_ids[user_ids])

    def sample_by_key_ids(self, key_ids, num):
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array(
                    [
                        i
                        for i, used, v in zip(
                            check_list,
                            self.used_ids[key_ids[check_list]],
                            value_ids[check_list],
                        )
                        if v in used
                    ]
                )
        return torch.tensor(value_ids)

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(user_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f"user_id [{user_id}] not exist.")
