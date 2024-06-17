from collections import OrderedDict

import torch
import numpy as np
from recbole.trainer import Trainer
from recbole.utils import calculate_valid_score
from recbole.data.interaction import Interaction
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from tqdm import tqdm

from recbole_pjf.data.dataloader import RFullSortEvalDataLoader, RNegSampleEvalDataLoader
from recbole_pjf.enum_type import InteractDirect

class MultiDirectTrainer(Trainer):
    def __init__(self, config, model):
        super(MultiDirectTrainer, self).__init__(config, model)
        setattr(self.eval_collector.register,'rec.items', True)
    
    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        # Overall look at the bilateral recommendation task

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location = self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)
            
        self.model.eval()

        eval_data[0].bilateral_type = True
        eval_data[1].bilateral_type = True
        self.all_rec = torch.LongTensor().to(self.device)
        self.all_rec_rank = torch.LongTensor().to(self.device)

        if isinstance(eval_data[0], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[0].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        self.eval_direct = InteractDirect.SIDE_A
        struct_A = self.collect_bilateral_info(eval_data[0], eval_func, show_progress, self.eval_direct)
        result_A = self.evaluator.evaluate(struct_A)

        self.config.change_direction()
        if isinstance(eval_data[1], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[1].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        self.eval_direct = InteractDirect.SIDE_B
        struct_B = self.collect_bilateral_info(eval_data[1], eval_func, show_progress, self.eval_direct)
        result_B = self.evaluator.evaluate(struct_B)
        self.config.change_direction()

        arr = self.all_rec.cpu().numpy()
        uni, idx, count = np.unique(arr, return_index=True, return_counts=True,axis=0)
        dup_idx = []
        self.all_rec_rank = [i.item() for i in self.all_rec_rank]
        for i, c in enumerate(count):
            if c > 1:
                dup_idx.extend(self.all_rec_rank[idx[i]: idx[i] + c])
                
        unique_rec = torch.unique(self.all_rec, dim=0)
        unique_rec_num = unique_rec.shape[0]

        total_positive_pairs = struct_A.get('rec.topk').sum(dim=0)[-1]
        n_users_A = struct_A.get('rec.topk').shape[0]
        n_users_B = struct_B.get('rec.topk').shape[0]
        n_topk = self.config['topk'][0]

        # calculate CRecall 
        crecall = unique_rec_num / total_positive_pairs
        # calculate CPrecision
        cprecision = torch.tensor(unique_rec_num / n_topk / (n_users_A + n_users_B))
        # calculate SRecall
        srecall = torch.tensor(2 * (self.all_rec.shape[0] - unique_rec_num) / total_positive_pairs)
        # calculate SPrecision
        sprecision = torch.tensor(2 * (self.all_rec.shape[0] - unique_rec_num) / n_topk / (n_users_A + n_users_B))
        # calculate Rndcg
        rndcg = torch.tensor(result_A[f'ndcg@{n_topk}'] * n_users_A + result_B[f'ndcg@{n_topk}'] * n_users_B) / (n_users_A + n_users_B)

        result_dict = dict()
        result_dict['crecall'] = crecall
        result_dict['cprecision'] = cprecision
        result_dict['srecall'] = srecall
        result_dict['sprecision'] = sprecision
        result_dict[f'rndcg@{n_topk}'] = rndcg

        result_A.update(result_dict)
        result_B.update(result_dict)

        return result_A, result_B
    
        # import matplotlib.pyplot as plt
        # y_lables = [i / 10 for i in range(0,r)]
        # plt.yticks(y_lables, fontsize=15)
        # plt.xticks(fontsize=15)  # 默认字体大小为10
        # plt.yticks(fontsize=15)
        # plt.xlabel('Ranking Position', fontsize=15)
        # plt.ylabel('Count', fontsize=15)
        # plt.hist(dup_idx, bins = np.arange(min(dup_idx), max(dup_idx) + 2))
        # plt.savefig("figure.pdf")

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data

        inter_len = len(interaction)
        new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
        batch_size = len(new_inter)
        new_inter.update(self.item_tensor.repeat(inter_len))
        if batch_size <= self.test_batch_size:
            try:
                scores = self.model.predict(new_inter, self.eval_direct.value)
            except:
                scores = self.model.predict(new_inter)
        else:
            try:
                scores = self.model.predict(new_inter, self.eval_direct.value)
            except:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def collect_bilateral_info(self, eval_data, eval_func, show_progress, direct):
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data[:4])
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

            # get crecall infor
            uid_list = torch.LongTensor(batched_data[4])
            _, topk_idx = torch.topk(scores, self.config['topk'][0], dim=-1)
            pos_matrix = torch.zeros_like(scores, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            real_rec_positive = topk_idx[pos_idx == 1]

            rank_idx = torch.cat([torch.arange(1, self.config['topk'][0] + 1).to(self.device).unsqueeze(0)] * pos_idx.shape[0], dim=0)
            rank_idx = rank_idx[pos_idx == 1]

            uid = torch.repeat_interleave(uid_list.cpu(), pos_idx.cpu().sum(dim=1), dim=0)
            if direct == InteractDirect.SIDE_A:
                cur = torch.stack((uid, real_rec_positive.cpu()), dim=1).to(self.device)
            elif direct == InteractDirect.SIDE_B:
                cur = torch.stack((real_rec_positive.cpu(), uid), dim=1).to(self.device)
                
            self.all_rec_rank = torch.cat((self.all_rec_rank, rank_idx), dim=0)
            self.all_rec = torch.cat((self.all_rec, cur), dim=0)

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        return struct

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result_all = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_g_score = calculate_valid_score(valid_result_all[0], self.valid_metric)
        valid_j_score = calculate_valid_score(valid_result_all[1], self.valid_metric)
        valid_score = (valid_g_score + valid_j_score) / 2

        valid_result = OrderedDict()
        valid_result['For side A'] = valid_result_all[0]
        valid_result['\nFor side B'] = valid_result_all[1]
        return valid_score, valid_result
