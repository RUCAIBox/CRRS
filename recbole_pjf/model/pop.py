import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class Pop(GeneralRecommender):
    r"""Pop is an fundamental model that always recommend the most popular item."""
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(Pop, self).__init__(config, dataset)

        self.item_cnt = torch.zeros(
            self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.user_cnt = torch.zeros(
            self.n_users, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.max_cnt_i = None
        self.max_cnt_u = None
        self.DIRECT_FIELD = config['DIRECT_FIELD']
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["item_cnt", "max_cnt_i", "user_cnt", "max_cnt_u"]

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        self.item_cnt[item, :] = self.item_cnt[item, :] + 1
        self.user_cnt[user, :] = self.user_cnt[user, :] + 1

        self.max_cnt_i = torch.max(self.item_cnt, dim=0)[0]
        self.max_cnt_u = torch.max(self.user_cnt, dim=0)[0]

        return torch.nn.Parameter(torch.zeros(1)).to(self.device)

    def predict(self, interaction, direct=0):
        item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        if direct == 0:
            direct = interaction[self.DIRECT_FIELD]
        result_i = torch.true_divide(self.item_cnt[item, :], self.max_cnt_i)
        result_u = torch.true_divide(self.user_cnt[user, :], self.max_cnt_u)
        return result_i.squeeze(-1) * (direct == 1) + result_u.squeeze(-1) * (direct == 2)

    def full_sort_predict(self, interaction):
        batch_user_num = interaction[self.USER_ID].shape[0]
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)
        return result.view(-1)