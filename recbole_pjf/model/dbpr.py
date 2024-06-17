import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class DBPR(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DBPR, self).__init__(config, dataset)
        self.neg_user_id = self.NEG_USER_ID = config['NEG_PREFIX'] + self.USER_ID
        self.embedding_size = config['embedding_size']
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.DIRECT_FIELD = config["DIRECT_FIELD"]

        # define layers and loss
        self.user_embedding = nn.Embedding(2 * self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(2 * self.n_items, self.embedding_size)

        self.loss = BPRLoss()
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item, direct):
        u_e = self.user_embedding(2 * user + direct - 1)
        i_e = self.item_embedding(2 * item + direct - 1)
        return torch.mul(u_e, i_e).sum(dim=1)

    def calculate_loss(self, interaction):
        pos_user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_user = interaction[self.NEG_USER_ID]
        direct = interaction[self.DIRECT_FIELD]

        score_pos = self.forward(pos_user, pos_item, direct)
        score_neg_i = self.forward(pos_user, neg_item, direct)
        # score_neg_u = self.forward(neg_user, pos_item, direct)

        loss = self.loss(score_pos, score_neg_i)
        # loss = self.loss(score_pos, score_neg_i) + self.loss(score_pos, score_neg_u)
        return loss

    def predict(self, interaction, direct = 0):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if direct == 0:
            direct = interaction[self.DIRECT_FIELD]
        return self.forward(user, item, direct)


