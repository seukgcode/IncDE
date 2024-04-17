from src.utils import *
from torch_scatter import scatter_add, scatter_mean, scatter_max

class BaseModel(nn.Module):
    def __init__(self, args, kg) -> None:
        super(BaseModel, self).__init__()
        self.args = args
        self.kg = kg

        """ initialize the entity and relation embeddings for the first snapshot """
        self.ent_embeddings = nn.Embedding(self.kg.snapshots[0].num_ent, self.args.emb_dim).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.snapshots[0].num_rel, self.args.emb_dim).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)

        """ loss function """
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")

    def reinit_param(self):
        """ Reinit all model parameters """
        for name, param in self.named_parameters():
            if param.requires_grad:
                xavier_normal_(param)

    def expand_embedding_size(self):
        """ init entity and relation embeddings for next snapshot """
        """ Since preparing the next snapshot, snapshot should be added one. """
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, self.args.emb_dim).to(self.args.device).double()
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim).to(self.args.device).double()
        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return deepcopy(ent_embeddings), deepcopy(rel_embeddings)

    def switch_snapshot(self):
        """ After the training process of a snapshot, prepare for next snapshot """
        pass

    def pre_snapshot(self):
        """ Process before training on a snapshot """
        pass

    def epoch_post_processing(self, size=None):
        """ Post process after a training iteration """
        pass

    def snapshot_post_processing(self):
        """ Post after training on a snapshot """
        pass

    def store_old_parameters(self):
        """ Store the learned model after training on a snapshot """
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def initialize_old_data(self):
        """ initialize the storage of old parameters """
        for name, param in self.named_parameters():
            if param.requires_grad:
                name = name.replace('.', '_')
                self.register_buffer(f'old_data_{name}', param.data.clone())

    def embedding(self, stage=None):
        """ stage: Train, Valid, Test """
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def new_loss(self, head, rel, tail=None, label=None):
        """ return loss of new facts """
        return self.margin_loss(head, rel, tail, label) / head.size(0)

    def margin_loss(self, head, rel, tail, label=None):
        """ Pair wise margin loss: L1-norm (h + r - t) """
        ent_embeddings, rel_embeddings = self.embedding('Train')
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, rel)
        t = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def split_pn_score(self, score, label):
        """
        split postive triples and negtive triples
        :param score: scores of all facts
        :param label: postive facts: 1, negtive facts: -1
        """
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def score_fun(self, h, r, t):
        """ Score function: L1-norm (h + r - t) """
        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t = self.norm_ent(t)
        return torch.norm(h + r - t, 1, -1)

    def predict(self, head, relation, stage='Valid'):
        """ Score all candidate facts for evaluation """
        if stage != 'Test':
            num_ent = self.kg.snapshots[self.args.snapshot_valid].num_ent
        else:
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
        ent_embeddings, rel_embeddings = self.embedding(stage)
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t_all = ent_embeddings[:num_ent]
        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t_all = self.norm_ent(t_all)

        """ h + r - t """
        pred_t = h + r
        score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)
        score = torch.sigmoid(score)
        return score

    def norm_rel(self, r):
        return F.normalize(r, 2, -1)

    def norm_ent(self, e):
        return F.normalize(e, 2, -1)