from torch.utils.data import Dataset
from src.utils import *

class TrainDatasetMarginLoss(Dataset):
    def __init__(self, args, kg):
        super(TrainDatasetMarginLoss, self).__init__()
        self.args = args
        self.kg = kg
        self.facts, self.facts_new = self.build_facts()

    def __len__(self):
        if self.args.train_new:
            return len(self.facts_new[self.args.snapshot])
        else:
            return len(self.facts[self.args.snapshot])

    def __getitem__(self, index):
        if self.args.train_new:
            ele = self.facts_new[self.args.snapshot][index]
        else:
            ele = self.facts[self.args.snapshot][index]
        fact, label = ele['fact'], ele['label']

        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        ss_id = self.args.snapshot
        h, r, t = fact
        prob = 0.5

        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label

    def build_facts(self):
        """ build postive training data for each snapshot """
        facts, facts_new = [], []
        for ss_id in range(int(self.args.snapshot_num)):
            facts_, facts_new_ = [], []
            """ for LKGE and other baselines """
            for h, r, t in self.kg.snapshots[ss_id].train:
                facts_new_.append({'fact': (h, r, t), 'label': 1})
                facts_new_.append({'fact': (t, r + 1, h), 'label': 1})
            """ for retraining """
            for h, r, t in self.kg.snapshots[ss_id].train_all:
                facts_.append({'fact': (h, r, t), 'label': 1})
                facts_.append({'fact': (h, r + 1, t), 'label': 1})
            facts.append(facts_)
            facts_new.append(facts_new_)
        return facts, facts_new


class TestDataset(Dataset):
    def __init__(self, args, kg):
        super(TestDataset, self).__init__()
        self.args = args
        self.kg = kg

        self.valid, self.test = self.build_facts()

    def __len__(self):
        if self.args.valid:
            return len(self.valid[self.args.snapshot_valid])
        else:
            return len(self.test[self.args.snapshot_test])

    def __getitem__(self, index):
        if self.args.valid:
            element = self.valid[self.args.snapshot_valid][index]
        else:
            element = self.test[self.args.snapshot_test][index]
        fact, label = torch.LongTensor(element['fact']), element['label']
        """ The previous label is a set, and the length of the label can be unified here """
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label
    
    def get_label(self, label):
        """ for valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        if self.args.valid:
            y = np.zeros([self.kg.snapshots[self.args.snapshot_valid].num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.kg.snapshots[self.args.snapshot_test].num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


    def build_facts(self):
        """ build positive data """
        valid, test = [], []
        for ss_id in range(int(self.args.snapshot_num)):
            valid_, test_ = [], []
            if self.args.train_new:
                for (h, r, t) in self.kg.snapshots[ss_id].valid:
                    valid_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            else:
                for (h, r, t) in self.kg.snapshots[ss_id].valid_all:
                    valid_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            if self.args.train_new:
                for (h, r, t) in self.kg.snapshots[ss_id].valid:
                    valid_.append({'fact': (t, r + 1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r + 1)]})
            else:
                for (h, r, t) in self.kg.snapshots[ss_id].valid_all:
                    valid_.append({'fact': (t, r + 1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r + 1)]})
            
            for (h, r, t) in self.kg.snapshots[ss_id].test:
                test_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            for (h, r, t) in self.kg.snapshots[ss_id].test:
                test_.append({'fact': (t, r + 1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r + 1)]})
            valid.append(valid_)
            test.append(test_)
        return valid, test




# """ for test, move it to filedir outside """
# if __name__ == "__main__":
#     from src.parse_args import args
#     from src.data_load.KnowledgeGraph import KnowledgeGraph
#     """ Set data path """
#     if not os.path.exists(args.data_path):
#         os.mkdir(args.data_path)
#     args.data_path = args.data_path + args.dataset + "/"
#     """ Set device """
#     torch.cuda.set_device(int(args.gpu))
#     _ = torch.tensor([1]).cuda()
#     args.device = _.device

#     kg = KnowledgeGraph(args)
#     train_data = TrainDatasetMarginLoss(args, kg)
#     print(train_data.facts_new[4])