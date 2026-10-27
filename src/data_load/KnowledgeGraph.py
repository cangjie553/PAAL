from ..utils import *
from copy import deepcopy as dcopy
import numpy as np
import random


class KnowledgeGraph():
    def __init__(self, args):
        self.args = args

        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()


        self.enable_noise_injection = getattr(args, 'enable_noise_injection', False)
        self.enable_noise_train = getattr(args, 'enable_noise_train', True)
        self.enable_noise_eval = getattr(args, 'enable_noise_eval', True)
        self.noise_injection_mode = getattr(args, 'noise_injection_mode', 'replace')
        self.noise_category = getattr(args, 'noise_category', 'none')
        self.noise_type = getattr(args, 'noise_type', 'none')
        self.noise_ratio = getattr(args, 'noise_ratio', 0.1)
        self.drift_speed = getattr(args, 'drift_speed', 0.05)

        self.snapshots = {i: SnapShot(self.args) for i in range(int(self.args.snapshot_num))}
        self.load_data()

    def load_data(self):
        '''
        load data from all snapshot file
        '''
        sr2o_all = dict()
        train_all, valid_all, test_all = [], [], []
        for ss_id in range(int(self.args.snapshot_num)):
            self.new_entities = set()
            '''load facts'''
            train_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'train.txt')
            test_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'test.txt')
            valid_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'valid.txt')

            '''extract entities & relations from facts'''
            self.expand_entity_relation(train_facts)
            self.expand_entity_relation(valid_facts)
            self.expand_entity_relation(test_facts)

            '''read train/test/valid data'''
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts, order=True)
            test = self.fact2id(test_facts, order=True)


            if self.enable_noise_injection:
                if self.enable_noise_train:
                    train = self._inject_noise_to_triples(train, ss_id, 'train')
                if self.enable_noise_eval:
                    valid = self._inject_noise_to_triples(valid, ss_id, 'valid')
                    test = self._inject_noise_to_triples(test, ss_id, 'test')

            '''
            Get edge_index and edge_type for GCN"
                edge_index = [[s_1, s_2, ... s_n],[o_1, o_2, ..., o_n]]
                edge_type = [r_1, r_2, ..., r_n]
            '''
            edge_s, edge_r, edge_o = [], [], []
            edge_s, edge_o, edge_r = self.expand_kg(train, 'train', edge_s, edge_o, edge_r, sr2o_all)
            edge_s, edge_o, edge_r = self.expand_kg(valid, 'valid', edge_s, edge_o, edge_r, sr2o_all)
            edge_s, edge_o, edge_r = self.expand_kg(test, 'test', edge_s, edge_o, edge_r, sr2o_all)

            '''prepare data for 're-training' model'''
            train_all += train
            valid_all += valid
            test_all += test

            '''store this snapshot'''
            self.store_snapshot(ss_id, train, train_all, test, test_all, valid, valid_all, edge_s, edge_o, edge_r, sr2o_all)
            self.new_entities.clear()

    def expand_entity_relation(self, facts):
        '''extract entities and relations from new facts'''
        for (s, r, o) in facts:
            '''extract entities'''
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_ent
                self.num_ent += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_ent
                self.num_ent += 1

            '''extract relations'''
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.relation2inv[self.num_rel] = self.num_rel + 1
                self.relation2inv[self.num_rel + 1] = self.num_rel
                self.num_rel += 2

    def fact2id(self, facts, order=False):
        '''(s name, r name, o name)-->(s id, r id, o id)'''
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (s, r, o) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
                i = i + 2
        else:
            for (s, r, o) in facts:
                fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    def expand_kg(self, facts, split, edge_s, edge_o, edge_r, sr2o_all):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        def add_key2val(dict, key, val):
            '''add {key: value} to dict'''
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)

        for (h, r, t) in facts:
            self.new_entities.add(h)
            self.new_entities.add(t)
            if split == 'train':
                '''edge_index'''
                edge_s.append(h)
                edge_r.append(r)
                edge_o.append(t)
            '''sr2o'''
            add_key2val(sr2o_all, (h, r), t)
            add_key2val(sr2o_all, (t, self.relation2inv[r]), h)
        return edge_s, edge_o, edge_r

    def store_snapshot(self, ss_id, train_new, train_all, test, test_all, valid, valid_all, edge_s, edge_o, edge_r, sr2o_all):
        '''store snapshot data'''
        self.snapshots[ss_id].num_ent = dcopy(self.num_ent)
        self.snapshots[ss_id].num_rel = dcopy(self.num_rel)

        '''train, valid and test data'''
        self.snapshots[ss_id].train_new = dcopy(train_new)
        self.snapshots[ss_id].train_all = dcopy(train_all)
        self.snapshots[ss_id].test = dcopy(test)
        self.snapshots[ss_id].valid = dcopy(valid)
        self.snapshots[ss_id].valid_all = dcopy(valid_all)
        self.snapshots[ss_id].test_all = dcopy(test_all)

        '''edge_index, edge_type (for GCN of MEAN and LAN)'''
        self.snapshots[ss_id].edge_s = dcopy(edge_s)
        self.snapshots[ss_id].edge_r = dcopy(edge_r)
        self.snapshots[ss_id].edge_o = dcopy(edge_o)

        '''sr2o (to filter golden facts)'''
        self.snapshots[ss_id].sr2o_all = dcopy(sr2o_all)
        self.snapshots[ss_id].edge_index = build_edge_index(edge_s, edge_o).to(self.args.device)
        self.snapshots[ss_id].edge_type = torch.cat(
            [torch.LongTensor(edge_r), torch.LongTensor(edge_r) + 1]).to(self.args.device)
        self.snapshots[ss_id].new_entities = dcopy(list(self.new_entities))

        if self.args.lifelong_name in ['LAN', 'MEAN']:
            self.snapshots[ss_id].ent2neigh, self.snapshots[ss_id].edge_index_sample, self.snapshots[ss_id].edge_type_sample, self.snapshots[ss_id].ent_neigh_num = self.snapshots[ss_id].sample_neighbor()

    def _inject_noise_to_triples(self, triples, snapshot_id, data_type):
        """
        向三元组数据注入噪声
        Args:
            triples: 三元组列表 [(h, r, t), ...] 或 [(h, r, t, label), ...]
            snapshot_id: snapshot ID
            data_type: 数据类型 ('train', 'valid', 'test')
        Returns:
            注入噪声后的三元组列表
        """
        if self.noise_category == 'none':
            return triples

        logger = getattr(self.args, 'logger', None)
        if logger:
            if self.noise_category == 'graph_structure':
                noise_desc = "missing_links+systematic_bias+temporal_drift"
            elif self.noise_category == 'random_semantic':
                noise_desc = "random_flip"
            else:
                noise_desc = "none"

            logger.info(f"[Noise Injection] Applying {self.noise_category}({noise_desc}) noise to {data_type} data "
                       f"(ratio={self.noise_ratio}, snapshot={snapshot_id})")

        base_triples = triples


        if self.noise_category == 'none':
            noisy = triples
        elif self.noise_category == 'graph_structure':

            noisy = self._apply_graph_structure_noise(triples, snapshot_id)
        elif self.noise_category == 'random_semantic':

            noisy = self._apply_random_flip_noise(triples, snapshot_id)
        else:
            noisy = triples

        if self.noise_injection_mode == 'append':

            return base_triples + noisy
        else:

            return noisy

    def _apply_random_flip_noise(self, triples, snapshot_id):
        """随机翻转噪声 - 随机改变头实体或尾实体"""
        np.random.seed(snapshot_id + 42)

        noisy_triples = []
        for triple in triples:
            if len(triple) == 3:
                h, r, t = triple
                label = None
            else:
                h, r, t, label = triple

            if np.random.random() < self.noise_ratio:
                if np.random.random() > 0.5:

                    new_h = np.random.randint(0, self.num_ent)
                    if label is not None:
                        noisy_triples.append((new_h, r, t, label))
                    else:
                        noisy_triples.append((new_h, r, t))
                else:

                    new_t = np.random.randint(0, self.num_ent)
                    if label is not None:
                        noisy_triples.append((h, r, new_t, label))
                    else:
                        noisy_triples.append((h, r, new_t))
            else:
                noisy_triples.append(triple)

        return noisy_triples

    def _apply_systematic_bias_noise(self, triples, snapshot_id):
        """系统偏差噪声 - 基于snapshot创建系统性偏差"""

        np.random.seed(snapshot_id + 100)


        num_biased_relations = max(1, int(self.num_rel * 0.3))
        biased_relations = set(np.random.choice(self.num_rel, num_biased_relations, replace=False))

        noisy_triples = []
        for triple in triples:
            if len(triple) == 3:
                h, r, t = triple
                label = None
            else:
                h, r, t, label = triple

            if r in biased_relations and np.random.random() < self.noise_ratio:

                if snapshot_id % 2 == 0:

                    offset = (snapshot_id % 5) + 1
                    new_h = (h + offset) % self.num_ent
                    if label is not None:
                        noisy_triples.append((new_h, r, t, label))
                    else:
                        noisy_triples.append((new_h, r, t))
                else:

                    offset = (snapshot_id % 3) + 1
                    new_t = (t + offset) % self.num_ent
                    if label is not None:
                        noisy_triples.append((h, r, new_t, label))
                    else:
                        noisy_triples.append((h, r, new_t))
            else:
                noisy_triples.append(triple)

        return noisy_triples

    def _apply_missing_links_noise(self, triples, snapshot_id):
        """缺失链接噪声 - 随机删除三元组"""
        np.random.seed(snapshot_id + 200)

        keep_ratio = 1.0 - self.noise_ratio
        num_keep = int(len(triples) * keep_ratio)

        if num_keep <= 0:
            return []

        keep_indices = np.random.choice(len(triples), num_keep, replace=False)
        return [triples[i] for i in keep_indices]

    def _apply_temporal_drift_noise(self, triples, snapshot_id):
        """时间漂移噪声 - 随着时间逐渐改变分布"""
        np.random.seed(snapshot_id + 300)

        drift_factor = 1.0 + (snapshot_id * self.drift_speed)

        noisy_triples = []
        for triple in triples:
            if len(triple) == 3:
                h, r, t = triple
                label = None
            else:
                h, r, t, label = triple

            if np.random.random() < self.noise_ratio:

                drift_offset = int(drift_factor * np.random.randint(1, 4))
                new_h = min(h + drift_offset, self.num_ent - 1)
                new_t = min(t + drift_offset, self.num_ent - 1)

                if label is not None:
                    noisy_triples.append((new_h, r, new_t, label))
                else:
                    noisy_triples.append((new_h, r, new_t))
            else:
                noisy_triples.append(triple)

        return noisy_triples

    def _apply_combined_noise(self, triples, snapshot_id):
        """组合噪声 - 多种噪声类型的组合"""

        triples = self._apply_missing_links_noise(triples, snapshot_id)
        triples = self._apply_systematic_bias_noise(triples, snapshot_id)
        triples = self._apply_temporal_drift_noise(triples, snapshot_id)
        return triples

    def _apply_graph_structure_noise(self, triples, snapshot_id):
        """
        图结构噪声 - 融合missing_links + systematic_bias + temporal_drift

        按顺序应用三种图结构噪声，模拟全面的图结构干扰
        """

        triples = self._apply_missing_links_noise(triples, snapshot_id)


        triples = self._apply_systematic_bias_noise(triples, snapshot_id)


        triples = self._apply_temporal_drift_noise(triples, snapshot_id)

        return triples


class SnapShot():
    def __init__(self, args):
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train_new, self.train_all, self.test, self.valid, self.valid_all, self.test_all = list(), list(), list(), list(), list(), list()
        self.edge_s, self.edge_r, self.edge_o = [], [], []
        self.sr2o_all = dict()
        self.edge_index, self.edge_type = None, None
        self.new_entities = []

    def sample_neighbor(self):
        '''sample neighbor for MEAN or LAN'''
        num = 64
        res = []
        triples = self.train_new
        ent2triples = {i:list() for i in range(self.num_ent)}
        edge_index_sample, edge_type_sample = [], []
        ent_neigh_num = torch.zeros(self.num_ent).to(self.args.device)
        for triple in triples:
            h, r, t = triple
            ent2triples[h].append((h, r, t))
            ent2triples[t].append((t, r+1, h))
        for ent in range(self.num_ent):
            ent2triples[ent].append((ent, self.num_rel, ent))
            if len(ent2triples[ent]) > num:
                ent_neigh_num[ent] = num
                samples = [ent2triples[ent][i] for i in np.random.choice(range(len(ent2triples[ent])), num, replace=False)]
            else:
                samples = ent2triples[ent]
                ent_neigh_num[ent] = len(ent2triples[ent])
                for i in range(num - len(ent2triples[ent])):
                    samples.append((self.num_ent, self.num_rel+1, self.num_ent))
            res.append(samples)
            for hrt in samples:
                h, r, t = hrt
                if r == self.num_rel+1:
                    pass
                else:
                    edge_index_sample.append([ent, t])
                    edge_type_sample.append(r)

        return torch.LongTensor(res).to(self.args.device), torch.LongTensor(edge_index_sample).to(self.args.device).t(), torch.LongTensor(edge_type_sample).to(self.args.device), ent_neigh_num

