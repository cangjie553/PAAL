from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable

class TrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        '''prepare data'''
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        total_cfr_loss = 0.0
        cfr_batch_count = 0
        num_forget_triples = len(getattr(model, 'global_forget_set', set()))
        
        for idx_b, batch in enumerate(self.data_loader):
            '''get loss'''
            bh, br, bt, by = batch
            


            
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                       br.to(self.args.device),
                                       bt.to(self.args.device),
                                       by.to(self.args.device) if by is not None else by).float()

            '''update'''
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            '''post processing'''
            model.epoch_post_processing(bh.size(0))
        return total_loss

    def evaluate_on_training_data(self, model):
        model.eval()
        results = {}
        num = 0
        retrain_data = []


        all_training_data = []
        all_entity = []
        all_relation = []
        all_data_head = torch.tensor([], dtype=torch.int64)
        all_data_relation = torch.tensor([], dtype=torch.int64)
        all_data_tail = torch.tensor([], dtype=torch.int64)
        all_data_label = torch.tensor([], dtype=torch.int64)

        new_data_head = []
        new_data_relation = []
        new_data_tail = []
        new_data_label = []
        for b_id, batch in enumerate(self.data_loader):
            """ Get loss   y中的值是0或者1，1表示这个样本是正确的，0表示这个样本是错误的 """
            head, relation, tail, label = batch


            all_entity = all_entity + head.tolist()
            all_entity = all_entity + tail.tolist()
            all_relation = all_relation + relation.tolist()
            all_data_head = torch.cat([all_data_head, head], dim=0)
            all_data_relation = torch.cat([all_data_relation, relation], dim=0)
            all_data_tail = torch.cat([all_data_tail, tail], dim=0)
            all_data_label = torch.cat([all_data_label, label], dim=0)

        for idx in range(len(all_data_head)):
            if all_data_label[idx] == 1:
                new_data_head.append(all_data_head[idx].item())
                new_data_relation.append(all_data_relation[idx].item())
                new_data_tail.append(all_data_tail[idx].item())
                new_data_label.append(all_data_label[idx].item())
        new_data_head = torch.tensor(new_data_head, dtype=torch.int64)
        new_data_relation = torch.tensor(new_data_relation, dtype=torch.int64)
        new_data_tail = torch.tensor(new_data_tail, dtype=torch.int64)
        new_data_label = torch.tensor(new_data_label, dtype=torch.int64)

        all_training_data = [new_data_head, new_data_relation, new_data_tail, new_data_label]
        all_entity = list(set(all_entity))
        all_relation = list(set(all_relation))

        def chunks(lst, n):
            for i in range(0, len(lst[0]), n):
                yield (lst[0][i:i + n], lst[1][i:i + n], lst[2][i:i + n], lst[3][i:i + n])
        
        all_prob = []
        for chunk in chunks(all_training_data, int(self.args.batch_size/32)):
            head, relation, tail, label = chunk
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)

            num += len(head)
            pred = model.predict(head, relation, stage='Valid')
            pred = pred.to(self.args.device)



            new_label = torch.zeros_like(pred)
            for idx in range(tail.shape[0]):
                new_label[idx, tail[idx]] = 1 
            """ filter: If there is more than one tail in the label, we only think that the tail in this triple is right """
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(new_label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred
            
            """ rank all candidate entities """
            """ Two sorts can be optimized into one """

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float()



            target_logit = pred[batch_size_range, tail]

            if self.args.use_difficult_samples:
                all_prob += torch.exp(1-target_logit).tolist()
            else:
                all_prob += torch.nn.Softmax(target_logit).dim.tolist()

   

        all_prob = np.array(all_prob)
        sum_prob = 0.
        for prob in all_prob:
            sum_prob += prob
        for idx, prob in enumerate(all_prob):
            all_prob[idx] = prob / sum_prob



        num_samples = min(self.args.num_samples_for_retrain, len(all_prob)/3)
        chosen_index = np.random.choice(len(all_prob), p=all_prob, size=num_samples, replace=False)


        for idx in chosen_index:
            head = new_data_head[idx].item()
            relation = new_data_relation[idx].item()
            tail = new_data_tail[idx].item()
            if head in all_entity:
                all_entity.remove(head)
            if tail in all_entity:
                all_entity.remove(tail)
            if relation in all_relation:
                all_relation.remove(relation)
            retrain_data.append((head, relation, tail, new_data_label[idx].item()))

        for idx in range(len(new_data_head)):
            if idx%1000 == 0:
                print(idx)
            if new_data_head[idx] in all_entity:
                retrain_data.append((new_data_head[idx].item(), new_data_relation[idx].item(), new_data_tail[idx].item(), new_data_label[idx]))
                all_entity.remove(new_data_head[idx].item())
                if new_data_relation[idx] in all_relation:
                    all_relation.remove(new_data_relation[idx].item())
                continue
            if new_data_tail[idx] in all_entity:
                retrain_data.append((new_data_head[idx].item(), new_data_relation[idx].item(), new_data_tail[idx].item(), new_data_label[idx]))
                all_entity.remove(new_data_tail[idx].item())
                if new_data_relation[idx] in all_relation:
                    all_relation.remove(new_data_relation[idx].item())
                continue
            if new_data_relation[idx] in all_relation:
                retrain_data.append((new_data_head[idx].item(), new_data_relation[idx].item(), new_data_tail[idx].item(), new_data_label[idx]))
                all_relation.remove(new_data_relation[idx].item())
            if all_entity == [] and all_relation == []:
                break



            break

        return retrain_data



class DevBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        self.batch_size = 100
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        sr2o = self.kg.snapshots[self.args.snapshot].sr2o_all
        '''start evaluation'''
        num = 0
        for step, batch in enumerate(self.data_loader):
            sub, rel, obj, label = batch
            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                if k not in [0, 2, 4, 9]:
                    continue
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
        if 'count' in results and results['count'] > 0:
            count = float(results['count'])
            for key, val in results.items():
                if key != 'count':
                    results[key] = round(val / count, 4)
        else:

            results = {'count': 0.0, 'mr': 0.0, 'mrr': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits5': 0.0, 'hits10': 0.0}
        return results

    def process_epoch_again(self, model):
        model.eval()
        num = 0
        results = dict()
        '''start evaluation'''
        for step, batch in enumerate(self.data_loader):
            sub, rel, obj, label = batch
            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            model.args.expand_dim = True
            pred = model.predict(sub, rel, stage=stage)
            model.args.expand_dim = False

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                if k not in [0, 2, 4, 9]:
                    continue
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results


class DevBatchProcessor_MEANandLAN():
    '''
    To save memory, we collect the queries with the same relation and then perform evaluation.
    '''
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        self.batch_size = 1
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        '''start evaluation'''
        sub, rel, obj, label = None, None, None, None

        for step, batch in enumerate(self.data_loader):
            sub_, rel_, obj_, label_ = batch
            if sub == None:
                sub, rel, obj, label = sub_, rel_, obj_, label_
                continue
            elif rel[0] == rel_ and rel.size(0) <= 50:
                sub = torch.cat((sub, sub_), dim=0)
                rel = torch.cat((rel, rel_), dim=0)
                obj = torch.cat((obj, obj_), dim=0)
                label = torch.cat((label, label_), dim=0)
                continue

            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
            sub, rel, obj, label = None, None, None, None

        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results