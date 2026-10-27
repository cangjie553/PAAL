from src.utils import *
from torch_scatter import scatter_add, scatter_mean, scatter_max


class BaseModel(nn.Module):
    def __init__(self, args, kg):
        super(BaseModel, self).__init__()
        self.args = args
        self.kg = kg

        '''initialize the entity and relation embeddings for the first snapshot'''
        self.ent_embeddings = nn.Embedding(self.kg.snapshots[0].num_ent, self.args.emb_dim).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.snapshots[0].num_rel, self.args.emb_dim).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)

        '''loss function'''
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")







        self.embedding_dim = self.args.emb_dim
        

        self.global_forget_set = set()

    def merge_parameters(self):
        dim_list = self.get_dim_list()

        this_ss_ent_embedding = self.mapping(self.this_snapshot_ent_embeddings.data)
        this_ss_rel_embedding = self.mapping(self.this_snapshot_rel_embeddings.data)


        self.ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot].num_ent, dim_list[self.args.snapshot + 1]).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot].num_rel, dim_list[self.args.snapshot + 1]).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)

        self.ent_embeddings.weight.data = this_ss_ent_embedding
        self.rel_embeddings.weight.data = this_ss_rel_embedding






    def reinit_param(self):
        '''
        Re-initialize all model parameters
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                xavier_normal_(p)

    def get_dim_list(self):

        if hasattr(self.args, 'dim_list') and self.args.dim_list is not None:
            return self.args.dim_list
        

        self.args.emb_dim1 = self.args.emb_dim
        dim_list = []
        dim_list.append(self.args.emb_dim1)
        dim_list.append(self.args.emb_dim2)
        dim_list.append(self.args.emb_dim3)
        dim_list.append(self.args.emb_dim4)
        dim_list.append(self.args.emb_dim5)
        return dim_list
    
    def expand_embedding_size(self):
        '''
        Initialize entity and relation embeddings for next snapshot
        '''

        dim_list = self.get_dim_list()
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, dim_list[self.args.snapshot + 1]).to(
            self.args.device).double()
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, dim_list[self.args.snapshot + 1]).to(
            self.args.device).double()


        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return deepcopy(ent_embeddings), deepcopy(rel_embeddings)

    def switch_snapshot(self):
        '''
        After the training process of a snapshot, prepare for next snapshot
        '''
        pass

    def pre_snapshot(self):
        '''
        Preprocess before training on a snapshot
        '''
        pass

    def epoch_post_processing(self, size=None):
        '''
        Post process after a training iteration
        '''
        pass

    def snapshot_post_processing(self):
        '''
        Post process after training on a snapshot
        '''
        pass

    def store_old_parameters(self):
        '''
        Store the learned model after training on a snapshot
        '''
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer('old_data_{}'.format(name), value.clone().detach())

    def initialize_old_data(self):
        '''
        Initialize the storage of old parameters
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')
                self.register_buffer('old_data_{}'.format(n), p.data.clone())

    def embedding(self, stage=None):
        '''
        :param stage: Train / Valid / Test
        :return: entity and relation embeddings
        '''
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def new_loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: loss of new facts
        '''
        return self.margin_loss(head, rel, tail, label)/head.size(0)
    
    def compute_continuous_forgetting_loss(self):
        '''
        Compute Continuous Forgetting Regularizer (CFR):
        L_CFR = sum( sigmoid(-score(h,r,t)) for all (h,r,t) in global_forget_set )
        
        This encourages the model to forget triples in the global forget set by
        increasing their distances (lower scores).
        
        Returns:
            cfr_loss: scalar tensor, or None if global_forget_set is empty
        '''
        if not hasattr(self, 'global_forget_set') or len(self.global_forget_set) == 0:
            return None
        
        try:

            forget_triples = list(self.global_forget_set)
            if len(forget_triples) == 0:
                return None
            

            current_snapshot = getattr(self.args, 'snapshot', 0)
            current_num_ent = self.kg.snapshots[current_snapshot].num_ent
            current_num_rel = self.kg.snapshots[current_snapshot].num_rel
            




            ent_embeddings, rel_embeddings = self.embedding('Train')
            actual_ent_size = ent_embeddings.shape[0]
            actual_rel_size = rel_embeddings.shape[0]
            

            valid_triples = []
            for t in forget_triples:
                h, r, t_idx = t[0], t[1], t[2]

                if 0 <= h < actual_ent_size and 0 <= r < actual_rel_size and 0 <= t_idx < actual_ent_size:
                    valid_triples.append(t)
            
            if len(valid_triples) == 0:

                logger = getattr(self.args, 'logger', None)
                if logger:
                    logger.warning(f"[CFR Debug] No valid triples (current_snapshot={current_snapshot}, "
                                  f"current_num_ent={current_num_ent}, current_num_rel={current_num_rel}, "
                                  f"actual_ent_size={actual_ent_size}, actual_rel_size={actual_rel_size}, "
                                  f"total forget triples={len(forget_triples)})")

                    if len(forget_triples) > 0:
                        sample_triples = list(forget_triples)[:5]
                        logger.warning(f"[CFR Debug] Sample forget triples: {sample_triples}")
                return None
            

            heads = torch.tensor([t[0] for t in valid_triples], dtype=torch.long, device=self.args.device)
            relations = torch.tensor([t[1] for t in valid_triples], dtype=torch.long, device=self.args.device)
            tails = torch.tensor([t[2] for t in valid_triples], dtype=torch.long, device=self.args.device)
            

            

            s = torch.index_select(ent_embeddings, 0, heads)
            r = torch.index_select(rel_embeddings, 0, relations)
            o = torch.index_select(ent_embeddings, 0, tails)
            scores = self.score_fun(s, r, o)
            





            



            score_scale = 5.0
            scaled_scores = scores / score_scale
            sigmoid_neg_scores = torch.sigmoid(-scaled_scores)
            cfr_loss = sigmoid_neg_scores.sum()
            


            

            logger = getattr(self.args, 'logger', None)
            if logger and hasattr(self, '_cfr_debug_logged'):

                pass
            elif logger:
                self._cfr_debug_logged = True
                logger.info(f"[CFR Score Analysis] num_triples={len(scores)}, "
                           f"score_mean={scores.mean().item():.4f}, score_std={scores.std().item():.4f}, "
                           f"score_min={scores.min().item():.4f}, score_max={scores.max().item():.4f}, "
                           f"sigmoid_neg_mean={sigmoid_neg_scores.mean().item():.6f}, "
                           f"sigmoid_neg_min={sigmoid_neg_scores.min().item():.6f}, "
                           f"sigmoid_neg_max={sigmoid_neg_scores.max().item():.6f}")
            
            return cfr_loss
        except Exception as e:

            logger = getattr(self.args, 'logger', None)
            if logger:
                logger.warning(f"[CFR] Error computing continuous forgetting loss: {e}")
            return None

    def margin_loss(self, head, rel, tail, label=None):
        '''
        Pair Wise Margin loss: L1-norm(s + r - o)
        :param head:
        :param rel:
        :param tail:
        :param label:
        :return:
        '''
        ent_embeddings, rel_embeddings = self.embedding('Train')

        s = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, rel)
        o = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(s, r, o)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        loss = self.margin_loss_func(p_score, n_score, y)
        return loss

    def split_pn_score(self, score, label):
        '''
        Get the scores of positive and negative facts
        :param score: scores of all facts
        :param label: positive facts: 1, negative facts: -1
        :return:
        '''
        p_score = score[torch.where(label>0)]
        n_score = (score[torch.where(label<0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def score_fun(self, s, r, o):
        '''
        score function f(s, r, o) = L1-norm(s + r - o)
        :param h:
        :param r:
        :param t:
        :return:
        '''
        s = self.norm_ent(s)
        r = self.norm_rel(r)
        o = self.norm_ent(o)
        return torch.norm(s + r - o, 1, -1)

    def predict(self, sub, rel, stage='Valid'):
        '''
        Scores all candidate facts for evaluation
        :param head: subject entity id
        :param rel: relation id
        :param stage: object entity id
        :return: scores of all candidate facts
        '''

        '''get entity and relation embeddings'''
        if self.args.expand_dim == False:
            if stage != 'Test':
                num_ent = self.kg.snapshots[self.args.snapshot].num_ent
            else:
                num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
            ent_embeddings, rel_embeddings = self.embedding(stage)
            s = torch.index_select(ent_embeddings, 0, sub)
            r = torch.index_select(rel_embeddings, 0, rel)
            o_all = ent_embeddings[:num_ent]
            s = self.norm_ent(s)
            r = self.norm_rel(r)
            o_all = self.norm_ent(o_all)

            '''s + r - o'''
            pred_o = s + r
            score = 9.0 - torch.norm(pred_o.unsqueeze(1) - o_all, p=1, dim=2)
            score = torch.sigmoid(score)

            return score
        
        else:
            if stage != 'Test':
                num_ent = self.kg.snapshots[self.args.snapshot].num_ent
            else:
                num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
            
            head = sub
            relation = rel
            tail = None








            t_all = self.ent_embeddings.weight[:num_ent]


            h = torch.index_select(self.ent_embeddings.weight, 0, head)
            r = torch.index_select(self.rel_embeddings.weight, 0, relation)

            h = self.norm_ent(h)
            r = self.norm_rel(r)
            t_all = self.norm_ent(t_all)

            """ h + r - t """

            pred_t = h + r
            score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)
            score = torch.sigmoid(score)
            return score

    def norm_rel(self, r):
        return nn.functional.normalize(r, 2, -1)

    def norm_ent(self, e):
        return nn.functional.normalize(e, 2, -1)
    
