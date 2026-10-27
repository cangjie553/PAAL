from .BaseModel import *
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PAAL import detect_candidates, split_candidates, create_forget_loss_module


class LKGE(BaseModel):
    def __init__(self, args, kg):
        super(LKGE, self).__init__(args, kg)
        self.init_old_weight()
        self.mse_loss_func = nn.MSELoss(size_average=False)
        self.ent_weight, self.rel_weight, self.other_weight = None, None, None
        self.margin_loss_func = nn.MarginRankingLoss(float(self.args.margin), size_average=False).to(self.args.device)
        

        if getattr(args, 'enable_unlearning', False):
            self.forget_loss_module = create_forget_loss_module(self, args, kg)


            self.F_pre = set()
            self.F_post = set()
            self.tsf_enabled = getattr(args, 'enable_tsf', False)
        else:
            self.forget_loss_module = None


        self.fixed_initial_graph = getattr(args, 'fixed_initial_graph_constraint', False)
        self.initial_edge_index = None
        self.initial_edge_type = None
        self.initial_num_ent = None
        self.initial_num_rel = None
        self._logged_fixed_graph_for_snapshot = -1


        self.adaptive_mae_weight = getattr(args, 'adaptive_mae_weight', False)
        self.drift_alpha = getattr(args, 'drift_alpha', 1.0)
        self.drift_metrics = getattr(args, 'drift_metrics', 'degree,relation').split(',')
        self.drift_adaptive_lambda0 = getattr(args, 'drift_adaptive_lambda0', 0.1)
        self.initial_structural_stats = None
        self._logged_adaptive_weight_for_snapshot = -1


        self.fixed_initial_regularization = getattr(args, 'fixed_initial_regularization', True)
        self.initial_parameters = {}
        

        self.enable_learnable_noise_scale = getattr(args, 'enable_learnable_noise_scale', True)
        if self.enable_learnable_noise_scale:

            initial_theta_r = getattr(args, 'noise_scale_init', 0.1)


            import math
            logit_init = math.log(initial_theta_r / (1 - initial_theta_r + 1e-8))
            self.noise_scale_logit = nn.Parameter(torch.tensor(logit_init, device=args.device))
        else:
            self.noise_scale_logit = None


        self.enable_learnable_noise_factor = getattr(args, 'enable_learnable_noise_factor', True)
        if self.enable_learnable_noise_factor:

            initial_theta_s = getattr(args, 'noise_sampling_factor_init', 0.5)

            logit_init_s = math.log(initial_theta_s / (1 - initial_theta_s + 1e-8))
            self.noise_factor_logit = nn.Parameter(torch.tensor(logit_init_s, device=args.device))
        else:
            self.noise_factor_logit = None
        


        paa_enable_str = getattr(args, 'enable_paa', 'False')
        self.enable_paa = paa_enable_str in ['True', True]


        paa_loss_modulation_str = getattr(args, 'enable_paa_loss_modulation', 'False')
        self.enable_paa_loss_modulation = paa_loss_modulation_str in ['True', True]
        self.paa_mae_modulation_weight = getattr(args, 'paa_mae_modulation_weight', 1.0)
        self.paa_regularization_modulation_weight = getattr(args, 'paa_regularization_modulation_weight', 1.0)
        self.paa_loss_modulation_update_freq = getattr(args, 'paa_loss_modulation_update_freq', 100)
        
        if self.enable_paa:
            from .paa_module import PatternAnchorAlignment, DimensionProjector
            self.paa_module = PatternAnchorAlignment(args, kg)
            self.paa_update_frequency = getattr(args, 'paa_update_frequency', 'snapshot')
            

            if self.paa_module.enable_dim_projection:
                self.paa_projector = self.paa_module.projector

                for input_dim, proj in self.paa_projector.projectors.items():
                    self.add_module(f'paa_projector_{input_dim}', proj)
            
            if getattr(args, 'logger', None):
                proj_str = f", projection_dim={self.paa_module.projection_dim}" if self.paa_module.enable_dim_projection else ""
                args.logger.info(f"[PAA] Initialized Pattern-Anchor Alignment module "
                               f"(update_frequency={self.paa_update_frequency}{proj_str})")


                if self.enable_paa_loss_modulation:
                    args.logger.info(f"[PAA Loss Modulation] Enabled - "
                                   f"MAE_weight={self.paa_mae_modulation_weight:.1f}, "
                                   f"Reg_weight={self.paa_regularization_modulation_weight:.1f}, "
                                   f"Update_freq={self.paa_loss_modulation_update_freq}")
                else:
                    args.logger.info(f"[PAA Loss Modulation] Disabled")
        else:
            self.paa_module = None
        


        self.rca_cache = {}
        if getattr(args, 'logger', None):
            args.logger.info("[RCA Cache] Initialized cache for Rule Consistent Anchors optimization")

    def store_initial_graph_structure(self):
        '''
        存储第一个snapshot的图结构信息，用于固定图结构约束
        Store initial graph structure for fixed graph constraint
        '''
        if self.fixed_initial_graph and self.initial_edge_index is None:

            self.initial_edge_index = self.kg.snapshots[0].edge_index.clone()
            self.initial_edge_type = self.kg.snapshots[0].edge_type.clone()
            self.initial_num_ent = self.kg.snapshots[0].num_ent
            self.initial_num_rel = self.kg.snapshots[0].num_rel

            logger = getattr(self.args, 'logger', None)
            if logger:
                logger.info(f"[Fixed Initial Graph] Saved initial graph structure: "
                           f"{self.initial_edge_index.size(1)} edges, "
                           f"{self.initial_num_ent} entities, {self.initial_num_rel} relations")


            if self.adaptive_mae_weight:
                self.initial_structural_stats = self.compute_structural_stats(self.kg.snapshots[0])
                if logger:
                    logger.info(f"[Adaptive MAE Weight] Saved initial structural statistics for drift calculation")

    def compute_structural_stats(self, snapshot):
        '''
        计算图结构的统计信息，用于drift计算
        Compute structural statistics for drift calculation
        '''
        stats = {}

        if 'degree' in self.drift_metrics:

            degree_stats = self.compute_degree_distribution(snapshot)
            stats['degree'] = degree_stats

        if 'relation' in self.drift_metrics:

            relation_stats = self.compute_relation_frequency(snapshot)
            stats['relation'] = relation_stats

        if 'path' in self.drift_metrics:

            path_stats = self.compute_path_distribution(snapshot)
            stats['path'] = path_stats

        return stats

    def compute_degree_distribution(self, snapshot):
        '''
        计算实体的度分布
        '''
        import torch

        edge_index = snapshot.edge_index
        num_ent = snapshot.num_ent


        device = edge_index.device


        out_degree = torch.zeros(num_ent, dtype=torch.float, device=device)
        in_degree = torch.zeros(num_ent, dtype=torch.float, device=device)


        out_degree.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))

        in_degree.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1], dtype=torch.float))


        total_degree = out_degree + in_degree
        degree_dist = total_degree / (total_degree.max() + 1e-8)

        return degree_dist

    def compute_relation_frequency(self, snapshot):
        '''
        计算关系类型的频率分布
        '''
        import torch

        edge_type = snapshot.edge_type
        num_rel = snapshot.num_rel


        device = edge_type.device


        relation_freq = torch.zeros(num_rel, dtype=torch.float, device=device)
        relation_freq.scatter_add_(0, edge_type, torch.ones_like(edge_type, dtype=torch.float))


        relation_dist = relation_freq / (relation_freq.sum() + 1e-8)

        return relation_dist

    def compute_path_distribution(self, snapshot):
        '''
        计算简单的路径统计（例如三角形等）
        简化为边的类型组合统计
        '''
        import torch

        edge_index = snapshot.edge_index
        edge_type = snapshot.edge_type



        path_stats = torch.zeros(10, dtype=torch.float)


        unique_edges = torch.unique(torch.stack([edge_index[0], edge_index[1], edge_type]), dim=1)
        path_stats[0] = len(unique_edges[0])


        path_stats = path_stats / (path_stats.sum() + 1e-8)

        return path_stats

    def compute_js_divergence(self, p, q):
        '''
        计算JS散度: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        其中 M = 0.5 * (P + Q)
        '''
        import torch
        import torch.nn.functional as F


        p = p / (p.sum() + 1e-8)
        q = q / (q.sum() + 1e-8)


        m = 0.5 * (p + q)


        kl_pm = F.kl_div(torch.log(m + 1e-8), p, reduction='sum')
        kl_qm = F.kl_div(torch.log(m + 1e-8), q, reduction='sum')

        js_div = 0.5 * (kl_pm + kl_qm)

        return js_div.item()

    def compute_structural_drift(self, current_snapshot):
        '''
        计算当前snapshot与初始snapshot的结构drift
        '''
        if not self.adaptive_mae_weight or self.initial_structural_stats is None:
            return 0.0

        current_stats = self.compute_structural_stats(current_snapshot)

        total_drift = 0.0
        metric_count = 0

        for metric in self.drift_metrics:
            if metric in self.initial_structural_stats and metric in current_stats:
                initial_dist = self.initial_structural_stats[metric]
                current_dist = current_stats[metric]


                min_len = min(len(initial_dist), len(current_dist))
                initial_dist = initial_dist[:min_len]
                current_dist = current_dist[:min_len]


                js_div = self.compute_js_divergence(initial_dist, current_dist)
                total_drift += js_div
                metric_count += 1


        avg_drift = total_drift / max(metric_count, 1)

        return avg_drift

    def get_adaptive_mae_weight(self, current_snapshot):
        '''
        根据结构drift计算自适应的MAE权重
        λ_i = λ₀ * exp(-α * D_i)
        '''
        if not self.adaptive_mae_weight:
            return self.drift_adaptive_lambda0


        drift_distance = self.compute_structural_drift(current_snapshot)


        adaptive_weight = self.drift_adaptive_lambda0 * torch.exp(torch.tensor(-self.drift_alpha * drift_distance))


        logger = getattr(self.args, 'logger', None)
        if logger and self.args.snapshot > 0 and getattr(self, '_logged_adaptive_weight_for_snapshot', -1) != self.args.snapshot:
            logger.info(f"[Adaptive MAE Weight] Snapshot {self.args.snapshot}: "
                       f"Drift={drift_distance:.4f}, λ={adaptive_weight:.6f} "
                       f"(λ₀={self.drift_adaptive_lambda0}, α={self.drift_alpha})")
            self._logged_adaptive_weight_for_snapshot = self.args.snapshot

        return adaptive_weight.item()

    def store_initial_parameters(self):
        '''
        存储第一个snapshot的参数快照，用于固定初始参数正则化
        Store initial snapshot parameters for fixed regularization
        '''
        if self.fixed_initial_regularization and not self.initial_parameters:
            for name, param in self.named_parameters():
                name_ = name.replace('.', '_')
                if 'mapping' in name_:
                    continue
                if 'trainable' in name_:
                    continue
                if 'frozen' in name_:
                    continue

                self.initial_parameters[name_] = param.data.clone()

            logger = getattr(self.args, 'logger', None)
            if logger:
                logger.info(f"[Fixed Initial Regularization] Saved initial parameters snapshot")

    def store_old_parameters(self):
        '''
        Store learned paramters and weights for regularization.
        '''
        self.args.snapshot -= 1
        param_weight = self.get_new_weight()
        self.args.snapshot += 1
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            value = param.data
            if 'mapping' in name:
                continue
            if 'trainable' in name:
                continue
            if 'frozen' in name:
                continue
            try:
                old_weight = getattr(self, 'old_weight_{}'.format(name))
            except AttributeError:

                if '_embeddings' in name:
                    old_weight = torch.tensor([[]])
                else:
                    old_weight = torch.tensor(0.0)
                self.register_buffer('old_weight_{}'.format(name), old_weight)

            new_weight = param_weight[name]
            self.register_buffer('old_data_{}'.format(name), value)
            if '_embeddings' in name:
                if self.args.snapshot == 0:
                    old_weight = torch.zeros_like(new_weight)
                else:
                    old_weight = torch.cat([old_weight, torch.zeros(new_weight.size(0) - old_weight.size(0), 1).to(self.args.device)], dim=0)

            self.register_buffer('old_weight_{}'.format(name), old_weight + new_weight)

    def init_old_weight(self):
        '''
        Initialize the learned parameters for storage.
        '''
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'ent_embeddings' in name_:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))
            elif 'rel_embeddings' in name_:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))
            else:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor(0.0))
                self.register_buffer('old_data_{}'.format(name_), param.data)

    def switch_snapshot(self):
        '''
        Prepare for the training on next snapshot.
        '''


        '''store old parameters'''
        self.store_old_parameters()
        '''expand embedding size for new entities and relations'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        '''inherit learned paramters'''
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)
        '''embedding transfer'''
        if self.args.using_embedding_transfer == 'True':
            reconstruct_ent_embeddings, reconstruct_rel_embeddings = self.reconstruct()
            new_ent_embeddings[self.kg.snapshots[self.args.snapshot].num_ent:] = reconstruct_ent_embeddings[self.kg.snapshots[self.args.snapshot].num_ent:]
            new_rel_embeddings[self.kg.snapshots[self.args.snapshot].num_rel:] = reconstruct_rel_embeddings[self.kg.snapshots[self.args.snapshot].num_rel:]
            self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)
        '''store the total number of facts containing each entity or relation'''
        new_ent_weight, new_rel_weight, new_other_weight = self.get_weight()
        self.register_buffer('new_weight_ent_embeddings_weight', new_ent_weight.clone().detach())
        self.register_buffer('new_weight_rel_embeddings_weight', new_rel_weight.clone().detach())
        '''get regularization weights'''
        self.new_weight_other_weight = new_other_weight

    def reconstruct(self):
        '''
        Reconstruct the entity and relation embeddings.
        '''
        num_ent, num_rel = self.kg.snapshots[self.args.snapshot+1].num_ent, self.kg.snapshots[self.args.snapshot+1].num_rel
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot+1].edge_index, self.kg.snapshots[self.args.snapshot+1].edge_type
        try:
            old_entity_weight = self.old_weight_entity_embeddings
            old_relation_weight = self.old_weight_relation_embeddings
            old_x = self.old_data_entity_embeddings
            old_r = self.old_data_relation_embeddings
        except:
            old_entity_weight, old_relation_weight = None, None
            old_x, old_r = None, None
        new_embeddings, rel_embeddings = self.gcn(self.ent_embeddings.weight, self.rel_embeddings.weight, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return new_embeddings, rel_embeddings

    def get_new_weight(self):
        '''
        Calculate the regularization weights for entities and relations.
        :return: weights for entities and relations.
        '''

        ent_weight, rel_weight, other_weight = self.get_weight()
        weight = dict()
        for name, param in self.named_parameters():
            name_ = name.replace('.','_')
            if 'ent_embeddings' in name_:
                weight[name_] = ent_weight
            elif 'rel_embeddings' in name_:
                weight[name_] = rel_weight
            else:
                weight[name_] = other_weight
        return weight

    def new_loss(self, head, rel, tail=None, label=None):
        return self.margin_loss(head, rel, tail, label).mean()

    def lkge_regular_loss(self, paa_score_avg=None):
        '''
        Calculate regularization loss to avoid catastrophic forgetting.
        Support optional fixed initial parameter regularization and PAA modulation.

        Args:
            paa_score_avg: Average PAA score for current batch (optional)
                         Used for PAA modulation: L_new = L_old * (1 + PAA) / 2

        :return: regularization loss (potentially modulated by PAA).
        '''
        if self.args.snapshot == 0:
            return 0.0

        losses = []


        if self.fixed_initial_regularization and self.initial_parameters:

            use_initial_params = True
            '''get samples number of entities and relations for weighting'''
            new_ent_weight, new_rel_weight, new_other_weight = self.new_weight_ent_embeddings_weight, self.new_weight_rel_embeddings_weight, self.new_weight_other_weight
        else:

            use_initial_params = False
            '''get samples number of entities and relations'''
            new_ent_weight, new_rel_weight, new_other_weight = self.new_weight_ent_embeddings_weight, self.new_weight_rel_embeddings_weight, self.new_weight_other_weight

        '''calculate regularization loss'''
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'ent_embeddings' in name_:
                new_weight = new_ent_weight
            elif 'rel_embeddings' in name_:
                new_weight = new_rel_weight
            else:
                new_weight = new_other_weight

            if 'mapping' in name_:
                continue
            if 'trainable' in name_:
                continue
            if 'frozen' in name_:
                continue

            if use_initial_params and name_ in self.initial_parameters:

                old_data = self.initial_parameters[name_]

                if old_data.shape != param.shape:

                    continue
                new_data = param


                old_weight = torch.ones(old_data.size(0), 1, device=self.args.device)
                if type(new_weight) != int:
                    new_weight = new_weight[:old_weight.size(0)]
                    new_data = new_data[:old_data.size(0)]
                losses.append((((new_data - old_data) * old_weight / (new_weight+old_weight)) ** 2).sum())
            else:

                old_weight = getattr(self, 'old_weight_{}'.format(name_))
                old_data = getattr(self, 'old_data_{}'.format(name_))
                new_data = param
                if type(new_weight) != int:
                    new_weight = new_weight[:old_weight.size(0)]
                    new_data = new_data[:old_data.size(0)]
                losses.append((((new_data - old_data) * old_weight / (new_weight+old_weight)) ** 2).sum())

        regularization_loss = sum(losses)




        if self.enable_paa_loss_modulation and paa_score_avg is not None and self.paa_regularization_modulation_weight > 0.0:

            paa_score_avg = torch.clamp(paa_score_avg, -1.0, 1.0)



            modulation_factor = (1.0 + paa_score_avg) / 2.0

            modulation_factor = 1.0 + self.paa_regularization_modulation_weight * (modulation_factor - 1.0)

            regularization_loss = regularization_loss * modulation_factor


            if self.args.logger and (self.args.snapshot == 0 and torch.rand(1).item() < 0.001):
                self.args.logger.info(
                    f"[PAA Regularization Modulation] avg_paa={paa_score_avg.item():.4f}, "
                    f"modulation_factor={modulation_factor.item():.4f}, "
                    f"original_reg={regularization_loss.item()/modulation_factor.item():.6f}, "
                    f"modulated_reg={regularization_loss.item():.6f}"
                )

        return regularization_loss
    
    def compute_paa_scores(self, head, rel, tail):
        """
        计算Pattern-Anchor Alignment (PAA)分数作为调控信号
        
        Compute PAA scores for the current batch of triples.
        PAA scores are used as continuous control signals to guide DSP loss 
        and noise sampling loss, NOT as a standalone loss term.
        
        Args:
            head: 头实体ID tensor (batch_size,)
            rel: 关系ID tensor (batch_size,)
            tail: 尾实体ID tensor (batch_size,)
        
        Returns:
            paa_scores: PAA分数 (batch_size,), 范围[-1, 1]
                        高分数表示三元组与关系原型高度对齐
                        如果PAA未启用或无原型，返回None
        """
        if not self.enable_paa or self.paa_module is None:
            return None
        

        if len(self.paa_module.relation_prototypes) == 0 and len(self.paa_module.previous_prototypes) == 0:
            return None
        
        try:

            ent_emb, rel_emb_weight = self.embedding('Train')
            

            head_emb = ent_emb[head]
            rel_emb = rel_emb_weight[rel]
            tail_emb = ent_emb[tail]
            

            composition_mode = getattr(self.args, 'paa_composition_mode', 'concat')
            triple_emb = self.paa_module.compute_triple_embedding(
                head_emb, rel_emb, tail_emb, mode=composition_mode
            )
            

            paa_scores = self.paa_module.compute_paa(triple_emb, rel)
            
            return paa_scores
            
        except Exception as e:
            if self.args.logger:
                self.args.logger.warning(f"[PAA] Error computing PAA scores: {e}")
            return None
    
    def pre_snapshot(self):
        '''
        Preprocess before training on a snapshot
        
        Note: PAA prototypes are NOT updated here. They are updated only in 
        snapshot_post_processing() to ensure:
        - Snapshot N uses prototypes from Snapshot N-1 (learned patterns)
        - This allows PAA to detect pattern drift between snapshots
        - Snapshot 0 has no prototypes (cold start), which is expected
        '''
        pass
    
    def snapshot_post_processing(self):
        '''
        Post process after training on a snapshot
        Updates PAA prototypes if enabled (for next snapshot's use)
        '''
        if self.enable_paa and self.paa_module and self.paa_update_frequency == 'snapshot':
            if self.args.logger:
                self.args.logger.info(f"[PAA] Post-training: Updating prototypes for snapshot {self.args.snapshot}")
            self.paa_module.update_prototypes(self, self.args.snapshot)
    
    def epoch_post_processing(self, size=None):
        '''
        Post process after a training iteration
        Updates PAA prototypes if epoch-level update is enabled
        '''
        if self.enable_paa and self.paa_module and self.paa_update_frequency == 'epoch':
            self.paa_module.update_prototypes(self, self.args.snapshot)


class TransE(LKGE):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)
        self.gcn = MAE(args, kg)

    def MAE_loss(self, paa_score_avg=None):
        '''
        Calculate the MAE loss by masking and reconstructing embeddings.
        Support optional fixed initial graph constraint and PAA modulation.

        Args:
            paa_score_avg: Average PAA score for current batch (optional)
                         Used for PAA modulation: L_new = L_old * (1 + PAA) / 2

        :return: MAE loss (potentially modulated by PAA)
        '''

        if self.fixed_initial_graph and self.initial_edge_index is not None:

            edge_index = self.initial_edge_index
            edge_type = self.initial_edge_type
            num_ent = self.initial_num_ent
            num_rel = self.initial_num_rel


            logger = getattr(self.args, 'logger', None)
            if logger and self.args.snapshot > 0 and self._logged_fixed_graph_for_snapshot != self.args.snapshot:
                logger.info(f"[Fixed Initial Graph] Using initial graph structure "
                           f"for snapshot {self.args.snapshot}")
                self._logged_fixed_graph_for_snapshot = self.args.snapshot
        else:

            num_ent = self.kg.snapshots[self.args.snapshot].num_ent
            num_rel = self.kg.snapshots[self.args.snapshot].num_rel
            edge_index = self.kg.snapshots[self.args.snapshot].edge_index
            edge_type = self.kg.snapshots[self.args.snapshot].edge_type

        '''reconstruct'''
        ent_embeddings, rel_embeddings = self.embedding('Train')

        try:
            old_entity_weight = self.old_weight_entity_embeddings
            old_relation_weight = self.old_weight_relation_embeddings
            old_x = self.old_data_entity_embeddings
            old_r = self.old_data_relation_embeddings
        except:
            old_entity_weight, old_relation_weight = None, None
            old_x, old_r = None, None

        ent_embeddings_reconstruct, rel_embeddings_reconstruct = self.gcn(
            ent_embeddings, rel_embeddings, edge_index, edge_type,
            num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r
        )


        current_num_ent = ent_embeddings.size(0)
        current_num_rel = rel_embeddings.size(0)


        constrained_num_ent = min(num_ent, current_num_ent)
        constrained_num_rel = min(num_rel, current_num_rel)

        loss_ent = self.mse_loss_func(
            ent_embeddings_reconstruct, ent_embeddings[:constrained_num_ent]
        ) / constrained_num_ent

        loss_rel = self.mse_loss_func(
            rel_embeddings_reconstruct, rel_embeddings[:constrained_num_rel]
        ) / constrained_num_rel

        return loss_ent + loss_rel


    def MAE_loss_protected(self, F_triples=None):
        """
        计算带保护的 MAE reconstruction 损失
        
        关键: 排除 F 中的三元组，只对 N(F) 计算蒸馏损失
        
        Args:
            F_triples: Set[(h,r,t), ...] 需要排除的遗忘集三元组
        
        Returns:
            mae_loss: MAE 重构损失
        """
        if F_triples is None:
            F_triples = set()
        

        ent_embeddings, rel_embeddings = self.embedding('Train')
        num_ent = self.kg.snapshots[self.args.snapshot].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot].num_rel
        

        edge_index = self.kg.snapshots[self.args.snapshot].edge_index
        edge_type = self.kg.snapshots[self.args.snapshot].edge_type
        

        if len(F_triples) > 0:
            filtered_edges = []
            filtered_types = []
            
            for i in range(edge_index.size(1)):
                h = edge_index[0, i].item()
                r = edge_type[i].item()
                t = edge_index[1, i].item()
                

                if (h, r, t) not in F_triples:
                    filtered_edges.append([h, t])
                    filtered_types.append(r)
            
            if len(filtered_edges) > 0:
                edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().to(self.args.device)
                edge_type = torch.tensor(filtered_types, dtype=torch.long).to(self.args.device)
            else:
                return torch.tensor(0.0, device=self.args.device)
        

        try:
            old_entity_weight = self.old_weight_entity_embeddings
            old_relation_weight = self.old_weight_relation_embeddings
            old_x = self.old_data_entity_embeddings
            old_r = self.old_data_relation_embeddings
        except:
            old_entity_weight, old_relation_weight = None, None
            old_x, old_r = None, None
        

        ent_embeddings_reconstruct, rel_embeddings_reconstruct = self.gcn(
            ent_embeddings, rel_embeddings, edge_index, edge_type,
            num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r
        )
        

        mae_loss = self.mse_loss_func(ent_embeddings_reconstruct, ent_embeddings[:num_ent]) / num_ent + \
                   self.mse_loss_func(rel_embeddings_reconstruct, rel_embeddings[:num_rel]) / num_rel




        if self.enable_paa_loss_modulation and paa_score_avg is not None and self.paa_mae_modulation_weight > 0.0:

            paa_score_avg = torch.clamp(paa_score_avg, -1.0, 1.0)



            modulation_factor = (1.0 + paa_score_avg) / 2.0

            modulation_factor = 1.0 + self.paa_mae_modulation_weight * (modulation_factor - 1.0)

            mae_loss = mae_loss * modulation_factor


            if self.args.logger and (self.args.snapshot == 0 and torch.rand(1).item() < 0.001):
                self.args.logger.info(
                    f"[PAA MAE Modulation] avg_paa={paa_score_avg.item():.4f}, "
                    f"modulation_factor={modulation_factor.item():.4f}, "
                    f"original_mae={mae_loss.item()/modulation_factor.item():.6f}, "
                    f"modulated_mae={mae_loss.item():.6f}"
                )

        return mae_loss


    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss + MAE loss + regularization loss + DSP loss (repair + forget)
        '''

        new_loss = self.new_loss(head, rel, tail, label)/head.size(0)
        loss = new_loss



        paa_score_avg = None
        if self.enable_paa_loss_modulation and self.enable_paa and self.paa_module is not None:

            has_prototypes = (len(self.paa_module.relation_prototypes) > 0 or
                            len(self.paa_module.previous_prototypes) > 0)
            if has_prototypes:


                if not hasattr(self, '_paa_batch_counter'):
                    self._paa_batch_counter = 0
                self._paa_batch_counter += 1


                update_freq = getattr(self, 'paa_loss_modulation_update_freq', 100)
                if self._paa_batch_counter % update_freq == 1 or not hasattr(self, '_cached_paa_avg'):
                    try:
                        paa_scores = self.compute_paa_scores(head, rel, tail)
                        if paa_scores is not None:
                            self._cached_paa_avg = paa_scores.mean()

                            if self.args.logger and self._paa_batch_counter % (update_freq * 10) == 1:
                                self.args.logger.info(f"[PAA Modulation] Computed fresh PAA scores for batch {self._paa_batch_counter} (every {update_freq} batches)")
                    except Exception as e:
                        if self.args.logger:
                            self.args.logger.warning(f"[PAA Loss Modulation] Failed to compute PAA scores: {e}")
                        self._cached_paa_avg = None


                paa_score_avg = self._cached_paa_avg

        if self.args.using_reconstruct_loss == 'True':
            MAE_loss = self.MAE_loss(paa_score_avg=paa_score_avg)

            if self.adaptive_mae_weight:
                adaptive_weight = self.get_adaptive_mae_weight(self.kg.snapshots[self.args.snapshot])
                loss += adaptive_weight * MAE_loss
            else:
                loss += float(self.args.reconstruct_weight) * MAE_loss
        if self.args.using_regular_loss == 'True':
            regular_loss = self.lkge_regular_loss(paa_score_avg=paa_score_avg)
            loss += float(self.args.regular_weight)*regular_loss
        




        if hasattr(self.args, 'lambda_dsp') and self.args.lambda_dsp > 0:
            dsp_loss_value, forget_loss_value = self.compute_dsp_loss()
            

            if getattr(self.args, 'adaptive_dsp', True) and hasattr(self, 'current_adaptive_weight'):
                effective_lambda_dsp = float(self.args.lambda_dsp) * self.current_adaptive_weight
            else:
                effective_lambda_dsp = float(self.args.lambda_dsp)
            
            weighted_dsp_loss = effective_lambda_dsp * dsp_loss_value
            loss += weighted_dsp_loss
            

            if getattr(self.args, 'enable_unlearning', False) and forget_loss_value.item() > 0:
                lambda_forget = getattr(self.args, 'lambda_forget', 0.05)
                

                if getattr(self.args, 'adaptive_dsp', True) and hasattr(self, 'current_adaptive_weight'):
                    effective_lambda_forget = lambda_forget * self.current_adaptive_weight
                else:
                    effective_lambda_forget = lambda_forget
                
                weighted_forget_loss = effective_lambda_forget * forget_loss_value
                loss += weighted_forget_loss
                

                if hasattr(self.args, 'epoch') and self.args.epoch % 10 == 0:
                    self.args.logger.info(
                        f"[PAAL] Epoch {self.args.epoch}: "
                        f"DSP Loss={weighted_dsp_loss.item():.4f} (λ={effective_lambda_dsp:.4f}), "
                        f"Forget Loss={weighted_forget_loss.item():.4f} (λ={effective_lambda_forget:.4f}), "
                        f"Q_avg={getattr(self, 'last_mean_Q', 0):.3f}"
                    )
            else:

                if hasattr(self.args, 'epoch') and self.args.epoch % 10 == 0 and weighted_dsp_loss.item() > 0:

                    current_epoch = self.args.epoch
                    if not hasattr(self, '_last_logged_dsp_epoch') or self._last_logged_dsp_epoch != current_epoch:
                        self._last_logged_dsp_epoch = current_epoch
                        self.args.logger.info(
                            f"[PAAL] Epoch {current_epoch}: "
                            f"DSP Loss={weighted_dsp_loss.item():.4f} | "
                            f"λ_eff={effective_lambda_dsp:.4f} | "
                            f"Q_avg={getattr(self, 'last_mean_Q', 0):.3f}"
                        )
        

        enable_noise_loss = getattr(self.args, 'enable_noise_sampling_loss', False)
        enable_noise_sampling = getattr(self.args, 'enable_noise_sampling', True)

        if enable_noise_sampling and enable_noise_loss:
            noise_loss = self.compute_noise_sampling_loss()
            noise_weight = getattr(self.args, 'noise_sampling_loss_weight', 10.0)
            if noise_loss.item() > 0:
                loss += noise_weight * noise_loss


                if hasattr(self.args, 'epoch') and self.args.epoch % 10 == 0:

                    current_epoch = self.args.epoch
                    if not hasattr(self, '_last_logged_noise_epoch') or self._last_logged_noise_epoch != current_epoch:
                        self._last_logged_noise_epoch = current_epoch
                        self.args.logger.info(
                            f"[Noise Sampling] Epoch {current_epoch}: "
                            f"Noise Loss={noise_loss.item():.6f} | "
                            f"Weight={noise_weight:.4f} | "
                            f"Weighted Loss={(noise_weight * noise_loss).item():.6f}"
                        )



            reg_added = False
            if self.enable_learnable_noise_scale and self.noise_scale_logit is not None:
                theta_r = torch.sigmoid(self.noise_scale_logit)

                reg_loss_r = 0.0001 * torch.relu(0.01 - theta_r) + 0.0001 * torch.relu(theta_r - 0.5)
                loss += reg_loss_r
                reg_added = True

            if self.enable_learnable_noise_factor and self.noise_factor_logit is not None:
                theta_s = torch.sigmoid(self.noise_factor_logit)

                reg_loss_s = 0.0001 * torch.relu(0.1 - theta_s) + 0.0001 * torch.relu(theta_s - 0.9)
                loss += reg_loss_s
                reg_added = True


            if reg_added and hasattr(self.args, 'epoch') and self.args.epoch == 0 and torch.rand(1).item() < 0.01:
                print(f"TEMP REG: Regularization added to loss")
                

                if hasattr(self.args, 'epoch') and self.args.epoch % 10 == 0:
                    current_theta_r = self.get_current_noise_scale(as_tensor=False)
                    current_theta_s = self.get_current_noise_factor(as_tensor=False)
                    learnable_r = "learnable" if self.enable_learnable_noise_scale else "fixed"
                    learnable_s = "learnable" if self.enable_learnable_noise_factor else "fixed"
                    self.args.logger.info(
                        f"[Noise Sampling] Epoch {self.args.epoch}: "
                        f"Noise Loss={noise_loss.item():.4f} (weight={noise_weight:.4f}), "
                        f"Triples={len(getattr(self, 'current_high_quality_triples', []))}, "
                        f"θr={current_theta_r:.4f} ({learnable_r}), θs={current_theta_s:.4f} ({learnable_s})"
                    )

        return loss

    def detect_low_quality_triples(self, triples, num_mc=5, gamma=0.1, tau_Q=0.3, tau_U=0.01):
        '''
        Detect low-quality facts using DQA logic
        For TransE: score is distance (lower is better)
        Q = sigmoid(-gamma * score) means lower distance -> higher quality
        '''
        import math
        low_quality_triples = []
        

        was_training = self.training
        self.train()
        
        Q_values = []
        U_values = []
        score_values = []
        

        dsp_gamma = getattr(self.args, 'dqa_gamma', 0.1)
        enable_temporal_dsp = getattr(self.args, 'enable_temporal_score', False)
        enable_structural_dsp = getattr(self.args, 'enable_structural_score', False)

        for triple in triples:
            h, r, t = triple
            h_tensor = torch.tensor([h], dtype=torch.long, device=self.args.device)
            r_tensor = torch.tensor([r], dtype=torch.long, device=self.args.device)
            t_tensor = torch.tensor([t], dtype=torch.long, device=self.args.device)


            with torch.no_grad():
                ent_embeddings, rel_embeddings = self.embedding('Train')
                h_emb = torch.index_select(ent_embeddings, 0, h_tensor)
                r_emb = torch.index_select(rel_embeddings, 0, r_tensor)
                t_emb = torch.index_select(ent_embeddings, 0, t_tensor)
                score = self.score_fun(h_emb, r_emb, t_emb).item()


            Q = self.compute_multi_dimensional_quality(
                triple, score, dsp_gamma,
                use_temporal=enable_temporal_dsp,
                use_structural=enable_structural_dsp
            )



            U = 0.0

            Q_values.append((Q, triple))
            U_values.append(U)
            score_values.append(score)
        

        if not was_training:
            self.eval()
        

        lq_ratio = getattr(self.args, 'lq_ratio', 0.3)
        

        Q_values.sort(key=lambda x: x[0])
        num_low_quality = max(1, int(len(Q_values) * lq_ratio))
        low_quality_triples = [triple for _, triple in Q_values[:num_low_quality]]
        

        mean_Q = sum(q for q, _ in Q_values) / len(Q_values) if Q_values else 0.0
        mean_U = sum(U_values) / len(U_values) if U_values else 0.0
        

        return low_quality_triples, mean_Q, mean_U
    
    def compute_temporal_score(self, triple, current_snapshot=None):
        '''
        Compute temporal quality score for a triple
        Based on age penalty and frequency penalty

        Args:
            triple: (h, r, t) tuple
            current_snapshot: Current snapshot index (for age calculation)
        Returns:
            temporal_score: Quality score based on temporal features (0-1, higher is better)
        '''
        h, r, t = triple


        age_penalty = getattr(self.args, 'temporal_age_penalty', 0.01)
        freq_penalty = getattr(self.args, 'temporal_freq_penalty', 0.1)





        if current_snapshot is not None:


            age_factor = min(1.0, current_snapshot / 10.0)
            age_score = 1.0 - age_penalty * age_factor
        else:
            age_score = 1.0



        freq_factor = 0.5
        freq_score = 1.0 - freq_penalty * freq_factor


        temporal_score = (age_score + freq_score) / 2.0

        return max(0.0, min(1.0, temporal_score))

    def compute_structural_score(self, triple):
        '''
        Compute structural quality score for a triple
        Based on degree penalty and relation frequency penalty

        Args:
            triple: (h, r, t) tuple
        Returns:
            structural_score: Quality score based on structural features (0-1, higher is better)
        '''
        h, r, t = triple


        degree_penalty = getattr(self.args, 'structural_degree_penalty', 0.05)
        relation_penalty = getattr(self.args, 'structural_relation_penalty', 0.02)






        h_degree = 7
        t_degree = 6


        optimal_degree = 8.0
        h_degree_score = max(0.0, 1.0 - degree_penalty * abs(h_degree - optimal_degree))
        t_degree_score = max(0.0, 1.0 - degree_penalty * abs(t_degree - optimal_degree))
        degree_score = (h_degree_score + t_degree_score) / 2.0



        common_relations = {0, 1, 2}
        relation_freq_score = 1.0 - relation_penalty if r in common_relations else 0.8 - relation_penalty


        structural_score = (degree_score + relation_freq_score) / 2.0

        return max(0.0, min(1.0, structural_score))

    def compute_multi_dimensional_quality(self, triple, transE_score, gamma=0.1, use_temporal=True, use_structural=True):
        '''
        Compute multi-dimensional quality score combining TransE, temporal, and structural features

        Args:
            triple: (h, r, t) tuple
            transE_score: Raw TransE distance score
            gamma: Sensitivity parameter for TransE quality conversion
            use_temporal: Whether to include temporal score
            use_structural: Whether to include structural score
        Returns:
            Q_total: Combined quality score (0-1, higher is better)
        '''
        import math


        Q_transE = 1.0 / (1.0 + math.exp(gamma * transE_score))


        temporal_weight = getattr(self.args, 'temporal_weight', 0.1) if use_temporal else 0.0
        structural_weight = getattr(self.args, 'structural_weight', 0.1) if use_structural else 0.0


        total_weight = 1.0 + temporal_weight + structural_weight
        w_transE = 1.0 / total_weight
        w_temporal = temporal_weight / total_weight
        w_structural = structural_weight / total_weight


        Q_temporal = self.compute_temporal_score(triple) if use_temporal else 0.0
        Q_structural = self.compute_structural_score(triple) if use_structural else 0.0


        Q_total = w_transE * Q_transE + w_temporal * Q_temporal + w_structural * Q_structural

        return Q_total

    def detect_high_quality_triples(self, triples, num_mc=5, gamma=0.1):
        '''
        Detect high-quality facts for noise sampling using percentile selection
        Select top percentile% triples by quality score

        Args:
            triples: List of triples to evaluate
            num_mc: Number of Monte Carlo samples (not used for TransE)
            gamma: Sensitivity parameter for quality score
        Returns:
            high_quality_triples: List of (Q, triple) tuples selected by percentile
            mean_Q: Average quality score of selected triples
        '''
        import math


        percentile = getattr(self.args, 'high_quality_percentile', 0.3)
        gamma = getattr(self.args, 'high_quality_gamma', 0.1)
        enable_temporal = getattr(self.args, 'enable_temporal_score', False)
        enable_structural = getattr(self.args, 'enable_structural_score', False)

        Q_values = []


        was_training = self.training
        self.train()

        for triple in triples:
            h, r, t = triple
            h_tensor = torch.tensor([h], dtype=torch.long, device=self.args.device)
            r_tensor = torch.tensor([r], dtype=torch.long, device=self.args.device)
            t_tensor = torch.tensor([t], dtype=torch.long, device=self.args.device)


            with torch.no_grad():
                ent_embeddings, rel_embeddings = self.embedding('Train')
                h_emb = torch.index_select(ent_embeddings, 0, h_tensor)
                r_emb = torch.index_select(rel_embeddings, 0, r_tensor)
                t_emb = torch.index_select(ent_embeddings, 0, t_tensor)
                score = self.score_fun(h_emb, r_emb, t_emb).item()


            Q = self.compute_multi_dimensional_quality(
                triple, score, gamma,
                use_temporal=enable_temporal,
                use_structural=enable_structural
            )

            Q_values.append((Q, triple))


        if not was_training:
            self.eval()


        Q_values.sort(key=lambda x: x[0], reverse=True)


        num_to_select = max(1, int(len(Q_values) * percentile))
        high_quality_triples = Q_values[:num_to_select]


        max_count = getattr(self.args, 'high_quality_max_count', 50)
        if len(high_quality_triples) > max_count:
            high_quality_triples = high_quality_triples[:max_count]


        mean_Q_all = sum(q for q, _ in Q_values) / len(Q_values) if Q_values else 0.0
        mean_Q_high = sum(q for q, _ in high_quality_triples) / len(high_quality_triples) if high_quality_triples else 0.0

        return high_quality_triples, mean_Q_high
    
    def apply_gaussian_noise_sampling(self, embedding, theta_r, phi_m=None, mu_m=None):
        '''
        Apply Gaussian noise sampling: Ng = θr (φm ⊙ g + µm)
        where g ~ K(0, Z) is Gaussian noise
        '''
        device = embedding.device
        emb_dim = embedding.size(-1)
        

        g = torch.randn_like(embedding)
        
        if phi_m is None:

            phi_m = torch.std(embedding, dim=-1, keepdim=True) if embedding.dim() > 1 else torch.std(embedding)
        if mu_m is None:

            mu_m = torch.mean(embedding, dim=-1, keepdim=True) if embedding.dim() > 1 else torch.mean(embedding)
        

        if embedding.dim() == 1:
            phi_m = phi_m if isinstance(phi_m, torch.Tensor) else torch.tensor(phi_m, device=device)
            mu_m = mu_m if isinstance(mu_m, torch.Tensor) else torch.tensor(mu_m, device=device)
        else:
            if phi_m.dim() == 0:
                phi_m = phi_m.unsqueeze(0).expand_as(embedding)
            if mu_m.dim() == 0:
                mu_m = mu_m.unsqueeze(0).expand_as(embedding)
        

        Ng = theta_r * (phi_m * g + mu_m)
        
        return Ng
    
    def apply_uniform_noise_sampling(self, embedding, theta_r, beta_min=-0.1, beta_max=0.1):
        '''
        Apply uniform noise sampling: Na = θr [βmin + (βmax - βmin) ⊙ U]
        where U ~ Uniform[0, 1]
        '''
        device = embedding.device
        

        U = torch.rand_like(embedding)
        

        Na = theta_r * (beta_min + (beta_max - beta_min) * U)
        
        return Na
    
    def adaptive_noise_sampling(self, embedding, theta_r, theta_n, mode='gaussian', 
                                 phi_m=None, mu_m=None, beta_min=-0.1, beta_max=0.1):
        '''
        Apply adaptive noise sampling strategy:
        Vn = (1 - θr) * Vo + Nm if ρ > θn else Vo
        
        Args:
            embedding: Original embedding vector Vo
            theta_r: Learnable noise scale factor
            theta_n: Adaptive sampling factor (0-1)
            mode: 'gaussian', 'uniform', or 'mixed'
        Returns:
            Noisy embedding Vn
        '''
        device = embedding.device
        


        batch_size = embedding.size(0)
        rho = torch.rand(batch_size, device=device)


        noise_mask = (rho > theta_n).float().unsqueeze(-1)


        if mode == 'gaussian':
            Nm = self.apply_gaussian_noise_sampling(embedding, theta_r, phi_m, mu_m)
        elif mode == 'uniform':
            Nm = self.apply_uniform_noise_sampling(embedding, theta_r, beta_min, beta_max)
        elif mode == 'mixed':

            mode_choice = torch.rand(batch_size, device=device) > 0.5
            Nm_gaussian = self.apply_gaussian_noise_sampling(embedding, theta_r, phi_m, mu_m)
            Nm_uniform = self.apply_uniform_noise_sampling(embedding, theta_r, beta_min, beta_max)
            Nm = torch.where(mode_choice.unsqueeze(-1), Nm_gaussian, Nm_uniform)
        else:
            raise ValueError(f"Unknown noise mode: {mode}")



        Vn = (1 - theta_r) * embedding + theta_r * Nm
        Vn = torch.where(noise_mask.bool(), Vn, embedding)
        
        return Vn
    
    def get_current_noise_scale(self, as_tensor=False):
        '''
        Get current noise scale factor θr
        If learnable, returns sigmoid(noise_scale_logit) which adapts during training
        Otherwise, returns fixed initial value

        Args:
            as_tensor: If True, returns tensor (for gradient computation);
                      If False, returns scalar value (for logging)
        '''
        if self.enable_learnable_noise_scale and self.noise_scale_logit is not None:

            theta_r = torch.sigmoid(self.noise_scale_logit)
            if as_tensor:
                return theta_r
            else:
                return theta_r.item()
        else:

            fixed_value = getattr(self.args, 'noise_scale_init', 0.1)
            if as_tensor:
                return torch.tensor(fixed_value, device=self.args.device)
            else:
                return fixed_value

    def get_current_noise_factor(self, as_tensor=False):
        '''
        Get current noise sampling factor θs
        If learnable, returns sigmoid(noise_factor_logit) which adapts during training
        Otherwise, returns fixed initial value

        Args:
            as_tensor: If True, returns tensor (for gradient computation);
                      If False, returns scalar value (for logging)
        '''
        if self.enable_learnable_noise_factor and self.noise_factor_logit is not None:

            theta_s = torch.sigmoid(self.noise_factor_logit)
            if as_tensor:
                return theta_s
            else:
                return theta_s.item()
        else:

            fixed_value = getattr(self.args, 'noise_sampling_factor_init', 0.5)
            if as_tensor:
                return torch.tensor(fixed_value, device=self.args.device)
            else:
                return fixed_value
    
    def apply_noise_sampling_to_high_quality_triples(self, high_quality_triples):
        '''
        Apply adaptive noise sampling to high-quality triples to enhance training
        Returns list of (noisy_embeddings, original_triple) pairs
        '''
        if not high_quality_triples:
            return []
        

        enable_noise = getattr(self.args, 'enable_noise_sampling', True)
        if not enable_noise:
            return []
        
        noise_mode = getattr(self.args, 'noise_sampling_mode', 'gaussian')

        theta_r = self.get_current_noise_scale(as_tensor=True)
        theta_s = self.get_current_noise_factor(as_tensor=True)
        theta_n = theta_s

        beta_min = getattr(self.args, 'noise_beta_min', -0.1)
        beta_max = getattr(self.args, 'noise_beta_max', 0.1)


        sampled_triples = high_quality_triples
        

        ent_embeddings, rel_embeddings = self.embedding('Train')
        
        noisy_triples = []
        for q, (h, r, t) in sampled_triples:
            h_tensor = torch.tensor([h], dtype=torch.long, device=self.args.device)
            r_tensor = torch.tensor([r], dtype=torch.long, device=self.args.device)
            t_tensor = torch.tensor([t], dtype=torch.long, device=self.args.device)
            

            h_emb = torch.index_select(ent_embeddings, 0, h_tensor).squeeze(0)
            r_emb = torch.index_select(rel_embeddings, 0, r_tensor).squeeze(0)
            t_emb = torch.index_select(ent_embeddings, 0, t_tensor).squeeze(0)
            

            phi_m_h = torch.std(h_emb)
            mu_m_h = torch.mean(h_emb)
            phi_m_t = torch.std(t_emb)
            mu_m_t = torch.mean(t_emb)
            

            h_emb_noisy = self.adaptive_noise_sampling(
                h_emb, theta_r, theta_n, mode=noise_mode,
                phi_m=phi_m_h, mu_m=mu_m_h, beta_min=beta_min, beta_max=beta_max
            )
            t_emb_noisy = self.adaptive_noise_sampling(
                t_emb, theta_r, theta_n, mode=noise_mode,
                phi_m=phi_m_t, mu_m=mu_m_t, beta_min=beta_min, beta_max=beta_max
            )
            

            noisy_triples.append({
                'triple': (h, r, t),
                'h_emb': h_emb_noisy,
                'r_emb': r_emb,
                't_emb': t_emb_noisy,
                'quality': q
            })
        
        return noisy_triples
    
    def compute_noise_sampling_loss(self):
        '''
        Compute robustness loss for high-quality triples.
        
        策略：对高质量三元组应用噪声扰动，计算鲁棒性损失
        - 不使用锚点（简化设计）
        - 损失 = |score_noisy - score_clean|
        - 使用可学习的θr和θs自动调节噪声强度
        '''
        if not hasattr(self, 'current_high_quality_triples') or not self.current_high_quality_triples:
            return torch.tensor(0.0, device=self.args.device)

        total_loss = torch.tensor(0.0, device=self.args.device)
        count = 0
        

        max_process = min(getattr(self.args, 'noise_sampling_max_process', 10), 
                         len(self.current_high_quality_triples))

        for triple_data in self.current_high_quality_triples[:max_process]:
            if len(triple_data) >= 2:
                high_quality_triple = triple_data[1]
                

                loss_val = self.compute_noise_sampling_loss_for_pair(
                    high_quality_triple, 
                    anchor_triple=None,
                    beta=None,
                    lambda_temp=None
                )
                total_loss += loss_val
                count += 1

        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=self.args.device)

    def compute_noise_sampling_loss_for_pair(self, high_quality_triple, anchor_triple, beta=3.0, lambda_temp=1.0):
        '''
        Compute robustness loss for high-quality triples with PAA-controlled noise perturbation.
        
        核心思想：鲁棒性损失 + PAA控制的自适应噪声
        - PAA控制噪声方差：sigma^2 = sigma_max^2 * (1 + PAA) / 2
        - sigma_max由可学习的θr和θs决定
        - 损失 = |score_noisy - score_clean|
        
        Args:
            high_quality_triple: 高质量三元组 (h, r, t)
            anchor_triple: 不再使用锚点（占位参数，保持接口兼容）
            beta: 未使用（保持接口兼容）
            lambda_temp: 未使用（保持接口兼容）
        
        Returns:
            loss: 鲁棒性损失标量
        '''
        h_hq, r_hq, t_hq = high_quality_triple
        

        h_hq_tensor = torch.tensor([h_hq], dtype=torch.long, device=self.args.device)
        r_hq_tensor = torch.tensor([r_hq], dtype=torch.long, device=self.args.device)
        t_hq_tensor = torch.tensor([t_hq], dtype=torch.long, device=self.args.device)
        

        ent_embeddings, rel_embeddings = self.embedding('Train')
        h_hq_emb = torch.index_select(ent_embeddings, 0, h_hq_tensor)
        r_hq_emb = torch.index_select(rel_embeddings, 0, r_hq_tensor)
        t_hq_emb = torch.index_select(ent_embeddings, 0, t_hq_tensor)
        

        f_clean = self.score_fun(h_hq_emb, r_hq_emb, t_hq_emb)
        

        paa_score = None
        if hasattr(self, 'enable_paa') and self.enable_paa and self.paa_module is not None:
            if len(self.paa_module.relation_prototypes) > 0 or len(self.paa_module.previous_prototypes) > 0:
                paa_score = self.compute_paa_scores(h_hq_tensor, r_hq_tensor, t_hq_tensor)
                if paa_score is not None:
                    paa_score = paa_score.detach()
        

        theta_r = self.get_current_noise_scale(as_tensor=True)
        theta_s = self.get_current_noise_factor(as_tensor=True)
        

        sigma_max = theta_r * theta_s
        





        if paa_score is not None:
            paa_modulation = torch.sqrt((1.0 + paa_score) / 2.0)
            sigma = sigma_max * paa_modulation
        else:
            sigma = sigma_max
        

        noise_h = torch.randn_like(h_hq_emb) * sigma
        noise_t = torch.randn_like(t_hq_emb) * sigma
        

        h_hq_emb_noisy = h_hq_emb + noise_h
        t_hq_emb_noisy = t_hq_emb + noise_t
        

        f_noisy = self.score_fun(h_hq_emb_noisy, r_hq_emb, t_hq_emb_noisy)
        

        robustness_loss = torch.abs(f_noisy - f_clean)
        

        if torch.rand(1).item() < 0.01:
            if paa_score is not None:
                print(f"[DEBUG PAA+Noise] triple=({h_hq},{r_hq},{t_hq}), "
                      f"f_clean={f_clean.item():.4f}, f_noisy={f_noisy.item():.4f}, "
                      f"diff={robustness_loss.item():.4f}, "
                      f"PAA={paa_score.item():.4f}, sigma={sigma.item():.4f}, "
                      f"theta_r={theta_r.item():.4f}, theta_s={theta_s.item():.4f}")
            else:
                print(f"[DEBUG Noise] triple=({h_hq},{r_hq},{t_hq}), "
                      f"f_clean={f_clean.item():.4f}, f_noisy={f_noisy.item():.4f}, "
                      f"diff={robustness_loss.item():.4f}, "
                      f"theta_r={theta_r.item():.4f}, theta_s={theta_s.item():.4f}")
        
        return robustness_loss.squeeze()
    
    def build_structural_anchors(self, triple, num_anchors=2, candidates=30):
        '''
        Build structural contrastive anchors for a given triple using multiple strategies:
        - TDA (Tail Distance Anchors): Original strategy, farthest tail entities
        - RCA (Rule Consistent Anchors): Common tails for the relation
        - CMEA (Counterfactual Minimal-Edit Anchors): Nearest embedding neighbors
        '''
        h, r, t = triple
        all_anchors = []
        

        ent_embeddings, _ = self.embedding('Train')
        num_entities = self.kg.snapshots[self.args.snapshot].num_ent
        

        if getattr(self.args, 'sca_tda', True):
            tda_anchors = self._build_tda_anchors(h, r, t, ent_embeddings, num_entities, num_anchors, candidates)
            all_anchors.extend(tda_anchors)
        

        if getattr(self.args, 'sca_rca', False):
            rca_anchors = self._build_rca_anchors(h, r, t, num_anchors)
            all_anchors.extend(rca_anchors)
        

        if getattr(self.args, 'sca_cmea', False):
            cmea_anchors = self._build_cmea_anchors(h, r, t, ent_embeddings, num_entities, num_anchors)
            all_anchors.extend(cmea_anchors)
        

        seen = set()
        unique_anchors = []
        for anchor in all_anchors:
            if anchor not in seen:
                seen.add(anchor)
                unique_anchors.append(anchor)
        
        return unique_anchors if unique_anchors else [(h, r, t)]
    
    def _build_tda_anchors(self, h, r, t, ent_embeddings, num_entities, num_anchors, candidates):
        '''
        TDA: Tail Distance Anchors - select farthest tail entities
        '''
        t_emb = ent_embeddings[t].unsqueeze(0)
        

        distances = torch.norm(ent_embeddings[:num_entities] - t_emb, dim=1)
        

        distances[t] = float('-inf')
        

        _, farthest_indices = torch.topk(distances, k=min(candidates, num_entities - 1), largest=True)
        

        anchor_entities = farthest_indices[:num_anchors]
        

        return [(h, r, t_alt.item()) for t_alt in anchor_entities]
    
   
    
    def _build_rca_anchors(self, h, r, t, num_anchors):
        '''
        RCA: Rule Consistent Anchors - use common tails for relation r
        Find the most frequent tail entities that appear with relation r
        
        优化版本：使用预计算的缓存，O(1) 查询而非 O(N) 遍历
        '''
        snapshot_id = self.args.snapshot
        

        if snapshot_id not in self.rca_cache:
            self._build_rca_cache(snapshot_id)
        

        if r not in self.rca_cache[snapshot_id]:

            return []
        
        sorted_tails = self.rca_cache[snapshot_id][r]
        

        top_tails = []
        for tail, count in sorted_tails:
            if tail != t:
                top_tails.append(tail)
                if len(top_tails) >= num_anchors:
                    break
        

        return [(h, r, t_alt) for t_alt in top_tails]
    
    def _build_cmea_anchors(self, h, r, t, ent_embeddings, num_entities, num_anchors):
        '''
        CMEA: Counterfactual Minimal-Edit Anchors
        Replace head or tail with nearest embedding neighbors
        '''
        anchors = []
        

        h_emb = ent_embeddings[h].unsqueeze(0)
        t_emb = ent_embeddings[t].unsqueeze(0)
        

        t_distances = torch.norm(ent_embeddings[:num_entities] - t_emb, dim=1)
        t_distances[t] = float('inf')
        _, nearest_t_indices = torch.topk(t_distances, k=min(num_anchors, num_entities - 1), largest=False)
        
        for t_alt in nearest_t_indices:
            anchors.append((h, r, t_alt.item()))
        

        h_distances = torch.norm(ent_embeddings[:num_entities] - h_emb, dim=1)
        h_distances[h] = float('inf')
        _, nearest_h_indices = torch.topk(h_distances, k=min(num_anchors, num_entities - 1), largest=False)
        
        for h_alt in nearest_h_indices:
            anchors.append((h_alt.item(), r, t))
        
        return anchors
    
    
    def compute_dsp_loss_for_pair(self, bad_triple, good_triple, beta=3.0, lambda_temp=1.0, paa_score=None):

        h_bad, r_bad, t_bad = bad_triple
        h_good, r_good, t_good = good_triple
        
        h_bad_tensor = torch.tensor([h_bad], dtype=torch.long, device=self.args.device)
        r_bad_tensor = torch.tensor([r_bad], dtype=torch.long, device=self.args.device)
        t_bad_tensor = torch.tensor([t_bad], dtype=torch.long, device=self.args.device)
        h_good_tensor = torch.tensor([h_good], dtype=torch.long, device=self.args.device)
        r_good_tensor = torch.tensor([r_good], dtype=torch.long, device=self.args.device)
        t_good_tensor = torch.tensor([t_good], dtype=torch.long, device=self.args.device)
        

        ent_embeddings, rel_embeddings = self.embedding('Train')
        

        h_bad_emb = torch.index_select(ent_embeddings, 0, h_bad_tensor)
        r_bad_emb = torch.index_select(rel_embeddings, 0, r_bad_tensor)
        t_bad_emb = torch.index_select(ent_embeddings, 0, t_bad_tensor)
        f_bad = self.score_fun(h_bad_emb, r_bad_emb, t_bad_emb)
        
        h_good_emb = torch.index_select(ent_embeddings, 0, h_good_tensor)
        r_good_emb = torch.index_select(rel_embeddings, 0, r_good_tensor)
        t_good_emb = torch.index_select(ent_embeddings, 0, t_good_tensor)
        f_good = self.score_fun(h_good_emb, r_good_emb, t_good_emb)
        


        s_bad = torch.sigmoid(lambda_temp * f_bad)
        s_good = torch.sigmoid(lambda_temp * f_good)
        


        log_s_good = torch.log(s_good + 1e-10)
        log_s_bad = torch.log(s_bad + 1e-10)
        diff = log_s_good - log_s_bad
        
        inner = beta * diff
        sigmoid_inner = torch.sigmoid(inner)
        
        L = -torch.log(sigmoid_inner + 1e-10)
        

        if paa_score is not None and hasattr(self, 'enable_paa') and self.enable_paa:





            paa_weight = 1.0 - (1.0 + paa_score) / 2.0
            L = L * paa_weight
            

            if torch.rand(1).item() < 0.01:
                print(f"[DEBUG DSP+PAA] triple=({h_bad},{r_bad},{t_bad}), anchor=({h_good},{r_good},{t_good}), "
                      f"f_bad={f_bad.item():.4f}, f_good={f_good.item():.4f}, "
                      f"s_bad={s_bad.item():.4f}, s_good={s_good.item():.4f}, diff={diff.item():.4f}, "
                      f"PAA={paa_score.item():.4f}, w_repel={paa_weight.item():.4f}, "
                      f"loss_before={(-torch.log(sigmoid_inner + 1e-10)).item():.4f}, "
                      f"loss_after={L.item():.4f}")
        
        return L.squeeze()
    
    def compute_dsp_loss(self):
        '''
        Compute DSP loss with F/R split:
        - If unlearning is enabled: split C into F (forget) and R (repair)
        - Otherwise: use original DSP logic for all low-quality triples
        '''
        import random
        

        triples = list(self.kg.snapshots[self.args.snapshot].train_new)
        
        if len(triples) == 0:
            return torch.tensor(0.0, device=self.args.device), torch.tensor(0.0, device=self.args.device)
        

        enable_unlearning = getattr(self.args, 'enable_unlearning', False)
        
        if enable_unlearning and self.forget_loss_module is not None:

            

            sample_size = min(getattr(self.args, 'dsp_sample_size', 20), len(triples))
            sampled_triples = random.sample(triples, sample_size)
            
            gamma = getattr(self.args, 'dqa_gamma', 0.1)
            low_quality, mean_Q, mean_U = self.detect_low_quality_triples(
                sampled_triples, num_mc=5, gamma=gamma, tau_Q=0.3, tau_U=0.01
            )
            

            enable_noise_sampling = getattr(self.args, 'enable_noise_sampling', True)
            if enable_noise_sampling:

                hq_sample_size = min(getattr(self.args, 'high_quality_sample_size', 100), len(triples))
                hq_sampled_triples = random.sample(triples, hq_sample_size)


                high_quality, mean_Q_high = self.detect_high_quality_triples(
                    hq_sampled_triples, num_mc=5, gamma=getattr(self.args, 'high_quality_gamma', 0.1)
                )

                if not hasattr(self, 'current_high_quality_triples'):
                    self.current_high_quality_triples = []
                self.current_high_quality_triples = high_quality
                

                if hasattr(self.args, 'epoch') and self.args.epoch % 10 == 0:
                    percentile = getattr(self.args, 'high_quality_percentile', 0.3)
                    max_count = getattr(self.args, 'high_quality_max_count', 50)
                    current_theta_r = self.get_current_noise_scale(as_tensor=False)
                    current_theta_s = self.get_current_noise_factor(as_tensor=False)
                    learnable_r = "learnable" if self.enable_learnable_noise_scale else "fixed"
                    learnable_s = "learnable" if self.enable_learnable_noise_factor else "fixed"
                    self.args.logger.info(
                        f"[High-Quality Selection] Epoch {self.args.epoch}: "
                        f"Detected={len(high_quality)}, Selected={len(high_quality)}, "
                        f"Percentile={percentile:.2f}, MaxCount={max_count}, "
                        f"MeanQ={mean_Q_high:.3f}, θr={current_theta_r:.4f} ({learnable_r}), "
                        f"θs={current_theta_s:.4f} ({learnable_s})"
                    )
            
            self.last_mean_Q = mean_Q
            
            if len(low_quality) == 0:
                self.current_adaptive_weight = 1.0
                return torch.tensor(0.0, device=self.args.device), torch.tensor(0.0, device=self.args.device)
            

            total_dsp_loss = torch.tensor(0.0, device=self.args.device)
            count = 0
            
            max_process = min(getattr(self.args, 'dsp_max_process', 3), len(low_quality))
            num_anchors = getattr(self.args, 'dsp_num_anchors', 2)
            beta = getattr(self.args, 'dsp_beta', 3.0)
            
            for triple in low_quality[:max_process]:

                paa_score = None
                if hasattr(self, 'enable_paa') and self.enable_paa and hasattr(self, 'paa_module') and self.paa_module is not None:
                    h, r, t = triple
                    paa_score = self.compute_paa_scores(
                        torch.tensor([h], device=self.args.device),
                        torch.tensor([r], device=self.args.device),
                        torch.tensor([t], device=self.args.device)
                    )
                    if paa_score is not None:
                        paa_score = paa_score[0]
                
                anchors = self.build_structural_anchors(triple, num_anchors=num_anchors, candidates=30)
                lambda_temp = getattr(self.args, 'dsp_temperature', 1.0)
                

                if len(anchors) > 0:
                    batch_loss = self.compute_dsp_loss_for_pair_batch(triple, anchors, beta=beta, lambda_temp=lambda_temp, paa_score=paa_score)
                    total_dsp_loss += batch_loss
                    count += len(anchors)
            
            if count > 0:
                dsp_loss = total_dsp_loss / count
            else:
                dsp_loss = torch.tensor(0.0, device=self.args.device)
            

            if getattr(self.args, 'adaptive_dsp', True):
                alpha = getattr(self.args, 'alpha_dsp', 0.5)
                adaptive_weight = alpha * mean_U + (1 - alpha) * (1 - mean_Q)
                self.current_adaptive_weight = adaptive_weight
            else:
                self.current_adaptive_weight = 1.0
            


            forget_loss = torch.tensor(0.0, device=self.args.device)
            


            
            return dsp_loss, forget_loss
        
        else:

            

            sample_size = min(getattr(self.args, 'dsp_sample_size', 20), len(triples))
            sampled_triples = random.sample(triples, sample_size)
            

            gamma = getattr(self.args, 'dqa_gamma', 0.1)
            low_quality, mean_Q, mean_U = self.detect_low_quality_triples(
                sampled_triples, num_mc=5, gamma=gamma, tau_Q=0.3, tau_U=0.01
            )
            

            enable_noise_sampling = getattr(self.args, 'enable_noise_sampling', True)
            if enable_noise_sampling:

                hq_sample_size = min(getattr(self.args, 'high_quality_sample_size', 100), len(triples))
                hq_sampled_triples = random.sample(triples, hq_sample_size)


                high_quality, mean_Q_high = self.detect_high_quality_triples(
                    hq_sampled_triples, num_mc=5, gamma=getattr(self.args, 'high_quality_gamma', 0.1)
                )

                if not hasattr(self, 'current_high_quality_triples'):
                    self.current_high_quality_triples = []
                self.current_high_quality_triples = high_quality
                

                if hasattr(self.args, 'epoch') and self.args.epoch % 10 == 0:
                    percentile = getattr(self.args, 'high_quality_percentile', 0.3)
                    max_count = getattr(self.args, 'high_quality_max_count', 50)
                    current_theta_r = self.get_current_noise_scale(as_tensor=False)
                    current_theta_s = self.get_current_noise_factor(as_tensor=False)
                    learnable_r = "learnable" if self.enable_learnable_noise_scale else "fixed"
                    learnable_s = "learnable" if self.enable_learnable_noise_factor else "fixed"
                    self.args.logger.info(
                        f"[High-Quality Selection] Epoch {self.args.epoch}: "
                        f"Detected={len(high_quality)}, Selected={len(high_quality)}, "
                        f"Percentile={percentile:.2f}, MaxCount={max_count}, "
                        f"MeanQ={mean_Q_high:.3f}, θr={current_theta_r:.4f} ({learnable_r}), "
                        f"θs={current_theta_s:.4f} ({learnable_s})"
                    )
            

            if len(low_quality) > sample_size * 0.5:
                low_quality = low_quality[:max(1, len(low_quality) // 2)]
            

            self.last_mean_Q = mean_Q
            
            if not low_quality:
                self.current_adaptive_weight = 1.0
                return torch.tensor(0.0, device=self.args.device), torch.tensor(0.0, device=self.args.device)
            

            total_dsp_loss = torch.tensor(0.0, device=self.args.device)
            count = 0
            

            max_process = min(getattr(self.args, 'dsp_max_process', 3), len(low_quality))
            

            num_anchors = getattr(self.args, 'dsp_num_anchors', 2)
            beta = getattr(self.args, 'dsp_beta', 3.0)
            
            for triple in low_quality[:max_process]:

                paa_score = None
                if hasattr(self, 'enable_paa') and self.enable_paa and hasattr(self, 'paa_module') and self.paa_module is not None:
                    h, r, t = triple
                    paa_score = self.compute_paa_scores(
                        torch.tensor([h], device=self.args.device),
                        torch.tensor([r], device=self.args.device),
                        torch.tensor([t], device=self.args.device)
                    )
                    if paa_score is not None:
                        paa_score = paa_score[0]
                
                anchors = self.build_structural_anchors(triple, num_anchors=num_anchors, candidates=30)
                lambda_temp = getattr(self.args, 'dsp_temperature', 1.0)
                

                if len(anchors) > 0:
                    batch_loss = self.compute_dsp_loss_for_pair_batch(triple, anchors, beta=beta, lambda_temp=lambda_temp, paa_score=paa_score)
                    total_dsp_loss += batch_loss
                    count += len(anchors)
            
            if count > 0:
                avg_dsp_loss = total_dsp_loss / count
                

                if getattr(self.args, 'adaptive_dsp', True):
                    alpha = getattr(self.args, 'alpha_dsp', 0.5)
                    adaptive_weight = alpha * mean_U + (1 - alpha) * (1 - mean_Q)
                    self.current_adaptive_weight = adaptive_weight
                else:
                    self.current_adaptive_weight = 1.0
                

                if hasattr(self.args, 'epoch') and self.args.epoch % 10 == 0:

                    current_epoch = self.args.epoch
                    if not hasattr(self, '_last_logged_dsp_detail_epoch') or self._last_logged_dsp_detail_epoch != current_epoch:
                        self._last_logged_dsp_detail_epoch = current_epoch
                        self.args.logger.info(
                            f"[PAAL] Epoch {current_epoch}: "
                            f"Sampled={sample_size}, detected={len(low_quality)}, "
                            f"Q_avg={mean_Q:.3f}, DSP_loss={avg_dsp_loss.item():.4f}"
                        )
                
                return avg_dsp_loss, torch.tensor(0.0, device=self.args.device)
            else:
                self.current_adaptive_weight = 1.0
                return torch.tensor(0.0, device=self.args.device), torch.tensor(0.0, device=self.args.device)

    def get_weight(self):
        '''get the total number of samples containing each entity or relation'''
        num_ent = self.kg.snapshots[self.args.snapshot+1].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot+1].num_rel
        ent_weight, rel_weight, other_weight = self.gcn.get_weight(num_ent, num_rel)
        return ent_weight, rel_weight, other_weight


class MAE(nn.Module):
    def __init__(self, args, kg):
        super(MAE, self).__init__()
        self.args = args
        self.kg = kg
        '''masked KG auto encoder'''
        self.conv_layers = nn.ModuleList()
        for i in range(args.num_layer):
            self.conv_layers.append(ConvLayer(args, kg))

    def forward(self, ent_embeddings, rel_embeddings, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r):

        x, r = ent_embeddings, rel_embeddings
        for i in range(self.args.num_layer):

            x, r = self.conv_layers[i](x, r, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return x, r

    def get_weight(self, num_ent, num_rel):
        '''get the total number of samples containing each entity or relation'''
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot+1].edge_index, self.kg.snapshots[self.args.snapshot+1].edge_type
        other_weight = edge_index.size(1)
        ent_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_index[0], dim_size=num_ent)
        rel_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_type, dim_size=num_rel)
        return ent_weight + 1, rel_weight + 1, other_weight

class ConvLayer(nn.Module):
    def __init__(self, args, kg):
        super(ConvLayer, self).__init__()
        self.args = args
        self.kg = kg


    def add_loop_edge(self, edge_index, edge_type, num_ent, num_rel):

        '''add self-loop edge for entities'''
        u, v = torch.arange(0, num_ent).unsqueeze(0).to(self.args.device), torch.arange(0, num_ent).unsqueeze(0).to(self.args.device)
        r = torch.zeros(num_ent).to(self.args.device).long()
        loop_edge = torch.cat([u, v], dim=0)
        edge_index = torch.cat([edge_index, loop_edge], dim=-1)
        edge_type = torch.cat([edge_type, r+num_rel], dim=-1)
        return edge_index, edge_type





