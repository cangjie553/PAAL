import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Tuple, Dict, Optional


class ForgetLossModule:
    """遗忘损失模块"""
    
    def __init__(self, model, args, kg):
        """
        初始化遗忘损失模块
        
        Args:
            model: LKGE 模型实例
            args: 命令行参数
            kg: 知识图谱数据
        """
        self.model = model
        self.args = args
        self.kg = kg
        self.device = args.device
        

        self.beta_forget = getattr(args, 'beta_forget', 10.0)
        self.margin_forget = getattr(args, 'margin_forget', 16.0)
        self.lambda_forget = getattr(args, 'lambda_forget', 1.0)
        self.forget_anchor_strategy = getattr(args, 'forget_anchor_strategy', 'random')
        

        self.beta_repair = getattr(args, 'dsp_beta', 3.0)
        
    def compute_forget_loss(self, F: List[Dict], num_negatives: int = 10) -> torch.Tensor:
        """
        计算遗忘损失
        
        目标: 主动遗忘 F 中的三元组，使模型无法再预测它们
        
        策略:
        1. 分数抑制损失: 直接惩罚 score(h,r,t)，使其增大 (TransE 距离增大)
        2. 排名降级损失: 对比损失，使 (h,r,t) 的分数高于随机负样本
        
        Args:
            F: 遗忘集 [{'triple': (h,r,t), 'Q': float}, ...]
            num_negatives: 每个三元组的负样本数量
            
        Returns:
            forget_loss: 遗忘损失 (标量)
        """
        if len(F) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0
        

        ent_embeddings, rel_embeddings = self.model.embedding('Train')
        
        for item in F:
            h, r, t = item['triple']
            

            h_tensor = torch.tensor([h], dtype=torch.long, device=self.device)
            r_tensor = torch.tensor([r], dtype=torch.long, device=self.device)
            t_tensor = torch.tensor([t], dtype=torch.long, device=self.device)
            

            h_emb = torch.index_select(ent_embeddings, 0, h_tensor)
            r_emb = torch.index_select(rel_embeddings, 0, r_tensor)
            t_emb = torch.index_select(ent_embeddings, 0, t_tensor)
            


            score_ht = self.model.score_fun(h_emb, r_emb, t_emb)
            L_suppress = torch.clamp(self.margin_forget - score_ht, min=0)
            



            neg_tails = self._sample_negatives(h, r, t, num_negatives)
            

            scores_neg = []
            for t_neg in neg_tails:
                t_neg_tensor = torch.tensor([t_neg], dtype=torch.long, device=self.device)
                t_neg_emb = torch.index_select(ent_embeddings, 0, t_neg_tensor)
                score_neg = self.model.score_fun(h_emb, r_emb, t_neg_emb)
                scores_neg.append(score_neg)
            




            if len(scores_neg) > 0:
                scores_neg_tensor = torch.stack(scores_neg)
                


                margin_rank = 2.0
                L_rank = torch.clamp(margin_rank - (score_ht - scores_neg_tensor), min=0).mean()
            else:
                L_rank = torch.tensor(0.0, device=self.device)
            

            loss_item = L_suppress + L_rank

            if loss_item.dim() > 0:
                loss_item = loss_item.mean()
            total_loss += loss_item
            count += 1
        
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_repair_loss(self, R: List[Dict], num_anchors: int = 2) -> torch.Tensor:
        """
        计算修复损失 (当前的 DSP 损失)
        
        目标: 温和地修复 R 中的三元组，将其推向更合理的嵌入位置
        
        Args:
            R: 修复集 [{'triple': (h,r,t), 'Q': float}, ...]
            num_anchors: 每个三元组的锚点数量
            
        Returns:
            repair_loss: 修复损失 (标量)
        """
        if len(R) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        for item in R:
            triple = item['triple']
            

            anchors = self.model.build_structural_anchors(
                triple, 
                num_anchors=num_anchors, 
                candidates=30
            )
            

            for anchor in anchors:
                loss_val = self._compute_dsp_loss_for_pair(
                    triple, anchor, beta=self.beta_repair
                )
                total_loss += loss_val
                count += 1
        
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=self.device)
    
    def build_forget_anchors(self, triple: Tuple[int, int, int], 
                            num_anchors: int = 5,
                            strategy: str = 'random') -> List[Tuple[int, int, int]]:
        """
        生成遗忘锚点
        
        策略:
        - 'random': 随机负样本 (破坏结构)
        - 'mixed': 混合策略 (50% 随机 + 50% 结构化)
        
        Args:
            triple: 三元组 (h, r, t)
            num_anchors: 锚点数量
            strategy: 锚点生成策略
            
        Returns:
            anchors: 锚点三元组列表
        """
        h, r, t = triple
        anchors = []
        
        if strategy == 'random':

            anchors = self._generate_random_anchors(h, r, t, num_anchors)
        
        elif strategy == 'mixed':

            num_random = num_anchors // 2
            num_structural = num_anchors - num_random
            

            random_anchors = self._generate_random_anchors(h, r, t, num_random)
            anchors.extend(random_anchors)
            

            structural_anchors = self.model.build_structural_anchors(
                triple, num_anchors=num_structural, candidates=30
            )
            anchors.extend(structural_anchors)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return anchors
    
    def _generate_random_anchors(self, h: int, r: int, t: int, 
                                num_anchors: int) -> List[Tuple[int, int, int]]:
        """生成随机负样本锚点"""
        anchors = []
        num_entities = self.kg.snapshots[self.args.snapshot].num_ent
        
        for _ in range(num_anchors):

            t_neg = random.randint(0, num_entities - 1)

            while t_neg == t:
                t_neg = random.randint(0, num_entities - 1)
            
            anchors.append((h, r, t_neg))
        
        return anchors
    
    def _sample_negatives(self, h: int, r: int, t: int, 
                         num_negatives: int) -> List[int]:
        """采样负样本尾实体"""
        num_entities = self.kg.snapshots[self.args.snapshot].num_ent
        negatives = []
        
        for _ in range(num_negatives):
            t_neg = random.randint(0, num_entities - 1)

            while t_neg == t:
                t_neg = random.randint(0, num_entities - 1)
            negatives.append(t_neg)
        
        return negatives
    






def create_forget_loss_module(model, args, kg):
    """
    创建遗忘损失模块 (便捷函数)
    
    Args:
        model: LKGE 模型实例
        args: 命令行参数
        kg: 知识图谱数据
        
    Returns:
        module: ForgetLossModule 实例
    """
    return ForgetLossModule(model, args, kg)

