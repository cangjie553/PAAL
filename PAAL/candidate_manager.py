import os
import json
import csv
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch


class CandidateManager:
    """管理候选三元组的检测、划分和持久化"""
    
    def __init__(self, seed=42):
        """
        初始化候选管理器
        
        Args:
            seed: 随机种子，保证可复现性
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def compute_quality_score(self, model, triple: Tuple[int, int, int],
                             gamma: float = 0.1, args=None, kg=None) -> float:
        """
        计算单个三元组的质量分数 Q

        支持单维度和多维度评分：
        - 单维度：Q = 1 / (1 + exp(gamma * confidence_score))
        - 多维度：Q = w1 * confidence_score + w2 * temporal_score

        其中：
        - confidence_score：基于TransE距离的质量分数
        - temporal_score：基于时间特征的质量分数
        - w1, w2：各维度权重

        Args:
            model: 嵌入模型 (需要有 score 方法)
            triple: (h, r, t) 三元组
            gamma: 质量分数敏感度参数
            args: 参数对象（用于多维度评分）

        Returns:
            Q: 质量分数，范围 [0, 1]
        """
        h, r, t = triple


        confidence_score = self._compute_confidence_score(model, triple, gamma)


        enable_multi_criteria = getattr(args, 'enable_multi_criteria_scoring', False) if args else False
        enable_temporal = getattr(args, 'enable_temporal_score', False) if args else False
        enable_structural = getattr(args, 'enable_structural_score', False) if args else False

        if enable_multi_criteria or enable_temporal or enable_structural:

            temporal_score = self._compute_temporal_score(triple, args, kg) if (enable_multi_criteria or enable_temporal) else 0.0
            structural_score = self._compute_structural_score(triple, args, kg) if (enable_multi_criteria or enable_structural) else 0.0


            temporal_weight = getattr(args, 'temporal_weight', 0.1) if (enable_multi_criteria or enable_temporal) else 0.0
            structural_weight = getattr(args, 'structural_weight', 0.1) if (enable_multi_criteria or enable_structural) else 0.0


            used_weight = temporal_weight + structural_weight
            confidence_weight = max(0.0, 1.0 - used_weight)


            combined_score = (confidence_weight * confidence_score +
                            temporal_weight * temporal_score +
                            structural_weight * structural_score)

            return combined_score
        else:

            return confidence_score

    def _compute_confidence_score(self, model, triple: Tuple[int, int, int],
                                  gamma: float = 0.1) -> float:
        """
        计算基于模型置信度的质量分数

        Args:
            model: 嵌入模型
            triple: (h, r, t) 三元组
            gamma: 质量分数敏感度参数

        Returns:
            confidence_score: 置信度质量分数，范围 [0, 1]
        """
        h, r, t = triple


        with torch.no_grad():

            ent_embeddings, rel_embeddings = model.embedding('Train')


            device = ent_embeddings.device


            num_ent = ent_embeddings.size(0)
            num_rel = rel_embeddings.size(0)

            if h >= num_ent or t >= num_ent or r >= num_rel:

                return 0.0


            h_tensor = torch.tensor([h], dtype=torch.long, device=device)
            r_tensor = torch.tensor([r], dtype=torch.long, device=device)
            t_tensor = torch.tensor([t], dtype=torch.long, device=device)


            h_emb = torch.index_select(ent_embeddings, 0, h_tensor)
            r_emb = torch.index_select(rel_embeddings, 0, r_tensor)
            t_emb = torch.index_select(ent_embeddings, 0, t_tensor)


            score = model.score_fun(h_emb, r_emb, t_emb).item()


        try:
            Q = 1.0 / (1.0 + math.exp(gamma * score))
        except OverflowError:

            Q = 0.0

        return Q

    def _compute_temporal_score(self, triple: Tuple[int, int, int], args, kg=None) -> float:
        """
        计算基于时间特征的质量分数

        评分逻辑：
        - 越新的三元组质量越低（temporal_age_penalty）
        - 越频繁更新的三元组质量越低（temporal_freq_penalty）

        Args:
            triple: (h, r, t) 三元组
            args: 参数对象
            kg: 知识图谱对象，用于获取历史信息

        Returns:
            temporal_score: 时间质量分数，范围 [0, 1]
        """
        age_penalty = getattr(args, 'temporal_age_penalty', 0.01)
        freq_penalty = getattr(args, 'temporal_freq_penalty', 0.1)

        if kg is not None:

            age_factor, freq_factor = self._get_temporal_features_from_kg(triple, kg, args)
        else:

            current_snapshot = getattr(args, 'snapshot', 0)
            total_snapshots = getattr(args, 'snapshot_num', 5)


            age_factor = current_snapshot / max(1, total_snapshots - 1)
            freq_factor = 0.5


        temporal_penalty = age_penalty * age_factor + freq_penalty * freq_factor
        temporal_score = max(0.0, min(1.0, 1.0 - temporal_penalty))

        return temporal_score

    def _get_temporal_features_from_kg(self, triple: Tuple[int, int, int], kg, args):
        """
        从知识图谱中提取三元组的时间特征

        Args:
            triple: (h, r, t) 三元组
            kg: 知识图谱对象
            args: 参数对象

        Returns:
            age_factor: 年龄因子 [0,1]，1表示最新
            freq_factor: 频率因子 [0,1]，1表示最高频率
        """
        h, r, t = triple
        current_snapshot = getattr(args, 'snapshot', 0)
        total_snapshots = getattr(args, 'snapshot_num', 5)


        appearance_counts = []
        first_appearance = -1

        for ss_id in range(total_snapshots):
            if ss_id < len(kg.snapshots):
                snapshot = kg.snapshots[ss_id]

                if hasattr(snapshot, 'train_all') and snapshot.train_all:
                    if triple in snapshot.train_all:
                        appearance_counts.append(1)
                        if first_appearance == -1:
                            first_appearance = ss_id
                    else:
                        appearance_counts.append(0)
                else:
                    appearance_counts.append(0)
            else:
                appearance_counts.append(0)


        if first_appearance >= 0:
            age_factor = (current_snapshot - first_appearance) / max(1, total_snapshots - 1)
            age_factor = min(1.0, age_factor)
        else:
            age_factor = 0.5


        total_appearances = sum(appearance_counts)
        freq_factor = total_appearances / total_snapshots if total_snapshots > 0 else 0.5

        return age_factor, freq_factor

    def _compute_structural_score(self, triple: Tuple[int, int, int], args, kg=None) -> float:
        """
        计算基于结构特征的质量分数

        评分逻辑：
        - 异常度数的实体降低质量分数（structural_degree_penalty）
        - 罕见关系的降低质量分数（structural_relation_penalty）

        Args:
            triple: (h, r, t) 三元组
            args: 参数对象
            kg: 知识图谱对象，用于提取结构信息

        Returns:
            structural_score: 结构质量分数，范围 [0, 1]
        """
        degree_penalty = getattr(args, 'structural_degree_penalty', 0.05)
        relation_penalty = getattr(args, 'structural_relation_penalty', 0.02)

        if kg is not None:

            degree_factor, relation_factor = self._get_structural_features_from_kg(triple, kg, args)
        else:

            degree_factor, relation_factor = 0.5, 0.5


        structural_penalty = degree_penalty * degree_factor + relation_penalty * relation_factor
        structural_score = max(0.0, min(1.0, 1.0 - structural_penalty))

        return structural_score

    def _get_structural_features_from_kg(self, triple: Tuple[int, int, int], kg, args):
        """
        从知识图谱中提取三元组的结构特征

        Args:
            triple: (h, r, t) 三元组
            kg: 知识图谱对象
            args: 参数对象

        Returns:
            degree_factor: 度数异常因子 [0,1]，1表示高度异常
            relation_factor: 关系频率因子 [0,1]，1表示非常罕见
        """
        h, r, t = triple
        current_snapshot = getattr(args, 'snapshot', 0)


        if hasattr(kg, 'snapshots') and current_snapshot < len(kg.snapshots):
            snapshot = kg.snapshots[current_snapshot]
            if hasattr(snapshot, 'train_all') and snapshot.train_all:
                train_triples = snapshot.train_all
            else:
                train_triples = []
        else:
            train_triples = []

        if not train_triples:
            return 0.5, 0.5


        entity_degrees = {}
        relation_counts = {}

        for triple_h, triple_r, triple_t in train_triples:

            for entity in [triple_h, triple_t]:
                entity_degrees[entity] = entity_degrees.get(entity, 0) + 1


            relation_counts[triple_r] = relation_counts.get(triple_r, 0) + 1


        degrees = list(entity_degrees.values())
        if degrees:
            avg_degree = sum(degrees) / len(degrees)

            variance = sum((d - avg_degree) ** 2 for d in degrees) / len(degrees)
            std_degree = variance ** 0.5 if variance > 0 else 1.0
        else:
            avg_degree = 1.0
            std_degree = 1.0


        h_degree = entity_degrees.get(h, 0)
        t_degree = entity_degrees.get(t, 0)


        h_zscore = abs(h_degree - avg_degree) / std_degree if std_degree > 0 else 0
        t_zscore = abs(t_degree - avg_degree) / std_degree if std_degree > 0 else 0


        degree_zscore = (h_zscore + t_zscore) / 2.0
        degree_factor = min(1.0, degree_zscore / 3.0)


        total_relations = len(relation_counts)
        if total_relations > 0:
            r_count = relation_counts.get(r, 0)
            r_frequency = r_count / sum(relation_counts.values())


            if r_frequency > 0:
                relation_factor = 1.0 / (r_frequency * total_relations)
                relation_factor = min(1.0, relation_factor / 10.0)
            else:
                relation_factor = 1.0
        else:
            relation_factor = 0.5

        return degree_factor, relation_factor

    def detect_candidates(self,
                         model,
                         triples: List[Tuple[int, int, int]],
                         tau_Q: float = 0.2,
                         percentile: Optional[float] = None,
                         gamma: float = 0.1,
                         max_samples: Optional[int] = None,
                         args=None,
                         kg=None) -> List[Dict]:
        """
        从历史三元组中检测低质量候选集 C
        
        Args:
            model: 嵌入模型
            triples: 历史三元组列表 [(h, r, t), ...]
            tau_Q: 质量阈值 (当 percentile=None 时使用)
            percentile: 百分位数 (如果设置，则选择 Q 值最低的 percentile% 三元组)
            gamma: 质量分数敏感度
            max_samples: 最大采样数量 (如果 None，则处理所有三元组)
            args: 参数对象 (用于多维度评分)
            kg: 知识图谱对象 (用于提取时间特征)
            
        Returns:
            candidates: 候选集 C，每个元素为 {'triple': (h,r,t), 'Q': float}
        """
        if len(triples) == 0:
            return []
        

        if max_samples is not None and len(triples) > max_samples:
            sampled_triples = random.sample(triples, max_samples)
        else:
            sampled_triples = triples
        

        candidates_with_q = []
        for triple in sampled_triples:
            Q = self.compute_quality_score(model, triple, gamma, args, kg)
            candidates_with_q.append({
                'triple': triple,
                'Q': Q
            })
        

        if percentile is not None:


            candidates_with_q.sort(key=lambda x: x['Q'])
            

            k = max(1, int(len(candidates_with_q) * percentile))
            candidates = candidates_with_q[:k]
        else:

            candidates = [c for c in candidates_with_q if c['Q'] < tau_Q]
        
        return candidates
    
    def split_candidates(self,
                        candidates: List[Dict],
                        p_forget: float = 0.005,
                        p_repair: float = 0.02) -> Tuple[List[Dict], List[Dict]]:
        """
        将候选集 C 划分为 F (遗忘集) 和 R (修复集)
        
        策略:
        - 按 Q 值升序排序 (质量从差到好)
        - F = 最差的 p_forget 百分比
        - R = 接下来的 p_repair 百分比
        
        Args:
            candidates: 候选集 C
            p_forget: 遗忘集比例 (相对于 C 的大小)
            p_repair: 修复集比例 (相对于 C 的大小)
            
        Returns:
            F: 遗忘集 (质量最差的)
            R: 修复集 (质量次差的)
        """
        if len(candidates) == 0:
            return [], []
        

        sorted_candidates = sorted(candidates, key=lambda x: x['Q'])
        

        n_total = len(sorted_candidates)
        n_forget = max(1, int(n_total * p_forget))
        n_repair = max(1, int(n_total * p_repair))
        

        if n_forget + n_repair > n_total:

            ratio = n_total / (n_forget + n_repair)
            n_forget = max(1, int(n_forget * ratio))
            n_repair = n_total - n_forget
        

        F = sorted_candidates[:n_forget]
        R = sorted_candidates[n_forget:n_forget + n_repair]
        
        return F, R
    
    def persist_candidate_sets(self,
                              snapshot_id: int,
                              C: List[Dict],
                              F: List[Dict],
                              R: List[Dict],
                              out_dir: str = "PAAL/candidates") -> Dict[str, str]:
        """
        持久化候选集 C, F, R 到文件系统
        
        文件格式:
        - CSV: h,r,t,Q,tag
        - JSON: 完整的元数据
        
        Args:
            snapshot_id: 快照 ID
            C: 候选集
            F: 遗忘集
            R: 修复集
            out_dir: 输出根目录
            
        Returns:
            paths: 保存的文件路径字典
        """

        snapshot_dir = os.path.join(out_dir, f"snapshot_{snapshot_id:03d}")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        paths = {}
        

        stats = {
            'snapshot_id': snapshot_id,
            'n_candidates': len(C),
            'n_forget': len(F),
            'n_repair': len(R),
            'Q_stats': {
                'C': self._compute_q_stats(C),
                'F': self._compute_q_stats(F),
                'R': self._compute_q_stats(R)
            }
        }
        
        stats_path = os.path.join(snapshot_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        paths['stats'] = stats_path
        

        if len(C) > 0:
            csv_path = os.path.join(snapshot_dir, "candidates_C.csv")
            self._save_to_csv(C, csv_path, tag='C')
            paths['C_csv'] = csv_path
            
            json_path = os.path.join(snapshot_dir, "candidates_C.json")
            self._save_to_json(C, json_path, tag='C')
            paths['C_json'] = json_path
        

        if len(F) > 0:
            csv_path = os.path.join(snapshot_dir, "forget_F.csv")
            self._save_to_csv(F, csv_path, tag='F')
            paths['F_csv'] = csv_path
            
            json_path = os.path.join(snapshot_dir, "forget_F.json")
            self._save_to_json(F, json_path, tag='F')
            paths['F_json'] = json_path
        

        if len(R) > 0:
            csv_path = os.path.join(snapshot_dir, "repair_R.csv")
            self._save_to_csv(R, csv_path, tag='R')
            paths['R_csv'] = csv_path
            
            json_path = os.path.join(snapshot_dir, "repair_R.json")
            self._save_to_json(R, json_path, tag='R')
            paths['R_json'] = json_path
        
        return paths
    
    def _compute_q_stats(self, candidates: List[Dict]) -> Dict:
        """计算 Q 值的统计信息"""
        if len(candidates) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        q_values = [c['Q'] for c in candidates]
        return {
            'mean': float(np.mean(q_values)),
            'std': float(np.std(q_values)),
            'min': float(np.min(q_values)),
            'max': float(np.max(q_values)),
            'median': float(np.median(q_values))
        }
    
    def _save_to_csv(self, candidates: List[Dict], path: str, tag: str):
        """保存候选集到 CSV 文件"""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['h', 'r', 't', 'Q', 'tag'])

            for c in candidates:
                h, r, t = c['triple']
                Q = c['Q']
                writer.writerow([h, r, t, f"{Q:.6f}", tag])
    
    def _save_to_json(self, candidates: List[Dict], path: str, tag: str):
        """保存候选集到 JSON 文件"""
        data = {
            'tag': tag,
            'count': len(candidates),
            'triples': [
                {
                    'h': c['triple'][0],
                    'r': c['triple'][1],
                    't': c['triple'][2],
                    'Q': c['Q']
                }
                for c in candidates
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def compute_quality_scores_batch(self, model, triples: List[Tuple[int, int, int]], 
                                    gamma: float = 0.1, batch_size: int = 1000) -> List[Dict]:
        """
        批量计算三元组的质量分数（向量化，高效）
        
        Args:
            model: 嵌入模型
            triples: 三元组列表 [(h, r, t), ...]
            gamma: 质量分数敏感度
            batch_size: 批处理大小
            
        Returns:
            candidates: [{'triple': (h,r,t), 'Q': float}, ...]
        """
        all_candidates = []
        
        with torch.no_grad():
            ent_embeddings, rel_embeddings = model.embedding('Train')
            device = ent_embeddings.device
            num_ent = ent_embeddings.size(0)
            num_rel = rel_embeddings.size(0)
            
            for i in range(0, len(triples), batch_size):
                batch_triples = triples[i:i+batch_size]
                

                h_arr = np.array([t[0] for t in batch_triples])
                r_arr = np.array([t[1] for t in batch_triples])
                t_arr = np.array([t[2] for t in batch_triples])
                
                valid_mask = (h_arr < num_ent) & (t_arr < num_ent) & (r_arr < num_rel)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    continue
                

                h_tensor = torch.from_numpy(h_arr[valid_indices]).long().to(device)
                r_tensor = torch.from_numpy(r_arr[valid_indices]).long().to(device)
                t_tensor = torch.from_numpy(t_arr[valid_indices]).long().to(device)
                

                h_emb = ent_embeddings[h_tensor]
                r_emb = rel_embeddings[r_tensor]
                t_emb = ent_embeddings[t_tensor]
                

                scores = model.score_fun(h_emb, r_emb, t_emb)
                

                Q_values = 1.0 / (1.0 + torch.exp(gamma * scores))
                Q_values = torch.clamp(Q_values, min=0.0, max=1.0)
                Q_np = Q_values.cpu().numpy()
                

                for idx, Q in zip(valid_indices, Q_np):
                    all_candidates.append({
                        'triple': batch_triples[idx],
                        'Q': float(Q)
                    })
        
        return all_candidates
    






_default_manager = CandidateManager(seed=42)


def detect_candidates(model,
                     triples: List[Tuple[int, int, int]],
                     tau_Q: float = 0.2,
                     percentile: Optional[float] = None,
                     gamma: float = 0.1,
                     max_samples: Optional[int] = None,
                     args=None,
                     kg=None) -> List[Dict]:
    """
    检测候选集 C (便捷函数)

    详细文档见 CandidateManager.detect_candidates
    """
    return _default_manager.detect_candidates(
        model, triples, tau_Q, percentile, gamma, max_samples, args, kg
    )


def split_candidates(candidates: List[Dict],
                    p_forget: float = 0.005,
                    p_repair: float = 0.02) -> Tuple[List[Dict], List[Dict]]:
    """
    划分候选集为 F 和 R (便捷函数)
    
    详细文档见 CandidateManager.split_candidates
    """
    return _default_manager.split_candidates(candidates, p_forget, p_repair)


def persist_candidate_sets(snapshot_id: int,
                          C: List[Dict],
                          F: List[Dict],
                          R: List[Dict],
                          out_dir: str = "PAAL/candidates") -> Dict[str, str]:
    """
    持久化候选集 (便捷函数)
    
    详细文档见 CandidateManager.persist_candidate_sets
    """
    return _default_manager.persist_candidate_sets(snapshot_id, C, F, R, out_dir)


def load_candidate_set(snapshot_id: int,
                      set_type: str = 'C',
                      out_dir: str = "PAAL/candidates") -> List[Dict]:
    """
    加载候选集 (便捷函数)
    
    详细文档见 CandidateManager.load_candidate_set
    """
    return _default_manager.load_candidate_set(snapshot_id, set_type, out_dir)

