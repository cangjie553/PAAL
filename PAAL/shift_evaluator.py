import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


class UnlearningEvaluator:
    """遗忘评估器"""
    
    def __init__(self, model, args, kg):
        """
        初始化评估器
        
        Args:
            model: LKGE 模型实例
            args: 命令行参数
            kg: 知识图谱数据
        """
        self.model = model
        self.args = args
        self.kg = kg
        self.device = args.device
        

        self.eval_batch_size = 64
        
    def construct_retained_set(self, test_triples: List[Tuple], 
                               F: List[Dict]) -> List[Tuple]:
        """
        构造保留集 T_ret = T_test \ F
        
        Args:
            test_triples: 测试集三元组
            F: 遗忘集
            
        Returns:
            T_ret: 保留集
        """

        F_set = set(item['triple'] for item in F)
        

        T_ret = [t for t in test_triples if t not in F_set]
        
        return T_ret
    
    def construct_neighborhood(self, F: List[Dict], 
                              test_triples: List[Tuple],
                              hop: int = 1) -> List[Tuple]:
        """
        构造 F 的邻域 N(F)
        
        定义: 与 F 中三元组共享至少一个实体或关系的三元组
        
        Args:
            F: 遗忘集
            test_triples: 测试集三元组
            hop: 邻域跳数 (默认 1)
            
        Returns:
            N_F: 邻域三元组集合
        """

        F_entities = set()
        F_relations = set()
        
        for item in F:
            h, r, t = item['triple']
            F_entities.add(h)
            F_entities.add(t)
            F_relations.add(r)
        

        N_F = []
        F_set = set(item['triple'] for item in F)
        
        for triple in test_triples:
            h, r, t = triple
            

            if triple in F_set:
                continue
            

            if h in F_entities or t in F_entities or r in F_relations:
                N_F.append(triple)
        
        return N_F
    
    def evaluate_triples(self, triples: List[Tuple], 
                        snapshot_id: int) -> Dict[str, float]:
        """
        在给定的三元组集合上评估链接预测性能
        
        Args:
            triples: 三元组列表
            snapshot_id: 快照 ID
            
        Returns:
            metrics: 评估指标字典
        """
        if len(triples) == 0:
            return {
                'MRR': 0.0,
                'Hits@1': 0.0,
                'Hits@3': 0.0,
                'Hits@5': 0.0,
                'Hits@10': 0.0,
                'count': 0
            }
        
        self.model.eval()
        
        ranks = []
        
        with torch.no_grad():

            ent_embeddings, rel_embeddings = self.model.embedding('Test')
            num_entities = self.kg.snapshots[snapshot_id].num_ent
            

            for i in range(0, len(triples), self.eval_batch_size):
                batch = triples[i:i + self.eval_batch_size]
                

                h_batch = torch.tensor([t[0] for t in batch], dtype=torch.long, device=self.device)
                r_batch = torch.tensor([t[1] for t in batch], dtype=torch.long, device=self.device)
                t_batch = torch.tensor([t[2] for t in batch], dtype=torch.long, device=self.device)
                

                h_emb = torch.index_select(ent_embeddings, 0, h_batch)
                r_emb = torch.index_select(rel_embeddings, 0, r_batch)
                o_all = ent_embeddings[:num_entities]
                


                batch_size = h_batch.size(0)
                h_r = h_emb.unsqueeze(1) + r_emb.unsqueeze(1)
                o_all_expanded = o_all.unsqueeze(0)
                scores = torch.norm(h_r - o_all_expanded, p=2, dim=2)
                




                sorted_indices = torch.argsort(scores, dim=1)
                rank_indices = torch.argsort(sorted_indices, dim=1)
                

                batch_range = torch.arange(batch_size, device=self.device)
                ranks_batch = rank_indices[batch_range, t_batch] + 1
                ranks.extend(ranks_batch.cpu().tolist())
        

        ranks = np.array(ranks)
        
        metrics = {
            'MRR': float(np.mean(1.0 / ranks)),
            'Hits@1': float(np.mean(ranks <= 1)),
            'Hits@3': float(np.mean(ranks <= 3)),
            'Hits@5': float(np.mean(ranks <= 5)),
            'Hits@10': float(np.mean(ranks <= 10)),
            'count': len(triples),
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks))
        }
        
        return metrics
    
    def eval_retained_performance(self, test_triples: List[Tuple],
                                 F: List[Dict],
                                 snapshot_id: int,
                                 baseline_metrics: Optional[Dict] = None) -> Dict:
        """
        评估保留知识的性能
        
        Args:
            test_triples: 测试集三元组
            F: 遗忘集
            snapshot_id: 快照 ID
            baseline_metrics: 遗忘前的基线指标
            
        Returns:
            results: 评估结果
        """

        T_ret = self.construct_retained_set(test_triples, F)
        

        metrics = self.evaluate_triples(T_ret, snapshot_id)
        

        if baseline_metrics is not None:
            delta_MRR = metrics['MRR'] - baseline_metrics.get('MRR', 0)
            delta_Hits10 = metrics['Hits@10'] - baseline_metrics.get('Hits@10', 0)
        else:
            delta_MRR = 0.0
            delta_Hits10 = 0.0
        
        results = {
            'T_ret_size': len(T_ret),
            'MRR_ret': metrics['MRR'],
            'Hits@1_ret': metrics['Hits@1'],
            'Hits@3_ret': metrics['Hits@3'],
            'Hits@10_ret': metrics['Hits@10'],
            'mean_rank_ret': metrics['mean_rank'],
            'delta_MRR': delta_MRR,
            'delta_Hits@10': delta_Hits10,
            'performance_maintained': delta_MRR >= -0.02
        }
        
        return results
    
    def eval_forgetting_effectiveness(self, F: List[Dict],
                                     snapshot_id: int,
                                     baseline_metrics: Optional[Dict] = None) -> Dict:
        """
        评估遗忘效果
        
        Args:
            F: 遗忘集
            snapshot_id: 快照 ID
            baseline_metrics: 遗忘前的基线指标
            
        Returns:
            results: 评估结果
        """
        if len(F) == 0:
            return {
                'F_size': 0,
                'MRR_F': 0.0,
                'forgetting_effective': False
            }
        

        F_triples = [item['triple'] for item in F]
        

        metrics = self.evaluate_triples(F_triples, snapshot_id)
        

        if baseline_metrics is not None:
            MRR_before = baseline_metrics.get('MRR', 0)
            MRR_after = metrics['MRR']
            delta_MRR = MRR_after - MRR_before
            forgetting_rate = (MRR_before - MRR_after) / MRR_before if MRR_before > 0 else 0
        else:
            delta_MRR = 0.0
            forgetting_rate = 0.0
        

        mean_rank = metrics['mean_rank']
        num_entities = self.kg.snapshots[snapshot_id].num_ent
        rank_percentile = mean_rank / num_entities
        
        results = {
            'F_size': len(F),
            'MRR_F': metrics['MRR'],
            'Hits@1_F': metrics['Hits@1'],
            'Hits@3_F': metrics['Hits@3'],
            'Hits@10_F': metrics['Hits@10'],
            'mean_rank_F': mean_rank,
            'median_rank_F': metrics['median_rank'],
            'rank_percentile': rank_percentile,
            'delta_MRR_F': delta_MRR,
            'forgetting_rate': forgetting_rate,
            'forgetting_effective': forgetting_rate > 0.3
        }
        
        return results
    
    def eval_local_stability(self, test_triples: List[Tuple],
                            F: List[Dict],
                            snapshot_id: int,
                            baseline_metrics: Optional[Dict] = None) -> Dict:
        """
        评估局部稳定性 (邻域性能)
        
        Args:
            test_triples: 测试集三元组
            F: 遗忘集
            snapshot_id: 快照 ID
            baseline_metrics: 遗忘前的基线指标
            
        Returns:
            results: 评估结果
        """

        N_F = self.construct_neighborhood(F, test_triples)
        
        if len(N_F) == 0:
            return {
                'N_F_size': 0,
                'MRR_N_F': 0.0,
                'stable': True
            }
        

        metrics = self.evaluate_triples(N_F, snapshot_id)
        

        if baseline_metrics is not None:
            delta_MRR = metrics['MRR'] - baseline_metrics.get('MRR', 0)
        else:
            delta_MRR = 0.0
        
        results = {
            'N_F_size': len(N_F),
            'MRR_N_F': metrics['MRR'],
            'Hits@10_N_F': metrics['Hits@10'],
            'mean_rank_N_F': metrics['mean_rank'],
            'delta_MRR_N_F': delta_MRR,
            'stable': delta_MRR >= -0.05
        }
        
        return results
    
    def compute_unlearning_quality_score(self, 
                                        retained_results: Dict,
                                        forgetting_results: Dict,
                                        stability_results: Dict,
                                        w1: float = 0.5,
                                        w2: float = 0.3,
                                        w3: float = 0.2) -> Dict:
        """
        计算综合遗忘质量分数 (UQS)
        
        UQS = w1 × (保留性能) + w2 × (遗忘效果) + w3 × (局部稳定性)
        
        Args:
            retained_results: 保留性能结果
            forgetting_results: 遗忘效果结果
            stability_results: 局部稳定性结果
            w1, w2, w3: 权重
            
        Returns:
            uqs_results: UQS 结果
        """

        delta_MRR_ret = retained_results.get('delta_MRR', 0)
        MRR_ret = retained_results.get('MRR_ret', 0)
        if MRR_ret > 0:
            retention_score = max(0, 1 + delta_MRR_ret / MRR_ret)
        else:
            retention_score = 0
        

        forgetting_rate = forgetting_results.get('forgetting_rate', 0)
        forgetting_score = min(1.0, forgetting_rate)
        

        delta_MRR_N_F = stability_results.get('delta_MRR_N_F', 0)
        MRR_N_F = stability_results.get('MRR_N_F', 0)
        if MRR_N_F > 0:
            stability_score = max(0, 1 + delta_MRR_N_F / MRR_N_F)
        else:
            stability_score = 1.0
        

        UQS = w1 * retention_score + w2 * forgetting_score + w3 * stability_score
        
        results = {
            'UQS': UQS,
            'retention_score': retention_score,
            'forgetting_score': forgetting_score,
            'stability_score': stability_score,
            'weights': {'w1': w1, 'w2': w2, 'w3': w3}
        }
        
        return results
    
    def save_unlearning_report(self, snapshot_id: int,
                               C: List[Dict],
                               F: List[Dict],
                               R: List[Dict],
                               retained_results: Dict,
                               forgetting_results: Dict,
                               stability_results: Dict,
                               uqs_results: Dict,
                               save_dir: str = None) -> str:
        """
        保存详细的遗忘报告
        
        Args:
            snapshot_id: 快照 ID
            C, F, R: 候选集
            retained_results: 保留性能结果
            forgetting_results: 遗忘效果结果
            stability_results: 局部稳定性结果
            uqs_results: UQS 结果
            save_dir: 保存目录
            
        Returns:
            report_path: 报告文件路径
        """
        if save_dir is None:
            save_dir = os.path.join(self.args.save_path, 'unlearning_reports')
        
        os.makedirs(save_dir, exist_ok=True)
        

        report = {
            'snapshot_id': snapshot_id,
            'timestamp': self._get_timestamp(),
            'dataset': self.args.dataset,
            

            'candidate_sets': {
                'C_size': len(C),
                'F_size': len(F),
                'R_size': len(R),
                'Q_stats': {
                    'C': self._compute_q_stats(C),
                    'F': self._compute_q_stats(F),
                    'R': self._compute_q_stats(R)
                }
            },
            

            'evaluation': {
                'retained_performance': retained_results,
                'forgetting_effectiveness': forgetting_results,
                'local_stability': stability_results,
                'unlearning_quality': uqs_results
            },
            

            'parameters': {
                'candidate_percentile': getattr(self.args, 'candidate_percentile', 0.2),
                'forget_ratio': getattr(self.args, 'forget_ratio', 0.3),
                'repair_ratio': getattr(self.args, 'repair_ratio', 0.5),
                'beta_forget': getattr(self.args, 'beta_forget', 10.0),
                'margin_forget': getattr(self.args, 'margin_forget', 16.0),
                'lambda_forget': getattr(self.args, 'lambda_forget', 1.0)
            }
        }
        

        report_path = os.path.join(save_dir, f'unlearning_report_snapshot_{snapshot_id}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        

        txt_path = os.path.join(save_dir, f'unlearning_report_snapshot_{snapshot_id}.txt')
        self._save_readable_report(report, txt_path)
        
        return report_path
    
    def _compute_q_stats(self, candidates: List[Dict]) -> Dict:
        """计算 Q 值统计"""
        if len(candidates) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        q_values = [c['Q'] for c in candidates]
        return {
            'mean': float(np.mean(q_values)),
            'std': float(np.std(q_values)),
            'min': float(np.min(q_values)),
            'max': float(np.max(q_values))
        }
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _save_readable_report(self, report: Dict, path: str):
        """保存可读的文本报告"""
        with open(path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PAAL Unlearning Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Snapshot: {report['snapshot_id']}\n")
            f.write(f"Dataset: {report['dataset']}\n")
            f.write(f"Time: {report['timestamp']}\n\n")
            

            f.write("-" * 80 + "\n")
            f.write("Candidate Sets\n")
            f.write("-" * 80 + "\n")
            cs = report['candidate_sets']
            f.write(f"|C| = {cs['C_size']}\n")
            f.write(f"|F| = {cs['F_size']}\n")
            f.write(f"|R| = {cs['R_size']}\n\n")
            

            f.write("-" * 80 + "\n")
            f.write("Evaluation Results\n")
            f.write("-" * 80 + "\n\n")
            
            eval_data = report['evaluation']
            
            f.write("1. Retained Performance:\n")
            ret = eval_data['retained_performance']
            f.write(f"   MRR_ret: {ret['MRR_ret']:.4f}\n")
            f.write(f"   Hits@10_ret: {ret['Hits@10_ret']:.4f}\n")
            f.write(f"   ΔMRR: {ret['delta_MRR']:.4f}\n")
            f.write(f"   Performance Maintained: {ret['performance_maintained']}\n\n")
            
            f.write("2. Forgetting Effectiveness:\n")
            forg = eval_data['forgetting_effectiveness']
            f.write(f"   MRR_F: {forg['MRR_F']:.4f}\n")
            f.write(f"   Forgetting Rate: {forg['forgetting_rate']:.2%}\n")
            f.write(f"   Forgetting Effective: {forg['forgetting_effective']}\n\n")
            
            f.write("3. Local Stability:\n")
            stab = eval_data['local_stability']
            f.write(f"   MRR_N(F): {stab['MRR_N_F']:.4f}\n")
            f.write(f"   ΔMRR_N(F): {stab['delta_MRR_N_F']:.4f}\n")
            f.write(f"   Stable: {stab['stable']}\n\n")
            
            f.write("4. Unlearning Quality Score (UQS):\n")
            uqs = eval_data['unlearning_quality']
            f.write(f"   UQS: {uqs['UQS']:.4f}\n")
            f.write(f"   Retention Score: {uqs['retention_score']:.4f}\n")
            f.write(f"   Forgetting Score: {uqs['forgetting_score']:.4f}\n")
            f.write(f"   Stability Score: {uqs['stability_score']:.4f}\n\n")
            
            f.write("=" * 80 + "\n")






def create_unlearning_evaluator(model, args, kg):
    """创建遗忘评估器 (便捷函数)"""
    return UnlearningEvaluator(model, args, kg)

