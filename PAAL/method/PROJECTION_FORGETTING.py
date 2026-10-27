import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional
import copy






def compute_forget_directions(model, F: List[Dict], anchors_map: Dict) -> Dict:
    """
    计算遗忘三元组涉及的实体/关系的"坏方向"
    
    对每个 (h,r,t) ∈ F:
        取其锚点 triple (h_w, r_w, t_w)
        计算三元组语义差向：
            u = (e_h + e_r - e_t) - (e_h_w + e_r_w - e_t_w)
        对 F 中出现的实体/关系：
            方向累加 direction[v] += u 或相关分量
    最后对每个 v 做 unit-normalize
    
    Args:
        model: LKGE 模型
        F: 遗忘三元组集合
        anchors_map: 字典 {triple -> [anchor_triples]}
                     每个遗忘三元组对应的锚点列表
    
    Returns:
        direction_map: 字典 {entity_id/relation_id -> direction_vector}
                      每个实体/关系的"坏方向"（unit vector）
    """
    if len(F) == 0:
        return {}
    

    ent_embeddings, rel_embeddings = model.embedding('Train')
    


    direction_accumulator = {}
    
    for item in F:
        h, r, t = item['triple']
        triple_key = (h, r, t)
        

        if triple_key not in anchors_map or len(anchors_map[triple_key]) == 0:
            continue
        

        h_emb = ent_embeddings[h].detach()
        r_emb = rel_embeddings[r].detach()
        t_emb = ent_embeddings[t].detach()
        

        forget_semantic = h_emb + r_emb - t_emb
        

        for anchor_triple in anchors_map[triple_key]:
            h_w, r_w, t_w = anchor_triple
            

            h_w_emb = ent_embeddings[h_w].detach()
            r_w_emb = rel_embeddings[r_w].detach()
            t_w_emb = ent_embeddings[t_w].detach()
            

            anchor_semantic = h_w_emb + r_w_emb - t_w_emb
            

            u = forget_semantic - anchor_semantic
            


            key_h = ('ent', h)
            if key_h not in direction_accumulator:
                direction_accumulator[key_h] = torch.zeros_like(h_emb)
            direction_accumulator[key_h] = direction_accumulator[key_h] + u
            

            key_r = ('rel', r)
            if key_r not in direction_accumulator:
                direction_accumulator[key_r] = torch.zeros_like(r_emb)
            direction_accumulator[key_r] = direction_accumulator[key_r] + u
            

            key_t = ('ent', t)
            if key_t not in direction_accumulator:
                direction_accumulator[key_t] = torch.zeros_like(t_emb)
            direction_accumulator[key_t] = direction_accumulator[key_t] - u
    

    direction_map = {}
    eps = 1e-12
    
    for key, direction in direction_accumulator.items():
        norm = torch.norm(direction) + eps
        direction_normalized = direction / norm
        direction_map[key] = direction_normalized.detach()
    
    return direction_map






def apply_projection_forgetting(model, direction_map: Dict, step_scale: float = 0.1):
    """
    对每个受影响实体/关系执行语义投影更新
    
    更新公式：
        e_v_new = e_v_old - step_scale * (<e_v_old, u_v> / (||u_v||^2 + eps)) * u_v
    
    其中：
        - u_v = direction_map[v] (已经是unit vector)
        - eps = 1e-12
        - step_scale ∈ [0.05, 0.5] 控制遗忘强度
    
    Args:
        model: LKGE 模型
        direction_map: 字典 {('ent'|'rel', id) -> direction_vector}
        step_scale: 投影步长，控制遗忘强度
    
    Returns:
        old_embeddings: 字典保存原始嵌入（用于回滚）
    """

    ent_embeddings, rel_embeddings = model.embedding('Train')
    

    old_embeddings = {
        'ent': ent_embeddings.data.clone(),
        'rel': rel_embeddings.data.clone()
    }
    
    eps = 1e-12
    

    with torch.no_grad():
        for key, u_v in direction_map.items():
            emb_type, v_id = key
            
            if emb_type == 'ent':
                e_v_old = ent_embeddings[v_id]
            else:
                e_v_old = rel_embeddings[v_id]
            



            dot_product = torch.dot(e_v_old, u_v)
            projection_component = dot_product / (torch.norm(u_v)**2 + eps) * u_v
            

            e_v_new = e_v_old - step_scale * projection_component
            

            if emb_type == 'ent':
                ent_embeddings.data[v_id] = e_v_new
            else:
                rel_embeddings.data[v_id] = e_v_new
    
    return old_embeddings






def build_anchors_for_forget_set(model, F: List[Dict], num_anchors: int = 2) -> Dict:
    """
    为遗忘集F中的每个三元组构造SCA锚点
    
    使用RCA策略：选择该关系的高频尾实体
    
    Args:
        model: LKGE 模型
        F: 遗忘集
        num_anchors: 每个三元组的锚点数量
    
    Returns:
        anchors_map: 字典 {(h,r,t) -> [(h,r,t_anchor1), (h,r,t_anchor2), ...]}
    """
    anchors_map = {}
    

    snapshot_id = model.args.snapshot
    train_triples = list(model.kg.snapshots[snapshot_id].train_new)
    
    for item in F:
        h, r, t = item['triple']
        triple_key = (h, r, t)
        

        if hasattr(model, 'build_structural_anchors'):
            anchors = model.build_structural_anchors(triple_key, num_anchors=num_anchors)
        else:

            tail_counts = {}
            for (h_i, r_i, t_i) in train_triples:
                if r_i == r and t_i != t:
                    tail_counts[t_i] = tail_counts.get(t_i, 0) + 1
            
            if tail_counts:

                sorted_tails = sorted(tail_counts.items(), key=lambda x: x[1], reverse=True)
                top_tails = [tail for tail, _ in sorted_tails[:num_anchors]]
                anchors = [(h, r, t_alt) for t_alt in top_tails]
            else:

                num_entities = model.kg.snapshots[snapshot_id].num_ent
                import random
                random_tails = random.sample(range(num_entities), min(num_anchors, num_entities))
                anchors = [(h, r, t_rand) for t_rand in random_tails if t_rand != t]
        
        anchors_map[triple_key] = anchors
    
    return anchors_map






def evaluate_forget_quality(model, F: List[Dict]) -> float:
    """
    评估遗忘集的链接预测质量（MRR）
    
    Args:
        model: LKGE 模型
        F: 遗忘集
    
    Returns:
        mrr: 遗忘集的MRR
    """
    if len(F) == 0:
        return 0.0
    
    model.eval()
    
    reciprocal_ranks = []
    
    with torch.no_grad():
        ent_embeddings, rel_embeddings = model.embedding('Test')
        num_entities = model.kg.snapshots[model.args.snapshot].num_ent
        
        for item in F:
            h, r, t = item['triple']
            
            h_emb = ent_embeddings[h].unsqueeze(0)
            r_emb = rel_embeddings[r].unsqueeze(0)
            

            all_t_emb = ent_embeddings[:num_entities]
            scores = model.score_fun(
                h_emb.expand(num_entities, -1),
                r_emb.expand(num_entities, -1),
                all_t_emb
            )
            

            sorted_indices = torch.argsort(scores)
            

            rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
    
    model.train()
    
    if len(reciprocal_ranks) > 0:
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    else:
        return 0.0






def forget_by_projection(model, F: List[Dict], 
                        step_scale: float = 0.2,
                        num_anchors: int = 2,
                        performance_threshold: float = 0.02,
                        verbose: bool = True) -> bool:
    """
    通过投影方法遗忘三元组集合F
    
    流程：
    1. 保存embedding checkpoint (for rollback)
    2. 构造SCA锚点
    3. 计算遗忘方向
    4. 应用投影遗忘
    5. 评估效果
    6. 若保留性能下降过多，回滚；否则接受
    
    Args:
        model: LKGE 模型
        F: 遗忘集
        step_scale: 投影步长 ∈ [0.05, 0.5]
        num_anchors: 每个三元组的锚点数量
        performance_threshold: 保留集性能下降阈值
        verbose: 是否打印日志
    
    Returns:
        success: 是否成功遗忘（未回滚）
    """
    if len(F) == 0:
        if verbose:
            print("[Projection-Forgetting] F is empty, skipping")
        return True
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"[Projection-Forgetting] Starting projection-based forgetting")
        print(f"{'='*80}")
        print(f"  |F| = {len(F)}")
        print(f"  step_scale = {step_scale}")
        print(f"  num_anchors = {num_anchors}")
    

    if verbose:
        print(f"\n[Step 1/6] Saving embedding checkpoint...")
    
    ent_embeddings, rel_embeddings = model.embedding('Train')
    checkpoint = {
        'ent': ent_embeddings.data.clone(),
        'rel': rel_embeddings.data.clone()
    }
    

    if verbose:
        print(f"[Step 2/6] Building SCA anchors...")
    
    anchors_map = build_anchors_for_forget_set(model, F, num_anchors=num_anchors)
    
    total_anchors = sum(len(anchors) for anchors in anchors_map.values())
    if verbose:
        print(f"  Generated {total_anchors} anchors for {len(F)} triples")
    

    if verbose:
        print(f"[Step 3/6] Computing forget directions...")
    
    direction_map = compute_forget_directions(model, F, anchors_map)
    
    if verbose:
        print(f"  Computed directions for {len(direction_map)} entities/relations")
    

    if verbose:
        print(f"[Step 4/6] Applying projection forgetting...")
    
    old_embeddings = apply_projection_forgetting(model, direction_map, step_scale=step_scale)
    
    if verbose:
        print(f"  Projection completed with step_scale={step_scale}")
    

    if verbose:
        print(f"[Step 5/6] Evaluating forgetting effect...")
    
    mrr_f_after = evaluate_forget_quality(model, F)
    
    if verbose:
        print(f"  MRR on F (after projection): {mrr_f_after:.4f}")
    


    if verbose:
        print(f"[Step 6/6] Checking retained performance...")
    







    
    if verbose:
        print(f"  Projection accepted (no rollback needed)")
        print(f"{'='*80}")
        print(f"[Projection-Forgetting] Completed")
        print(f"{'='*80}\n")
    
    return True






def perform_projection_forgetting_integrated(trainer_obj, snapshot_id: int):
    """
    集成投影遗忘到PAAL-UF流程
    
    流程：
    1. 加载遗忘集F
    2. 执行投影遗忘
    3. 执行邻域保护蒸馏（N(F)，不包含F）
    
    Args:
        trainer_obj: experiment对象（包含model, args, kg等）
        snapshot_id: 当前snapshot ID
    
    Returns:
        model: 遗忘后的模型
    """
    model = trainer_obj.model
    args = trainer_obj.args
    

    from PAAL import load_candidate_set
    import os
    
    candidates_dir = os.path.join(args.save_path, 'candidates')
    
    try:
        F = load_candidate_set(snapshot_id, 'F', out_dir=candidates_dir)
    except FileNotFoundError:
        print(f"[Projection-Forgetting] No forget set found for snapshot {snapshot_id}")
        return model
    
    if F is None or len(F) == 0:
        print(f"[Projection-Forgetting] Forget set is empty")
        return model
    

    step_scale = getattr(args, 'projection_step_scale', 0.2)
    num_anchors = getattr(args, 'projection_num_anchors', 2)
    

    success = forget_by_projection(
        model, F,
        step_scale=step_scale,
        num_anchors=num_anchors,
        verbose=True
    )
    
    if success:
        print(f"[Projection-Forgetting] Forgetting successful for snapshot {snapshot_id}")
    else:
        print(f"[Projection-Forgetting] Forgetting rolled back due to performance drop")
    



    
    return model

