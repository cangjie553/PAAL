import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def select_low_quality_triples_pre_training(model, all_triples, ratio=0.1):
    """
    Stage-1: 训练前选择低质量三元组（不参与训练）
    
    策略：计算所有三元组的Q-score，选择最差的ratio%作为F_pre
    
    Args:
        model: LKGE模型（初始化后）
        all_triples: 当前snapshot的所有训练三元组 [(h,r,t), ...]
        ratio: 遗忘比例（默认0.1 = 10%）
    
    Returns:
        F_pre: 预遗忘集（不参与训练）
        train_triples: 剩余训练集（参与训练）
    """

    logger = getattr(model.args, 'logger', None)
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[TSF Stage-1] Pre-training low-quality selection")
    logger.info(f"{'='*80}")
    logger.info(f"  Total triples: {len(all_triples)}")
    logger.info(f"  Forget ratio: {ratio:.1%}")
    

    q_scores = []
    with torch.no_grad():
        model.eval()
        
        batch_size = 1024
        for i in range(0, len(all_triples), batch_size):
            batch = all_triples[i:i+batch_size]
            

            heads = torch.tensor([t[0] for t in batch], dtype=torch.long, device=model.args.device)
            relations = torch.tensor([t[1] for t in batch], dtype=torch.long, device=model.args.device)
            tails = torch.tensor([t[2] for t in batch], dtype=torch.long, device=model.args.device)
            

            ent_embeddings, rel_embeddings = model.embedding('Train')
            

            s = torch.index_select(ent_embeddings, 0, heads)
            r = torch.index_select(rel_embeddings, 0, relations)
            o = torch.index_select(ent_embeddings, 0, tails)
            scores = model.score_fun(s, r, o)
            


            distances = scores
            q_batch = 1.0 / (1.0 + distances.detach().cpu().numpy())
            q_scores.extend(q_batch.tolist())
    

    if ratio <= 0:
        logger.info(f"  Ratio is 0, skipping pre-training removal")
        return set(), all_triples
    
    num_forget = max(1, int(len(all_triples) * ratio))
    

    sorted_indices = sorted(range(len(q_scores)), key=lambda i: q_scores[i])
    forget_indices = sorted_indices[:num_forget]
    
    F_pre = set()
    filtered_train_triples = []
    
    for idx, triple in enumerate(all_triples):
        if idx in forget_indices:
            F_pre.add(triple)
        else:
            filtered_train_triples.append(triple)
    
    logger.info(f"  Selected {len(F_pre)} low-quality triples as F_pre")
    logger.info(f"  Remaining {len(filtered_train_triples)} triples for training")
    
    return F_pre, filtered_train_triples




def forget_dsp_loss(model, F_triples, beta=1.0):
    """
    F-DSP遗忘损失: L_FDSP = -log(σ(-β * s_F))
    
    目标：降低遗忘集F的分数（增大距离）
    """
    if len(F_triples) == 0:
        return torch.tensor(0.0, device=model.args.device, requires_grad=True)
    

    heads = torch.tensor([t[0] for t in F_triples], dtype=torch.long, device=model.args.device)
    relations = torch.tensor([t[1] for t in F_triples], dtype=torch.long, device=model.args.device)
    tails = torch.tensor([t[2] for t in F_triples], dtype=torch.long, device=model.args.device)
    

    ent_embeddings, rel_embeddings = model.embedding('Train')
    

    s = torch.index_select(ent_embeddings, 0, heads)
    r = torch.index_select(rel_embeddings, 0, relations)
    o = torch.index_select(ent_embeddings, 0, tails)
    scores = model.score_fun(s, r, o)
    


    loss = -F.logsigmoid(-beta * scores).mean()
    
    return loss


def retain_protect_loss(model, R_triples, old_scores_dict):
    """
    保留保护损失: L_Rprotect = mean((old_score - new_score)^2)
    
    目标：保持保留集R的分数接近原始值
    """
    if len(R_triples) == 0:
        return torch.tensor(0.0, device=model.args.device, requires_grad=True)
    

    heads = torch.tensor([t[0] for t in R_triples], dtype=torch.long, device=model.args.device)
    relations = torch.tensor([t[1] for t in R_triples], dtype=torch.long, device=model.args.device)
    tails = torch.tensor([t[2] for t in R_triples], dtype=torch.long, device=model.args.device)
    

    ent_embeddings, rel_embeddings = model.embedding('Train')
    

    s = torch.index_select(ent_embeddings, 0, heads)
    r = torch.index_select(rel_embeddings, 0, relations)
    o = torch.index_select(ent_embeddings, 0, tails)
    new_scores = model.score_fun(s, r, o)
    

    old_scores = torch.tensor([old_scores_dict.get(triple, 0.0) for triple in R_triples], 
                             dtype=torch.float64, device=model.args.device)
    

    loss = ((old_scores - new_scores) ** 2).mean()
    
    return loss


def build_retain_set_from_sca(model, F_triples, num_anchors=5):
    """
    使用SCA（Stable Anchor）策略构造保留集R
    
    Args:
        model: LKGE模型
        F_triples: 遗忘集
        num_anchors: 每个遗忘三元组的锚点数量
    
    Returns:
        R_triples: 保留集三元组列表
        old_scores: 旧分数字典 {(h,r,t): score}
    """
    logger = getattr(model.args, 'logger', None)
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    R_triples = []
    old_scores = {}
    

    all_entities = set()
    all_relations = set()
    for (h, r, t) in F_triples:
        all_entities.add(h)
        all_entities.add(t)
        all_relations.add(r)
    

    for (h, r, t) in F_triples:


        candidate_entities = list(all_entities - {h, t})
        if len(candidate_entities) >= num_anchors:
            anchor_entities = random.sample(candidate_entities, num_anchors)
        else:
            anchor_entities = candidate_entities
        

        for anchor in anchor_entities:

            anchor_triple1 = (anchor, r, t)
            anchor_triple2 = (h, r, anchor)
            

            with torch.no_grad():
                model.eval()
                ent_embeddings, rel_embeddings = model.embedding('Train')
                

                h_emb = torch.index_select(ent_embeddings, 0, torch.tensor([anchor_triple1[0]], dtype=torch.long, device=model.args.device))
                r_emb = torch.index_select(rel_embeddings, 0, torch.tensor([anchor_triple1[1]], dtype=torch.long, device=model.args.device))
                o_emb = torch.index_select(ent_embeddings, 0, torch.tensor([anchor_triple1[2]], dtype=torch.long, device=model.args.device))
                score1 = model.score_fun(h_emb, r_emb, o_emb).item()
                

                h_emb = torch.index_select(ent_embeddings, 0, torch.tensor([anchor_triple2[0]], dtype=torch.long, device=model.args.device))
                r_emb = torch.index_select(rel_embeddings, 0, torch.tensor([anchor_triple2[1]], dtype=torch.long, device=model.args.device))
                o_emb = torch.index_select(ent_embeddings, 0, torch.tensor([anchor_triple2[2]], dtype=torch.long, device=model.args.device))
                score2 = model.score_fun(h_emb, r_emb, o_emb).item()
                
                R_triples.append(anchor_triple1)
                old_scores[anchor_triple1] = score1
                
                R_triples.append(anchor_triple2)
                old_scores[anchor_triple2] = score2
    

    R_triples = list(set(R_triples))
    
    logger.info(f"  Built retain set with {len(R_triples)} triples from SCA anchors")
    
    return R_triples, old_scores


def run_fdsp_finetuning(model, optimizer, F_combined, R_triples, old_scores,
                        uf_steps=10, lambda_f=1.0, lambda_r=1.0, beta=1.0,
                        verbose=True):
    """
    最终F-DSP微调阶段
    
    对合并后的遗忘集F = F_pre ∪ F_post应用F-DSP损失
    现在包含Continuous Forgetting Regularizer (CFR)
    """

    logger = getattr(model.args, 'logger', None)
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"[TSF Final] F-DSP Fine-tuning on combined forget set")
        logger.info(f"{'='*80}")
        logger.info(f"  |F_combined| = {len(F_combined)} (forget set)")
        logger.info(f"  |R| = {len(R_triples)} (retain set)")
        logger.info(f"  UF steps = {uf_steps}")
        logger.info(f"  λF = {lambda_f}, λR = {lambda_r}, β = {beta}")
    
    model.train()
    


    enable_cfr = getattr(model.args, 'enable_cfr', 'False')
    has_cfr = False
    if enable_cfr == 'True' or enable_cfr is True:
        cfr_weight = getattr(model.args, 'cfr_weight', 0.01)
        if cfr_weight > 0:
            if not hasattr(model, 'global_forget_set'):
                model.global_forget_set = set()
            has_cfr = len(model.global_forget_set) > 0
    else:
        cfr_weight = 0.0
    
    if verbose:
        if enable_cfr == 'True' or enable_cfr is True:
            if has_cfr:
                logger.info(f"  [CFR] Enabled with weight={cfr_weight}, Global forget set size: {len(model.global_forget_set)}")
            else:
                logger.info(f"  [CFR] Enabled but global_forget_set is empty (weight={cfr_weight})")
        else:
            logger.info(f"  [CFR] Disabled (enable_cfr={enable_cfr})")
    

    if hasattr(model, '_cfr_debug_logged'):
        delattr(model, '_cfr_debug_logged')
    
    for step in range(uf_steps):
        optimizer.zero_grad()
        

        loss_f = forget_dsp_loss(model, F_combined, beta=beta)
        

        loss_r = retain_protect_loss(model, R_triples, old_scores)
        

        loss_cfr = None
        if has_cfr:
            cfr_loss = model.compute_continuous_forgetting_loss()
            if cfr_loss is not None:



                loss_cfr = cfr_weight * cfr_loss

                if verbose and step == 0:
                    num_forget = len(model.global_forget_set)
                    logger.info(f"  [CFR Debug] Raw CFR loss (sum): {cfr_loss.item():.6f}, Weighted: {loss_cfr.item():.6f}, num_forget={num_forget}, cfr_weight={cfr_weight}")
            else:

                if verbose and step == 0:
                    logger.warning(f"  [CFR Debug] compute_continuous_forgetting_loss returned None (global_forget_set size: {len(model.global_forget_set)}, enable_cfr={enable_cfr})")
        else:

            if verbose and step == 0:
                logger.warning(f"  [CFR Debug] Disabled: has_cfr={has_cfr}, enable_cfr={enable_cfr}, cfr_weight={cfr_weight}, global_forget_set size: {len(getattr(model, 'global_forget_set', set()))})")
        

        total_loss = lambda_f * loss_f + lambda_r * loss_r
        if loss_cfr is not None:
            total_loss = total_loss + loss_cfr
        

        total_loss.backward()
        optimizer.step()
        

        if verbose:
            log_msg = f"[TSF Final] Step {step:2d}/{uf_steps}: " \
                     f"Total={total_loss.item():.4f}, " \
                     f"L_FDSP={loss_f.item():.4f}, " \
                     f"L_Rprotect={loss_r.item():.4f}"
            if loss_cfr is not None:
                log_msg += f", L_CFR={loss_cfr.item():.9f}"
            else:
                log_msg += f", L_CFR=0.000000000"
            logger.info(log_msg)
    
    if verbose:
        logger.info(f"{'='*80}")
        logger.info(f"[TSF Final] ✅ F-DSP fine-tuning completed ({uf_steps} steps)")
        logger.info(f"{'='*80}\n")
    
    return model

