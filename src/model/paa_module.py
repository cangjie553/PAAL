import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionProjector(nn.Module):
    """
    Learnable linear projection layer to handle variable embedding dimensions across snapshots.
    Maps embeddings from arbitrary dimensions to a fixed target dimension.
    """
    def __init__(self, target_dim=512, device='cpu'):
        super(DimensionProjector, self).__init__()
        self.target_dim = target_dim
        self.device = device
        self.projectors = {}
        
    def project(self, embedding, input_dim=None):
        """
        Project embedding to target dimension.
        
        Args:
            embedding: tensor of shape (..., input_dim)
            input_dim: if None, inferred from embedding shape
        
        Returns:
            projected embedding of shape (..., target_dim)
        """
        if input_dim is None:
            input_dim = embedding.shape[-1]
        

        if input_dim not in self.projectors:
            projector = nn.Linear(input_dim, self.target_dim, device=self.device)

            nn.init.xavier_uniform_(projector.weight, gain=0.01)
            nn.init.zeros_(projector.bias)
            self.projectors[input_dim] = projector
        
        projector = self.projectors[input_dim]
        return projector(embedding)
    
    def get_projector(self, input_dim):
        """Get or create projector for given input dimension."""
        if input_dim not in self.projectors:
            projector = nn.Linear(input_dim, self.target_dim, device=self.device)
            nn.init.xavier_uniform_(projector.weight, gain=0.01)
            nn.init.zeros_(projector.bias)
            self.projectors[input_dim] = projector
        return self.projectors[input_dim]
    
    def parameters(self):
        """Return all parameters from all projectors."""
        for projector in self.projectors.values():
            yield from projector.parameters()


class PatternAnchorAlignment:
    """
    Maintains relation-level pattern prototypes and computes alignment scores.
    
    - Prototypes are updated once per snapshot (not per batch)
    - Falls back to previous snapshot's prototype if current data is insufficient
    - PAA score is a continuous scalar in [-1, 1], not a hard classifier
    """
    
    def __init__(self, args, kg):
        """
        Args:
            args: argument namespace containing configuration
            kg: knowledge graph object
        """
        self.args = args
        self.kg = kg
        self.device = args.device
        



        self.min_samples = getattr(args, 'paa_min_quality_triples', 5)

        self.composition_mode = getattr(args, 'paa_composition_mode', 'concat')
        self.batch_size = getattr(args, 'paa_batch_size', 512)
        

        self.enable_dim_projection = getattr(args, 'paa_enable_dim_projection', True)
        self.projection_dim = getattr(args, 'paa_projection_dim', 512)
        

        if self.enable_dim_projection:
            self.projector = DimensionProjector(target_dim=self.projection_dim, device=self.device)
        else:
            self.projector = None
        


        self.relation_prototypes = {}
        

        self.previous_prototypes = {}
        

        self.current_embedding_dim = None
    
    def compute_paa(self, triple_emb, relation_id):
        """
        Compute Pattern-Anchor Alignment score.
        
        Args:
            triple_emb: triple embedding tensor, shape (emb_dim,) or (batch_size, emb_dim)
            relation_id: int or tensor of relation IDs
        
        Returns:
            PAA score(s) in [-1, 1]. Higher values indicate better pattern alignment.
            Returns 0.0 if no prototype exists.
        """

        is_batch = len(triple_emb.shape) == 2
        

        if self.enable_dim_projection and self.projector is not None:
            with torch.no_grad():
                input_dim = triple_emb.shape[-1]
                projector = self.projector.get_projector(input_dim)

                triple_emb_input = triple_emb.to(dtype=projector.weight.dtype)
                triple_emb_proj = projector(triple_emb_input)
        else:

            triple_emb_proj = triple_emb.float()
        
        if not is_batch:

            relation_id = int(relation_id) if torch.is_tensor(relation_id) else relation_id
            
            if relation_id not in self.relation_prototypes:

                if hasattr(self, '_debug_no_proto_count'):
                    self._debug_no_proto_count += 1
                else:
                    self._debug_no_proto_count = 1
                    
                if self.args.logger and torch.rand(1).item() < 0.01:
                    self.args.logger.debug(
                        f"[PAA DEBUG] Relation {relation_id} has NO prototype. "
                        f"Total prototypes={len(self.relation_prototypes)}"
                    )
                return torch.tensor(0.0, device=self.device)
            
            prototype = self.relation_prototypes[relation_id]
            

            paa = F.cosine_similarity(
                triple_emb_proj.unsqueeze(0),
                prototype.unsqueeze(0),
                dim=1
            ).squeeze()
            

            if torch.rand(1).item() < 0.01:
                if self.args.logger:
                    self.args.logger.info(
                        f"[PAA DEBUG] Relation {relation_id}: PAA={paa.item():.4f}, "
                        f"triple_emb_dim={triple_emb.shape[-1]}, "
                        f"triple_norm={torch.norm(triple_emb).item():.2f}, "
                        f"proto_norm={torch.norm(prototype).item():.2f}"
                    )
            
            return paa
        
        else:

            batch_size = triple_emb_proj.shape[0]
            paa_scores = torch.zeros(batch_size, device=self.device)
            
            if torch.is_tensor(relation_id):
                relation_ids = relation_id.cpu().tolist()
            else:
                relation_ids = [relation_id] * batch_size
            
            no_proto_count = 0
            non_zero_paa_count = 0
            paa_values = []
            
            for i, rel_id in enumerate(relation_ids):
                if rel_id in self.relation_prototypes:
                    prototype = self.relation_prototypes[rel_id]
                    

                    score = F.cosine_similarity(
                        triple_emb_proj[i].unsqueeze(0),
                        prototype.unsqueeze(0),
                        dim=1
                    ).squeeze()
                    paa_scores[i] = score
                    if score.item() != 0.0:
                        non_zero_paa_count += 1
                        paa_values.append(score.item())
                else:
                    no_proto_count += 1
                    paa_scores[i] = 0.0
            

            if self.args.logger and torch.rand(1).item() < 0.01:
                min_paa = min(paa_values) if paa_values else 0.0
                max_paa = max(paa_values) if paa_values else 0.0
                log_msg = (
                    f"[PAA DEBUG Batch] size={batch_size}, no_proto={no_proto_count}, "
                    f"non_zero_paa={non_zero_paa_count}, PAA range=[{min_paa:.4f}, {max_paa:.4f}]"
                )
                if self.enable_dim_projection:
                    log_msg += f" (projected_dim={self.projection_dim})"
                self.args.logger.debug(log_msg)
            
            return paa_scores
    
    def compute_triple_embedding(self, h_emb, r_emb, t_emb, mode='concat'):
        """
        Compose triple embedding from head, relation, and tail embeddings.
        
        Args:
            h_emb: head entity embedding
            r_emb: relation embedding
            t_emb: tail entity embedding
            mode: composition mode ('concat', 'sum', or 'hadamard')
        
        Returns:
            triple_emb: composed triple embedding
        """
        if mode == 'concat':
            return torch.cat([h_emb, r_emb, t_emb], dim=-1)
        elif mode == 'sum':
            return h_emb + r_emb + t_emb
        elif mode == 'hadamard':
            return h_emb * r_emb * t_emb
        else:
            raise ValueError(f"Unknown composition mode: {mode}")
    
    def update_prototypes(self, model, current_snapshot_id):
        """
        Update relation prototypes for the current snapshot.
        
        Uses ALL triples from current snapshot to compute "dominant structural configuration",
        not just high-quality triples. This ensures:
        1. Sufficient samples for statistical stability (min_samples=5)
        2. Prototypes represent typical structure, not just optimal cases
        3. Most relations can create prototypes (not limited by high-quality selection)
        
        Args:
            model: the LKGE model instance (used to access embeddings)
            current_snapshot_id: current snapshot ID
        """

        with torch.no_grad():
            ent_emb, rel_emb = model.embedding('Train')
            current_emb_dim = ent_emb.shape[1]
            

            if self.composition_mode == 'concat':
                expected_triple_dim = 3 * current_emb_dim
            else:
                expected_triple_dim = current_emb_dim
            

            if self.current_embedding_dim is not None and self.current_embedding_dim != current_emb_dim:
                if self.args.logger:
                    old_triple_dim = 3 * self.current_embedding_dim if self.composition_mode == 'concat' else self.current_embedding_dim
                    self.args.logger.warning(
                        f"[PAA] Embedding dimension changed: "
                        f"emb_dim {self.current_embedding_dim} → {current_emb_dim}, "
                        f"triple_emb_dim {old_triple_dim} → {expected_triple_dim}. "
                        f"Clearing {len(self.relation_prototypes)} old prototypes."
                    )
                self.previous_prototypes.clear()
                self.relation_prototypes.clear()
            
            self.current_embedding_dim = current_emb_dim
            
            if self.args.logger:
                self.args.logger.info(
                    f"[PAA] Snapshot {current_snapshot_id}: "
                    f"embedding_dim={current_emb_dim}, "
                    f"expected_triple_emb_dim={expected_triple_dim} ({self.composition_mode} mode)"
                )

        

        self.previous_prototypes = self.relation_prototypes.copy()
        self.relation_prototypes = {}
        

        try:
            snapshot = self.kg.snapshots[current_snapshot_id]
            edge_index = snapshot.edge_index
            edge_type = snapshot.edge_type
            

            all_triples = []
            num_edges = edge_index.size(1)
            for i in range(num_edges):
                h = edge_index[0, i].item()
                t = edge_index[1, i].item()
                r = edge_type[i].item()
                all_triples.append((h, r, t))
            
            if not all_triples:
                if self.args.logger:
                    self.args.logger.info(f"[PAA] No triples in snapshot {current_snapshot_id}")
                return
                
        except Exception as e:
            if self.args.logger:
                self.args.logger.warning(f"[PAA] Error accessing snapshot {current_snapshot_id}: {e}")
            return
        

        relation_groups = {}
        for h, r, t in all_triples:
            if r not in relation_groups:
                relation_groups[r] = []
            relation_groups[r].append((h, r, t))
        

        if self.args.logger:
            total_triples = len(all_triples)
            num_relations = len(relation_groups)
            triples_per_relation = [len(triples) for triples in relation_groups.values()]
            max_triples = max(triples_per_relation) if triples_per_relation else 0
            min_triples = min(triples_per_relation) if triples_per_relation else 0
            avg_triples = sum(triples_per_relation) / len(triples_per_relation) if triples_per_relation else 0
            
            self.args.logger.info(
                f"[PAA] Snapshot {current_snapshot_id}: "
                f"Total triples={total_triples}, "
                f"Relations={num_relations}, "
                f"Triples/Relation: min={min_triples}, max={max_triples}, avg={avg_triples:.1f}, "
                f"min_samples={self.min_samples}"
            )
        

        updated_count = 0
        skipped_count = 0
        
        with torch.no_grad():
            ent_emb, rel_emb = model.embedding('Train')
            
            for rel_id, triples in relation_groups.items():
                if len(triples) < self.min_samples:

                    skipped_count += 1
                    if rel_id in self.previous_prototypes:
                        self.relation_prototypes[rel_id] = self.previous_prototypes[rel_id]
                    continue
                

                embeddings = []
                for h, r, t in triples:
                    h_emb = ent_emb[h]
                    r_emb = rel_emb[r]
                    t_emb = ent_emb[t]
                    triple_emb = self.compute_triple_embedding(h_emb, r_emb, t_emb, mode=self.composition_mode)
                    embeddings.append(triple_emb)
                
                embeddings = torch.stack(embeddings)
                prototype = embeddings.mean(dim=0)
                

                if self.enable_dim_projection and self.projector is not None:
                    input_dim = prototype.shape[-1]
                    projector = self.projector.get_projector(input_dim)

                    prototype = prototype.to(dtype=projector.weight.dtype, device=self.device)
                    prototype = projector(prototype)
                else:

                    prototype = prototype.float()
                
                self.relation_prototypes[rel_id] = prototype
                updated_count += 1
        
        if self.args.logger:
            log_msg = (
                f"[PAA] Snapshot {current_snapshot_id}: "
                f"Updated {updated_count} prototypes, "
                f"Skipped {skipped_count} (insufficient triples), "
                f"Total prototypes={len(self.relation_prototypes)}"
            )
            if self.enable_dim_projection:
                log_msg += f" (projected to {self.projection_dim}D)"
            self.args.logger.info(log_msg)
    
    def get_prototype(self, relation_id):
        """Get prototype for a specific relation (for debugging/analysis)."""
        return self.relation_prototypes.get(relation_id, None)
    
    def has_prototype(self, relation_id):
        """Check if a prototype exists for a relation."""
        return relation_id in self.relation_prototypes

