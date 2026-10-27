import numpy as np
import torch
from typing import Dict, List, Tuple, Set


class DimensionEstimator:
    """
    Estimates embedding dimensions based on knowledge graph scale.
    Supports both total scale N and effective scale N_eff for CKGE-TU settings.
    """
    
    def __init__(self, args, kg):
        """
        Initialize the dimension estimator.
        
        Args:
            args: Arguments containing configuration
            kg: KnowledgeGraph object containing snapshot information
        """
        self.args = args
        self.kg = kg
        

        self.forgotten_triples_accumulated = set()
        

        self.dimension_history = []
        

        self.use_effective_scale = args.use_effective_scale
        self.formula = args.dim_estimation_formula
        self.base_constant = args.dim_base_constant
        self.scale_factor = args.dim_scale_factor
        
    def add_forgotten_triples(self, forgotten_triples: List[Tuple[int, int, int]]):
        """
        Add forgotten triples to the accumulated set.
        
        Args:
            forgotten_triples: List of (head, relation, tail) triples that are forgotten
        """
        for triple in forgotten_triples:
            self.forgotten_triples_accumulated.add(tuple(triple))
    
    def load_forgotten_triples_from_file(self, snapshot_id: int, file_path: str = None):
        """
        Load forgotten triples from a file for a specific snapshot.
        
        Args:
            snapshot_id: Current snapshot ID
            file_path: Path to the forgotten triples file (optional)
        """
        if file_path is None:
            file_path = f"{self.args.data_path}{snapshot_id}/forgotten.txt"
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        h, r, t = parts

                        if hasattr(self.kg, 'entity2id') and hasattr(self.kg, 'relation2id'):
                            h_id = self.kg.entity2id.get(h, None)
                            r_id = self.kg.relation2id.get(r, None)
                            t_id = self.kg.entity2id.get(t, None)
                            if h_id is not None and r_id is not None and t_id is not None:
                                self.forgotten_triples_accumulated.add((h_id, r_id, t_id))
        except FileNotFoundError:

            pass
    
    def compute_total_scale(self, snapshot_id: int) -> int:
        """
        Compute the total scale N (traditional approach).
        
        N = |E_i| + |R_i| + |T_i|
        
        Args:
            snapshot_id: Current snapshot ID
            
        Returns:
            Total scale N
        """
        snapshot = self.kg.snapshots[snapshot_id]
        num_entities = snapshot.num_ent
        num_relations = snapshot.num_rel
        num_triples = len(snapshot.train_all) if hasattr(snapshot, 'train_all') else len(snapshot.train_new)
        
        total_scale = num_entities + num_relations + num_triples
        return total_scale
    
    def compute_effective_scale(self, snapshot_id: int) -> int:
        """
        Compute the effective scale N_eff for CKGE-TU setting.
        
        N_eff = |E_i| + |R_i| + |T_i \ D_{f,accumulated}|
        
        where D_{f,accumulated} is the set of all forgotten triples up to current snapshot.
        
        Args:
            snapshot_id: Current snapshot ID
            
        Returns:
            Effective scale N_eff
        """
        snapshot = self.kg.snapshots[snapshot_id]
        num_entities = snapshot.num_ent
        num_relations = snapshot.num_rel
        

        all_triples = set()
        if hasattr(snapshot, 'train_all'):
            all_triples.update([tuple(t) for t in snapshot.train_all])
        else:
            all_triples.update([tuple(t) for t in snapshot.train_new])
        

        effective_triples = all_triples - self.forgotten_triples_accumulated
        num_effective_triples = len(effective_triples)
        
        effective_scale = num_entities + num_relations + num_effective_triples
        return effective_scale
    
    def estimate_dimension_logarithmic(self, scale: int) -> int:
        """
        Estimate dimension using logarithmic relationship.
        
        d_{i+1} = base_constant + scale_factor * log(N)
        
        Args:
            scale: Knowledge graph scale (N or N_eff)
            
        Returns:
            Estimated dimension
        """
        if scale <= 0:
            return int(self.args.emb_dim)
        
        dimension = self.base_constant + self.scale_factor * np.log(scale)
        dimension = int(np.round(dimension))
        

        dimension = max(dimension, int(self.args.emb_dim))
        
        return dimension
    
    def estimate_dimension_for_snapshot(self, snapshot_id: int) -> int:
        """
        Estimate the required dimension for a given snapshot.
        
        Args:
            snapshot_id: Snapshot ID
            
        Returns:
            Estimated embedding dimension
        """

        if self.use_effective_scale:
            self.load_forgotten_triples_from_file(snapshot_id)
        

        if self.use_effective_scale:
            scale = self.compute_effective_scale(snapshot_id)
            scale_type = "N_eff"
        else:
            scale = self.compute_total_scale(snapshot_id)
            scale_type = "N"
        

        if self.formula == "log":
            dimension = self.estimate_dimension_logarithmic(scale)
        elif self.formula == "fixed":

            dimension = int(self.args.emb_dim)
        elif self.formula == "manual":

            dim_list = self._get_manual_dim_list()
            dimension = dim_list[snapshot_id] if snapshot_id < len(dim_list) else dim_list[-1]
        else:

            dim_list = self._get_manual_dim_list()
            dimension = dim_list[snapshot_id] if snapshot_id < len(dim_list) else dim_list[-1]
        

        self.dimension_history.append({
            'snapshot_id': snapshot_id,
            'scale': scale,
            'scale_type': scale_type,
            'dimension': dimension,
            'num_forgotten': len(self.forgotten_triples_accumulated)
        })
        

        if hasattr(self.args, 'logger'):
            self.args.logger.info(
                f"Snapshot {snapshot_id}: {scale_type}={scale}, "
                f"Forgotten triples={len(self.forgotten_triples_accumulated)}, "
                f"Estimated dimension={dimension}"
            )
        
        return dimension
    
    def _get_manual_dim_list(self) -> List[int]:
        """
        Get manually specified dimension list from args.
        
        Returns:
            List of dimensions for each snapshot
        """
        dim_list = [self.args.emb_dim]
        if hasattr(self.args, 'emb_dim2'):
            dim_list.append(self.args.emb_dim2)
        if hasattr(self.args, 'emb_dim3'):
            dim_list.append(self.args.emb_dim3)
        if hasattr(self.args, 'emb_dim4'):
            dim_list.append(self.args.emb_dim4)
        if hasattr(self.args, 'emb_dim5'):
            dim_list.append(self.args.emb_dim5)
        return dim_list
    
    def get_dimension_list_for_all_snapshots(self) -> List[int]:
        """
        Compute dimension list for all snapshots.
        
        Returns:
            List of dimensions for each snapshot
        """
        dim_list = []
        for ss_id in range(int(self.args.snapshot_num)):
            dimension = self.estimate_dimension_for_snapshot(ss_id)
            dim_list.append(dimension)
        return dim_list
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about dimension estimation.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            'dimension_history': self.dimension_history,
            'total_forgotten_triples': len(self.forgotten_triples_accumulated),
            'use_effective_scale': self.use_effective_scale,
            'formula': self.formula
        }
    
    def print_statistics(self):
        """
        Print dimension estimation statistics.
        """
        if hasattr(self.args, 'logger'):
            logger = self.args.logger
        else:
            import logging
            logger = logging.getLogger()
        
        logger.info("=" * 80)
        logger.info("Dimension Estimation Statistics")
        logger.info("=" * 80)
        logger.info(f"Use Effective Scale: {self.use_effective_scale}")
        logger.info(f"Formula: {self.formula}")
        logger.info(f"Total Forgotten Triples: {len(self.forgotten_triples_accumulated)}")
        logger.info("-" * 80)
        
        for entry in self.dimension_history:
            logger.info(
                f"Snapshot {entry['snapshot_id']}: "
                f"{entry['scale_type']}={entry['scale']}, "
                f"dim={entry['dimension']}, "
                f"forgotten={entry['num_forgotten']}"
            )
        
        logger.info("=" * 80)

