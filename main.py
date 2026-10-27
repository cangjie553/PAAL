import sys
sys.path.append('.')

from src.utils import *
import argparse
from src.train import *
from src.test import *
from src.config import *
from src.parse_args import args
from src.dimension_estimator import DimensionEstimator
from src.model.LKGE import TransE as LKGE_TransE
from src.model.GEM import TransE as GEM_TransE
from src.model.EMR import TransE as EMR_TransE
from src.model.CWR import TransE as CWR_TransE
from src.model.LAN import TransE as LAN_TransE
from src.model.PNN import TransE as PNN_TransE
from src.model.finetune import TransE as finetune_TransE
from src.model.MEAN import TransE as MEAN_TransE
from src.model.SI import TransE as SI_TransE
from src.model.Snapshot_only import TransE as Snapshot_TransE
from src.model.EWC import TransE as EWC_TransE
from src.model.retraining import TransE as retraining_TransE
from src.data_load.KnowledgeGraph import KnowledgeGraph
import shutil
from datetime import datetime
import json



import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from PAAL import create_unlearning_evaluator, load_candidate_set
from src.method.FDSP_IMPLEMENTATION import perform_f_dsp_forgetting, forget_mode_finetune
from src.method.PROJECTION_FORGETTING import perform_projection_forgetting_integrated
from src.method.TSF_MODULE_V2 import (select_low_quality_triples_pre_training, 
                            capture_low_quality_triples_post_training,
                            run_fdsp_finetuning, build_retain_set_from_sca)


class experiment():
    def __init__(self, args):
        self.args = args

        '''1. prepare data file path, model saving path and log path'''
        self.prepare()

        '''2. load data'''
        self.kg = KnowledgeGraph(args)

        '''3. initialize dimension estimator and set initial dimension'''
        self.dimension_estimator = DimensionEstimator(args, self.kg)

        if self.args.dim_estimation_formula == 'log':
            first_dim = self.dimension_estimator.estimate_dimension_for_snapshot(0)
            self.args.emb_dim = first_dim
            self.args.logger.info(f"Using estimated dimension for first snapshot: {first_dim}")

        '''4. create model and optimizer'''
        self.model, self.optimizer = self.create_model()
        if self.args.lifelong_name == 'GEM':
            self.args.optimizer = self.optimizer

        self.args.logger.info(self.args)

    def create_model(self):
        '''
        Initialize KG embedding model and optimizer.
        return: model, optimizer
        '''
        if self.args.lifelong_name == 'LKGE':
            model = LKGE_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'GEM':
            model = GEM_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'EMR':
            model = EMR_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'LAN':
            model = LAN_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'finetune':
            model = finetune_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'Snapshot':
            model = Snapshot_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'SI':
            model = SI_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'CWR':
            model = CWR_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'PNN':
            model = PNN_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'MEAN':
            model = MEAN_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'EWC':
            model = EWC_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'retraining':
            self.args.train_new = False
            model = retraining_TransE(self.args, self.kg)
        else:
            self.args.logger.info("Unknown lifelong model name", "f")
            exit()
        model.to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(
            self.args.learning_rate), weight_decay=self.args.l2)
        return model, optimizer

    def reset_model(self, model=False, optimizer=False):
        '''
        Reset the model or optimizer
        :param model: If True: reset the model
        :param optimizer: If True: reset the optimizer
        '''
        if model:
            self.model, self.optimizer = self.create_model()
            if self.args.lifelong_name == 'GEM':
                self.args.optimizer = self.optimizer
        if optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(
                self.args.learning_rate), weight_decay=self.args.l2)
            if self.args.lifelong_name == 'GEM':
                self.args.optimizer = self.optimizer

    def train(self):

        '''
        Training process
        :return: training time
        '''


        start_time = time.time()
        print("Start Training ===============================>")
        



        snapshot_id = self.args.snapshot
        enable_tsf = getattr(self.args, 'enable_tsf', 'False')
        if enable_tsf == 'True' or enable_tsf is True:

            all_triples = list(self.kg.snapshots[snapshot_id].train_new)
            

            F_pre, filtered_train_triples = select_low_quality_triples_pre_training(
                self.model, all_triples,
                ratio=getattr(self.args, 'tsf_pre_ratio', 0.1)
            )
            

            self.model.F_pre = F_pre
            

            self.kg.snapshots[snapshot_id].train_new = filtered_train_triples
            
            self.args.logger.info(f"[TSF Stage-1] Original: {len(all_triples)} triples")
            self.args.logger.info(f"[TSF Stage-1] F_pre excluded: {len(F_pre)} triples")
            self.args.logger.info(f"[TSF Stage-1] Training will use: {len(filtered_train_triples)} triples")
        
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)
        retrain_data = None

        '''Training iteration'''
        for epoch in range(int(self.args.epoch_num)):
            self.args.epoch = epoch
            '''training'''
            loss, valid_res = trainer.run_epoch()
            '''early stop'''
            if self.best_valid < valid_res[self.args.valid_metrics]:
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = max(0, self.stop_epoch-5)
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                self.save_model()
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info('Early Stopping! Snapshot:{} Epoch: {} Best Results: {}'.format(
                        self.args.snapshot, epoch, round(self.best_valid*100, 3)))

                    if getattr(self.args, 'enable_unlearning', False):
                        snapshot_id = getattr(self.args, 'snapshot', 0)
                        candidates_dir = os.path.join(self.args.save_path, 'candidates')
                        snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
                        forget_file = os.path.join(snapshot_dir, 'forget_F.json')
                        if not os.path.exists(forget_file):
                            self.args.logger.info(f"[PAAL Unlearning] Early stopping at epoch {epoch}, performing forgetting...")
                            self.perform_snapshot_forgetting()
                    break
            '''logging'''
            if epoch % 1 == 0:
                self.args.logger.info('Snapshot:{}\tEpoch:{}\tLoss:{}\tMRR:{}\tHits@10:{}\tBest:{}'.format(self.args.snapshot, epoch, round(
                    loss, 3), round(valid_res['mrr'] * 100, 2), round(valid_res['hits10'] * 100, 2), round(self.best_valid * 100, 2)))
        end_time = time.time()
        training_time = end_time - start_time
        
        snapshot_id = getattr(self.args, 'snapshot', 0)
        

        if getattr(self.args, 'enable_paa', 'False') in ['True', True] and hasattr(self.model, 'paa') and self.model.paa is not None:
            if hasattr(self.model, 'current_high_quality_triples') and self.model.current_high_quality_triples:

                high_quality_triples = []
                for triple_data in self.model.current_high_quality_triples:
                    if len(triple_data) >= 2:
                        h, r, t = triple_data[1]
                        high_quality_triples.append((h, r, t))
                
                if len(high_quality_triples) > 0:
                    self.args.logger.info(f"[PAA] Updating relation prototypes for snapshot {snapshot_id} with {len(high_quality_triples)} high-quality triples")
                    self.model.paa.update_prototypes(
                        high_quality_triples,
                        self.model.compute_triple_embedding
                    )
                    self.args.logger.info(f"[PAA] Updated {len(self.model.paa.relation_prototypes)} relation prototypes")
        

        if getattr(self.args, 'enable_unlearning', False):
            candidates_dir = os.path.join(self.args.save_path, 'candidates')
            snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
            forget_file = os.path.join(snapshot_dir, 'forget_F.json')
            

            if not os.path.exists(forget_file):
                self.args.logger.info(f"[遗忘操作] Training completed for snapshot {snapshot_id}, performing forgetting...")

                self.perform_snapshot_forgetting()
                


                enable_cfr = getattr(self.args, 'enable_cfr', 'False')
                if enable_cfr == 'True' or enable_cfr is True:

                    try:
                        from PAAL import load_candidate_set
                        F_loaded = load_candidate_set(snapshot_id, 'F', out_dir=candidates_dir)
                        if F_loaded is not None and len(F_loaded) > 0:
                            F_triples = [tuple(item['triple']) for item in F_loaded]
                            if not hasattr(self.model, 'global_forget_set'):
                                self.model.global_forget_set = set()
                            self.model.global_forget_set.update(F_triples)
                            self.args.logger.info(f"[Continuous Forgetting] Added {len(F_triples)} triples to global forget set. Total: {len(self.model.global_forget_set)}")
                    except Exception as e:
                        self.args.logger.warning(f"[Continuous Forgetting] Failed to load forget set for snapshot {snapshot_id}: {e}")
            else:
                self.args.logger.info(f"[遗忘操作] Forgetting already performed for snapshot {snapshot_id}, skipping.")
        




        enable_tsf = getattr(self.args, 'enable_tsf', 'False')
        if enable_tsf == 'True' or enable_tsf is True:
            self.args.logger.info(f"\n{'='*80}")
            self.args.logger.info(f"[TSF] Stage-2 + Final F-DSP Fine-tuning")
            self.args.logger.info(f"{'='*80}")
            
            try:

                train_triples = list(self.kg.snapshots[snapshot_id].train_new)
                F_post = capture_low_quality_triples_post_training(
                    self.model, train_triples,
                    ratio=getattr(self.args, 'tsf_post_ratio', 0.05)
                )
                

                self.model.F_post = F_post
                

                F_pre = getattr(self.model, 'F_pre', set())
                F_combined = list(F_pre | F_post)
                
                self.args.logger.info(f"[TSF] |F_pre| = {len(F_pre)} (excluded from training)")
                self.args.logger.info(f"[TSF] |F_post| = {len(F_post)} (captured after training)")
                self.args.logger.info(f"[TSF] |F_combined| = {len(F_combined)} (total forget set)")
                


                enable_cfr = getattr(self.args, 'enable_cfr', 'False')
                if enable_cfr == 'True' or enable_cfr is True:
                    if not hasattr(self.model, 'global_forget_set'):
                        self.model.global_forget_set = set()

                    F_combined_tuples = [tuple(t) if not isinstance(t, tuple) else t for t in F_combined]
                    self.model.global_forget_set.update(F_combined_tuples)
                    self.args.logger.info(f"[Continuous Forgetting] Added {len(F_combined)} triples to global forget set. Total: {len(self.model.global_forget_set)}")
                
                if len(F_combined) > 0:
                    self.args.logger.info(f"[TSF] F_combined is NOT empty, proceeding with F-DSP fine-tuning...")
                    

                    R_triples, old_scores = build_retain_set_from_sca(
                        self.model, F_combined,
                        num_anchors=getattr(self.args, 'tsf_num_anchors', 5)
                    )
                    
                    self.args.logger.info(f"[TSF] |R| = {len(R_triples)} (retain set with SCA anchors)")
                    

                    self.args.logger.info(f"[TSF] ✅ About to call run_fdsp_finetuning...")
                    self.args.logger.info(f"[TSF] Starting F-DSP fine-tuning...")
                    self.args.logger.info(f"[TSF]   - UF steps: {getattr(self.args, 'tsf_uf_steps', 10)}")
                    self.args.logger.info(f"[TSF]   - Lambda_F: {getattr(self.args, 'tsf_lambda_f', 1.0)}")
                    self.args.logger.info(f"[TSF]   - Lambda_R: {getattr(self.args, 'tsf_lambda_r', 1.0)}")
                    self.args.logger.info(f"[TSF]   - Beta: {getattr(self.args, 'tsf_beta', 1.0)}")
                    

                    try:
                        from src.method.TSF_MODULE_V2 import run_fdsp_finetuning
                        self.args.logger.info(f"[TSF] ✅ run_fdsp_finetuning imported successfully")
                    except ImportError as e:
                        self.args.logger.error(f"[TSF] ❌ Failed to import run_fdsp_finetuning: {e}")
                        raise
                    
                    self.args.logger.info(f"[TSF] ✅ Calling run_fdsp_finetuning now...")
                    self.model = run_fdsp_finetuning(
                        self.model, self.optimizer,
                        F_combined, R_triples, old_scores,
                        uf_steps=getattr(self.args, 'tsf_uf_steps', 10),
                        lambda_f=getattr(self.args, 'tsf_lambda_f', 1.0),
                        lambda_r=getattr(self.args, 'tsf_lambda_r', 1.0),
                        beta=getattr(self.args, 'tsf_beta', 1.0),
                        verbose=True
                    )
                    
                    self.args.logger.info(f"[TSF] F-DSP fine-tuning completed!")
                    

                    tsf_save_path = os.path.join(self.args.save_path, f'checkpoint_{snapshot_id}_tsf.pt')
                    torch.save(self.model.state_dict(), tsf_save_path)
                    self.args.logger.info(f'[TSF] Model saved to {tsf_save_path}')
                    

                    if getattr(self.args, 'eval_unlearning', True):
                        self.args.logger.info(f"\n[TSF] Re-evaluating snapshot {snapshot_id} after F-DSP fine-tuning...")
                        unlearning_res_after_tsf = self.evaluate_unlearning(snapshot_id)
                        
                        if unlearning_res_after_tsf is not None:

                            if not hasattr(self.args, 'tsf_unlearning_results'):
                                self.args.tsf_unlearning_results = {}
                            retained = unlearning_res_after_tsf.get('retained', {})
                            forgetting = unlearning_res_after_tsf.get('forgetting', {})
                            self.args.tsf_unlearning_results[snapshot_id] = {
                                'snapshot_id': snapshot_id,
                                'MRR_ret': retained.get('MRR_ret', 0.0),
                                'Hits@1_ret': retained.get('Hits@1_ret', 0.0),
                                'Hits@3_ret': retained.get('Hits@3_ret', 0.0),
                                'Hits@10_ret': retained.get('Hits@10_ret', 0.0),
                                'MRR_F': forgetting.get('MRR_F', 0.0),
                                'Hits@1_F': forgetting.get('Hits@1_F', 0.0),
                                'Hits@3_F': forgetting.get('Hits@3_F', 0.0),
                                'Hits@10_F': forgetting.get('Hits@10_F', 0.0),
                            }
                            self.args.logger.info(f"[TSF] ✅ Evaluation results saved for snapshot {snapshot_id}")
                    

                    try:

                        candidates_dir = os.path.join(self.args.save_path, 'candidates')
                        snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
                        os.makedirs(snapshot_dir, exist_ok=True)
                        


                        f_data = {
                            'tag': 'F',
                            'count': len(F_combined),
                            'strategy': 'TSF',
                            'triples': []
                        }
                        for t in F_combined:
                            if isinstance(t, (list, tuple)) and len(t) == 3:
                                h, r, t_val = t[0], t[1], t[2]
                            else:

                                h, r, t_val = t.get('h', 0), t.get('r', 0), t.get('t', 0)
                            f_data['triples'].append({
                                'h': int(h),
                                'r': int(r),
                                't': int(t_val),
                                'Q': 0.5
                            })
                        
                        f_path = os.path.join(snapshot_dir, 'forget_F.json')
                        with open(f_path, 'w') as f:
                            json.dump(f_data, f, indent=2)
                        self.args.logger.info(f"[TSF] ✅ Forget set saved to {f_path} (|F|={len(F_combined)})")
                    except Exception as e:
                        self.args.logger.warning(f"[TSF] ⚠️  Failed to save forget set to file: {e}")
                        import traceback
                        self.args.logger.warning(traceback.format_exc())
                    

                    self.model.F_pre.clear()
                    self.model.F_post.clear()
                else:
                    self.args.logger.warning(f"[TSF] ⚠️  F_combined is EMPTY (F_pre={len(F_pre)}, F_post={len(F_post)})!")
                    self.args.logger.warning(f"[TSF] ⚠️  Skipping F-DSP fine-tuning because there are no triples to forget.")
                    self.args.logger.warning(f"[TSF] ⚠️  To enable F-DSP, ensure tsf_pre_ratio > 0 OR tsf_post_ratio > 0")
                
                self.args.logger.info(f"{'='*80}")
                self.args.logger.info(f"[TSF] Two-Stage Forgetting completed")
                self.args.logger.info(f"{'='*80}\n")
                
            except Exception as e:
                self.args.logger.error(f"[TSF] Error: {str(e)}")
                import traceback
                self.args.logger.error(traceback.format_exc())
        

        enable_fdsp = getattr(self.args, 'enable_fdsp', 'False')
        enable_tsf_check = getattr(self.args, 'enable_tsf', 'False')
        

        if (enable_fdsp == 'True' or enable_fdsp is True) and not (enable_tsf_check == 'True' or enable_tsf_check is True):
                self.args.logger.info(f"\n{'='*80}")
                self.args.logger.info(f"[F-DSP] Starting F-DSP forgetting for snapshot {snapshot_id}")
                self.args.logger.info(f"{'='*80}")
                
                try:

                    from src.method.FDSP_IMPLEMENTATION import perform_f_dsp_forgetting_optimized
                    

                    self.model = perform_f_dsp_forgetting_optimized(self, snapshot_id)
                    

                    fdsp_save_path = os.path.join(self.args.save_path, f'checkpoint_{snapshot_id}_fdsp.pt')
                    torch.save(self.model.state_dict(), fdsp_save_path)
                    self.args.logger.info(f'[F-DSP] Model saved to {fdsp_save_path}')
                    
                    self.args.logger.info(f"{'='*80}")
                    self.args.logger.info(f"[F-DSP] F-DSP forgetting completed")
                    self.args.logger.info(f"{'='*80}\n")
                except Exception as e:
                    self.args.logger.error(f"[F-DSP] Error: {str(e)}")
                    import traceback
                    self.args.logger.error(traceback.format_exc())

        
        print('generating retrain data')

        if self.args.snapshot_num > self.args.snapshot + 1:
            retrain_data = trainer.train_processor.evaluate_on_training_data(
                self.model)

        return training_time, retrain_data

    def test(self):
        tester = Tester(self.args, self.kg, self.model)
        res = tester.test()
        return res

    def prepare(self):
        '''
        set the log path, the model saving path and device
        :return: None
        '''
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)

        '''set data path'''
        self.args.data_path = args.data_path + args.dataset + '/'
        self.args.save_path = args.save_path + args.dataset + '-' + \
            args.embedding_model + '-' + args.lifelong_name + '-' + args.loss_name

        '''add logging implement to model path for ablation_study'''
        if self.args.lifelong_name == 'LKGE':
            if self.args.using_regular_loss == 'False':
                self.args.save_path = self.args.save_path + '-WO_regularloss'
            if self.args.using_reconstruct_loss == 'False':
                self.args.save_path = self.args.save_path + '-WO_reconstructloss'
            if self.args.using_embedding_transfer == 'False':
                self.args.save_path = self.args.save_path + '-WO_transfer'
            if self.args.using_finetune == 'False':
                self.args.save_path = self.args.save_path + '-WO_finetune'
        if self.args.note != '':
            self.args.save_path = self.args.save_path

        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        self.args.log_path = args.log_path + datetime.now().strftime('%Y%m%d/')
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)

        self.args.log_path = args.log_path + args.dataset + '-formula' + args.formula

        '''add additional note to log name'''
        if self.args.note != '':
            self.args.log_path = self.args.log_path + self.args.note

        '''set logger'''
        logger = logging.getLogger()
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = args.log_path + '.log'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        '''set device'''
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device

    def load_forgotten_triples(self, snapshot_id):
        """
        Load forgotten triples for the current snapshot and update dimension estimator.
        
        Args:
            snapshot_id: Current snapshot ID
        """
        if self.args.use_effective_scale:
            forgotten_file = f"{self.args.data_path}{snapshot_id}/forgotten.txt"
            try:
                forgotten_triples = []
                with open(forgotten_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 3:
                            h, r, t = parts

                            if h in self.kg.entity2id and r in self.kg.relation2id and t in self.kg.entity2id:
                                h_id = self.kg.entity2id[h]
                                r_id = self.kg.relation2id[r]
                                t_id = self.kg.entity2id[t]
                                forgotten_triples.append((h_id, r_id, t_id))
                
                if forgotten_triples:
                    self.dimension_estimator.add_forgotten_triples(forgotten_triples)
                    self.args.logger.info(f"Loaded {len(forgotten_triples)} forgotten triples for snapshot {snapshot_id}")
            except FileNotFoundError:

                pass

    def next_snapshot_setting(self):
        '''
        Prepare for next snapshot
        '''

        if self.args.snapshot == 0 and hasattr(self.model, 'fixed_initial_graph') and self.model.fixed_initial_graph:
            self.model.store_initial_graph_structure()


        if self.args.snapshot == 0 and hasattr(self.model, 'fixed_initial_regularization') and self.model.fixed_initial_regularization:
            self.model.store_initial_parameters()

        self.model.switch_snapshot()




    def get_dim_list(self):
        """
        Get dimension list for all snapshots.
        
        If use_effective_scale is enabled, uses DimensionEstimator with N_eff.
        Otherwise, falls back to manual dimension specification.
        """
        if self.args.use_effective_scale or self.args.dim_estimation_formula == 'log':

            dim_list = self.dimension_estimator.get_dimension_list_for_all_snapshots()
            self.args.logger.info(f"Using dimension estimator: {dim_list}")
        else:

            self.args.emb_dim1 = self.args.emb_dim
            dim_list = []
            dim_list.append(self.args.emb_dim1)
            dim_list.append(self.args.emb_dim2)
            dim_list.append(self.args.emb_dim3)
            dim_list.append(self.args.emb_dim4)
            dim_list.append(self.args.emb_dim5)
            self.args.logger.info(f"Using manual dimensions: {dim_list}")
        
        return dim_list

    def continual_learning(self):
        '''
        The training process on all snapshots.
        :return:
        '''
        '''prepare'''
        report_results = PrettyTable()
        report_results.field_names = [
            'Snapshot', 'Time', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10']
        test_results = []
        training_times = []
        BWT, FWT = [], []
        first_learning_res = []
        dim_list = self.get_dim_list()

        self.args.dim_list = dim_list

        all_test_again_results = []
        all_test_results = []

        unlearning_results = []
        '''training process'''
        for ss_id in range(int(self.args.snapshot_num)):
            self.args.snapshot = ss_id
            self.args.snapshot_test = ss_id
            self.args.emb_dim = dim_list[ss_id]

            '''skip previous snapshots, train on the final snapshot'''
            if self.args.skip_previous == 'True' and self.args.snapshot < int(self.args.snapshot_num) - 1 and self.args.lifelong_name in ['Snapshot', 'retraining']:
                self.next_snapshot_setting()
                self.reset_model(optimizer=True)
                continue

            '''preprocess before training on a snapshot'''
            self.model.pre_snapshot()

            if ss_id > 0:

                if self.args.lifelong_name in ['MEAN', 'LAN']:
                    FWT.append(0)
                else:
                    self.args.test_FWT = True
                    res_before = self.test()
                    FWT.append(res_before['mrr'])
            self.args.test_FWT = False

            retrain_data = None
            '''training'''
            if ss_id == 0 or self.args.lifelong_name not in ['MEAN', 'LAN', 'LKGE'] or (self.args.lifelong_name == 'LKGE' and self.args.using_finetune == 'True'):
                training_time, retrain_data = self.train()
            else:
                training_time = 0

            '''prepare result table'''
            test_res = PrettyTable()
            test_res.field_names = [
                'Snapshot:'+str(ss_id), 'MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']

            '''save and reload model'''
            best_checkpoint = os.path.join(
                self.args.save_path, str(ss_id) + 'model_best.tar')
            self.load_checkpoint(best_checkpoint)

            '''post-snapshot correction'''
            if self.args.enable_correction:
                self.post_snapshot_correction()

            '''cleanup checkpoints: only keep best model for this snapshot'''
            self.cleanup_snapshot_checkpoints(ss_id)

            '''post processing'''
            self.model.snapshot_post_processing()

            '''evaluation'''
            print("start testing")
            tem_res = {}
            tem_res['trained_ssid'] = ss_id
            reses = []
            for test_ss_id in range(ss_id+1):
                self.args.snapshot_test = test_ss_id
                res = self.test()
                res['test_ssid'] = test_ss_id
                if test_ss_id == ss_id:
                    first_learning_res.append(res['mrr'])
                test_res.add_row(
                    [test_ss_id, res['mrr'], res['hits1'], res['hits3'], res['hits5'], res['hits10']])
                reses.append(res)
            tem_res['results'] = reses

            if ss_id == self.args.snapshot_num-1:
                for iid in range(self.args.snapshot_num-1):
                    BWT.append(reses[iid]['mrr']-first_learning_res[iid])

            '''record all results'''
            self.args.logger.info('\n{}'.format(test_res))
            test_results.append(test_res)

            '''record report results'''
            whole_mrr, whole_hits1, whole_hits3, whole_hits10 = self.get_report_result(
                reses)
            report_results.add_row(
                [ss_id, training_time, whole_mrr, whole_hits1, whole_hits3, whole_hits10])
            training_times.append(training_time)
            aggregated_result = {}
            aggregated_result['whole_mrr'] = whole_mrr
            aggregated_result['whole_hits1'] = whole_hits1
            aggregated_result['whole_hits3'] = whole_hits3
            aggregated_result['whole_hits10'] = whole_hits10
            tem_res['aggregated_result'] = aggregated_result
            all_test_results.append(tem_res)
            
            '''unlearning evaluation (在链接预测之后评估)'''

            enable_unlearning = getattr(self.args, 'enable_unlearning', False)
            enable_tsf = getattr(self.args, 'enable_tsf', 'False')
            enable_tsf_check = enable_tsf == 'True' or enable_tsf is True
            
            if (enable_unlearning or enable_tsf_check) and getattr(self.args, 'eval_unlearning', True):

                if enable_tsf_check and not enable_unlearning:

                    if hasattr(self.args, 'tsf_unlearning_results') and ss_id in getattr(self.args, 'tsf_unlearning_results', {}):

                        tsf_res = self.args.tsf_unlearning_results[ss_id]
                        unlearning_res = {
                            'retained': {
                                'MRR_ret': tsf_res['MRR_ret'],
                                'Hits@1_ret': tsf_res['Hits@1_ret'],
                                'Hits@3_ret': tsf_res['Hits@3_ret'],
                                'Hits@10_ret': tsf_res['Hits@10_ret'],
                            },
                            'forgetting': {
                                'MRR_F': tsf_res['MRR_F'],
                                'Hits@1_F': tsf_res['Hits@1_F'],
                                'Hits@3_F': tsf_res['Hits@3_F'],
                                'Hits@10_F': tsf_res['Hits@10_F'],
                            }
                        }
                        self.args.logger.info(f"[TSF] Using post-Stage-2 evaluation results for snapshot {ss_id}")
                    else:

                        self.args.logger.info(f"[TSF] Skipping early evaluation for snapshot {ss_id} (will evaluate after Stage-2)")
                        unlearning_res = None
                else:
                    unlearning_res = self.evaluate_unlearning(ss_id)
                
                if unlearning_res is not None:
                    retained = unlearning_res.get('retained', {})
                    forgetting = unlearning_res.get('forgetting', {})
                    


                    snapshot_report = PrettyTable()
                    snapshot_report.field_names = [
                        'Snapshot', 'Set', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10'
                    ]
                    snapshot_report.add_row([
                        ss_id,
                        'Retention',
                        round(retained.get('MRR_ret', 0.0), 3),
                        round(retained.get('Hits@1_ret', 0.0), 3),
                        round(retained.get('Hits@3_ret', 0.0), 3),
                        round(retained.get('Hits@10_ret', 0.0), 3),
                    ])
                    snapshot_report.add_row([
                        ss_id,
                        'Forgetting',
                        round(forgetting.get('MRR_F', 0.0), 3),
                        round(forgetting.get('Hits@1_F', 0.0), 3),
                        round(forgetting.get('Hits@3_F', 0.0), 3),
                        round(forgetting.get('Hits@10_F', 0.0), 3),
                    ])
                    
                    self.args.logger.info(f"\nSnapshot {ss_id} 遗忘评估结果:")
                    self.args.logger.info(f"{snapshot_report}")
                    self.args.logger.info(f"  保留集: MRR={retained.get('MRR_ret', 0.0):.4f}, Hits@1={retained.get('Hits@1_ret', 0.0):.4f}, Hits@3={retained.get('Hits@3_ret', 0.0):.4f}, Hits@10={retained.get('Hits@10_ret', 0.0):.4f}")
                    self.args.logger.info(f"  遗忘集: MRR={forgetting.get('MRR_F', 0.0):.4f}, Hits@1={forgetting.get('Hits@1_F', 0.0):.4f}, Hits@3={forgetting.get('Hits@3_F', 0.0):.4f}, Hits@10={forgetting.get('Hits@10_F', 0.0):.4f}")
                    

                    unlearning_results.append({
                        'snapshot_id': ss_id,

                        'MRR_ret': retained.get('MRR_ret', 0.0),
                        'Hits@1_ret': retained.get('Hits@1_ret', 0.0),
                        'Hits@3_ret': retained.get('Hits@3_ret', 0.0),
                        'Hits@10_ret': retained.get('Hits@10_ret', 0.0),

                        'MRR_F': forgetting.get('MRR_F', 0.0),
                        'Hits@1_F': forgetting.get('Hits@1_F', 0.0),
                        'Hits@3_F': forgetting.get('Hits@3_F', 0.0),
                        'Hits@10_F': forgetting.get('Hits@10_F', 0.0),
                    })
                else:

                    unlearning_results.append({
                        'snapshot_id': ss_id,
                        'MRR_ret': 0.0,
                        'Hits@1_ret': 0.0,
                        'Hits@3_ret': 0.0,
                        'Hits@10_ret': 0.0,
                        'MRR_F': 0.0,
                        'Hits@1_F': 0.0,
                        'Hits@3_F': 0.0,
                        'Hits@10_F': 0.0,
                    })

            if ss_id < int(self.args.snapshot_num) - 1:
                if dim_list[ss_id+1] == dim_list[ss_id]:
                    print('the same dimension, no need to train mapping matrix')
                else:


                    self.model.mapping = torch.nn.Linear(
                        dim_list[ss_id], dim_list[ss_id+1]-dim_list[ss_id], bias=False, dtype=torch.float64).to(self.args.device)
                    xavier_normal_(self.model.mapping.weight.data)


                    self.model.this_snapshot_ent_embeddings = self.model.ent_embeddings.weight.clone().to(
                        self.args.device)
                    self.model.this_snapshot_rel_embeddings = self.model.rel_embeddings.weight.clone(
                    ).to(self.args.device)

                    print('start training mapping matrix')
                    self.train_mapping_matrix(retrain_data)

                    self.model.ent_embeddings = nn.Embedding(
                        self.kg.snapshots[ss_id].num_ent, dim_list[ss_id+1]).to(self.args.device).double()
                    self.model.rel_embeddings = nn.Embedding(
                        self.kg.snapshots[ss_id].num_rel, dim_list[ss_id+1]).to(self.args.device).double()
                    self.model.ent_embeddings.weight.data[:, :dim_list[ss_id]
                                                          ] = self.model.this_snapshot_ent_embeddings.data
                    self.model.ent_embeddings.weight.data[:, dim_list[ss_id]:] = self.model.mapping(
                        self.model.this_snapshot_ent_embeddings)
                    self.model.rel_embeddings.weight.data[:, :dim_list[ss_id]
                                                          ] = self.model.this_snapshot_rel_embeddings.data
                    self.model.rel_embeddings.weight.data[:, dim_list[ss_id]:] = self.model.mapping(
                        self.model.this_snapshot_rel_embeddings)


                    print('start testing again')

                    ttem_res = {}
                    ttem_res['trained_ssid'] = ss_id
                    rreses = []
                    for test_ss_id in range(ss_id+1):
                        self.args.snapshot_test = test_ss_id
                        tester_again = Tester(self.args, self.kg, self.model)
                        result = tester_again.test_again()
                        result['test_ssid'] = test_ss_id
                        rreses.append(result)
                    ttem_res['results'] = rreses


                    wwhole_mrr, wwhole_hits1, wwhole_hits3, wwhole_hits10 = self.get_report_result(
                        rreses)


                    aggregated_result = {}
                    aggregated_result['whole_mrr'] = wwhole_mrr
                    aggregated_result['whole_hits1'] = wwhole_hits1
                    aggregated_result['whole_hits3'] = wwhole_hits3
                    aggregated_result['whole_hits10'] = wwhole_hits10
                    ttem_res['aggregated_result'] = aggregated_result
                    all_test_again_results.append(ttem_res)
                    print('whole mrr:', wwhole_mrr)





                    print('get out of testing again')
            '''prepare next snapshot'''
            if self.args.snapshot < int(self.args.snapshot_num) - 1:
                if self.args.lifelong_name in ['Snapshot', 'retraining']:
                    self.reset_model(model=True)
                self.next_snapshot_setting()
                self.reset_model(optimizer=True)







        RtF = 0.0
        if len(first_learning_res) == self.args.snapshot_num and len(all_test_results) > 0:

            final_results = all_test_results[-1]['results']
            RtF_values = []
            n = self.args.snapshot_num

            for iid in range(1, n):
                if iid < len(first_learning_res) and iid < len(final_results):
                    h_ii = first_learning_res[iid]
                    h_ni = final_results[iid]['mrr']
                    denominator = h_ii + h_ni
                    if denominator > 0:
                        rtf_i = h_ni / denominator
                        RtF_values.append(rtf_i)

            if RtF_values and n > 1:
                RtF = (1 / (n - 1)) * sum(RtF_values)
        

        fwt_value = sum(FWT)/len(FWT) if len(FWT) > 0 else 0.0
        bwt_value = sum(BWT)/len(BWT) if len(BWT) > 0 else 0.0
        
        all_test_results.append(
            {'FWT': fwt_value, 'BWT': bwt_value, 'RtF': RtF})
        

        res_dir = 'res_diff_datasets'
        os.makedirs(res_dir, exist_ok=True)
        
        test_data_path = os.path.join(res_dir, self.args.dataset + \
            '-num' + str(self.args.num_samples_for_retrain))
        test_again_data_path = os.path.join(res_dir, "AGAIN" + '-' + \
            self.args.dataset + '-num' + str(self.args.num_samples_for_retrain))
        if self.args.use_difficult_samples:
            test_data_path += '-difficult-'
            test_again_data_path += '-difficult-'
        else:
            test_data_path += '-easy-'
            test_again_data_path += '-easy-'
        test_data_path += args.dataset + '.json'
        test_again_data_path += args.dataset + '.json'
        
        with open(test_data_path, 'w') as f:
            json.dump(all_test_results, f)
        with open(test_again_data_path, 'w') as f:
            json.dump(all_test_again_results, f)
        self.args.logger.info('Final Result:\n{}'.format(test_results))
        self.args.logger.info('Report Result:\n{}'.format(report_results))
        self.args.logger.info(
            'Sum_Training_Time:{}'.format(sum(training_times)))
        self.args.logger.info('Every_Training_Time:{}'.format(training_times))

        fwt_avg = sum(FWT)/len(FWT) if len(FWT) > 0 else 0.0
        bwt_avg = sum(BWT)/len(BWT) if len(BWT) > 0 else 0.0
        self.args.logger.info('Forward transfer: {}  Backward transfer: {}  Resistance to Forgetting (RtF): {}'.format(
            fwt_avg, bwt_avg, RtF))
        



        enable_unlearning = getattr(self.args, 'enable_unlearning', False)
        enable_tsf = getattr(self.args, 'enable_tsf', 'False')
        enable_tsf_check = enable_tsf == 'True' or enable_tsf is True
        

        if enable_tsf_check and hasattr(self.args, 'tsf_unlearning_results'):
            for ss_id, tsf_res in self.args.tsf_unlearning_results.items():

                found = False
                for i, res in enumerate(unlearning_results):
                    if res['snapshot_id'] == ss_id:
                        unlearning_results[i] = tsf_res
                        found = True
                        break
                if not found:
                    unlearning_results.append(tsf_res)
            self.args.logger.info(f"[TSF] Updated unlearning_results with post-Stage-2 evaluations")
        
        if (enable_unlearning or enable_tsf_check) and getattr(self.args, 'eval_unlearning', True) and len(unlearning_results) > 0:
            self.args.logger.info('\n' + '='*80)
            self.args.logger.info('遗忘评估报告 (Unlearning Evaluation Report)')
            self.args.logger.info('格式与链接预测完全一致')
            self.args.logger.info('='*80)
            

            retention_report = PrettyTable()
            retention_report.field_names = [
                'Snapshot', 'Whole_MRR_Ret', 'Whole_Hits@1_Ret', 'Whole_Hits@3_Ret', 'Whole_Hits@10_Ret'
            ]
            

            forgetting_report = PrettyTable()
            forgetting_report.field_names = [
                'Snapshot', 'Whole_MRR_F', 'Whole_Hits@1_F', 'Whole_Hits@3_F', 'Whole_Hits@10_F'
            ]
            

            for res in unlearning_results:
                retention_report.add_row([
                    res['snapshot_id'],
                    round(res.get('MRR_ret', 0.0), 3),
                    round(res.get('Hits@1_ret', 0.0), 3),
                    round(res.get('Hits@3_ret', 0.0), 3),
                    round(res.get('Hits@10_ret', 0.0), 3),
                ])
                
                forgetting_report.add_row([
                    res['snapshot_id'],
                    round(res.get('MRR_F', 0.0), 3),
                    round(res.get('Hits@1_F', 0.0), 3),
                    round(res.get('Hits@3_F', 0.0), 3),
                    round(res.get('Hits@10_F', 0.0), 3),
                ])
            
            self.args.logger.info('\n保留集报告 (Retention Report - 链接预测结果即为保留集结果):\n{}'.format(retention_report))
            self.args.logger.info('\n遗忘集报告 (Forgetting Report - 被移除的训练数据):\n{}'.format(forgetting_report))
            



            if len(unlearning_results) > 0:

                last_snapshot_res = unlearning_results[-1]
                


                if 'MRR_ret' in last_snapshot_res and last_snapshot_res['MRR_ret'] > 0:
                    MRR_r = last_snapshot_res['MRR_ret']
                else:

                    if len(all_test_results) > 0:
                        MRR_r = all_test_results[-1]['aggregated_result']['whole_mrr']
                    else:
                        MRR_r = 0.0
                
                MRR_f = last_snapshot_res.get('MRR_F', 0.0)
                


                MRR_Avg = (MRR_r + (1 - MRR_f)) / 2.0
                

                denominator = MRR_r + (1 - MRR_f)
                if denominator > 0:
                    MRR_F1 = (2.0 * MRR_r * (1 - MRR_f)) / denominator
                else:
                    MRR_F1 = 0.0
                

                综合指标_report = PrettyTable()
                综合指标_report.field_names = ['Metric', 'Value', 'Description']
                综合指标_report.add_row([
                    'MRR_r (保留集)', 
                    round(MRR_r, 4),
                    'Retained set performance'
                ])
                综合指标_report.add_row([
                    'MRR_f (遗忘集)', 
                    round(MRR_f, 4),
                    'Forget set performance'
                ])
                综合指标_report.add_row([
                    '1 - MRR_f', 
                    round(1 - MRR_f, 4),
                    'Forgetting effectiveness'
                ])
                综合指标_report.add_row([
                    'MRR_Avg', 
                    round(MRR_Avg, 4),
                    '(MRR_r + (1-MRR_f)) / 2'
                ])
                综合指标_report.add_row([
                    'MRR_F1', 
                    round(MRR_F1, 4),
                    'Harmonic mean of retention and forgetting'
                ])
                
                self.args.logger.info('\n' + '='*80)
                self.args.logger.info('综合指标报告 (Comprehensive Metrics - Based on Final Snapshot)')
                self.args.logger.info('='*80)
                self.args.logger.info(f'Final Snapshot ID: {last_snapshot_res["snapshot_id"]}')
                self.args.logger.info('\n{}'.format(综合指标_report))
                self.args.logger.info('\n说明：')
                self.args.logger.info('  MRR_Avg: 平均指标，越高越好')
                self.args.logger.info('  MRR_F1: 调和平均，同时考虑保留和遗忘效果，越高越好')
                self.args.logger.info('='*80 + '\n')
            
            self.args.logger.info('='*80 + '\n')
            

            whole_results = self.evaluate_unlearning_whole()




            enable_retrain = getattr(self.args, 'enable_retrain_baseline', False)
            if enable_retrain:
                self.args.logger.info('\n' + '='*80)
                self.args.logger.info('开始执行 RETRAIN BASELINE')
                self.args.logger.info('='*80)
                retrain_results = self.perform_retrain_baseline()
                self.args.logger.info('RETRAIN BASELINE 执行完成')
                self.args.logger.info('='*80 + '\n')
            else:
                retrain_results = None


            unlearning_report_path = os.path.join(res_dir, self.args.dataset + '-unlearning-report.json')
            with open(unlearning_report_path, 'w') as f:
                json.dump(unlearning_results, f, indent=2)
            self.args.logger.info(f'Unlearning report saved to: {unlearning_report_path}')


            if retrain_results is not None:
                retrain_report_path = os.path.join(res_dir, self.args.dataset + '-retrain-report.json')
                with open(retrain_report_path, 'w') as f:
                    json.dump(retrain_results, f, indent=2)
                self.args.logger.info(f'Retrain report saved to: {retrain_report_path}')




        try:
            overall_results = self.evaluate_overall_performance()

            overall_entry = {
                'type': 'overall_performance',
                'results': [{
                    'test_ssid': 'overall',
                    'mrr': overall_results['mrr'],
                    'hits1': overall_results['hits1'],
                    'hits3': overall_results['hits3'],
                    'hits5': overall_results['hits5'],
                    'hits10': overall_results['hits10']
                }],
                'FWT': None,
                'BWT': None,
                'RtF': None
            }
            all_test_results.append(overall_entry)


            with open(test_data_path, 'w') as f:
                json.dump(all_test_results, f)
            with open(test_again_data_path, 'w') as f:
                json.dump(all_test_again_results, f)

            self.args.logger.info('Overall Performance评估完成并保存')

        except Exception as e:
            self.args.logger.error(f'Overall Performance评估失败: {str(e)}')
            import traceback
            self.args.logger.error(traceback.format_exc())

    def get_report_result(self, results):
        '''
        Get report results of the final model: mrr, hits@1, hits@3, hits@10
        :param results: Evaluation results dict: {mrr: hits@k}
        :return: mrr, hits@1, hits@3, hits@10
        '''
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.snapshots[idx].test))
        whole_mrr = sum([mrr * num_test[i]
                        for i, mrr in enumerate(mrrs)]) / sum(num_test)
        whole_hits1 = sum([hits1 * num_test[i]
                          for i, hits1 in enumerate(hits1s)]) / sum(num_test)
        whole_hits3 = sum([hits3 * num_test[i]
                          for i, hits3 in enumerate(hits3s)]) / sum(num_test)
        whole_hits10 = sum([hits10 * num_test[i]
                           for i, hits10 in enumerate(hits10s)]) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def save_model(self, is_best=False):
        '''
        Save trained model.
        :param is_best: If True, save it as the best model.
        After training on each snapshot, we will use the best model to evaluate.
        '''
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.model.state_dict()
        checkpoint_dict['epoch_id'] = self.args.epoch
        out_tar = os.path.join(self.args.save_path, str(
            self.args.snapshot) + 'checkpoint-{}.tar'.format(self.args.epoch))
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(self.args.save_path, str(
                self.args.snapshot) + 'model_best.tar')
            shutil.copyfile(out_tar, best_path)
    
    def cleanup_snapshot_checkpoints(self, snapshot_id):
        '''
        清理指定snapshot的所有checkpoint文件，只保留best model
        :param snapshot_id: snapshot编号
        '''
        try:
            import glob
            snapshot_prefix = str(snapshot_id)

            checkpoint_pattern = os.path.join(self.args.save_path, snapshot_prefix + 'checkpoint-*.tar')
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            deleted_count = 0
            for fpath in checkpoint_files:
                try:
                    os.remove(fpath)
                    deleted_count += 1

                except Exception as e:
                    logging.warning('删除checkpoint失败 %s: %s' % (fpath, str(e)))
            
            if deleted_count > 0:
                logging.info('Snapshot %d: 已清理 %d 个checkpoint文件，仅保留best model' % (snapshot_id, deleted_count))
        except Exception as e:
            logging.warning('清理snapshot %d checkpoint时出错: %s' % (snapshot_id, str(e)))

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            logging.info('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(
                input_file, map_location="cuda:{}".format(self.args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info('=> no checkpoint found at \'{}\''.format(input_file))

    def post_snapshot_correction(self):
        """
        Perform post-snapshot correction using integrated DSP methods.
        """

        triples = list(self.kg.snapshots[self.args.snapshot].train_new)



        low_quality, mean_Q, mean_U = self.model.detect_low_quality_triples(triples)


        low_quality_sample = low_quality[:20]

        if not low_quality_sample:
            self.args.logger.info("No low-quality triples detected, skipping correction")
            return


        correction_lr = self.args.correction_lr_multiplier * float(self.args.learning_rate)
        correction_optimizer = torch.optim.Adam(self.model.parameters(), lr=correction_lr, weight_decay=self.args.l2)

        correction_steps = 0


        for x in low_quality_sample:
            anchors = self.model.build_structural_anchors(x)
            for y_w in anchors:
                loss = self.model.compute_dsp_loss_for_pair(x, y_w)
                correction_optimizer.zero_grad()
                loss.backward()
                correction_optimizer.step()
                correction_steps += 1
                if correction_steps >= 3:
                    break
            if correction_steps >= 3:
                break

        self.args.logger.info(f"Performed {correction_steps} post-snapshot correction steps")
    
    def perform_snapshot_forgetting_v2(self):
        """
        训练结束后执行遗忘操作（支持两种策略）
        
        策略1: random - 随机选择
        策略2: quality_based - 基于嵌入质量分数选择排名靠后的
        """
        if not getattr(self.args, 'enable_unlearning', False):
            return
        
        snapshot_id = getattr(self.args, 'snapshot', 0)
        train_triples = list(self.kg.snapshots[snapshot_id].train_new)
        
        if len(train_triples) == 0:
            return
        

        forget_ratio = getattr(self.args, 'forget_ratio_of_trainset', 0.01)
        forget_count = max(1, int(len(train_triples) * forget_ratio))
        

        import random
        

        forget_strategy = getattr(self.args, 'forget_strategy', 'random')
        
        if forget_strategy == 'random':

            self.args.logger.info(f"[遗忘操作] 使用随机策略")
            random.seed(getattr(self.args, 'seed', 55))
            forget_triples = random.sample(train_triples, forget_count)
            
            self.args.logger.info(f"  训练集大小: {len(train_triples)}")
            self.args.logger.info(f"  随机选取遗忘: {forget_count} 个 ({forget_ratio*100:.2f}%)")
            
        elif forget_strategy in ['worst_quality', 'quality_based']:

            self.args.logger.info(f"[遗忘操作] 使用质量导向策略")
            
            from PAAL.candidate_manager import CandidateManager
            
            gamma = getattr(self.args, 'dqa_gamma', 0.1)
            batch_size = getattr(self.args, 'quality_batch_size', 5000)
            
            self.args.logger.info(f"  训练集大小: {len(train_triples)}")
            self.args.logger.info(f"  目标遗忘数量: {forget_count} ({forget_ratio*100:.2f}%)")
            self.args.logger.info(f"  正在计算质量分数（批量大小={batch_size}）...")
            
            manager = CandidateManager()
            

            all_candidates = manager.compute_quality_scores_batch(
                model=self.model,
                triples=train_triples,
                gamma=gamma,
                batch_size=batch_size
            )
            
            self.args.logger.info(f"  质量分数计算完成，共 {len(all_candidates)} 个三元组")
            

            all_candidates.sort(key=lambda x: x['Q'])
            

            forget_triples = [c['triple'] for c in all_candidates[:forget_count]]
            

            Q_values = [c['Q'] for c in all_candidates[:forget_count]]
            self.args.logger.info(f"  遗忘集 Q 值范围: [{min(Q_values):.6f}, {max(Q_values):.6f}]")
            self.args.logger.info(f"  遗忘集 Q 值均值: {sum(Q_values)/len(Q_values):.6f}")
        
        else:
            self.args.logger.error(f"Unknown forget strategy: {forget_strategy}")
            return
        

        candidates_dir = os.path.join(self.args.save_path, 'candidates')
        os.makedirs(candidates_dir, exist_ok=True)
        snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
        os.makedirs(snapshot_dir, exist_ok=True)
        
        f_data = {
            'tag': 'F',
            'count': len(forget_triples),
            'strategy': forget_strategy,
            'triples': [
                {'h': t[0], 'r': t[1], 't': t[2], 'Q': 0.5}
                for t in forget_triples
            ]
        }
        f_path = os.path.join(snapshot_dir, 'forget_F.json')
        with open(f_path, 'w') as f:
            json.dump(f_data, f, indent=2)
        
        self.args.logger.info(f"[遗忘操作] 完成:")
        self.args.logger.info(f"  遗忘集大小: {len(forget_triples)} 个")
        self.args.logger.info(f"  策略: {forget_strategy}")
        self.args.logger.info(f"  已保存: {f_path}")
    
    def select_and_remove_forgetting_triples(self, snapshot_id):
        """
        在训练开始前，随机选取要遗忘的三元组并从训练集中移除
        
        Args:
            snapshot_id: 当前快照 ID
        """
        import random

        

        train_triples = list(self.kg.snapshots[snapshot_id].train_new)
        
        if len(train_triples) == 0:
            return
        

        forget_ratio = getattr(self.args, 'forget_ratio_of_trainset', 0.01)
        forget_count = max(1, int(len(train_triples) * forget_ratio))
        

        random.seed(getattr(self.args, 'seed', 55))
        forget_triples = random.sample(train_triples, forget_count)
        

        forget_set = set(forget_triples)
        new_train = [t for t in train_triples if t not in forget_set]
        

        self.kg.snapshots[snapshot_id].train_new = new_train
        

        candidates_dir = os.path.join(self.args.save_path, 'candidates')
        os.makedirs(candidates_dir, exist_ok=True)
        snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
        os.makedirs(snapshot_dir, exist_ok=True)
        

        f_data = {
            'tag': 'F',
            'count': len(forget_triples),
            'strategy': 'random_removal',
            'triples': [
                {'h': t[0], 'r': t[1], 't': t[2], 'Q': 0.5}
                for t in forget_triples
            ]
        }
        f_path = os.path.join(snapshot_dir, 'forget_F.json')
        with open(f_path, 'w') as f:
            json.dump(f_data, f, indent=2)
        
        self.args.logger.info(f"[简化遗忘] Snapshot {snapshot_id}:")
        self.args.logger.info(f"  原始训练集: {len(train_triples)} 个三元组")
        self.args.logger.info(f"  随机选取遗忘: {forget_count} 个 ({forget_ratio*100:.2f}%)")
        self.args.logger.info(f"  新训练集: {len(new_train)} 个三元组")
        self.args.logger.info(f"  遗忘集已保存: {f_path}")
    
    def perform_snapshot_forgetting(self):
        """
        在训练结束时执行遗忘操作（优化版：支持快速采样模式）
        """
        if not getattr(self.args, 'enable_unlearning', False):
            return
        
        snapshot_id = getattr(self.args, 'snapshot', 0)
        triples = list(self.kg.snapshots[snapshot_id].train_new)
        
        if len(triples) == 0:
            return
        

        total_train_size = len(triples)
        forget_ratio_of_trainset = getattr(self.args, 'forget_ratio_of_trainset', 0.01)
        target_forget_count = max(1, int(total_train_size * forget_ratio_of_trainset))
        
        from PAAL.candidate_manager import CandidateManager

        import random
        import time
        
        gamma = getattr(self.args, 'dqa_gamma', 0.1)
        batch_size = getattr(self.args, 'quality_batch_size', 5000)
        

        use_fast = getattr(self.args, 'use_fast_forgetting', True)
        
        if use_fast:

            sample_ratio = getattr(self.args, 'forget_sample_ratio', 5.0)
            sample_size = int(target_forget_count * sample_ratio)
            

            if total_train_size < 5000:

                sample_size = total_train_size
                mode_str = "全局排名（小数据集）"
            elif sample_size >= total_train_size:

                sample_size = total_train_size
                mode_str = "全局排名"
            else:

                mode_str = f"快速采样（{sample_ratio}倍）"
            
            sampled_triples = random.sample(triples, sample_size) if sample_size < total_train_size else triples
            
            self.args.logger.info(
                f"[PAAL Unlearning] {mode_str}："
            )
            self.args.logger.info(
                f"  训练集大小: {total_train_size}"
            )
            self.args.logger.info(
                f"  采样大小: {sample_size} ({sample_size/total_train_size*100:.1f}%)"
            )
            self.args.logger.info(
                f"  目标遗忘数量: {target_forget_count} ({forget_ratio_of_trainset*100:.2f}%)"
            )
        else:

            sampled_triples = triples
            mode_str = "完整全局排名"
            
            self.args.logger.info(
                f"[PAAL Unlearning] {mode_str}："
            )
            self.args.logger.info(
                f"  训练集大小: {total_train_size}"
            )
            self.args.logger.info(
                f"  目标遗忘数量: {target_forget_count} ({forget_ratio_of_trainset*100:.2f}%)"
            )
        
        self.args.logger.info(
            f"  正在计算质量分数（批量大小={batch_size}）..."
        )
        
        manager = CandidateManager()
        

        start_time = time.time()
        

        all_candidates = manager.compute_quality_scores_batch(
            model=self.model,
            triples=sampled_triples,
            gamma=gamma,
            batch_size=batch_size
        )
        
        elapsed_time = time.time() - start_time
        
        self.args.logger.info(
            f"  质量分数计算完成，共 {len(all_candidates)} 个三元组，耗时 {elapsed_time:.2f}秒"
        )
        
        if len(all_candidates) > 0:

            all_candidates.sort(key=lambda x: x['Q'])
            

            F_count = min(target_forget_count, len(all_candidates))
            F = all_candidates[:F_count]
            

            candidate_multiplier = getattr(self.args, 'candidate_multiplier', 2.0)
            C_size = min(len(all_candidates), int(F_count * candidate_multiplier))
            C = all_candidates[:C_size]
            

            F_Q_values = [c['Q'] for c in F]
            C_Q_values = [c['Q'] for c in C]
            all_Q_values = [c['Q'] for c in all_candidates]
            
            self.args.logger.info(
                f"  遗忘集 F: {len(F)} 个三元组"
            )
            self.args.logger.info(
                f"    Q 值范围: [{min(F_Q_values):.6f}, {max(F_Q_values):.6f}]"
            )
            self.args.logger.info(
                f"    Q 值均值: {sum(F_Q_values)/len(F_Q_values):.6f}"
            )
            self.args.logger.info(
                f"  候选集 C: {len(C)} 个三元组 ({len(C)/len(all_candidates)*100:.1f}% of sampled)"
            )
            self.args.logger.info(
                f"    Q 值范围: [{min(C_Q_values):.6f}, {max(C_Q_values):.6f}]"
            )
            self.args.logger.info(
                f"  采样集 Q 值统计:"
            )
            self.args.logger.info(
                f"    范围: [{min(all_Q_values):.6f}, {max(all_Q_values):.6f}]"
            )
            self.args.logger.info(
                f"    均值: {sum(all_Q_values)/len(all_Q_values):.6f}"
            )
            

            candidates_dir = os.path.join(self.args.save_path, 'candidates')
            os.makedirs(candidates_dir, exist_ok=True)
            
            snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
            os.makedirs(snapshot_dir, exist_ok=True)
            

            if len(F) > 0:
                f_data = {
                    'tag': 'F',
                    'count': len(F),
                    'strategy': 'global_ranking',
                    'triples': [
                        {
                            'h': c['triple'][0],
                            'r': c['triple'][1],
                            't': c['triple'][2],
                            'Q': c['Q']
                        }
                        for c in F
                    ]
                }
                f_path = os.path.join(snapshot_dir, 'forget_F.json')
                with open(f_path, 'w') as f_file:
                    json.dump(f_data, f_file, indent=2)
            

            if len(C) > 0:
                c_data = {
                    'tag': 'C',
                    'count': len(C),
                    'strategy': 'global_ranking',
                    'triples': [
                        {
                            'h': c['triple'][0],
                            'r': c['triple'][1],
                            't': c['triple'][2],
                            'Q': c['Q']
                        }
                        for c in C
                    ]
                }
                c_path = os.path.join(snapshot_dir, 'candidates_C.json')
                with open(c_path, 'w') as c_file:
                    json.dump(c_data, c_file, indent=2)
            

            r_data = {'tag': 'R', 'count': 0, 'triples': []}
            r_path = os.path.join(snapshot_dir, 'repair_R.json')
            with open(r_path, 'w') as r_file:
                json.dump(r_data, r_file, indent=2)
            
            forget_percentage = (F_count / total_train_size) * 100
            self.args.logger.info(
                f"[PAAL Unlearning] ✅ 全局排名遗忘完成："
            )
            self.args.logger.info(
                f"  训练集: {total_train_size} 个三元组"
            )
            self.args.logger.info(
                f"  遗忘集 F: {F_count} 个 ({forget_percentage:.3f}%)"
            )
            self.args.logger.info(
                f"  候选集 C: {len(C)} 个 ({len(C)/total_train_size*100:.2f}%)"
            )
            self.args.logger.info(
                f"  保存路径: {snapshot_dir}"
            )
    
    def evaluate_forgetting_persistence(self):
        """
        评估遗忘和保留的持续性（Forgetting & Retention Persistence）
        
        像链接预测那样，展示在每个 snapshot 训练完后：
        1. 模型在所有历史保留集上的表现（确保不会灾难性遗忘）
        2. 模型在所有历史遗忘集上的表现（确保不会重新学习）
        
        目标：
        - MRR_Ret 应该保持高水平（像链接预测那样）
        - MRR_F 应该保持低水平（遗忘持续性）
        """
        if not getattr(self.args, 'enable_unlearning', False):
            return
        
        self.args.logger.info('\n' + '='*80)
        self.args.logger.info('遗忘和保留持续性评估 (Forgetting & Retention Persistence)')
        self.args.logger.info('像链接预测那样，检查每个 snapshot 训练后的性能变化')
        self.args.logger.info('='*80)
        
        try:
            from PAAL import create_unlearning_evaluator, load_candidate_set

            
            candidates_dir = os.path.join(self.args.save_path, 'candidates')
            

            persistence_report = PrettyTable()
            persistence_report.field_names = [
                'Trained_SS', 'Whole_MRR_Ret', 'Whole_Hits@1_Ret', 'Whole_Hits@10_Ret', 
                'Whole_MRR_F', 'Whole_Hits@1_F', 'Whole_Hits@10_F'
            ]
            

            for trained_ss_id in range(self.args.snapshot_num):

                checkpoint_path = os.path.join(self.args.save_path, f'{trained_ss_id}model_best.tar')
                
                if not os.path.exists(checkpoint_path):
                    self.args.logger.warning(f"Checkpoint not found for snapshot {trained_ss_id}")
                    continue
                

                model_state_backup = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                

                checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.args.snapshot = trained_ss_id
                self.model.args.snapshot = trained_ss_id
            

            evaluator = create_unlearning_evaluator(self.model, self.args, self.kg)
            

            all_Ret_results = []
            all_F_results = []
            
            for ss_id in range(trained_ss_id + 1):
                try:
                    F = load_candidate_set(ss_id, 'F', out_dir=candidates_dir)
                    if F is None or len(F) == 0:
                        continue
                    

                    test_triples = list(self.kg.snapshots[ss_id].test)
                    F_set = set(item['triple'] for item in F)
                    T_ret = [t for t in test_triples if t not in F_set]
                    

                    if len(T_ret) > 0:
                        ret_metrics = evaluator.evaluate_triples(T_ret, trained_ss_id)
                        all_Ret_results.append({
                            'MRR': ret_metrics['MRR'],
                            'Hits@1': ret_metrics['Hits@1'],
                            'Hits@10': ret_metrics['Hits@10'],
                            'size': len(T_ret)
                        })
                    

                    F_triples = [item['triple'] for item in F]
                    f_metrics = evaluator.evaluate_triples(F_triples, trained_ss_id)
                    all_F_results.append({
                        'MRR': f_metrics['MRR'],
                        'Hits@1': f_metrics['Hits@1'],
                        'Hits@10': f_metrics['Hits@10'],
                        'size': len(F)
                    })
                    
                except FileNotFoundError:
                    continue
            

            if len(all_Ret_results) > 0:
                total_ret_size = sum([r['size'] for r in all_Ret_results])
                whole_mrr_ret = sum([r['MRR'] * r['size'] for r in all_Ret_results]) / total_ret_size
                whole_hits1_ret = sum([r['Hits@1'] * r['size'] for r in all_Ret_results]) / total_ret_size
                whole_hits10_ret = sum([r['Hits@10'] * r['size'] for r in all_Ret_results]) / total_ret_size
            else:
                whole_mrr_ret = whole_hits1_ret = whole_hits10_ret = 0.0
            

            if len(all_F_results) > 0:
                total_f_size = sum([r['size'] for r in all_F_results])
                whole_mrr_f = sum([r['MRR'] * r['size'] for r in all_F_results]) / total_f_size
                whole_hits1_f = sum([r['Hits@1'] * r['size'] for r in all_F_results]) / total_f_size
                whole_hits10_f = sum([r['Hits@10'] * r['size'] for r in all_F_results]) / total_f_size
            else:
                whole_mrr_f = whole_hits1_f = whole_hits10_f = 0.0
            

            persistence_report.add_row([
                trained_ss_id,
                round(whole_mrr_ret, 3),
                round(whole_hits1_ret, 3),
                round(whole_hits10_ret, 3),
                round(whole_mrr_f, 3),
                round(whole_hits1_f, 3),
                round(whole_hits10_f, 3),
            ])
            

            self.model.load_state_dict(model_state_backup)
            
            self.args.logger.info('\n保留与遗忘持续性报告 (像链接预测的 Whole MRR 一样):\n{}'.format(persistence_report))
            self.args.logger.info('\n说明:')
            self.args.logger.info('  - Trained_SS: 训练完该 snapshot 后的模型')
            self.args.logger.info('  - Whole_MRR_Ret: 该模型在所有历史保留集上的加权平均 MRR')
            self.args.logger.info('    期望: 保持高水平，避免灾难性遗忘（与链接预测相同）')
            self.args.logger.info('  - Whole_MRR_F: 该模型在所有历史遗忘集上的加权平均 MRR')
            self.args.logger.info('    期望: 保持低水平，确保遗忘知识不被记起')
            self.args.logger.info('='*80 + '\n')
            
        except Exception as e:
            self.args.logger.error(f"Error in evaluate_forgetting_persistence: {str(e)}")
            import traceback
            self.args.logger.error(traceback.format_exc())

    def evaluate_overall_performance(self):
        """
        评估Overall Performance：用最终模型在所有snapshot测试集的并集上进行评估

        这与传统的CKGE评估不同，传统的评估是分别评估每个snapshot的测试集。
        Overall Performance评估的是模型在所有历史知识上的整体推理能力。

        Returns:
            dict: Overall Performance评估结果，包含MRR和Hits@K指标
        """
        self.args.logger.info('\n' + '='*80)
        self.args.logger.info('Overall Performance评估')
        self.args.logger.info('用最终模型在所有snapshot测试集的并集上进行评估')
        self.args.logger.info('='*80)


        final_snapshot_id = self.args.snapshot_num - 1
        final_model_path = os.path.join(self.args.save_path, str(final_snapshot_id) + 'model_best.tar')
        self.load_checkpoint(final_model_path)


        class OverallTestDataset(Dataset):
            def __init__(self, args, kg):
                self.args = args
                self.kg = kg



                final_snapshot_id = self.args.snapshot_num - 1
                self.num_entities = self.kg.snapshots[final_snapshot_id].num_ent


                self.all_test_facts = []
                for ss_id in range(self.args.snapshot_num):
                    test_facts = list(self.kg.snapshots[ss_id].test)

                    for fact in test_facts:

                        fact_with_ss = {
                            'fact': fact,
                            'snapshot_id': ss_id
                        }
                        self.all_test_facts.append(fact_with_ss)

                self.args.logger.info(f'Overall Performance评估数据总量: {len(self.all_test_facts)}')
                self.args.logger.info(f'使用实体数量基准: {self.num_entities}')

            def __len__(self):
                return len(self.all_test_facts)

            def __getitem__(self, idx):
                ele = self.all_test_facts[idx]
                ss_id = ele['snapshot_id']
                fact = ele['fact']


                if len(fact) == 3:
                    s, r, o = fact
                elif len(fact) == 2:

                    s, r = fact
                    o = 0
                else:
                    raise ValueError(f"Unexpected fact length: {len(fact)}")




                y = torch.zeros(self.num_entities, dtype=torch.float32)



                all_correct_objects = set()
                

                for hist_ss_id in range(self.args.snapshot_num):
                    hist_sr2o = self.kg.snapshots[hist_ss_id].sr2o_all
                    if (s, r) in hist_sr2o:
                        for obj_id in hist_sr2o[(s, r)]:
                            all_correct_objects.add(obj_id)


                for obj_id in all_correct_objects:
                    if isinstance(obj_id, int) and obj_id < self.num_entities and obj_id != o:
                        y[obj_id] = 1.0

                return torch.LongTensor([s]), torch.LongTensor([r]), torch.LongTensor([o]), y, ss_id

            @staticmethod
            def collate_fn(data):
                s = torch.stack([_[0] for _ in data], dim=0).squeeze(-1)
                r = torch.stack([_[1] for _ in data], dim=0).squeeze(-1)
                o = torch.stack([_[2] for _ in data], dim=0).squeeze(-1)

                l = [_[3] for _ in data]
                ss_ids = torch.tensor([_[4] for _ in data], dtype=torch.long)
                return s, r, o, l, ss_ids


        class OverallBatchProcessor():
            def __init__(self, args, kg):
                self.args = args
                self.kg = kg
                self.batch_size = 100


                self.dataset = OverallTestDataset(args, kg)
                self.data_loader = DataLoader(self.dataset,
                                            shuffle=False,
                                            batch_size=self.batch_size,
                                            collate_fn=self.dataset.collate_fn,
                                            generator=torch.Generator().manual_seed(int(args.seed)),
                                            pin_memory=True)

            def process_epoch(self, model):
                model.eval()
                results = dict()
                num = 0

                '''start evaluation'''
                ranks = []
                for step, batch in enumerate(self.data_loader):
                    sub, rel, obj, label, ss_ids = batch
                    sub = sub.to(self.args.device)
                    rel = rel.to(self.args.device)
                    obj = obj.to(self.args.device)

                    num += len(sub)

                    stage = 'Test'

                    '''link prediction'''
                    pred = model.predict(sub, rel, stage=stage)


                    batch_ranks = []
                    for i in range(len(sub)):

                        pred_i = pred[i:i+1]
                        obj_i = obj[i:i+1]
                        label_i = label[i].to(self.args.device)



                        if pred_i.size(1) > self.dataset.num_entities:
                            pred_i = pred_i[:, :self.dataset.num_entities]
                        elif pred_i.size(1) < self.dataset.num_entities:

                            padding_size = self.dataset.num_entities - pred_i.size(1)
                            padding = torch.full((1, padding_size), -float('inf'),
                                               device=pred_i.device)
                            pred_i = torch.cat([pred_i, padding], dim=1)


                        b_range = torch.arange(1, device=self.args.device)
                        target_pred = pred_i[b_range, obj_i]


                        pred_i = torch.where(label_i.bool(), -torch.ones_like(pred_i) * 10000000, pred_i)
                        pred_i[b_range, obj_i] = target_pred


                        rank = self.get_rank(pred_i, obj_i)
                        batch_ranks.append(rank)

                    if batch_ranks:
                        batch_ranks = torch.cat(batch_ranks)
                        ranks.append(batch_ranks)

                '''calculate metrics'''
                if ranks:
                    ranks = torch.cat(ranks)
                    results['count'] = num
                    results['mrr'] = torch.mean(1.0 / ranks).item()
                    results['hits1'] = torch.mean((ranks <= 1).float()).item()
                    results['hits3'] = torch.mean((ranks <= 3).float()).item()
                    results['hits5'] = torch.mean((ranks <= 5).float()).item()
                    results['hits10'] = torch.mean((ranks <= 10).float()).item()
                else:
                    results['count'] = 0
                    results['mrr'] = 0.0
                    results['hits1'] = 0.0
                    results['hits3'] = 0.0
                    results['hits5'] = 0.0
                    results['hits10'] = 0.0

                return results

            @staticmethod
            def get_rank(pred, obj):
                argsort = torch.argsort(pred, dim=1, descending=True)
                rank = torch.nonzero(argsort == obj.view(-1, 1))[:, 1] + 1
                return rank.float()


        overall_processor = OverallBatchProcessor(self.args, self.kg)
        overall_results = overall_processor.process_epoch(self.model)


        self.args.logger.info('Overall Performance Results:')
        self.args.logger.info(f'MRR: {overall_results["mrr"]:.4f}')
        self.args.logger.info(f'Hits@1: {overall_results["hits1"]:.4f}')
        self.args.logger.info(f'Hits@3: {overall_results["hits3"]:.4f}')
        self.args.logger.info(f'Hits@5: {overall_results["hits5"]:.4f}')
        self.args.logger.info(f'Hits@10: {overall_results["hits10"]:.4f}')

        return overall_results

    def evaluate_unlearning_whole(self):
        """
        像链接预测那样计算 Whole 遗忘指标：
        用最终模型分别评估每个 snapshot 的保留集和遗忘集，然后加权平均
        
        Returns:
            dict: Whole 遗忘评估结果
        """

        enable_unlearning = getattr(self.args, 'enable_unlearning', False)
        enable_tsf = getattr(self.args, 'enable_tsf', 'False')
        enable_tsf_check = enable_tsf == 'True' or enable_tsf is True
        
        if not enable_unlearning and not enable_tsf_check:
            return None
            
        final_snapshot_id = self.args.snapshot_num - 1
        
        self.args.logger.info(f"\n{'='*80}")
        self.args.logger.info(f"Computing Whole Unlearning Metrics with Final Model (Snapshot {final_snapshot_id})")
        self.args.logger.info(f"像链接预测那样：分别评估每个 snapshot 再加权平均")
        self.args.logger.info(f"{'='*80}")
        
        try:
            from PAAL import create_unlearning_evaluator, load_candidate_set
            

            evaluator = create_unlearning_evaluator(self.model, self.args, self.kg)
            

            ret_results = []
            f_results = []
            

            all_forget_triples = []
            all_retain_triples = []
            
            candidates_dir = os.path.join(self.args.save_path, 'candidates')
            
            for ss_id in range(self.args.snapshot_num):

                try:
                    F = load_candidate_set(ss_id, 'F', out_dir=candidates_dir)
                    if F is None or len(F) == 0:
                        self.args.logger.info(f"  Snapshot {ss_id}: No forget set, skipping")
                        continue
                    

                    test_triples = list(self.kg.snapshots[ss_id].test)
                    

                    F_set = set(item['triple'] for item in F)
                    T_ret = [t for t in test_triples if t not in F_set]
                    

                    if len(T_ret) > 0:
                        ret_metrics = evaluator.evaluate_triples(T_ret, final_snapshot_id)
                        ret_results.append({
                            'MRR': ret_metrics['MRR'],
                            'Hits@1': ret_metrics['Hits@1'],
                            'Hits@3': ret_metrics['Hits@3'],
                            'Hits@10': ret_metrics['Hits@10'],
                            'size': len(T_ret)
                        })
                    
                    if len(F) > 0:
                        F_triples = [item['triple'] for item in F]
                        f_metrics = evaluator.evaluate_triples(F_triples, final_snapshot_id)
                        f_results.append({
                            'MRR': f_metrics['MRR'],
                            'Hits@1': f_metrics['Hits@1'],
                            'Hits@3': f_metrics['Hits@3'],
                            'Hits@10': f_metrics['Hits@10'],
                            'size': len(F)
                        })

                        all_forget_triples.extend(F_triples)


                        if len(T_ret) > 0:
                            all_retain_triples.extend(T_ret)
                    
                    self.args.logger.info(
                        f"  Snapshot {ss_id}: |T_ret|={len(T_ret)}, MRR_ret={ret_metrics['MRR']:.4f} | "
                        f"|F|={len(F)}, MRR_F={f_metrics['MRR']:.4f}"
                    )
                    
                except FileNotFoundError:
                    self.args.logger.info(f"  Snapshot {ss_id}: No candidate sets found, skipping")
                    continue
            
            if len(ret_results) == 0 and len(f_results) == 0:
                self.args.logger.warning("No retained or forgotten triples found across all snapshots")
                return None
            

            if len(ret_results) > 0:
                total_ret_size = sum([r['size'] for r in ret_results])
                whole_mrr_ret = sum([r['MRR'] * r['size'] for r in ret_results]) / total_ret_size
                whole_hits1_ret = sum([r['Hits@1'] * r['size'] for r in ret_results]) / total_ret_size
                whole_hits3_ret = sum([r['Hits@3'] * r['size'] for r in ret_results]) / total_ret_size
                whole_hits10_ret = sum([r['Hits@10'] * r['size'] for r in ret_results]) / total_ret_size
            else:
                total_ret_size = 0
                whole_mrr_ret = whole_hits1_ret = whole_hits3_ret = whole_hits10_ret = 0.0
            
            if len(f_results) > 0:
                total_f_size = sum([r['size'] for r in f_results])
                whole_mrr_f = sum([r['MRR'] * r['size'] for r in f_results]) / total_f_size
                whole_hits1_f = sum([r['Hits@1'] * r['size'] for r in f_results]) / total_f_size
                whole_hits3_f = sum([r['Hits@3'] * r['size'] for r in f_results]) / total_f_size
                whole_hits10_f = sum([r['Hits@10'] * r['size'] for r in f_results]) / total_f_size
            else:
                total_f_size = 0
                whole_mrr_f = whole_hits1_f = whole_hits3_f = whole_hits10_f = 0.0
            



            overall_forget_metrics = None
            if len(all_forget_triples) > 0:

                all_forget_triples_unique = list(set(all_forget_triples))
                self.args.logger.info(f"\n评估总体遗忘集（所有snapshot合并）:")
                self.args.logger.info(f"  总遗忘三元组数（去重前）: {len(all_forget_triples)}")
                self.args.logger.info(f"  总遗忘三元组数（去重后）: {len(all_forget_triples_unique)}")
                


                overall_forget_metrics = evaluator.evaluate_triples(all_forget_triples_unique, final_snapshot_id)
                

                overall_forget_table = PrettyTable()
                overall_forget_table.field_names = ['Metric', 'Value', 'Description']
                overall_forget_table.add_row([
                    'Overall Forget Set Size',
                    len(all_forget_triples_unique),
                    'Total unique forgotten triples (all snapshots combined)'
                ])
                overall_forget_table.add_row([
                    'Overall MRR_F',
                    round(overall_forget_metrics['MRR'], 4),
                    'Final model performance on combined forget set'
                ])
                overall_forget_table.add_row([
                    'Overall Hits@1_F',
                    round(overall_forget_metrics['Hits@1'], 4),
                    'Final model Hits@1 on combined forget set'
                ])
                overall_forget_table.add_row([
                    'Overall Hits@3_F',
                    round(overall_forget_metrics['Hits@3'], 4),
                    'Final model Hits@3 on combined forget set'
                ])
                overall_forget_table.add_row([
                    'Overall Hits@10_F',
                    round(overall_forget_metrics['Hits@10'], 4),
                    'Final model Hits@10 on combined forget set'
                ])
                self.args.logger.info('\n总体遗忘集报告 (Overall Forget Set Report - 所有snapshot的遗忘集合并):\n{}'.format(overall_forget_table))




            overall_retain_metrics = None
            if len(all_retain_triples) > 0:

                all_retain_triples_unique = list(set(all_retain_triples))
                self.args.logger.info(f"\n评估总体保留集（所有snapshot合并）:")
                self.args.logger.info(f"  总保留三元组数（去重前）: {len(all_retain_triples)}")
                self.args.logger.info(f"  总保留三元组数（去重后）: {len(all_retain_triples_unique)}")



                overall_retain_metrics = evaluator.evaluate_triples(all_retain_triples_unique, final_snapshot_id)


                overall_retain_table = PrettyTable()
                overall_retain_table.field_names = ['Metric', 'Value', 'Description']
                overall_retain_table.add_row([
                    'Overall Retain Set Size',
                    len(all_retain_triples_unique),
                    'Total unique retained triples (all snapshots combined)'
                ])
                overall_retain_table.add_row([
                    'Overall MRR_Ret',
                    round(overall_retain_metrics['MRR'], 4),
                    'Final model performance on combined retain set'
                ])
                overall_retain_table.add_row([
                    'Overall Hits@1_Ret',
                    round(overall_retain_metrics['Hits@1'], 4),
                    'Final model Hits@1 on combined retain set'
                ])
                overall_retain_table.add_row([
                    'Overall Hits@3_Ret',
                    round(overall_retain_metrics['Hits@3'], 4),
                    'Final model Hits@3 on combined retain set'
                ])
                overall_retain_table.add_row([
                    'Overall Hits@10_Ret',
                    round(overall_retain_metrics['Hits@10'], 4),
                    'Final model Hits@10 on combined retain set'
                ])
                self.args.logger.info('\n总体保留集报告 (Overall Retain Set Report - 所有snapshot的保留集合并):\n{}'.format(overall_retain_table))



            

            retention_report = PrettyTable()
            retention_report.field_names = [
                'Snapshot', 'Whole_MRR_Ret', 'Whole_Hits@1_Ret', 'Whole_Hits@3_Ret', 'Whole_Hits@10_Ret'
            ]
            

            forgetting_report = PrettyTable()
            forgetting_report.field_names = [
                'Snapshot', 'Whole_MRR_F', 'Whole_Hits@1_F', 'Whole_Hits@3_F', 'Whole_Hits@10_F'
            ]
            

            for ss_id in range(self.args.snapshot_num):

                F_triples = []
                

                if getattr(self, '_enable_tsf_check', False):

                    F_pre = getattr(self.model, 'F_pre', set())
                    F_post = getattr(self.model, 'F_post', set())
                    F_combined = list(F_pre | F_post)
                    if len(F_combined) > 0:
                        F_triples = F_combined
                

                if len(F_triples) == 0:
                    try:
                        F = load_candidate_set(ss_id, 'F', out_dir=candidates_dir)
                        if F is not None and len(F) > 0:
                            F_triples = [item['triple'] for item in F]
                    except FileNotFoundError:
                        pass
                
                if len(F_triples) == 0:
                    continue
                

                test_triples = list(self.kg.snapshots[ss_id].test)
                

                F_set = set(F_triples)
                T_ret = [t for t in test_triples if t not in F_set]
                

                if len(T_ret) > 0:
                    ret_metrics = evaluator.evaluate_triples(T_ret, final_snapshot_id)
                    retention_report.add_row([
                        ss_id,
                        round(ret_metrics['MRR'], 3),
                        round(ret_metrics['Hits@1'], 3),
                        round(ret_metrics['Hits@3'], 3),
                        round(ret_metrics['Hits@10'], 3),
                    ])
                
                if len(F_triples) > 0:
                    f_metrics = evaluator.evaluate_triples(F_triples, final_snapshot_id)
                    forgetting_report.add_row([
                        ss_id,
                        round(f_metrics['MRR'], 3),
                        round(f_metrics['Hits@1'], 3),
                        round(f_metrics['Hits@3'], 3),
                        round(f_metrics['Hits@10'], 3),
                    ])
            

            if whole_mrr_ret is not None and whole_mrr_f is not None:
                MRR_Avg = (whole_mrr_ret + (1 - whole_mrr_f)) / 2.0
                denominator = whole_mrr_ret + (1 - whole_mrr_f)
                MRR_F1 = (2.0 * whole_mrr_ret * (1 - whole_mrr_f)) / denominator if denominator > 0 else 0.0
            else:
                MRR_Avg = 0.0
                MRR_F1 = 0.0


            self.args.logger.info('\n保留集报告 (Retention Report - 链接预测结果即为保留集结果):\n{}'.format(retention_report))
            self.args.logger.info('\n遗忘集报告 (Forgetting Report - 被移除的训练数据):\n{}'.format(forgetting_report))
            
            self.args.logger.info(f"\n加权平均结果:")
            self.args.logger.info(f"  Whole T_ret (n={total_ret_size}): MRR={whole_mrr_ret:.4f}, Hits@1={whole_hits1_ret:.4f}, Hits@10={whole_hits10_ret:.4f}")
            self.args.logger.info(f"  Whole F (n={total_f_size}): MRR={whole_mrr_f:.4f}, Hits@1={whole_hits1_f:.4f}, Hits@10={whole_hits10_f:.4f}")
            
            results = {
                'MRR_ret': whole_mrr_ret,
                'Hits@1_ret': whole_hits1_ret,
                'Hits@3_ret': whole_hits3_ret,
                'Hits@10_ret': whole_hits10_ret,
                'MRR_F': whole_mrr_f,
                'Hits@1_F': whole_hits1_f,
                'Hits@3_F': whole_hits3_f,
                'Hits@10_F': whole_hits10_f,
                'T_ret_total': total_ret_size,
                'F_total': total_f_size,
                'MRR_Avg': MRR_Avg,
                'MRR_F1': MRR_F1,

                'overall_retain': overall_retain_metrics,
                'overall_retain_size': len(all_retain_triples_unique) if 'all_retain_triples_unique' in locals() else 0,

                'overall_forget': overall_forget_metrics,
                'overall_forget_size': len(all_forget_triples_unique) if 'all_forget_triples_unique' in locals() else 0,
            }
            
            self.args.logger.info(f"Whole unlearning evaluation completed")
            return results
            
        except Exception as e:
            self.args.logger.error(f"Error in evaluate_unlearning_whole: {str(e)}")
            import traceback
            self.args.logger.error(traceback.format_exc())
            return None
    
    def evaluate_unlearning(self, snapshot_id):
        """
        遗忘操作后的评估（像链接预测那样详细展示）
        
        在遗忘操作后，评估：
        1. 保留集 (train - F) - 参与训练的数据
        2. 遗忘集 (F) - 标记为遗忘的数据
        
        Args:
            snapshot_id: 当前快照 ID
        """

        enable_unlearning = getattr(self.args, 'enable_unlearning', False)
        enable_tsf = getattr(self.args, 'enable_tsf', 'False')
        enable_tsf_check = enable_tsf == 'True' or enable_tsf is True
        
        if not enable_unlearning and not enable_tsf_check:
            return None
        
        if not getattr(self.args, 'eval_unlearning', True):
            return None
        
        self.args.logger.info(f"\n{'='*80}")
        self.args.logger.info(f"遗忘操作后评估 (Snapshot {snapshot_id})")
        self.args.logger.info(f"像链接预测那样：评估所有历史 snapshot 的保留集和遗忘集")
        self.args.logger.info(f"{'='*80}")
        
        try:
            from PAAL import create_unlearning_evaluator, load_candidate_set

            

            evaluator = create_unlearning_evaluator(self.model, self.args, self.kg)
            candidates_dir = os.path.join(self.args.save_path, 'candidates')
            

            F = None
            F_triples = []
            

            if enable_tsf_check:
                F_pre = getattr(self.model, 'F_pre', set())
                F_post = getattr(self.model, 'F_post', set())
                F_combined = list(F_pre | F_post)
                if len(F_combined) > 0:
                    F_triples = F_combined
                    self.args.logger.info(f"Using TSF forget set: |F_pre|={len(F_pre)}, |F_post|={len(F_post)}, |F_combined|={len(F_combined)}")
            

            if len(F_triples) == 0:
                try:
                    F = load_candidate_set(snapshot_id, 'F', out_dir=candidates_dir)
                    if F is not None and len(F) > 0:
                        F_triples = [item['triple'] for item in F]
                except FileNotFoundError:
                    pass
            
            if len(F_triples) == 0:
                self.args.logger.info(f"No forget set found for snapshot {snapshot_id} (neither TSF nor file)")
                return None
            

            train_triples = list(self.kg.snapshots[snapshot_id].train_new)
            

            F_set = set(F_triples)
            T_ret = [t for t in train_triples if t not in F_set]
            
            self.args.logger.info(f"\n评估 Snapshot {snapshot_id}:")
            self.args.logger.info(f"  训练集大小: {len(train_triples)}")
            self.args.logger.info(f"  遗忘集大小: {len(F_triples)}")
            self.args.logger.info(f"  保留集大小: {len(T_ret)}")
            

            if len(T_ret) > 0:
                ret_metrics = evaluator.evaluate_triples(T_ret, snapshot_id)
            else:
                ret_metrics = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}
            

            f_metrics = evaluator.evaluate_triples(F_triples, snapshot_id)
            

            snapshot_report = PrettyTable()
            snapshot_report.field_names = [
                'Snapshot', 'Set', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10'
            ]
            snapshot_report.add_row([
                snapshot_id,
                'Retention',
                round(ret_metrics['MRR'], 3),
                round(ret_metrics['Hits@1'], 3),
                round(ret_metrics['Hits@3'], 3),
                round(ret_metrics['Hits@10'], 3),
            ])
            snapshot_report.add_row([
                snapshot_id,
                'Forgetting',
                round(f_metrics['MRR'], 3),
                round(f_metrics['Hits@1'], 3),
                round(f_metrics['Hits@3'], 3),
                round(f_metrics['Hits@10'], 3),
            ])
            
            self.args.logger.info(f"\nSnapshot {snapshot_id} 遗忘评估结果:")
            self.args.logger.info(f"{snapshot_report}")
            self.args.logger.info(f"  保留集: MRR={ret_metrics['MRR']:.4f}, Hits@1={ret_metrics['Hits@1']:.4f}, Hits@3={ret_metrics['Hits@3']:.4f}, Hits@10={ret_metrics['Hits@10']:.4f}")
            self.args.logger.info(f"  遗忘集: MRR={f_metrics['MRR']:.4f}, Hits@1={f_metrics['Hits@1']:.4f}, Hits@3={f_metrics['Hits@3']:.4f}, Hits@10={f_metrics['Hits@10']:.4f}")
            

            retained_results = {
                'MRR_ret': ret_metrics['MRR'],
                'Hits@1_ret': ret_metrics['Hits@1'],
                'Hits@3_ret': ret_metrics['Hits@3'],
                'Hits@10_ret': ret_metrics['Hits@10'],
                'T_ret_size': len(T_ret),
            }
            
            forgetting_results = {
                'MRR_F': f_metrics['MRR'],
                'Hits@1_F': f_metrics['Hits@1'],
                'Hits@3_F': f_metrics['Hits@3'],
                'Hits@10_F': f_metrics['Hits@10'],
                'F_size': len(F_triples),
            }
            
            return {
                'retained': retained_results,
                'forgetting': forgetting_results,
            }
            
        except Exception as e:
            self.args.logger.error(f"Error in evaluate_unlearning: {str(e)}")
            import traceback
            self.args.logger.error(traceback.format_exc())
            return None

    def perform_retrain_baseline(self):
        """
        执行RETRAIN BASELINE：收集所有遗忘集，在去掉遗忘集的训练集上重训练
        """
        try:
            self.args.logger.info("[RETRAIN BASELINE] 开始收集所有遗忘集...")


            all_forget_triples = set()
            all_train_triples = set()

            candidates_dir = os.path.join(self.args.save_path, 'candidates')

            for snapshot_id in range(self.args.snapshot_num):

                if snapshot_id < len(self.kg.snapshots):
                    snapshot_train = list(self.kg.snapshots[snapshot_id].train_new)
                    all_train_triples.update(snapshot_train)


                snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
                forget_file = os.path.join(snapshot_dir, 'forget_F.json')

                if os.path.exists(forget_file):
                    try:
                        with open(forget_file, 'r') as f:
                            forget_data = json.load(f)
                        if forget_data and 'triples' in forget_data and len(forget_data['triples']) > 0:

                            forget_triples = [(item['h'], item['r'], item['t']) for item in forget_data['triples']]
                            all_forget_triples.update(forget_triples)
                            self.args.logger.info(f"[RETRAIN BASELINE] Snapshot {snapshot_id}: 遗忘集大小 {len(forget_triples)}")
                    except Exception as e:
                        self.args.logger.warning(f"[RETRAIN BASELINE] 加载snapshot {snapshot_id}遗忘集失败: {e}")

                        try:
                            with open(forget_file, 'r') as f:
                                content = f.read()[:200]
                                self.args.logger.warning(f"[RETRAIN BASELINE] 文件内容预览: {content}")
                        except:
                            pass

            self.args.logger.info(f"[RETRAIN BASELINE] 总训练集大小: {len(all_train_triples)}")
            self.args.logger.info(f"[RETRAIN BASELINE] 总遗忘集大小: {len(all_forget_triples)}")


            retrain_all_triples = list(all_train_triples - all_forget_triples)
            self.args.logger.info(f"[RETRAIN BASELINE] Retrain总数据集大小: {len(retrain_all_triples)}")

            if len(retrain_all_triples) == 0:
                self.args.logger.error("[RETRAIN BASELINE] Retrain训练集为空，无法进行重训练")
                return None


            import random
            random.seed(42)


            retrain_shuffled = retrain_all_triples.copy()
            random.shuffle(retrain_shuffled)


            val_ratio = 0.1
            val_size = max(1, int(len(retrain_shuffled) * val_ratio))
            retrain_val_triples = retrain_shuffled[:val_size]
            retrain_train_triples = retrain_shuffled[val_size:]

            self.args.logger.info(f"[RETRAIN BASELINE] Retrain训练集大小: {len(retrain_train_triples)}")
            self.args.logger.info(f"[RETRAIN BASELINE] Retrain验证集大小: {len(retrain_val_triples)}")



            from src.data_load.KnowledgeGraph import KnowledgeGraph


            original_snapshots = self.kg.snapshots
            original_snapshot_num = self.args.snapshot_num


            retrain_kg = KnowledgeGraph(self.args)


            class MockSnapshot:
                def __init__(self, train_triples, val_triples, num_ent, num_rel):

                    self.train_new = train_triples
                    self.train_all = train_triples
                    self.train = train_triples


                    self.valid = val_triples
                    self.valid_all = val_triples


                    self.test = []
                    self.test_all = []


                    self.num_ent = num_ent
                    self.num_rel = num_rel


                    self.edge_s = []
                    self.edge_r = []
                    self.edge_o = []
                    self.sr2o_all = {}
                    self.edge_index = None
                    self.edge_type = None
                    self.new_entities = []



            retrain_kg = self.kg


            all_entities = set()
            all_relations = set()
            sr2o_all = {}


            for h, r, t in retrain_train_triples:
                all_entities.add(h)
                all_entities.add(t)
                all_relations.add(r)


                if (h, r) not in sr2o_all:
                    sr2o_all[(h, r)] = set()
                sr2o_all[(h, r)].add(t)


                inv_r = r + 1
                if (t, inv_r) not in sr2o_all:
                    sr2o_all[(t, inv_r)] = set()
                sr2o_all[(t, inv_r)].add(h)


            for h, r, t in retrain_val_triples:
                all_entities.add(h)
                all_entities.add(t)
                all_relations.add(r)


                if (h, r) not in sr2o_all:
                    sr2o_all[(h, r)] = set()
                sr2o_all[(h, r)].add(t)


                inv_r = r + 1
                if (t, inv_r) not in sr2o_all:
                    sr2o_all[(t, inv_r)] = set()
                sr2o_all[(t, inv_r)].add(h)

            retrain_snapshot = MockSnapshot(
                retrain_train_triples,
                retrain_val_triples,
                retrain_kg.num_ent,
                retrain_kg.num_rel
            )


            retrain_snapshot.sr2o_all = sr2o_all


            retrain_kg.snapshots = [retrain_snapshot] * original_snapshot_num



            self.args.logger.info("[RETRAIN BASELINE] 创建随机初始化的retrain模型...")


            original_model = self.model
            original_optimizer = self.optimizer


            original_lifelong_name = self.args.lifelong_name
            original_enable_tsf = getattr(self.args, 'enable_tsf', 'False')
            original_snapshot_num = self.args.snapshot_num
            original_emb_dim = self.args.emb_dim

            self.args.lifelong_name = 'retraining'
            self.args.snapshot = 0
            self.args.snapshot_num = 1
            self.args.enable_tsf = 'False'


            self.args.emb_dim = original_emb_dim


            retrain_model, retrain_optimizer = self.create_model()


            original_kg = self.kg
            self.kg = retrain_kg
            self.model = retrain_model
            self.optimizer = retrain_optimizer


            self.args.logger.info("[RETRAIN BASELINE] 开始训练retrain模型...")


            original_epoch_num = self.args.epoch_num
            original_patience = self.args.patience



            self.args.epoch_num = max(100, self.args.epoch_num)
            self.args.patience = 10


            start_time = time.time()
            self.args.logger.info(f"[RETRAIN BASELINE] 使用 {len(retrain_train_triples)} 个三元组进行训练")


            training_time = self.train()

            end_time = time.time()
            total_training_time = end_time - start_time

            self.args.logger.info(f"[RETRAIN BASELINE] Retrain训练完成，总耗时: {total_training_time:.2f}秒")

            self.args.logger.info("[RETRAIN BASELINE] 开始评估retrain模型...")


            self.kg = original_kg
            self.args.snapshot_num = original_snapshot_num


            self.model = retrain_model


            from PAAL import create_unlearning_evaluator

            test_results = []
            test_lengths = []


            for snapshot_id in range(original_snapshot_num):
                if snapshot_id < len(original_snapshots) and hasattr(original_snapshots[snapshot_id], 'test'):
                    test_triples = original_snapshots[snapshot_id].test
                    if test_triples:
                        try:
                            evaluator = create_unlearning_evaluator(retrain_model, self.args, self.kg)
                            snapshot_test_results = evaluator.evaluate_triples(test_triples, snapshot_id)
                            test_results.append({
                                'MRR': snapshot_test_results.get('MRR', 0.0),
                                'Hits@1': snapshot_test_results.get('Hits@1', 0.0),
                                'Hits@3': snapshot_test_results.get('Hits@3', 0.0),
                                'Hits@10': snapshot_test_results.get('Hits@10', 0.0)
                            })
                            test_lengths.append(len(test_triples))
                        except Exception as e:
                            self.args.logger.warning(f"评估snapshot {snapshot_id}测试集失败: {e}")
                            test_results.append({'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0})
                            test_lengths.append(0)


            if test_lengths and sum(test_lengths) > 0:
                total_test_length = sum(test_lengths)
                test_metrics = {
                    'MRR': sum(r['MRR'] * test_lengths[i] for i, r in enumerate(test_results)) / total_test_length,
                    'Hits@1': sum(r['Hits@1'] * test_lengths[i] for i, r in enumerate(test_results)) / total_test_length,
                    'Hits@3': sum(r['Hits@3'] * test_lengths[i] for i, r in enumerate(test_results)) / total_test_length,
                    'Hits@10': sum(r['Hits@10'] * test_lengths[i] for i, r in enumerate(test_results)) / total_test_length
                }
            else:
                test_metrics = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}


            all_forget_triples = []
            all_retain_triples = []


            for snapshot_id in range(original_snapshot_num):

                candidates_dir = os.path.join(self.args.save_path, 'candidates')
                snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')
                forget_file = os.path.join(snapshot_dir, 'forget_F.json')

                if os.path.exists(forget_file):
                    try:
                        with open(forget_file, 'r') as f:
                            forget_data = json.load(f)
                        forget_triples = [(item['h'], item['r'], item['t']) for item in forget_data['triples']]
                        all_forget_triples.extend(forget_triples)
                    except Exception as e:
                        self.args.logger.warning(f"加载snapshot {snapshot_id}遗忘集失败: {e}")


                if snapshot_id < len(original_snapshots):
                    train_triples = set(original_snapshots[snapshot_id].train_new)
                    forget_triples_set = set((item['h'], item['r'], item['t']) for item in forget_data['triples']) if os.path.exists(forget_file) and 'forget_data' in locals() else set()
                    retain_triples = list(train_triples - forget_triples_set)
                    all_retain_triples.extend(retain_triples)


            forget_metrics = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}
            if all_forget_triples:
                try:
                    evaluator = create_unlearning_evaluator(retrain_model, self.args, self.kg)
                    forget_results = evaluator.evaluate_triples(all_forget_triples, 0)
                    forget_metrics = {
                        'MRR': forget_results.get('MRR', 0.0),
                        'Hits@1': forget_results.get('Hits@1', 0.0),
                        'Hits@3': forget_results.get('Hits@3', 0.0),
                        'Hits@10': forget_results.get('Hits@10', 0.0)
                    }
                except Exception as e:
                    self.args.logger.warning(f"评估遗忘集失败: {e}")


            retain_metrics = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}
            if all_retain_triples:
                try:
                    evaluator = create_unlearning_evaluator(retrain_model, self.args, self.kg)
                    retain_results = evaluator.evaluate_triples(all_retain_triples, 0)
                    retain_metrics = {
                        'MRR': retain_results.get('MRR', 0.0),
                        'Hits@1': retain_results.get('Hits@1', 0.0),
                        'Hits@3': retain_results.get('Hits@3', 0.0),
                        'Hits@10': retain_results.get('Hits@10', 0.0)
                    }
                except Exception as e:
                    self.args.logger.warning(f"评估保留集失败: {e}")


            retrain_results = [{
                'Test_MRR': test_metrics['MRR'],
                'Test_Hits@1': test_metrics['Hits@1'],
                'Test_Hits@3': test_metrics['Hits@3'],
                'Test_Hits@10': test_metrics['Hits@10'],
                'Retain_MRR': retain_metrics['MRR'],
                'Retain_Hits@1': retain_metrics['Hits@1'],
                'Retain_Hits@3': retain_metrics['Hits@3'],
                'Retain_Hits@10': retain_metrics['Hits@10'],
                'Forget_MRR': forget_metrics['MRR'],
                'Forget_Hits@1': forget_metrics['Hits@1'],
                'Forget_Hits@3': forget_metrics['Hits@3'],
                'Forget_Hits@10': forget_metrics['Hits@10'],
            }]


            self.args.logger.info("[RETRAIN BASELINE] 生成retrain评估报告...")


            from prettytable import PrettyTable


            summary_report = PrettyTable()
            summary_report.field_names = [
                'Type', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10'
            ]

            result = retrain_results[0]
            summary_report.add_row([
                'Test',
                round(result['Test_MRR'], 3),
                round(result['Test_Hits@1'], 3),
                round(result['Test_Hits@3'], 3),
                round(result['Test_Hits@10'], 3),
            ])
            summary_report.add_row([
                'Retain',
                round(result['Retain_MRR'], 3),
                round(result['Retain_Hits@1'], 3),
                round(result['Retain_Hits@3'], 3),
                round(result['Retain_Hits@10'], 3),
            ])
            summary_report.add_row([
                'Forget',
                round(result['Forget_MRR'], 3),
                round(result['Forget_Hits@1'], 3),
                round(result['Forget_Hits@3'], 3),
                round(result['Forget_Hits@10'], 3),
            ])


            result = retrain_results[0]
            whole_test_mrr = result['Test_MRR']
            whole_test_hits1 = result['Test_Hits@1']
            whole_test_hits3 = result['Test_Hits@3']
            whole_test_hits10 = result['Test_Hits@10']

            whole_retain_mrr = result['Retain_MRR']
            whole_retain_hits1 = result['Retain_Hits@1']
            whole_retain_hits3 = result['Retain_Hits@3']
            whole_retain_hits10 = result['Retain_Hits@10']

            whole_forget_mrr = result['Forget_MRR']
            whole_forget_hits1 = result['Forget_Hits@1']
            whole_forget_hits3 = result['Forget_Hits@3']
            whole_forget_hits10 = result['Forget_Hits@10']


            self.args.logger.info('\n' + '='*120)
            self.args.logger.info('RETRAIN BASELINE 评估报告')
            self.args.logger.info('='*120)
            self.args.logger.info('\nRetrain模型评估结果:')
            self.args.logger.info(f'{summary_report}')


            whole_report = PrettyTable()
            whole_report.field_names = [
                'Type', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10'
            ]

            whole_report.add_row([
                'Test', round(whole_test_mrr, 3), round(whole_test_hits1, 3),
                round(whole_test_hits3, 3), round(whole_test_hits10, 3)
            ])
            whole_report.add_row([
                'Retain', round(whole_retain_mrr, 3), round(whole_retain_hits1, 3),
                round(whole_retain_hits3, 3), round(whole_retain_hits10, 3)
            ])
            whole_report.add_row([
                'Forget', round(whole_forget_mrr, 3), round(whole_forget_hits1, 3),
                round(whole_forget_hits3, 3), round(whole_forget_hits10, 3)
            ])

            self.args.logger.info('\n整体结果汇总:')
            self.args.logger.info(f'{whole_report}')
            self.args.logger.info(f'\n训练时间: {total_training_time:.2f}秒')
            self.args.logger.info(f'训练三元组数: {len(retrain_train_triples)}')
            self.args.logger.info(f'遗忘三元组数: {len(all_forget_triples)}')
            self.args.logger.info('='*120 + '\n')


            self.model = original_model
            self.optimizer = original_optimizer
            self.kg = original_kg
            self.args.lifelong_name = original_lifelong_name
            self.args.enable_tsf = original_enable_tsf
            self.args.snapshot_num = original_snapshot_num
            self.args.emb_dim = original_emb_dim
            self.args.epoch_num = original_epoch_num
            self.args.patience = original_patience


            return {
                'retrain_results': retrain_results,
                'whole_metrics': {
                    'test': {
                        'MRR': whole_test_mrr,
                        'Hits@1': whole_test_hits1,
                        'Hits@3': whole_test_hits3,
                        'Hits@10': whole_test_hits10
                    },
                    'retain': {
                        'MRR': whole_retain_mrr,
                        'Hits@1': whole_retain_hits1,
                        'Hits@3': whole_retain_hits3,
                        'Hits@10': whole_retain_hits10
                    },
                    'forget': {
                        'MRR': whole_forget_mrr,
                        'Hits@1': whole_forget_hits1,
                        'Hits@3': whole_forget_hits3,
                        'Hits@10': whole_forget_hits10
                    }
                },
                'training_time': total_training_time,
                'train_triples_count': len(retrain_train_triples),
                'forget_triples_count': len(all_forget_triples)
            }

        except Exception as e:
            self.args.logger.error(f"[RETRAIN BASELINE] 执行失败: {str(e)}")
            import traceback
            self.args.logger.error(traceback.format_exc())
            return None

    def evaluate_triples_on_model(self, snapshot_id, triple_type, model):
        """
        使用指定模型评估指定类型的三元组
        """
        try:
            candidates_dir = os.path.join(self.args.save_path, 'candidates')
            snapshot_dir = os.path.join(candidates_dir, f'snapshot_{snapshot_id:03d}')

            if triple_type == 'forgetting':
                triples_file = os.path.join(snapshot_dir, 'forget_F.json')
            elif triple_type == 'retaining':
                triples_file = os.path.join(snapshot_dir, 'retain_R.json')
            else:
                return None

            if not os.path.exists(triples_file):
                return None


            with open(triples_file, 'r') as f:
                triples_data = json.load(f)

            if not triples_data:
                return None


            if isinstance(triples_data, dict) and 'triples' in triples_data:

                triples = [(item['h'], item['r'], item['t']) for item in triples_data['triples']]
            elif isinstance(triples_data, list):

                triples = [(item['triple'] if 'triple' in item else item) for item in triples_data]
            else:
                self.args.logger.warning(f"未知的JSON格式: {type(triples_data)}")
                return None


            from PAAL import create_unlearning_evaluator
            evaluator = create_unlearning_evaluator(
                snapshot_id, model, self.kg, self.args
            )


            metrics = evaluator.evaluate_triples(triples, snapshot_id)

            return {
                'MRR': metrics.get('mrr', 0.0),
                'Hits@1': metrics.get('hits@1', 0.0),
                'Hits@3': metrics.get('hits@3', 0.0),
                'Hits@10': metrics.get('hits@10', 0.0)
            }

        except Exception as e:
            self.args.logger.warning(f"评估 {triple_type} 三元组失败: {e}")
            return None



if __name__ == '__main__':
    same_seeds(int(args.seed))
    config(args)
    E = experiment(args)
    E.continual_learning()
