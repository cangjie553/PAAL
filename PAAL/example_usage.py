import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PAAL import (
    CandidateManager,
    detect_candidates,
    split_candidates,
    persist_candidate_sets,
    load_candidate_set
)


def example_basic_usage(model, historical_triples, snapshot_id):
    """
    基本使用示例
    
    Args:
        model: 训练好的嵌入模型
        historical_triples: 历史三元组列表 [(h, r, t), ...]
        snapshot_id: 当前快照 ID
    """
    print("=" * 60)
    print("示例 1: 基本使用流程")
    print("=" * 60)
    

    print("\n步骤 1: 检测低质量候选集 C")
    print("-" * 60)
    
    C = detect_candidates(
        model=model,
        triples=historical_triples,
        tau_Q=0.3,
        percentile=None,
        gamma=0.1,
        max_samples=1000
    )
    
    print(f"历史三元组数: {len(historical_triples)}")
    print(f"检测到的候选集大小: {len(C)}")
    
    if len(C) > 0:
        Q_values = [c['Q'] for c in C]
        print(f"Q 值范围: [{min(Q_values):.4f}, {max(Q_values):.4f}]")
        print(f"平均 Q 值: {sum(Q_values)/len(Q_values):.4f}")
    

    print("\n步骤 2: 划分遗忘集 F 和修复集 R")
    print("-" * 60)
    
    F, R = split_candidates(
        candidates=C,
        p_forget=0.3,
        p_repair=0.5
    )
    
    print(f"|F| (遗忘集): {len(F)}")
    print(f"|R| (修复集): {len(R)}")
    
    if len(F) > 0:
        Q_F = [c['Q'] for c in F]
        print(f"F 的 Q 值范围: [{min(Q_F):.4f}, {max(Q_F):.4f}]")
    
    if len(R) > 0:
        Q_R = [c['Q'] for c in R]
        print(f"R 的 Q 值范围: [{min(Q_R):.4f}, {max(Q_R):.4f}]")
    

    print("\n步骤 3: 持久化到文件系统")
    print("-" * 60)
    
    paths = persist_candidate_sets(
        snapshot_id=snapshot_id,
        C=C,
        F=F,
        R=R,
        out_dir="PAAL/candidates"
    )
    
    print("保存的文件:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    
    print("\n✓ 基本流程完成!")


def example_percentile_strategy(model, historical_triples, snapshot_id):
    """
    使用百分位数策略的示例
    
    Args:
        model: 训练好的嵌入模型
        historical_triples: 历史三元组列表
        snapshot_id: 当前快照 ID
    """
    print("\n" + "=" * 60)
    print("示例 2: 使用百分位数策略")
    print("=" * 60)
    

    print("\n使用百分位数策略 (选择最差的 20%)")
    print("-" * 60)
    
    C = detect_candidates(
        model=model,
        triples=historical_triples,
        percentile=0.2,
        gamma=0.1
    )
    
    print(f"历史三元组数: {len(historical_triples)}")
    print(f"候选集大小: {len(C)} ({len(C)/len(historical_triples)*100:.1f}%)")
    

    F, R = split_candidates(C, p_forget=0.2, p_repair=0.5)
    
    print(f"|F|: {len(F)} (候选集的 20%)")
    print(f"|R|: {len(R)} (候选集的 50%)")
    
    print("\n✓ 百分位数策略完成!")


def example_load_and_reuse(snapshot_id):
    """
    加载已保存的候选集示例
    
    Args:
        snapshot_id: 快照 ID
    """
    print("\n" + "=" * 60)
    print("示例 3: 加载已保存的候选集")
    print("=" * 60)
    

    print("\n加载候选集...")
    print("-" * 60)
    
    try:
        C = load_candidate_set(snapshot_id, 'C', out_dir="PAAL/candidates")
        F = load_candidate_set(snapshot_id, 'F', out_dir="PAAL/candidates")
        R = load_candidate_set(snapshot_id, 'R', out_dir="PAAL/candidates")
        
        print(f"加载的候选集 C: {len(C)} 个三元组")
        print(f"加载的遗忘集 F: {len(F)} 个三元组")
        print(f"加载的修复集 R: {len(R)} 个三元组")
        

        print("\n使用加载的集合:")
        print("-" * 60)
        
        if len(F) > 0:
            print(f"遗忘集示例 (前 3 个):")
            for i, c in enumerate(F[:3]):
                h, r, t = c['triple']
                Q = c['Q']
                print(f"  {i+1}. ({h}, {r}, {t}) - Q={Q:.4f}")
        
        print("\n✓ 加载成功!")
        
    except FileNotFoundError:
        print(f"⚠ 快照 {snapshot_id} 的候选集文件不存在")
        print("  请先运行示例 1 或 2 生成文件")


def example_integration_with_training():
    """
    与训练流程集成的伪代码示例
    """
    print("\n" + "=" * 60)
    print("示例 4: 与训练流程集成 (伪代码)")
    print("=" * 60)
    
    code = """


for snapshot_id in range(num_snapshots):
    print(f"\\n训练 Snapshot {snapshot_id}")
    

    train_data = load_snapshot_data(snapshot_id)
    historical_triples = get_historical_triples(snapshot_id)
    

    for epoch in range(num_epochs):

        loss = model.train_step(train_data)
    

    print(f"\\n检测 Snapshot {snapshot_id} 的候选集...")
    
    C = detect_candidates(
        model=model,
        triples=historical_triples,
        percentile=0.2,
        gamma=args.dqa_gamma
    )
    
    F, R = split_candidates(
        candidates=C,
        p_forget=args.forget_ratio,
        p_repair=args.repair_ratio
    )
    

    paths = persist_candidate_sets(
        snapshot_id=snapshot_id,
        C=C, F=F, R=R,
        out_dir=os.path.join(args.save_path, "candidates")
    )
    
    print(f"检测到 {len(C)} 个候选三元组")
    print(f"  遗忘集 F: {len(F)} 个")
    print(f"  修复集 R: {len(R)} 个")
    



"""
    
    print(code)
    print("\n✓ 集成示例完成!")


def example_advanced_usage():
    """
    高级使用示例
    """
    print("\n" + "=" * 60)
    print("示例 5: 高级使用 - 自定义管理器")
    print("=" * 60)
    

    manager = CandidateManager(seed=123)
    
    print("\n使用自定义管理器的好处:")
    print("-" * 60)
    print("1. 可以使用不同的随机种子")
    print("2. 可以在多个地方复用同一个管理器实例")
    print("3. 可以继承 CandidateManager 类并扩展功能")
    
    code = """

class MyCustomManager(CandidateManager):
    def __init__(self, seed=42):
        super().__init__(seed)
    
    def custom_quality_score(self, model, triple):

        Q = self.compute_quality_score(model, triple)

        return Q
    
    def custom_split_strategy(self, candidates):


        pass


manager = MyCustomManager(seed=42)
C = manager.detect_candidates(model, triples)
F, R = manager.split_candidates(C)
"""
    
    print(code)
    print("\n✓ 高级示例完成!")


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print(" " * 20 + "PAAL Candidate Manager 使用示例")
    print("=" * 80)
    


    
    print("\n⚠ 注意: 以下示例需要实际的模型和数据")
    print("这里展示的是 API 的使用方式和集成方法\n")
    

    example_integration_with_training()
    

    example_advanced_usage()
    
    print("\n" + "=" * 80)
    print("所有示例完成!")
    print("=" * 80)
    
    print("\n📚 更多信息:")
    print("  - 查看 test_candidate_manager.py 了解完整的测试用例")
    print("  - 查看 candidate_manager.py 了解详细的 API 文档")
    print("  - 运行 'python PAAL/test_candidate_manager.py' 运行测试")


if __name__ == '__main__':
    main()

