import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import Optional, Tuple, Union

from data_utils import get_ssl_dataloaders, transform_weak, transform_strong
from model import get_resnet18_cifar
from ssl_methods_train_step import mean_teacher_train_step, pseudo_labeling_train_step, update_ema_variables, fixmatch_train_step, pi_model_train_step, vat_train_step


# --- 配置 ---
N_LABELED_PER_CLASS = 4000
BATCH_SIZE_LABELED = 128
BATCH_SIZE_UNLABELED = 256 # 调试时减小
EPOCHS = 100 # 增加 epoch 数量以便观察效果
EVAL_STEP = 1 # 每隔多少个 epoch 评估一次
LR = 3e-3 # 学习率
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# --- SSL Method Specific Config ---
SSL_METHOD = 'vat' # 'supervised-only', 'pai-model', 'vat', 'mean-teacher', 'fixmatch', 'pseudo-labeling'

# Pi-Model specific
LAMBDA_PI_CONSISTENCY = 10.0

# Mean Teacher specific
EMA_DECAY = 0.99 # Mean Teacher 的 EMA decay 参数
LAMBDA_MT_CONSISTENCY = 1.0 # Mean Teacher 一致性损失的权重

# VAT specific
LAMBDA_VAT = 1.0
VAT_EPSILON = 8.0
VAT_XI = 1e-6
VAT_POWER_ITERATIONS = 1

# FixMatch specific
CONFIDENCE_THRESHOLD = 0.95 # FixMatch/PL 伪标签置信度阈值
LAMBDA_FIXMATCH_CONSISTENCY = 1.0 # FixMatch 一致性损失权重

# Pseudo-Labeling specific
LAMBDA_PL = 0.5 # Pseudo-Labeling 伪标签损失权重

# --- 评估函数 ---
def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    '''
    在测试集上评估模型
    '''
    model.eval()
    acc_num = 0
    total_num = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            acc_num += torch.sum(pred == labels).item()
            total_num += labels.size(0)
    acc = acc_num / total_num if total_num > 0 else 0.0
    print(f"评估 - 平均准确率: {acc:.4f} ({acc_num}/{total_num})")
    model.train() # 评估后切换回训练模式
    return acc

# --- 统一的训练步骤函数 ---
def train_step(student_model: nn.Module, teacher_model: Optional[nn.Module],
               optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
               ssl_method: str,
               loss_supervised_ce: nn.CrossEntropyLoss, loss_consistency_mse: nn.MSELoss,
               images_l: torch.Tensor, labels_l: torch.Tensor,
               batch_unlabeled: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
               lambda_mt_consistency: float, ema_decay: float,
               lambda_fixmatch_consistency: float, confidence_threshold: float,
               lambda_pl: float,
               lambda_pi_consistency: float,
               lambda_vat: float, vat_epsilon: float, vat_xi: float, vat_power_iterations: int,
               current_step: int, device: str):
    '''
    统一的训练步骤
    完成损失计算, 梯度下降更新
    Args:
        student_model: 'pseudo-labeling', 'fixmatch'方法都只是用一个模型
        teacher_model: 'mean-teacher'方法 需要同时使用学生模型和教师模型
        
        optimizer: 使用SGD方法
        scheduler: 用于模拟退火
        ssl_method: 包括 'mean-teacher', 'pseudo-labeling', 'fixmatch' 方法
        
        loss_supervised_ce: 在 'pseudo-labeling', 'fixmatch' 中使用
        loss_supervised_mse: 特别用于 'mean-teacher'方法中 衡量 Teacher-Student 模型间的一致性损失
        images_l, labels_l: 用于计算监督损失
        batch_unlabeled: 用于不同SSL方法的无标签数据, 不同的方法有不同的解包数据
        lambad_mt_consistency: Mean-Teacher 的一致性损失的权重系数
        ema_decay: 教师模型更新新旧模型的权重系数
        lambad_fixmatch_consistency: FixMatch 的一致性损失的权重系数
        confidence_threshold: 伪标签的权重系数
        lambda_pl: pseudo label 的一致性损失的权重系数
        lambda_pi_consistency: Pai-Model 的一致性损失的权重系数
        lambda_vat: VAT 的一致性损失的权重系数
        vat_epsilon:
        vat_xi:
        vat_power_iterations:
        current_step: (可选) 在Teacher-Student 的EMA更新中, 原论文使用了随时间变化的更新权重
        device: 训练设备
    Returns:
        total_loss, loss_s, loss_u (总损失, 监督损失, 一致性损失)
    '''
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    loss_s = torch.tensor(0.0, device=device) # 初始化监督损失
    loss_u = torch.tensor(0.0, device=device) # 初始化无监督损失

    # --- 根据 SSL 方法解包无标注数据并调用相应的训练步骤 ---
    if ssl_method == 'pai-model':
        if not (isinstance(batch_unlabeled, (list, tuple)) and len(batch_unlabeled) == 2):
             raise TypeError("Pi-Model expects a tuple (weak_aug1, weak_aug2) for unlabeled data.")
        images_u_w, images_u_s = batch_unlabeled
        images_u_w, images_u_s = images_u_w.to(device), images_u_s.to(device)
        total_loss, loss_s, loss_u = pi_model_train_step(
            student_model, loss_supervised_ce, loss_consistency_mse,
            images_l, labels_l, images_u_w, images_u_s,
            lambda_pi_consistency
        )

    elif ssl_method == 'mean-teacher':
        if not isinstance(batch_unlabeled, torch.Tensor):
             raise TypeError("Mean Teacher expects a single Tensor (weakly augmented) for unlabeled data.")
        if teacher_model is None:
             raise ValueError("Mean Teacher requires a teacher model.")
        images_u_weak = batch_unlabeled.to(device)
        total_loss, loss_s, loss_u = mean_teacher_train_step(
            student_model, teacher_model, loss_supervised_ce, loss_consistency_mse,
            images_l, labels_l, images_u_weak, lambda_mt_consistency
        )

    elif ssl_method == 'pseudo-labeling':
        if not isinstance(batch_unlabeled, torch.Tensor):
             raise TypeError("Pseudo-Labeling expects a single Tensor (weakly augmented) for unlabeled data.")
        images_u_weak = batch_unlabeled.to(device)
        total_loss, loss_s, loss_u = pseudo_labeling_train_step(
            student_model, loss_supervised_ce, images_l, labels_l,
            images_u_weak, lambda_pl, confidence_threshold
        )

    elif ssl_method == 'vat':
        if not isinstance(batch_unlabeled, torch.Tensor):
             raise TypeError("VAT expects a single Tensor (weakly augmented) for unlabeled data.")
        images_u_weak = batch_unlabeled.to(device)
        total_loss, loss_s, loss_u = vat_train_step(
            student_model, loss_supervised_ce, images_l, labels_l,
            images_u_weak, lambda_vat, vat_epsilon, vat_xi, vat_power_iterations,
            device
        )
    elif ssl_method == 'fixmatch':
        if not (isinstance(batch_unlabeled, (list, tuple)) and len(batch_unlabeled) == 2):
             raise TypeError("FixMatch expects a tuple (weak_aug, strong_aug) for unlabeled data.")
        images_u_weak, images_u_strong = batch_unlabeled
        images_u_weak, images_u_strong = images_u_weak.to(device), images_u_strong.to(device)
        total_loss, loss_s, loss_u = fixmatch_train_step(
            student_model, loss_supervised_ce, images_l, labels_l,
            images_u_weak, images_u_strong, lambda_fixmatch_consistency, confidence_threshold
        )

    else:
        raise ValueError(f"未知的 SSL 方法: {ssl_method}")

    # --- 反向传播和优化 ---
    total_loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step() # 如果使用学习率调度器

    # --- 更新 Teacher Model ---
    if ssl_method == 'mean-teacher' and teacher_model is not None:
        update_ema_variables(student_model, teacher_model, ema_decay, current_step)

    # 返回损失值用于记录 (统一返回 total, supervised, unsupervised)
    return total_loss.item(), loss_s.item(), loss_u.item()


# --- 主训练函数 ---
def train(ssl_method: str, epochs: int, device: str) -> nn.Module:
    """
    训练主循环
    Args:
        ssl_method: 使用完全的Supervised的方法 or 使用的 SSL 方法 ('pai-model', 'mean-teacher', 'vat', 'fixmatch', 'pseudo-labeling')
        epochs: 训练轮数
        device: 设备

    Returns:
        训练好的模型 Mean Teacher or Student Model
    """
    # --- 初始化模型 ---
    # 默认不使用教师模型
    student_model = get_resnet18_cifar(num_classes=10).to(device)
    teacher_model = None
    model_to_evaluate = student_model

    # 若使用mean-teacher方法则需要单独配置
    if ssl_method == 'mean-teacher':
        # 创建 Teacher 模型，初始权重与 Student 相同
        teacher_model = get_resnet18_cifar(num_classes=10).to(device)
        teacher_model.load_state_dict(student_model.state_dict())
        # 确保 Teacher 模型不需要梯度
        for param in teacher_model.parameters():
            param.requires_grad = False
        # MT 方法中评估的是教师模型
        model_to_evaluate = teacher_model
        print("已初始化 Student 和 Teacher 模型 (用于 Mean Teacher)")
    else:
        print(f"已初始化 Student 模型 (用于 {ssl_method})")


    # --- 优化器和损失函数 ---
    # 优化器只优化 student_model 的参数
    optimizer = optim.SGD(student_model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
    # 使用 Cosine Annealing 调度器
    if ssl_method == 'supervised-only':
        step_per_epoch = len(labeled_loader)
    else:
        step_per_epoch = len(unlabeled_loader)
    total_steps = epochs * step_per_epoch # Use unlabeled loader length as reference
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    loss_supervised_ce = nn.CrossEntropyLoss().to(device)
    loss_consistency_mse = nn.MSELoss().to(device) # Only used by Mean Teacher

    # --- 训练循环 ---
    global_step = 0
    best_acc = 0.0
    best_model_state = None # 存储最佳模型的状态字典

    for epoch_idx in range(epochs):
        print(f"\n--- Epoch {epoch_idx + 1}/{epochs} ---")
        student_model.train() # 确保 student 在训练模式
        # 确保 Teacher 在评估模式
        if teacher_model:
             teacher_model.eval()

        # --- 初始化一个Epoch中的损失记录 ---
        running_loss = 0.0
        running_loss_s = 0.0
        running_loss_u = 0.0 # Unsupervised loss
        num_batches = 0
        
        # ======================================================================
        # BRANCH 1: Supervised Only Training Logic
        # ======================================================================
        if ssl_method == 'supervised-only':
            for iter_idx, (images_l, labels_l) in enumerate(labeled_loader):
                images_l, labels_l = images_l.to(device), labels_l.to(device)

                optimizer.zero_grad()
                outputs_l = student_model(images_l)
                loss = loss_supervised_ce(outputs_l, labels_l)

                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                running_loss += loss.item()
                running_loss_s += loss.item()
                num_batches += 1
                global_step += 1

                if iter_idx % 50 == 0:
                    print(f"  Iter {iter_idx}/{len(labeled_loader)} - "
                          f"Loss: {loss.item():.4f} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ======================================================================
        # BRANCH 2: SSL Training Logic (MT, FixMatch, PL)
        # ======================================================================
        else:
            # 使用迭代器处理长度可能不同的 dataloader
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            num_iterations = len(unlabeled_loader) # 以无标注数据迭代次数为准

            for iter_idx in range(num_iterations):
                # --- 获取数据 ---
                try:
                    images_l, labels_l = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader) # 重新初始化
                    images_l, labels_l = next(labeled_iter)

                try:
                    batch_unlabeled = next(unlabeled_iter)
                except StopIteration:
                    print("Warning: Unlabeled data iterator reset unexpectedly.")
                    unlabeled_iter = iter(unlabeled_loader)
                    batch_unlabeled = next(unlabeled_iter)

                # --- 数据移动到设备 ---
                images_l, labels_l = images_l.to(device), labels_l.to(device)

                # --- 执行一步训练 ---
                loss, loss_s, loss_u = train_step(
                    student_model=student_model,
                    teacher_model=teacher_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ssl_method=ssl_method,
                    loss_supervised_ce=loss_supervised_ce,
                    loss_consistency_mse=loss_consistency_mse,
                    images_l=images_l,
                    labels_l=labels_l,
                    batch_unlabeled=batch_unlabeled,
                    lambda_mt_consistency=LAMBDA_MT_CONSISTENCY,
                    ema_decay=EMA_DECAY,
                    lambda_fixmatch_consistency=LAMBDA_FIXMATCH_CONSISTENCY,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    lambda_pl=LAMBDA_PL,
                    lambda_pi_consistency=LAMBDA_PI_CONSISTENCY,
                    lambda_vat=LAMBDA_VAT,
                    vat_epsilon=VAT_EPSILON,
                    vat_xi=VAT_XI,
                    vat_power_iterations=VAT_POWER_ITERATIONS,
                    current_step=global_step,
                    device=device
                )

                running_loss += loss
                running_loss_s += loss_s
                running_loss_u += loss_u
                num_batches += 1
                global_step += 1

                # --- 每 50 个 batch 打印一次信息 ---
                if iter_idx % 50 == 0:
                    print(f"  Iter {iter_idx}/{num_iterations} - "
                        f"Loss: {loss:.4f} (S: {loss_s:.4f}, U: {loss_u:.4f}) | " # U for Unsupervised
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- Epoch 结束 ---
        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_loss_s = running_loss_s / num_batches if num_batches > 0 else 0
        avg_loss_u = running_loss_u / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch_idx + 1} 结束 - 平均损失: {avg_loss:.4f} (S: {avg_loss_s:.4f}, U: {avg_loss_u:.4f})")

        # --- 评估 ---
        current_acc = evaluate(model_to_evaluate, test_loader, device=device)

        # --- 保存最佳模型 ---
        if current_acc > best_acc:
            best_acc = current_acc
            # 使用深拷贝确保保存的是当前最佳状态
            best_model_state = copy.deepcopy(model_to_evaluate.state_dict())
            print(f"*** 新的最佳准确率: {best_acc:.4f}，模型状态已保存 ***")


    print(f"\n训练完成. 最佳测试准确率: {best_acc:.4f}")

    # 加载最佳状态到最终要返回的模型上
    if best_model_state:
        print("加载最佳模型状态...")
        model_to_evaluate.load_state_dict(best_model_state)
    else:
        print("警告: 未找到最佳模型状态，返回最后状态的模型。")

    return model_to_evaluate

# --- 开始训练 ---
if __name__ == "__main__":
    SEED = 42

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # --- 获取数据加载器 ---
    print(f"为 {SSL_METHOD} 配置数据加载器...")
    if SSL_METHOD == 'supervised-only':
        labeled_loader, _, test_loader = get_ssl_dataloaders(
            n_labeled_per_class=N_LABELED_PER_CLASS,
            batch_size_labeled=BATCH_SIZE_LABELED,
            batch_size_unlabeled=BATCH_SIZE_UNLABELED,
            unlabeled_transform=transform_weak
        )
    elif SSL_METHOD == 'pai-model':
        labeled_loader, unlabeled_loader, test_loader = get_ssl_dataloaders(
            n_labeled_per_class=N_LABELED_PER_CLASS,
            batch_size_labeled=BATCH_SIZE_LABELED,
            batch_size_unlabeled=BATCH_SIZE_UNLABELED,
            unlabeled_transform=(transform_weak, transform_strong)
        )
    elif SSL_METHOD == 'mean-teacher':
        labeled_loader, unlabeled_loader, test_loader = get_ssl_dataloaders(
            n_labeled_per_class=N_LABELED_PER_CLASS,
            batch_size_labeled=BATCH_SIZE_LABELED,
            batch_size_unlabeled=BATCH_SIZE_UNLABELED,
            unlabeled_transform=transform_weak,
        )
    elif SSL_METHOD == 'pseudo-labeling':
        labeled_loader, unlabeled_loader, test_loader = get_ssl_dataloaders(
            n_labeled_per_class=N_LABELED_PER_CLASS,
            batch_size_labeled=BATCH_SIZE_LABELED,
            batch_size_unlabeled=BATCH_SIZE_UNLABELED,
            unlabeled_transform=transform_weak
        )
    elif SSL_METHOD == 'vat':
        labeled_loader, unlabeled_loader, test_loader = get_ssl_dataloaders(
            n_labeled_per_class=N_LABELED_PER_CLASS,
            batch_size_labeled=BATCH_SIZE_LABELED,
            batch_size_unlabeled=BATCH_SIZE_UNLABELED,
            unlabeled_transform=transform_weak
        )
    elif SSL_METHOD == 'fixmatch':
        # FixMatch 需要强弱两种增强
        labeled_loader, unlabeled_loader, test_loader = get_ssl_dataloaders(
            n_labeled_per_class=N_LABELED_PER_CLASS,
            batch_size_labeled=BATCH_SIZE_LABELED,
            batch_size_unlabeled=BATCH_SIZE_UNLABELED,
            unlabeled_transform=(transform_weak, transform_strong) # <--- 返回 (weak, strong) 元组
        )
    else:
        raise ValueError(f"未知的 SSL 方法: {SSL_METHOD}")

    print("数据加载器配置完成.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备 {device}")
    
    print(f"\n===== 开始训练: {SSL_METHOD} =====")

    # 执行训练
    trained_model = train(ssl_method=SSL_METHOD, epochs=EPOCHS, device=device)

    # 训练结束后再次评估最佳模型
    print("\n===== 最终评估 (加载的最佳模型) =====")
    final_acc = evaluate(trained_model, test_loader, device=device)
    print(f"最终模型在测试集上的准确率: {final_acc:.4f}")

    # 保存最终模型
    save_path = f'{SSL_METHOD}_best_model.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"最终 (最佳) 模型已保存为 {save_path}")