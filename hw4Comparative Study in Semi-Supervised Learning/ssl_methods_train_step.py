import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

# --- Pi-Model 训练步骤 ---
def pi_model_train_step(model: nn.Module,
                        loss_supervised_ce: nn.CrossEntropyLoss, loss_consistency_mse: nn.MSELoss,
                        images_l: torch.Tensor, labels_l: torch.Tensor,
                        images_u_aug1: torch.Tensor, images_u_aug2: torch.Tensor, 
                        lambda_consistency: float
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs a single training step for the Pi-Model."""
    # 1. 监督损失
    logits_l = model(images_l)
    loss_s = loss_supervised_ce(logits_l, labels_l)

    # 2. 一致性损失
    logits_u_aug1 = model(images_u_aug1)
    logits_u_aug2 = model(images_u_aug2)

    loss_u = loss_consistency_mse(
        F.softmax(logits_u_aug1, dim=1),
        F.softmax(logits_u_aug2, dim=1)
    )

    # 3. 总损失
    total_loss = loss_s + lambda_consistency * loss_u

    return total_loss, loss_s, loss_u * lambda_consistency

# --- Helper: KL 散度 ---
def kl_divergence_with_logits(q_logits, p_logits):
    """
    Calculates KL divergence KL(P || Q) directly from logits.
    P is the target distribution.
    Q is the distribution being regularized.

    Args:
        q_logits: Logits of distribution Q. Shape: (batch_size, num_classes)
        p_logits: Logits of distribution P. Shape: (batch_size, num_classes)

    Returns:
        KL divergence averaged over the batch.
    """
    p_softmax = F.softmax(p_logits.detach(), dim=1)
    q_log_softmax = F.log_softmax(q_logits, dim=1)
    kl_div = F.kl_div(q_log_softmax, p_softmax, reduction='none') # Shape: (batch_size, num_classes)
    kl_div = kl_div.sum(dim=1) # Sum over classes. Shape: (batch_size,)
    return kl_div.mean()


# --- Helper: VAT 扰动计算 ---
def _normalize_perturbation(d: torch.Tensor) -> torch.Tensor:
    """Normalizes the perturbation tensor d to have unit L2 norm."""

    dims = tuple(range(1, d.dim()))
    norm = torch.sqrt(torch.sum(d ** 2, dim=dims, keepdim=True))
    norm = torch.clamp(norm, min=1e-12)
    return d / norm

def _compute_vat_perturbation(model: nn.Module, x: torch.Tensor, xi: float, eps: float, num_iterations: int) -> torch.Tensor:
    """
    Computes the VAT adversarial perturbation r_adv using power iteration method.
    """
    model.eval()
    with torch.no_grad():
        logits_p = model(x).detach()

    d = torch.randn_like(x, requires_grad=False)
    d = _normalize_perturbation(d) * xi

    for _ in range(num_iterations):
        d.requires_grad_(True)
        x_perturbed = x + d
        logits_q = model(x_perturbed)

        # Calculate KL divergence KL(P || Q)
        kl_div = kl_divergence_with_logits(logits_q, logits_p)

        # Calculate gradient of KL
        (grad_d,) = torch.autograd.grad(kl_div, d, retain_graph=False)

        d = grad_d.detach()
        d = _normalize_perturbation(d)

    model.train()

    r_adv = d * eps
    return r_adv.detach()

# --- VAT 训练步骤 ---
def vat_train_step(model: nn.Module, loss_supervised_ce: nn.CrossEntropyLoss,
                   images_l: torch.Tensor, labels_l: torch.Tensor,
                   images_u: torch.Tensor,
                   lambda_vat: float, vat_epsilon: float, vat_xi: float, vat_power_iterations: int,
                   device: str
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs a single training step for Virtual Adversarial Training (VAT)."""
    # 1. 监督损失
    logits_l = model(images_l)
    loss_s = loss_supervised_ce(logits_l, labels_l)

    # 2. VAT Loss 
    r_adv_l = _compute_vat_perturbation(model, images_l, vat_xi, vat_epsilon, vat_power_iterations)
    logits_l_perturbed = model(images_l + r_adv_l.to(device))
    vat_loss_l = kl_divergence_with_logits(logits_l_perturbed, logits_l.detach()) # KL(P_orig || P_perturbed)

    with torch.no_grad():
        logits_u = model(images_u).detach()
    r_adv_u = _compute_vat_perturbation(model, images_u, vat_xi, vat_epsilon, vat_power_iterations)
    logits_u_perturbed = model(images_u + r_adv_u.to(device))
    vat_loss_u = kl_divergence_with_logits(logits_u_perturbed, logits_u) # KL(P_orig || P_perturbed)

    total_vat_loss = vat_loss_l + vat_loss_u
    loss_u = total_vat_loss

    # 3. 总损失
    total_loss = loss_s + lambda_vat * loss_u

    return total_loss, loss_s, loss_u * lambda_vat


# --- EMA 更新函数 ---
@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    """
    更新 Teacher模型的权重 (EMA)
    Args:
        model: Student 模型
        ema_model: Teacher 模型
        alpha: EMA decay 因子
        global_step: 当前训练步数 (用于调整 alpha，可选)
    """
    # alpha 可以是固定的，也可以是随训练变化的
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        
        
# --- Mean Teacher 训练步骤 ---
def mean_teacher_train_step(student_model: nn.Module, teacher_model: nn.Module,
                            loss_supervised: nn.CrossEntropyLoss, loss_consistency: nn.MSELoss,
                            images_l: torch.Tensor, labels_l: torch.Tensor,
                            images_u_weak: torch.Tensor, lambda_consistency: float) -> torch.Tensor:
    """计算 Mean Teacher 的一步损失"""
    # 1. 监督损失 (Student)
    outputs_l = student_model(images_l)
    loss_s = loss_supervised(outputs_l, labels_l)

    # 2. 一致性损失
    # Student 对弱增强无标注数据的预测 (需要计算梯度)
    outputs_u_student_logits = student_model(images_u_weak)
    outputs_u_student_prob = torch.softmax(outputs_u_student_logits, dim=1)

    # Teacher 对弱增强无标注数据的预测 (不需要计算梯度)
    with torch.no_grad():
        outputs_u_teacher_logits = teacher_model(images_u_weak)
        outputs_u_teacher_prob = torch.softmax(outputs_u_teacher_logits, dim=1)

    # 计算 MSE 损失 (作用于概率)
    loss_c = loss_consistency(outputs_u_student_prob, outputs_u_teacher_prob)

    # 总损失
    total_loss = loss_s + lambda_consistency * loss_c
    return total_loss, loss_s, loss_c # 返回各部分损失，便于记录


# --- Pseudo-Labeling 训练步骤 ---
def pseudo_labeling_train_step(model: nn.Module,
                               loss_supervised_ce: nn.CrossEntropyLoss,
                               images_l: torch.Tensor, labels_l: torch.Tensor,
                               images_u_weak: torch.Tensor,
                               lambda_pl: float, confidence_threshold: float
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 Pseudo-Labeling 的一步损失"""
    # 1. 监督损失
    outputs_l = model(images_l)
    loss_s = loss_supervised_ce(outputs_l, labels_l)

    # 2. 伪标签损失
    loss_pl = torch.tensor(0.0, device=images_l.device)
    num_confident = 0

    with torch.no_grad():
        # 生成伪标签 (使用弱增强)
        logits_u_weak_no_grad = model(images_u_weak) # No grad for pseudo-label generation
        probs_u_weak = torch.softmax(logits_u_weak_no_grad, dim=1)
        max_probs, targets_u = torch.max(probs_u_weak, dim=1)
        mask = max_probs.ge(confidence_threshold).float()
        num_confident = mask.sum().item()

    if num_confident > 0:
        # 计算模型对 *相同* 弱增强图像的预测 logits (这次需要梯度)
        logits_u_weak_grad = model(images_u_weak)
        # 计算 CE 损失 (模型对弱增强的预测 vs 高置信度伪标签)
        loss_pl_per_sample = F.cross_entropy(logits_u_weak_grad, targets_u, reduction='none')
        loss_pl = (loss_pl_per_sample * mask).mean() # 只对置信度高的样本计算平均损失

    # 总损失
    total_loss = loss_s + lambda_pl * loss_pl
    return total_loss, loss_s, loss_pl

def fixmatch_train_step(model: nn.Module,
                        loss_supervised_ce: nn.CrossEntropyLoss, # Reusing CE loss
                        images_l: torch.Tensor, labels_l: torch.Tensor,
                        images_u_weak: torch.Tensor, images_u_strong: torch.Tensor,
                        lambda_consistency: float, confidence_threshold: float
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 FixMatch 的一步损失"""
    # 1. 监督损失
    outputs_l = model(images_l)
    loss_s = loss_supervised_ce(outputs_l, labels_l)

    # 2. 无监督一致性损失
    loss_u = torch.tensor(0.0, device=images_l.device)
    num_confident = 0

    with torch.no_grad():
        # 生成伪标签 (使用弱增强)
        logits_u_weak = model(images_u_weak)
        probs_u_weak = torch.softmax(logits_u_weak, dim=1)
        max_probs, targets_u = torch.max(probs_u_weak, dim=1)
        mask = max_probs.ge(confidence_threshold).float() # .ge() is >=
        num_confident = mask.sum().item()

    if num_confident > 0:
        # 计算强增强输出的 logits
        logits_u_strong = model(images_u_strong)
        # 计算 CE 损失 (模型对强增强的预测 vs 弱增强生成的高置信度伪标签)
        # 使用 reduction='none' 获取每个样本的损失，然后用 mask 加权平均
        loss_u_per_sample = F.cross_entropy(logits_u_strong, targets_u, reduction='none')
        loss_u = (loss_u_per_sample * mask).mean() # 只对置信度高的样本计算平均损失

    # 总损失
    total_loss = loss_s + lambda_consistency * loss_u
    #print(f"  FixMatch Step - Confident Samples: {num_confident}/{images_u_weak.size(0)}") # Debug
    return total_loss, loss_s, loss_u