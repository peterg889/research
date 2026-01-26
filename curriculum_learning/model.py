"""
Unified BERT Model for Curriculum Learning

Scalable BERT implementation using HuggingFace transformers with 
enhanced monitoring and training stability features.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from transformers import BertConfig, BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from config import Config


@dataclass
class ModelMetrics:
    """Model performance and resource metrics"""
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    gradient_norm: float
    is_stable: bool


class CurriculumBERT(nn.Module):
    """
    BERT model optimized for curriculum learning research.
    
    Features:
    - Multiple model sizes (mini, small, base)
    - Training stability monitoring
    - Gradient norm tracking
    - Scientific initialization
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        # Create HuggingFace BERT config
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_seq_length + 2,  # +2 for CLS/SEP
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            position_embedding_type="absolute",
            pad_token_id=0
        )
        
        # Create the BERT model
        self.bert = BertForMaskedLM(bert_config)
        
        # Initialize with scientific best practices
        self._initialize_weights()
        
        # Monitoring
        self.training_step = 0
        self.gradient_norms = []
        
        print(f"CurriculumBERT initialized")
        print(f"   Model: {config.model_size}")
        print(f"   Parameters: {self.get_metrics().total_parameters:,}")
        print(f"   Layers: {config.num_hidden_layers}")
        print(f"   Hidden size: {config.hidden_size}")
    
    def _initialize_weights(self):
        """Scientific weight initialization"""
        
        def init_weights(module):
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    nn.init.constant_(module.weight[module.padding_idx], 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
        self.bert.apply(init_weights)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> MaskedLMOutput:
        """Forward pass with monitoring"""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def create_mlm_labels(self, input_ids: torch.Tensor, tokenizer, mask_prob: float = 0.15) -> torch.Tensor:
        """Create labels for masked language modeling
        
        IMPORTANT: This returns labels only. The masking of input_ids should be done
        on a clone to avoid modifying the original data.
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Clone to avoid modifying original and ensure they're on the right device
        labels = input_ids.clone().to(device)
        masked_input_ids = input_ids.clone().to(device)
        
        # Create random mask on the same device
        probability_matrix = torch.full(labels.shape, mask_prob, device=device)
        
        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
        for special_id in tokenizer.all_special_ids:
            special_tokens_mask |= (labels == special_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Only compute loss on masked tokens
        labels[~masked_indices] = -100
        
        # Replace masked tokens with [MASK] token in the cloned tensor
        masked_input_ids[masked_indices] = tokenizer.mask_token_id
        
        # For MLM, we need to return the masked input_ids too
        # Store it as an attribute so the training loop can access it
        self._last_masked_input_ids = masked_input_ids
        
        return labels
    
    def compute_gradient_norm(self) -> float:
        """Compute and track gradient norm"""
        total_norm = 0.0
        param_count = 0
        
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Keep only recent norms
        if len(self.gradient_norms) > 100:
            self.gradient_norms = self.gradient_norms[-100:]
        
        return total_norm
    
    def check_training_stability(self) -> Dict[str, bool]:
        """Check if training is stable"""
        checks = {}
        
        # Check gradient norms
        if self.gradient_norms:
            recent_norm = sum(self.gradient_norms[-10:]) / len(self.gradient_norms[-10:])
            checks['gradient_norm_stable'] = recent_norm < 10.0
            checks['gradient_exploding'] = recent_norm > 100.0
        else:
            checks['gradient_norm_stable'] = True
            checks['gradient_exploding'] = False
        
        # Check for NaN parameters
        has_nan = any(torch.isnan(param.data).any() for param in self.parameters())
        checks['parameters_valid'] = not has_nan
        
        # Overall stability
        checks['is_stable'] = (checks['gradient_norm_stable'] and 
                              checks['parameters_valid'] and 
                              not checks['gradient_exploding'])
        
        return checks
    
    def get_metrics(self) -> ModelMetrics:
        """Get comprehensive model metrics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Average gradient norm
        avg_grad_norm = (sum(self.gradient_norms) / len(self.gradient_norms) 
                        if self.gradient_norms else 0.0)
        
        # Stability check
        stability = self.check_training_stability()
        
        return ModelMetrics(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            gradient_norm=avg_grad_norm,
            is_stable=stability.get('is_stable', True)
        )
    
    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter grouping"""
        
        # Group parameters (no weight decay for bias and LayerNorm)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, 
                        num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler with warmup"""
        
        def lr_lambda(current_step: int):
            # Warmup phase
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            
            # Cosine decay
            progress = float(current_step - self.config.warmup_steps) / float(
                max(1, num_training_steps - self.config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_model(config: Config, device: str = "auto", use_multi_gpu: bool = True) -> CurriculumBERT:
    """
    Factory function to create and setup model with multi-GPU support.
    
    Args:
        config: Experiment configuration
        device: Device to load model on ("auto", "cpu", "cuda", "cuda:0", etc.)
        use_multi_gpu: Enable multi-GPU training if available
    
    Returns:
        Initialized CurriculumBERT model (possibly wrapped in DataParallel)
    """
    
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = CurriculumBERT(config)
    
    # Check for multi-GPU setup
    if device.startswith("cuda") and use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"🖥️  Multi-GPU training enabled: {torch.cuda.device_count()} GPUs available")
        
        # Move model to primary device first
        primary_device = torch.device(device if ":" in device else "cuda:0")
        model = model.to(primary_device)
        
        # Wrap in DataParallel
        # NOTE: DataParallel maintains batch order, which preserves curriculum learning
        # Each GPU processes a portion of the batch, but the order is maintained
        model = torch.nn.DataParallel(model)
        
        print(f"📱 Model distributed across GPUs: {list(range(torch.cuda.device_count()))}")
        print("   Curriculum order preserved: DataParallel splits batches, not samples")
    else:
        # Single device training
        model = model.to(device)
        print(f"📱 Model loaded on: {device}")
    
    # Enable mixed precision if configured
    if config.mixed_precision and device.startswith("cuda"):
        print("⚡ Mixed precision training enabled")
    
    return model


if __name__ == "__main__":
    # Test the model
    from config import debug_config
    
    print("🧪 Testing CurriculumBERT...\n")
    
    # Test different model sizes
    for model_size in ["bert-mini", "bert-small"]:
        print(f"Testing {model_size}:")
        
        config = debug_config(model_size=model_size)
        model = create_model(config, device="cpu")
        
        # Get metrics
        metrics = model.get_metrics()
        print(f"  Parameters: {metrics.total_parameters:,}")
        print(f"  Size: {metrics.model_size_mb:.1f} MB")
        
        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            print(f"  Forward pass: {outputs.logits.shape}")
        
        # Test optimizer
        optimizer = model.prepare_optimizer()
        print(f"  Optimizer: {type(optimizer).__name__}")
        
        # Test scheduler
        scheduler = model.create_scheduler(optimizer, 1000)
        print(f"  Scheduler: {type(scheduler).__name__}")
        
        print()
    
    print("CurriculumBERT test completed!")