"""
Entropy-Based Speculative Decoding Implementation Architecture
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class EntropyConfig:
    """Configuration for entropy-based stopping criteria"""
    strategy: str = "absolute"  # absolute, relative, combined, adaptive
    theta_abs: float = 1.5
    theta_rel: float = 1.5
    window_size: int = 3
    min_draft_len: int = 1
    max_draft_len: int = 16
    entropy_type: str = "shannon"  # shannon, renyi, topk
    renyi_alpha: float = 2.0
    topk_k: int = 50
    adaptive_alpha: float = 0.1
    use_temperature: bool = True
    temperature: float = 1.0


class EntropyType(Enum):
    SHANNON = "shannon"
    RENYI = "renyi"
    TOPK = "topk"


class StoppingStrategy(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    WINDOW = "window"
    COMBINED = "combined"
    ADAPTIVE = "adaptive"


class EntropyComputer:
    """Efficient entropy computation with caching"""
    
    def __init__(self, config: EntropyConfig):
        self.config = config
        self.entropy_type = EntropyType(config.entropy_type)
        
    def compute(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy based on configuration"""
        if self.config.use_temperature:
            logits = logits / self.config.temperature
            
        if self.entropy_type == EntropyType.SHANNON:
            return self._shannon_entropy(logits)
        elif self.entropy_type == EntropyType.RENYI:
            return self._renyi_entropy(logits, self.config.renyi_alpha)
        elif self.entropy_type == EntropyType.TOPK:
            return self._topk_entropy(logits, self.config.topk_k)
        else:
            raise ValueError(f"Unknown entropy type: {self.entropy_type}")
    
    def _shannon_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Standard Shannon entropy: H(p) = -Σ p(x) * log(p(x))"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def _renyi_entropy(self, logits: torch.Tensor, alpha: float) -> torch.Tensor:
        """Rényi entropy: H_α(p) = (1/(1-α)) * log(Σ p(x)^α)"""
        probs = F.softmax(logits, dim=-1)
        if alpha == 1.0:
            return self._shannon_entropy(logits)
        sum_p_alpha = torch.sum(probs ** alpha, dim=-1)
        entropy = (1 / (1 - alpha)) * torch.log(sum_p_alpha)
        return entropy
    
    def _topk_entropy(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Entropy computed only over top-k tokens"""
        topk_logits, _ = torch.topk(logits, k, dim=-1)
        return self._shannon_entropy(topk_logits)


class StoppingCriterion(ABC):
    """Abstract base class for stopping criteria"""
    
    @abstractmethod
    def should_stop(self, 
                   current_entropy: float,
                   entropy_history: List[float],
                   position: int) -> bool:
        pass


class AbsoluteThresholdCriterion(StoppingCriterion):
    """Stop when entropy exceeds absolute threshold"""
    
    def __init__(self, threshold: float):
        self.threshold = threshold
        
    def should_stop(self, current_entropy: float, 
                   entropy_history: List[float], 
                   position: int) -> bool:
        return current_entropy > self.threshold


class RelativeIncreaseCriterion(StoppingCriterion):
    """Stop when entropy increases by relative factor"""
    
    def __init__(self, ratio_threshold: float):
        self.ratio_threshold = ratio_threshold
        
    def should_stop(self, current_entropy: float,
                   entropy_history: List[float],
                   position: int) -> bool:
        if not entropy_history:
            return False
        prev_entropy = entropy_history[-1]
        if prev_entropy < 1e-6:
            return False
        ratio = current_entropy / prev_entropy
        return ratio > self.ratio_threshold


class SlidingWindowCriterion(StoppingCriterion):
    """Stop when sliding window average exceeds threshold"""
    
    def __init__(self, window_size: int, threshold: float):
        self.window_size = window_size
        self.threshold = threshold
        
    def should_stop(self, current_entropy: float,
                   entropy_history: List[float],
                   position: int) -> bool:
        recent = entropy_history[-(self.window_size-1):] + [current_entropy]
        if len(recent) < self.window_size:
            return False
        avg_entropy = sum(recent) / len(recent)
        return avg_entropy > self.threshold


class AdaptiveThresholdCriterion(StoppingCriterion):
    """Adaptive threshold that increases with position"""
    
    def __init__(self, base_threshold: float, alpha: float):
        self.base_threshold = base_threshold
        self.alpha = alpha
        
    def should_stop(self, current_entropy: float,
                   entropy_history: List[float],
                   position: int) -> bool:
        adaptive_threshold = self.base_threshold + self.alpha * position
        return current_entropy > adaptive_threshold


class CombinedCriterion(StoppingCriterion):
    """Combine multiple criteria with AND/OR logic"""
    
    def __init__(self, criteria: List[StoppingCriterion], logic: str = "or"):
        self.criteria = criteria
        self.logic = logic
        
    def should_stop(self, current_entropy: float,
                   entropy_history: List[float],
                   position: int) -> bool:
        results = [c.should_stop(current_entropy, entropy_history, position) 
                  for c in self.criteria]
        
        if self.logic == "or":
            return any(results)
        elif self.logic == "and":
            return all(results)
        else:
            raise ValueError(f"Unknown logic: {self.logic}")


class EntropyBasedDrafter:
    """Drafter model with entropy-based generation"""
    
    def __init__(self, model, tokenizer, entropy_config: EntropyConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_config = entropy_config
        self.entropy_computer = EntropyComputer(entropy_config)
        self.stopping_criterion = self._build_stopping_criterion()
        
    def _build_stopping_criterion(self) -> StoppingCriterion:
        """Build stopping criterion based on configuration"""
        strategy = StoppingStrategy(self.entropy_config.strategy)
        
        if strategy == StoppingStrategy.ABSOLUTE:
            return AbsoluteThresholdCriterion(self.entropy_config.theta_abs)
        elif strategy == StoppingStrategy.RELATIVE:
            return RelativeIncreaseCriterion(self.entropy_config.theta_rel)
        elif strategy == StoppingStrategy.WINDOW:
            return SlidingWindowCriterion(
                self.entropy_config.window_size,
                self.entropy_config.theta_abs
            )
        elif strategy == StoppingStrategy.ADAPTIVE:
            return AdaptiveThresholdCriterion(
                self.entropy_config.theta_abs,
                self.entropy_config.adaptive_alpha
            )
        elif strategy == StoppingStrategy.COMBINED:
            criteria = [
                AbsoluteThresholdCriterion(self.entropy_config.theta_abs),
                RelativeIncreaseCriterion(self.entropy_config.theta_rel)
            ]
            return CombinedCriterion(criteria, logic="or")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    @torch.no_grad()
    def generate_draft(self, 
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      past_key_values: Optional[Any] = None) -> Dict[str, Any]:
        """Generate draft tokens with entropy-based stopping"""
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        draft_tokens = []
        draft_logits = []
        draft_entropies = []
        draft_probs = []
        
        current_ids = input_ids
        current_mask = attention_mask
        current_past = past_key_values
        
        for position in range(self.entropy_config.max_draft_len):
            # Forward pass
            outputs = self.model(
                input_ids=current_ids if position == 0 else current_ids[:, -1:],
                attention_mask=current_mask,
                past_key_values=current_past,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]
            current_past = outputs.past_key_values
            
            # Compute entropy
            entropy = self.entropy_computer.compute(logits)
            
            # Check stopping criterion (after minimum length)
            if position >= self.entropy_config.min_draft_len - 1:
                if self.stopping_criterion.should_stop(
                    entropy.item(), 
                    [e.item() for e in draft_entropies],
                    position
                ):
                    break
            
            # Sample next token
            probs = F.softmax(logits / self.entropy_config.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Store results
            draft_tokens.append(next_token)
            draft_logits.append(logits)
            draft_entropies.append(entropy)
            draft_probs.append(probs)
            
            # Update inputs
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if current_mask is not None:
                current_mask = torch.cat([
                    current_mask,
                    torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)
                ], dim=1)
        
        return {
            'tokens': torch.cat(draft_tokens, dim=1) if draft_tokens else torch.empty((batch_size, 0), device=device, dtype=torch.long),
            'logits': torch.stack(draft_logits, dim=1) if draft_logits else torch.empty((batch_size, 0, self.model.config.vocab_size), device=device),
            'entropies': torch.stack(draft_entropies) if draft_entropies else torch.empty(0, device=device),
            'probs': torch.stack(draft_probs, dim=1) if draft_probs else torch.empty((batch_size, 0, self.model.config.vocab_size), device=device),
            'length': len(draft_tokens),
            'past_key_values': current_past
        }


class EntropyBasedSpeculativeDecoder:
    """Main speculative decoder with entropy-based drafting"""
    
    def __init__(self, 
                 drafter_model,
                 verifier_model,
                 tokenizer,
                 entropy_config: EntropyConfig):
        self.drafter = EntropyBasedDrafter(drafter_model, tokenizer, entropy_config)
        self.verifier = verifier_model
        self.tokenizer = tokenizer
        self.entropy_config = entropy_config
        
        # Statistics tracking
        self.stats = {
            'total_drafted': 0,
            'total_accepted': 0,
            'draft_lengths': [],
            'acceptance_per_position': {},
            'entropy_trajectories': []
        }
    
    @torch.no_grad()
    def generate_token(self,
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      drafter_past: Optional[Any] = None,
                      verifier_past: Optional[Any] = None) -> Dict[str, Any]:
        """Generate tokens using entropy-based speculative decoding"""
        device = input_ids.device
        
        # Generate draft sequence
        draft_output = self.drafter.generate_draft(
            input_ids, attention_mask, drafter_past
        )
        
        draft_tokens = draft_output['tokens']
        draft_logits = draft_output['logits']
        draft_length = draft_output['length']
        
        if draft_length == 0:
            # Fall back to regular generation
            return self._generate_single_token(input_ids, attention_mask, verifier_past)
        
        # Verify draft tokens
        verify_ids = torch.cat([input_ids, draft_tokens], dim=1)
        verify_mask = attention_mask
        if verify_mask is not None:
            verify_mask = torch.cat([
                verify_mask,
                torch.ones((1, draft_length), device=device, dtype=verify_mask.dtype)
            ], dim=1)
        
        verify_outputs = self.verifier(
            input_ids=verify_ids,
            attention_mask=verify_mask,
            past_key_values=verifier_past,
            use_cache=True
        )
        
        verify_logits = verify_outputs.logits[:, -(draft_length+1):-1, :]
        
        # Acceptance checking
        accepted_tokens = []
        for i in range(draft_length):
            draft_prob = F.softmax(draft_logits[:, i, :], dim=-1)
            verify_prob = F.softmax(verify_logits[:, i, :], dim=-1)
            
            # Rejection sampling
            draft_token = draft_tokens[:, i]
            r = torch.rand(1, device=device)
            
            acceptance_prob = torch.minimum(
                torch.ones_like(r),
                verify_prob[:, draft_token] / (draft_prob[:, draft_token] + 1e-8)
            )
            
            if r < acceptance_prob:
                accepted_tokens.append(draft_token)
            else:
                # Rejection - sample from residual
                break
        
        # Update statistics
        self._update_stats(draft_length, len(accepted_tokens), draft_output['entropies'])
        
        # Prepare output
        if accepted_tokens:
            accepted_ids = torch.stack(accepted_tokens, dim=1)
            output_ids = torch.cat([input_ids, accepted_ids], dim=1)
        else:
            # Sample from verifier distribution at rejection point
            sample_logits = verify_outputs.logits[:, len(accepted_tokens), :]
            sample_probs = F.softmax(sample_logits / self.entropy_config.temperature, dim=-1)
            sampled_token = torch.multinomial(sample_probs, num_samples=1)
            output_ids = torch.cat([input_ids, sampled_token], dim=1)
            accepted_tokens = [sampled_token]
        
        return {
            'output_ids': output_ids,
            'accepted_length': len(accepted_tokens),
            'draft_length': draft_length,
            'verifier_past': verify_outputs.past_key_values
        }
    
    def _generate_single_token(self, input_ids, attention_mask, past_key_values):
        """Fallback to regular generation"""
        outputs = self.verifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits / self.entropy_config.temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return {
            'output_ids': torch.cat([input_ids, next_token], dim=1),
            'accepted_length': 1,
            'draft_length': 0,
            'verifier_past': outputs.past_key_values
        }
    
    def _update_stats(self, draft_length: int, accepted_length: int, entropies: torch.Tensor):
        """Update generation statistics"""
        self.stats['total_drafted'] += draft_length
        self.stats['total_accepted'] += accepted_length
        self.stats['draft_lengths'].append(draft_length)
        
        for i in range(accepted_length):
            if i not in self.stats['acceptance_per_position']:
                self.stats['acceptance_per_position'][i] = {'accepted': 0, 'total': 0}
            self.stats['acceptance_per_position'][i]['accepted'] += 1
            
        for i in range(draft_length):
            if i not in self.stats['acceptance_per_position']:
                self.stats['acceptance_per_position'][i] = {'accepted': 0, 'total': 0}
            self.stats['acceptance_per_position'][i]['total'] += 1
            
        if len(entropies) > 0:
            self.stats['entropy_trajectories'].append(entropies.cpu().numpy())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        stats = self.stats.copy()
        
        # Compute derived statistics
        if stats['total_drafted'] > 0:
            stats['overall_acceptance_rate'] = stats['total_accepted'] / stats['total_drafted']
        else:
            stats['overall_acceptance_rate'] = 0.0
            
        stats['avg_draft_length'] = np.mean(stats['draft_lengths']) if stats['draft_lengths'] else 0.0
        stats['position_acceptance_rates'] = {}
        
        for pos, counts in stats['acceptance_per_position'].items():
            if counts['total'] > 0:
                stats['position_acceptance_rates'][pos] = counts['accepted'] / counts['total']
                
        return stats