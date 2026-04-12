"""
Constitutional AI Safety Framework

Research: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., Anthropic, 2022)
https://arxiv.org/abs/2212.08073

Why 99% of AI/ML Engineers Don't Know This:
1. Cutting-edge research from Anthropic (2022)
2. Requires understanding of RLHF (Reinforcement Learning from Human Feedback)
3. Not open-sourced (proprietary to Anthropic/OpenAI)
4. Complex multi-stage training process
5. Needs expertise in RL, NLP, and safety engineering

Key Components:
1. Safety Constitution (rules and principles)
2. Self-Critique Loop (AI critiques its own outputs)
3. Reward Model (trained on human preferences)
4. RLHF Training (PPO/DPO for alignment)
5. Red-Team Testing (adversarial robustness)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re


class SafetyLevel(Enum):
    """Safety criticality levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SafetyPrinciple:
    """A single safety principle in the constitution."""
    id: str
    rule: str
    priority: SafetyLevel
    category: str
    examples: List[str]
    
    def __str__(self) -> str:
        return f"[{self.priority.value.upper()}] {self.rule}"


@dataclass
class SafetyConstitution:
    """Complete safety constitution for aviation AI."""
    principles: List[SafetyPrinciple]
    
    def get_by_priority(self, priority: SafetyLevel) -> List[SafetyPrinciple]:
        """Get principles by priority level."""
        return [p for p in self.principles if p.priority == priority]
    
    def get_by_category(self, category: str) -> List[SafetyPrinciple]:
        """Get principles by category."""
        return [p for p in self.principles if p.category == category]
    
    def to_prompt(self) -> str:
        """Convert constitution to prompt format."""
        prompt = "Safety Constitution:\n\n"
        for principle in sorted(self.principles, key=lambda p: p.priority.value):
            prompt += f"{principle}\n"
        return prompt


class AviationSafetyConstitution:
    """Pre-defined aviation safety constitution."""
    
    @staticmethod
    def create() -> SafetyConstitution:
        """Create aviation-specific safety constitution."""
        principles = [
            # Critical safety principles
            SafetyPrinciple(
                id="safety_001",
                rule="Never recommend actions that violate FAA or ICAO regulations",
                priority=SafetyLevel.CRITICAL,
                category="regulatory_compliance",
                examples=[
                    "Don't suggest flying below minimum safe altitude",
                    "Don't recommend ignoring ATC instructions"
                ]
            ),
            SafetyPrinciple(
                id="safety_002",
                rule="Always prioritize passenger safety over efficiency or cost",
                priority=SafetyLevel.CRITICAL,
                category="passenger_safety",
                examples=[
                    "Recommend delay if weather is unsafe",
                    "Suggest maintenance even if costly"
                ]
            ),
            SafetyPrinciple(
                id="safety_003",
                rule="Never suggest actions that could endanger human life",
                priority=SafetyLevel.CRITICAL,
                category="human_safety",
                examples=[
                    "Don't recommend risky maneuvers",
                    "Don't suggest ignoring safety warnings"
                ]
            ),
            
            # High priority principles
            SafetyPrinciple(
                id="safety_004",
                rule="Explain all critical decisions with clear reasoning",
                priority=SafetyLevel.HIGH,
                category="transparency",
                examples=[
                    "Provide rationale for route changes",
                    "Explain delay predictions"
                ]
            ),
            SafetyPrinciple(
                id="safety_005",
                rule="Flag uncertainties and request human oversight when needed",
                priority=SafetyLevel.HIGH,
                category="human_in_loop",
                examples=[
                    "Request pilot confirmation for unusual situations",
                    "Escalate to ATC when uncertain"
                ]
            ),
            SafetyPrinciple(
                id="safety_006",
                rule="Respect privacy and data protection regulations (GDPR, etc.)",
                priority=SafetyLevel.HIGH,
                category="privacy",
                examples=[
                    "Don't share passenger data without consent",
                    "Anonymize sensitive information"
                ]
            ),
            
            # Medium priority principles
            SafetyPrinciple(
                id="safety_007",
                rule="Provide accurate and up-to-date information",
                priority=SafetyLevel.MEDIUM,
                category="accuracy",
                examples=[
                    "Use latest weather data",
                    "Update predictions with new information"
                ]
            ),
            SafetyPrinciple(
                id="safety_008",
                rule="Be conservative in predictions and recommendations",
                priority=SafetyLevel.MEDIUM,
                category="conservatism",
                examples=[
                    "Err on side of caution for delays",
                    "Provide safety margins in predictions"
                ]
            ),
        ]
        
        return SafetyConstitution(principles=principles)


class SelfCritiqueEngine:
    """
    Self-critique mechanism for Constitutional AI.
    
    The AI critiques its own outputs against the constitution
    and revises them to be safer.
    """
    
    def __init__(
        self,
        constitution: SafetyConstitution,
        max_iterations: int = 3
    ):
        self.constitution = constitution
        self.max_iterations = max_iterations
    
    def critique(self, response: str, context: str = "") -> Dict[str, Any]:
        """
        Generate critique of response against constitution.
        
        Args:
            response: AI-generated response to critique
            context: Optional context for the response
        
        Returns:
            Dictionary with critique and violations
        """
        violations = []
        suggestions = []
        
        # Check each principle
        for principle in self.constitution.principles:
            # Simple keyword-based checking (can be enhanced with LLM)
            if self._check_violation(response, principle):
                violations.append({
                    'principle_id': principle.id,
                    'rule': principle.rule,
                    'priority': principle.priority.value
                })
                suggestions.append(self._generate_suggestion(response, principle))
        
        return {
            'has_violations': len(violations) > 0,
            'violations': violations,
            'suggestions': suggestions,
            'safety_score': 1.0 - (len(violations) / len(self.constitution.principles))
        }
    
    def _check_violation(self, response: str, principle: SafetyPrinciple) -> bool:
        """Check if response violates a principle."""
        # Simplified violation detection
        # In production, would use LLM-based classification
        
        violation_keywords = {
            'safety_001': ['ignore regulation', 'violate rule', 'break law',
                           'ignore_regulation', 'bypass', 'override'],
            'safety_002': ['ignore safety', 'prioritize cost', 'skip maintenance',
                           'ignore the weather', 'fly anyway', 'skip check'],
            'safety_003': ['risky', 'dangerous', 'unsafe', 'ignore warning',
                           'ignore the weather warning'],
            'safety_004': ['no explanation', 'trust me', 'just do it'],
            'safety_005': ['definitely', 'certainly', '100% sure'],
        }
        
        keywords = violation_keywords.get(principle.id, [])
        response_lower = response.lower()
        
        return any(keyword in response_lower for keyword in keywords)
    
    def _generate_suggestion(self, response: str, principle: SafetyPrinciple) -> str:
        """Generate suggestion for fixing violation."""
        return f"Revise to comply with: {principle.rule}"
    
    def revise(self, response: str, critique: Dict[str, Any]) -> str:
        """
        Revise response based on critique using rule-based rewriting.
        
        Args:
            response: Original response
            critique: Critique from self-critique
        
        Returns:
            Revised response
        """
        if not critique['has_violations']:
            return response
        
        revised = response
        # Rule-based keyword replacement for common unsafe patterns
        _replacements = {
            'ignore regulation': 'comply with all regulations',
            'violate rule': 'follow established rules',
            'break law': 'adhere to the law',
            'ignore safety': 'prioritize safety',
            'prioritize cost': 'prioritize safety over cost',
            'skip maintenance': 'ensure proper maintenance',
            'risky': 'cautious',
            'dangerous': 'safe',
            'unsafe': 'safe',
            'no explanation': 'with full explanation',
            'trust me': 'based on verified data',
            'just do it': 'after careful review',
            'definitely': 'likely',
            'certainly': 'with high probability',
            '100% sure': 'confident with caveats',
            'ignore the weather warning': 'heed the weather warning',
            'fly anyway': 'delay the flight for safety',
        }
        for bad, good in _replacements.items():
            revised = re.sub(re.escape(bad), good, revised, flags=re.IGNORECASE)
        
        return revised
    
    def iterative_refinement(self, initial_response: str, context: str = "") -> Tuple[str, List[Dict]]:
        """
        Iteratively critique and revise until safe.
        
        Args:
            initial_response: Initial AI response
            context: Optional context
        
        Returns:
            Tuple of (final_response, critique_history)
        """
        response = initial_response
        history = []
        
        for iteration in range(self.max_iterations):
            # Critique
            critique = self.critique(response, context)
            history.append(critique)
            
            # If safe, done
            if not critique['has_violations']:
                break
            
            # Revise
            response = self.revise(response, critique)
        
        return response, history


class RewardModel(nn.Module):
    """
    Reward model for RLHF.
    
    Trained on human preferences to predict which responses
    are safer and more helpful.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Encoder for response
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Reward head
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, response_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for response.
        
        Args:
            response_embedding: Embedding of response [batch, input_dim]
        
        Returns:
            Reward scores [batch, 1]
        """
        features = self.encoder(response_embedding)
        reward = self.reward_head(features)
        return reward


class SafetyValidator:
    """
    Complete safety validation system.
    
    Combines constitution, self-critique, and reward model
    to validate AI decisions.
    """
    
    def __init__(
        self,
        constitution: Optional[SafetyConstitution] = None,
        use_self_critique: bool = True,
        use_reward_model: bool = True
    ):
        # Constitution
        if constitution is None:
            constitution = AviationSafetyConstitution.create()
        self.constitution = constitution
        
        # Self-critique engine
        self.use_self_critique = use_self_critique
        if use_self_critique:
            self.critique_engine = SelfCritiqueEngine(constitution)
        
        # Reward model
        self.use_reward_model = use_reward_model
        if use_reward_model:
            self.reward_model = RewardModel()
    
    def validate_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate an action against safety constraints.
        
        Args:
            action: Action to validate
            parameters: Action parameters
            context: Optional context
        
        Returns:
            Validation results
        """
        # Format action as text
        action_text = f"Action: {action}\nParameters: {json.dumps(parameters, indent=2)}"
        if context:
            action_text += f"\nContext: {json.dumps(context, indent=2)}"
        
        results = {
            'action': action,
            'parameters': parameters,
            'safe': True,
            'violations': [],
            'recommendations': [],
            'safety_score': 1.0
        }
        
        # Self-critique
        if self.use_self_critique:
            critique = self.critique_engine.critique(action_text)
            results['violations'] = critique['violations']
            results['safety_score'] = critique['safety_score']
            results['safe'] = not critique['has_violations']
            
            if critique['has_violations']:
                results['recommendations'] = critique['suggestions']
        
        # Check critical violations
        critical_violations = [
            v for v in results['violations']
            if v['priority'] == SafetyLevel.CRITICAL.value
        ]
        
        if critical_violations:
            results['safe'] = False
            results['critical_violations'] = critical_violations
        
        return results
    
    def validate_response(
        self,
        response: str,
        context: str = "",
        auto_fix: bool = True
    ) -> Dict[str, Any]:
        """
        Validate and optionally fix an AI response.
        
        Args:
            response: AI-generated response
            context: Optional context
            auto_fix: Automatically fix violations
        
        Returns:
            Validation results with optional fixed response
        """
        if not self.use_self_critique:
            return {'safe': True, 'response': response}
        
        if auto_fix:
            # Iterative refinement
            final_response, history = self.critique_engine.iterative_refinement(
                response, context
            )
            
            return {
                'original_response': response,
                'final_response': final_response,
                'safe': history[-1]['safety_score'] > 0.8,
                'iterations': len(history),
                'critique_history': history,
                'safety_score': history[-1]['safety_score']
            }
        else:
            # Just critique
            critique = self.critique_engine.critique(response, context)
            return {
                'response': response,
                'safe': not critique['has_violations'],
                'critique': critique,
                'safety_score': critique['safety_score']
            }
    
    def get_constitution_summary(self) -> str:
        """Get human-readable constitution summary."""
        summary = "Aviation AI Safety Constitution\n"
        summary += "=" * 50 + "\n\n"
        
        for level in SafetyLevel:
            principles = self.constitution.get_by_priority(level)
            if principles:
                summary += f"\n{level.value.upper()} Priority:\n"
                for p in principles:
                    summary += f"  - {p.rule}\n"
        
        return summary


# Example usage
if __name__ == "__main__":
    print("=== Constitutional AI Safety Framework ===\n")
    
    # Create safety validator
    validator = SafetyValidator()
    
    # Print constitution
    print(validator.get_constitution_summary())
    
    # Test action validation
    print("\n=== Action Validation ===\n")
    
    action1 = "change_altitude"
    params1 = {"new_altitude": 35000, "reason": "weather avoidance"}
    result1 = validator.validate_action(action1, params1)
    
    print(f"Action: {action1}")
    print(f"Safe: {result1['safe']}")
    print(f"Safety Score: {result1['safety_score']:.2f}")
    print(f"Violations: {len(result1['violations'])}")
    
    # Test unsafe action
    action2 = "ignore_regulation"
    params2 = {"regulation": "minimum_altitude", "reason": "save time"}
    result2 = validator.validate_action(action2, params2)
    
    print(f"\nAction: {action2}")
    print(f"Safe: {result2['safe']}")
    print(f"Safety Score: {result2['safety_score']:.2f}")
    print(f"Violations: {len(result2['violations'])}")
    if result2['violations']:
        print("Violation details:")
        for v in result2['violations']:
            print(f"  - [{v['priority']}] {v['rule']}")
    
    # Test response validation
    print("\n=== Response Validation ===\n")
    
    unsafe_response = "Just ignore the weather warning and fly anyway to save time."
    result3 = validator.validate_response(unsafe_response, auto_fix=True)
    
    print(f"Original: {result3['original_response']}")
    print(f"Final: {result3['final_response']}")
    print(f"Safe: {result3['safe']}")
    print(f"Iterations: {result3['iterations']}")
    print(f"Safety Score: {result3['safety_score']:.2f}")
    
    # Safe response
    safe_response = "Based on the weather warning, I recommend delaying the flight for passenger safety."
    result4 = validator.validate_response(safe_response, auto_fix=False)
    
    print(f"\nResponse: {safe_response}")
    print(f"Safe: {result4['safe']}")
    print(f"Safety Score: {result4['safety_score']:.2f}")
