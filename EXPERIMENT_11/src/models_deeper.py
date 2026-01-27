#!/usr/bin/env python3
"""
Shared model architectures for DSA RL Experiment.
- AsymmetricActorCritic: Full model for training (Actor + Critic + Junction Head)
- ActorOnly: Lightweight model for inference (Actor + Junction Head)
"""
import torch
import torch.nn as nn

# ---------- HELPERS ----------
def gn(c): 
    """GroupNorm helper: groups=4"""
    return nn.GroupNorm(4, c)

# ---------- FULL MODEL (TRAINING) ----------
class AsymmetricActorCritic(nn.Module):
    """
    Deeper asymmetric Actor-Critic with multi-layer LSTMs.
    Now includes a 'Junction Head' to detect bifurcations.
    """
    def __init__(self, n_actions=9, K=16):
        super().__init__()

        # =====================================================================
        # ACTOR BRANCH (Shared by Policy and Junction Detection)
        # =====================================================================
        
        # 1. Visual Encoder (Deep CNN)
        # Input: (B, 4, 33, 33) -> Output: (B, 128)
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),

            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),

            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2. Sequential Memory (LSTM)
        # Input: Action History -> Output: (B, 128)
        self.actor_lstm = nn.LSTM(
            input_size=n_actions,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # 3. Policy Head (Movement)
        # Input: 256 features (128 CNN + 128 LSTM)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, n_actions)
        )

        # 4. Junction Head (Structure Detection) - NEW
        # Input: 256 features (128 CNN + 128 LSTM)
        # Output: 1 logit (Binary Classification: Junction vs No Junction)
        self.junction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

        # =====================================================================
        # CRITIC BRANCH (Privileged Information)
        # =====================================================================
        
        # 1. Visual Encoder (Input includes Ground Truth)
        # Input: (B, 5, 33, 33) -> Output: (B, 128)
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),

            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),

            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2. Sequential Memory
        self.critic_lstm = nn.LSTM(
            input_size=n_actions,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # 3. Value Head
        self.critic_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, actor_obs, critic_gt, ahist_onehot,
                hc_actor=None, hc_critic=None):
        """
        Returns:
            logits: Action probabilities (B, n_actions)
            value: State value (B,)
            junction_pred: Junction logit (B,) - NEW
            hc_actor: Actor hidden states
            hc_critic: Critic hidden states
        """
        # ----- ACTOR FLOW -----
        feat_a = self.actor_cnn(actor_obs).flatten(1)   # (B, 128)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        
        # Fuse Visual + History
        joint_a = torch.cat([feat_a, lstm_a[:, -1]], dim=1) # (B, 256)
        
        # Heads
        logits = self.actor_head(joint_a)
        junction_pred = self.junction_head(joint_a).squeeze(-1)

        # ----- CRITIC FLOW -----
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)  # (B, 128)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        
        # Fuse Visual + History
        joint_c = torch.cat([feat_c, lstm_c[:, -1]], dim=1) # (B, 256)
        
        value = self.critic_head(joint_c).squeeze(-1)

        return logits, value, junction_pred, hc_actor, hc_critic

# ---------- ACTOR-ONLY MODEL (INFERENCE) ----------
class ActorOnly(nn.Module):
    """
    Actor-only model for inference.
    Matches the Actor branch of AsymmetricActorCritic exactly.
    Used during the Tree Construction algorithm.
    """
    def __init__(self, n_actions=9, K=16):
        super().__init__()

        # Same CNN as training
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),

            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),

            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Same LSTM as training
        self.actor_lstm = nn.LSTM(
            input_size=n_actions,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # Movement Head
        self.actor_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, n_actions)
        )

        # Junction Head - NEW
        self.junction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, actor_obs, ahist_onehot, hc_actor=None):
        """
        Forward pass for inference.

        Returns:
            logits: (B, n_actions)
            junction_pred: (B,) - NEW
            hc_actor: updated hidden state
        """
        feat = self.actor_cnn(actor_obs).flatten(1)   # (B, 128)
        lstm_out, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        
        joint = torch.cat([feat, lstm_out[:, -1]], dim=1)
        
        logits = self.actor_head(joint)
        junction_pred = self.junction_head(joint).squeeze(-1)

        return logits, junction_pred, hc_actor