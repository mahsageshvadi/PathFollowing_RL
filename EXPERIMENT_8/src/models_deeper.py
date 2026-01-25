#!/usr/bin/env python3
"""
Shared model architectures for DSA RL Experiment.
- AsymmetricActorCritic: Full model for training (Actor + Critic)
- ActorOnly: Lightweight model for inference (Actor only, no Critic)
"""
import torch
import torch.nn as nn

# ---------- HELPERS ----------
def gn(c): 
    """GroupNorm helper"""
    return nn.GroupNorm(4, c)

# ---------- FULL MODEL (TRAINING) ----------
class AsymmetricActorCritic(nn.Module):
    """
    Deeper asymmetric Actor-Critic with multi-layer LSTMs
    """
    def __init__(self, n_actions=9, K=16):
        super().__init__()

        # ---------- ACTOR CNN (deeper) ----------
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),

            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),

            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ---------- ACTOR LSTM (longer memory) ----------
        self.actor_lstm = nn.LSTM(
            input_size=n_actions,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.actor_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, n_actions)
        )

        # ---------- CRITIC CNN (deeper + privileged) ----------
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),

            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),

            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ---------- CRITIC LSTM ----------
        self.critic_lstm = nn.LSTM(
            input_size=n_actions,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

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

        # ----- ACTOR -----
        feat_a = self.actor_cnn(actor_obs).flatten(1)   # (B, 128)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1]], dim=1)
        logits = self.actor_head(joint_a)

        # ----- CRITIC -----
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)  # (B, 128)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        joint_c = torch.cat([feat_c, lstm_c[:, -1]], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)

        return logits, value, hc_actor, hc_critic

# ---------- ACTOR-ONLY MODEL (INFERENCE) ----------
# ---------- ACTOR-ONLY MODEL (INFERENCE) ----------
class ActorOnly(nn.Module):
    """
    Actor-only model for inference.
    Architecture EXACTLY matches the actor part of AsymmetricActorCritic.
    """
    def __init__(self, n_actions=9, K=16):
        super().__init__()

        # ---------- ACTOR CNN (same as training actor) ----------
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),

            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),

            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ---------- ACTOR LSTM (longer memory) ----------
        self.actor_lstm = nn.LSTM(
            input_size=n_actions,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # ---------- ACTOR HEAD ----------
        self.actor_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, n_actions)
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

        Args:
            actor_obs: (B, 4, 33, 33)
            ahist_onehot: (B, K, n_actions)
            hc_actor: optional LSTM hidden state

        Returns:
            logits: (B, n_actions)
            hc_actor: updated hidden state
        """
        feat = self.actor_cnn(actor_obs).flatten(1)   # (B, 128)
        lstm_out, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint = torch.cat([feat, lstm_out[:, -1]], dim=1)
        logits = self.actor_head(joint)

        return logits, hc_actor
