import torch
import torch.nn as nn

def gn(c): 
    return nn.GroupNorm(4, c)

class AsymmetricActorCritic(nn.Module):
    """
    Input Channels (5):
    0: Current Local Crop (33x33)
    1: Prev Pos 1 (33x33)
    2: Prev Pos 2 (33x33)
    3: Path Mask (Memory)
    4: Wide-Angle Crop (65x65 resized to 33x33)
    """
    def __init__(self, n_actions=9, K=16):
        super().__init__()

        # --- SHARED VISUAL BACKBONE ---
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=128, num_layers=2, batch_first=True)

        # --- HEADS ---
        # 1. Policy (Movement)
        self.actor_head = nn.Sequential(nn.Linear(256, 256), nn.PReLU(), nn.Linear(256, n_actions))
        
        # 2. Junction Head (Soft Gaussian classification)
        self.junction_head = nn.Sequential(nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 1))
        
        # 3. Stop Head (Terminal Tip detection)
        self.stop_head = nn.Sequential(nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 1))

        # --- CRITIC (Privileged Info) ---
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), gn(32), nn.PReLU(), # 5 channels + 1 GT mask
            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.critic_head = nn.Sequential(nn.Linear(192, 128), nn.PReLU(), nn.Linear(128, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))

    def forward(self, actor_obs, critic_gt, ahist_onehot):
        # Actor
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, _ = self.actor_lstm(ahist_onehot)
        joint_a = torch.cat([feat_a, lstm_a[:, -1]], dim=1)

        logits = self.actor_head(joint_a)
        junc_pred = self.junction_head(joint_a).squeeze(-1)
        stop_pred = self.stop_head(joint_a).squeeze(-1)

        # Critic
        critic_in = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_in).flatten(1)
        joint_c = torch.cat([feat_c, lstm_a[:, -1]], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)

        return logits, value, junc_pred, stop_pred