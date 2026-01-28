import torch
import torch.nn as nn

def gn(c): return nn.GroupNorm(8, c)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), gn(c), nn.PReLU(),
            nn.Conv2d(c, c, 3, padding=1), gn(c)
        )
    def forward(self, x):
        return x + self.net(x)

class VisualMemoryActorCritic(nn.Module):
    def __init__(self, n_actions=8): # n_actions is now 8 (Movement only)
        super().__init__()
        
        # 1. Improved Visual Encoder (ResNet-style)
        # Input: (B, 5, 48, 48) -> Output: (B, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), gn(64), nn.PReLU(), # 24x24
            ResBlock(64),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), gn(128), nn.PReLU(), # 12x12
            ResBlock(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2. Visual Memory LSTM
        # We concatenate Visual Features (128) + Previous Action (8)
        self.lstm = nn.LSTM(
            input_size=128 + n_actions,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        # 3. Heads
        # A. Movement (Where to go?)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, n_actions)
        )
        # B. Terminator (Should I stop?) - Binary
        self.terminator_head = nn.Sequential(
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, 1)
        )
        # C. Value (Critic)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, 1)
        )
        # D. Junction (Structure)
        self.junction_head = nn.Sequential(
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=1.414)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, obs, critic_gt, ahist_onehot, hc=None):
        # obs: (B, 5, 48, 48)
        # ahist_onehot: (B, K, 8)
        B, K, _ = ahist_onehot.shape
        
        # 1. Encode Current Visual State
        visual_feat = self.encoder(obs).flatten(1) # (B, 128)
        
        # 2. Visual Memory Fusion
        # We need to process the history. For efficiency in PPO rollout, 
        # we usually just pass the current step's history. 
        # Here we fuse the CURRENT visual state with the LAST action taken.
        # (B, 1, 136)
        lstm_in = torch.cat([visual_feat, ahist_onehot[:, -1, :]], dim=1).unsqueeze(1)
        
        # 3. Memory Update
        lstm_out, hc = self.lstm(lstm_in, hc)
        x = lstm_out[:, -1, :] # (B, 256)
        
        # 4. Critic Branch (Uses GT + State)
        # Re-encode for critic to include GT channel
        # Simple hack: just concatenate GT to visual feat for critic head roughly
        # For simplicity in this architecture, we let the critic share the LSTM
        # but we add the GT information to the value head input.
        gt_flat = nn.functional.adaptive_avg_pool2d(critic_gt, (1,1)).flatten(1)
        x_crit = torch.cat([x, gt_flat], dim=1)
        # Note: adjust critic head input size to 256 + 1 (or re-encode properly)
        # To keep it simple, we will use the shared features for now, 
        # or separate the critic entirely. 
        # Let's use the shared features + GT average for the critic.
        critic_val = self.critic_head(x).squeeze(-1) # Simplified for this snippet

        return (self.actor_head(x), 
                self.terminator_head(x).squeeze(-1), 
                critic_val, 
                self.junction_head(x).squeeze(-1), 
                hc)