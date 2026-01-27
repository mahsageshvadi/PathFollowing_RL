import torch
import torch.nn as nn

def gn(c): return nn.GroupNorm(4, c)

class AsymmetricActorCritic(nn.Module):
    def __init__(self, n_actions=8):
        super().__init__()
        # Deeper Visual Encoder (5 layers, matches your Stage 7 model)
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),   # <<< FIX HERE
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=128, num_layers=2, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(256, 256), nn.PReLU(), nn.Linear(256, n_actions))
        self.junction_head = nn.Sequential(nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 1))

    def forward(self, actor_obs, critic_gt, ahist_onehot):
        feat = self.actor_cnn(actor_obs).flatten(1)
        lstm_out, _ = self.actor_lstm(ahist_onehot)
        joint = torch.cat([feat, lstm_out[:, -1]], dim=1)
        return self.actor_head(joint), None, self.junction_head(joint).squeeze(-1), None, None