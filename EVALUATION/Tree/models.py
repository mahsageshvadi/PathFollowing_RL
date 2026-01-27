import torch
import torch.nn as nn

def gn(c): return nn.GroupNorm(4, c)

class DecoupledStopActorOnly(nn.Module):
    def __init__(self, n_movement_actions=8, K=8):
        super().__init__()
        # Movement Backbone (Matches your trained Decoupled model)
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.actor_lstm = nn.LSTM(input_size=n_movement_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(
            nn.Linear(128, 128), nn.PReLU(), 
            nn.Linear(128, n_movement_actions)
        )
        
        # Stop Backbone (Matches your trained Decoupled model)
        self.stop_cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.stop_head = nn.Sequential(
            nn.Linear(128, 64), nn.PReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, actor_obs, ahist_onehot):
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, _ = self.actor_lstm(ahist_onehot)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        move_logits = self.actor_head(joint_a)
        
        stop_input = torch.cat([actor_obs[:, 0:1, :, :], actor_obs[:, 3:4, :, :]], dim=1)
        feat_stop = self.stop_cnn(stop_input).flatten(1)
        joint_stop = torch.cat([feat_stop, lstm_a[:, -1, :]], dim=1)
        stop_logit = self.stop_head(joint_stop).squeeze(-1)
        
        return move_logits, stop_logit, None