# 1404-05-25
# Moha,,ad Kadkhodaei
# ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__() # Initialize the parent nn.Module class
        self.fc1 = nn.Linear(state_dim, 256) # First fully connected layer: state_dim -> hidden_size
        self.fc2 = nn.Linear(256, 256) # Second fully connected layer: hidden_size -> hidden_size
        self.fc3 = nn.Linear(256, action_dim) # Output layer: hidden_size -> action_dim
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) # Output in [-1, 1]
        return x
