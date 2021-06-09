from torch import nn


class ResNet(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()

        self.l1 = nn.Linear(28 * 28, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.final = nn.Linear(hidden_size, 10)
        self.do = nn.Dropout(dropout_rate)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h1 + h2)
        h3 = nn.functional.relu(self.l3(do))
        do = self.do(h2 + h3)
        logits = self.final(do)
        return logits
