import jittor as jt
import jittor.nn as jnn

class JTResBlock(jnn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = jnn.Linear(hidden_size, hidden_size)
        self.linear.weight.stop_grad()
        self.act = jnn.SiLU()

    def execute(self, x):
        return x + self.act(self.linear(x))


class JTMedusaHead(jnn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers):
        super().__init__()
        blocks = [JTResBlock(hidden_size) for _ in range(num_layers)]
        self.net = jnn.Sequential(
            *blocks,
            jnn.Linear(hidden_size, vocab_size, bias=False),
        )

    def execute(self, x):
        return self.net(x)
