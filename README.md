import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is not None:
            output = self.transformer_encoder(src, src_mask)
        else:
            output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Hyperparameters
vocab_size = 10000  # Size of the vocabulary
d_model = 512       # Embedding size
nhead = 8           # Number of attention heads
dim_feedforward = 2048  # Feedforward network hidden size
num_layers = 6      # Number of Transformer layers

# Instantiate the model
model = TransformerModel(vocab_size, d_model, nhead, dim_feedforward, num_layers)

# Example input (batch_size, sequence_length)
sample_input = torch.randint(0, vocab_size, (32, 100))  # Random integers as input

# Forward pass
output = model(sample_input)

print("Output shape:", output.shape)  # Should be (batch_size, sequence_length, vocab_size)<p align="center">
  <a href="https://probot.github.io"><img src="/static/robot.svg" width="160" alt="Probot's logo, a cartoon robot" /></a>
</p>
<h3 align="center"><a href="https://probot.github.io">Probot</a></h3>
<p align="center">A framework for building GitHub Apps to automate and improve your workflow<p>
<p align="center"><a href="https://npmjs.com/package/probot"><img src="https://badgen.net/npm/v/probot" alt="npm"></a> <a href="https://github.com/probot/probot/actions?query=workflow%3ACI"><img src="https://github.com/probot/probot/workflows/CI/badge.svg" alt="Build Status"></a> <a href="https://codecov.io/gh/probot/probot/"><img src="https://badgen.now.sh/codecov/c/github/probot/probot" alt="Codecov"></a> <a href="https://twitter.com/ProbotTheRobot"><img src="https://img.shields.io/twitter/follow/ProbotTheRobot.svg?style=social&logo=twitter&label=Follow" alt="@ProbotTheRobot on Twitter"></a>

---

If you've ever thought, "wouldn't it be cool if GitHub could…"; I'm going to stop you right there. Most features can actually be added via [GitHub Apps](https://docs.github.com/en/developers/apps), which extend GitHub and can be installed directly on organizations and user accounts and granted access to specific repositories. They come with granular permissions and built-in webhooks. Apps are first class actors within GitHub.

## How it works

**Probot is a framework for building [GitHub Apps](https://docs.github.com/en/developers/apps) in [Node.js](https://nodejs.org/)**, written in [TypeScript](https://www.typescriptlang.org/). GitHub Apps can listen to webhook events sent by a repository or organization. Probot uses its internal event emitter to perform actions based on those events. A simple Probot App might look like this:

```js
export default (app) => {
  app.on("issues.opened", async (context) => {
    const issueComment = context.issue({
      body: "Thanks for opening this issue!",
    });
    return context.octokit.issues.createComment(issueComment);
  });

  app.onAny(async (context) => {
    context.log.info({ event: context.name, action: context.payload.action });
  });

  app.onError(async (error) => {
    app.log.error(error);
  });
};
```

## Building a Probot App

If you've landed in this GitHub repository and are looking to start building your own Probot App, look no further than [probot.github.io](https://probot.github.io/docs/)! The Probot website contains our extensive getting started documentation and will guide you through the set up process.

This repository hosts the code for the npm Probot package which is what all Probot Apps run on. Most folks who land in this repository are likely looking to get started [building their own app](https://probot.github.io/docs/).

## Contributing

Probot is built by people just like you! Most of the interesting things are built _with_ Probot, so consider starting by [writing a new app](https://probot.github.io/docs/) or improving one of the [existing ones](https://github.com/search?q=topic%3Aprobot-app&type=Repositories).

If you're interested in contributing to Probot itself, check out our [contributing docs](CONTRIBUTING.md) to get started.

Want to discuss with Probot users and contributors? [Discuss on GitHub](https://github.com/probot/probot/discussions)!

## Ideas

Have an idea for a cool new GitHub App (built with Probot)? That's great! If you want feedback, help, or just to share it with the world you can do so by [creating an issue in the `probot/ideas` repository](https://github.com/probot/ideas/issues/new)!
