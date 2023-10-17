from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from loss import ordinalentropy_loss


class Ordinal_entropy(nn.Module):
    """OER adapts a model by entropy minimization during testing.

    Once oered, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"  # if not steps >=0, then trigger error
        self.episodic = episodic
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory

        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure ordinal entropy of the model prediction, take gradients, and update params.
    """
    # forward
    optimizer.zero_grad()
    outputs, features = model(x)
    # outputs as pseudo-label
    outputs = outputs.detach()    # detach the target before computing the loss  https://stackoverflow.com/questions/72590591/the-derivative-for-target-is-not-implemented
    # adapt
    # loss = ordinalentropy_loss(features, outputs)
    loss = torch.nn.L1Loss()
    loss = loss(features, outputs)

    # loss_func = Depth_Loss(1, 1, 1, maxDepth=80)
    # loss = loss_func(features, outputs)
    print("loss: ", loss)
    # print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
    loss.backward()
    optimizer.step()
    # print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
    return outputs

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

