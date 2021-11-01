"""
Two one-hidden-layer baseline models are implemented here: 
- Fully Connected Neural Network (FCN)
- Graph Convolutional Neural Network (GCN)

For comparison, in the training process, both models take one batch per epoch, 
and that batch contains the entire dataset. 
"""

# load packages 
import numpy as np
import torch
from torch import nn, optim

# ============================
# ---------- FCN -------------
# ============================

class FCN(nn.Module):
    """ a one hidden layer fully connected network (fcn) """

    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs['input_shape']
        self.output_shape = kwargs['output_shape']
        self.hidden_layer_dim = kwargs['hidden_layer_dim']
        # layer construction
        self.hidden_layer = nn.Linear(
            in_features=self.input_shape,
            out_features=self.hidden_layer_dim,
            bias=False
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_layer_dim,
            out_features=self.output_shape,
            bias=False
        )

    def forward(self, features):
        """ forward propagation """
        hidden_features = self.hidden_layer(features)
        hidden_features_activation = torch.relu(hidden_features)
        output_logits = self.output_layer(hidden_features_activation)
        return output_logits


# ============================
# ---------- GCN -------------
# ============================

class GCN(FCN):
    """ a one hidden layer graph convolution neural network (gcn) """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_matrix = kwargs['adjacency_matrix']
        assert self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1]
        self.graph_based_weight = self.compute_graph_based_weight(
            self.adjacency_matrix)

    def compute_graph_based_weight(self, adjacency_matrix):
        """ 
        pre-process the graph and find A_hat, as it stays constant in all layers 
        :param adjacency_matrix: the adjacency matrix for the input graph 
        :return A_hat, defined in equation (9) in https://arxiv.org/pdf/1609.02907.pdf
        """
        num_nodes = adjacency_matrix.shape[0]
        A_tilde = adjacency_matrix + np.identity(num_nodes)
        degree_tilde = A_tilde.sum(axis=1)
        degree_tilde_half = 1 / np.sqrt(degree_tilde)
        D_tilde_half = np.diag(degree_tilde_half)
        A_hat = D_tilde_half @ A_tilde @ D_tilde_half
        A_hat = torch.from_numpy(A_hat).type(torch.float32)  # crucial to convert to float32
        return A_hat

    def forward(self, features):
        """ forward propagation, now apply the weights from graph structure """
        hidden_features = self.hidden_layer(self.graph_based_weight.matmul(features))
        hidden_features_activation = torch.relu(hidden_features)
        output_logits = self.output_layer(self.graph_based_weight.matmul(hidden_features_activation))
        return output_logits


# =======================
# ----- auxiliary -------
# =======================

def train_loop(model, features, labels, loss_fn, max_epoch=100, learning_rate=1, is_classification=True):
    """ 
    the train loop: forward and backward pass, and print the loss after each epoch 
    for this file in particular, each epoch contains only one batch. 
    :param model: FCN or GCN 
    :param features: the batch features 
    :param labels: the batch labels
    :param loss_fn: the loss function 
    :param is_classification: a bool, True if the model is for classification, otherwise regression
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(max_epoch):
        # forward
        optimizer.zero_grad()
        logits = model(features)
        raw_pred = logits[0]
        # backward
        train_loss = loss_fn(raw_pred, labels)
        train_loss.backward()
        optimizer.step()

        # report loss and accuracy each couple of iterations
        if (epoch + 1) % 10 == 0:
            # compute and report loss
            loss = train_loss / len(labels)
            if is_classification:  # include accuracy report for classification
                accuracy = (raw_pred.argmax(1) == labels).type(torch.float32).mean()
                print(f"epoch: {epoch + 1}/{max_epoch}, loss = {loss:.6f}, accuracy: {accuracy * 100:.2f}%")
            else:                  
                print(f"epoch: {epoch + 1}/{max_epoch}, loss={loss:.6f}")
