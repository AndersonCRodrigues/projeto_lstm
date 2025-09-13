import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_lstm_layers=2,
        num_dense_layers=2,
        dropout=0.3,
        use_attention=False,
        use_batch_norm=True,
        use_layer_norm=False,
        use_bidirectional=False,
        use_gru=False,
        activation="relu",
        attention_heads=8,
        lstm_dropout=0.0,
        dense_dropout=0.3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_bidirectional = use_bidirectional
        self.use_gru = use_gru

        activations = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        self.activation = activations.get(activation, nn.ReLU())

        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=use_bidirectional,
        )

        rnn_output_size = hidden_size * (2 if use_bidirectional else 1)

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_output_size,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )

        if use_batch_norm:
            self.rnn_batch_norm = nn.BatchNorm1d(rnn_output_size)
        if use_layer_norm:
            self.rnn_layer_norm = nn.LayerNorm(rnn_output_size)

        dense_layers_list = []
        current_size = rnn_output_size
        for i in range(num_dense_layers):
            next_size = (
                max(16, int(current_size * 0.5)) if i < num_dense_layers - 1 else 1
            )
            dense_layers_list.append(nn.Linear(current_size, next_size))
            if i < num_dense_layers - 1:
                dense_layers_list.append(self.activation)
                dense_layers_list.append(nn.Dropout(dense_dropout))
            current_size = next_size
        self.dense_layers = nn.Sequential(*dense_layers_list)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        num_directions = 2 if self.use_bidirectional else 1
        h0 = torch.zeros(
            self.num_lstm_layers * num_directions, batch_size, self.hidden_size
        ).to(device)

        if self.use_gru:
            rnn_out, _ = self.rnn(x, h0)
        else:
            c0 = torch.zeros(
                self.num_lstm_layers * num_directions, batch_size, self.hidden_size
            ).to(device)
            rnn_out, _ = self.rnn(x, (h0, c0))

        if self.use_attention:
            output, _ = self.attention(rnn_out, rnn_out, rnn_out)
            output = output[:, -1, :]
        else:
            output = rnn_out[:, -1, :]

        if self.use_batch_norm and output.size(0) > 1:
            output = self.rnn_batch_norm(output)
        if self.use_layer_norm:
            output = self.rnn_layer_norm(output)

        output = self.dense_layers(output)
        return output
