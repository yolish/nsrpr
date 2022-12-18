import torch.nn as nn
import torch
import torch.nn.functional as F


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)


class NSRPR(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_dim = config.get("input_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation='gelu', batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer,
                                                               num_layers=config.get("num_decoder_layers"),
                                                               norm=nn.LayerNorm(d_model))

        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.cls1 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.cls2 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))

        self.rel_regressor_x = PoseRegressor(d_model, 3)
        self.rel_regressor_q = PoseRegressor(d_model, 4)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, query, knn):

        knn = self.proj(knn)
        query = self.proj(query)

        # apply first classifier on knn
        knn_distr_before = self.cls1(knn)

        # apply decoder
        out = self.ln(self.transformer_decoder(knn, query.unsqueeze(1)))

        # apply second classifier on decoders outputs
        knn_distr_after = self.cls2(out)

        # apply regressors
        returned_value = {}
        num_neighbors = knn.shape[1]
        for i in range(num_neighbors):
            rel_x = self.rel_regressor_x(out[:, i, :])
            rel_q = self.rel_regressor_q(out[:, i, :])
            returned_value["rel_pose_{}".format(i)] = torch.cat((rel_x, rel_q), dim=1)
        returned_value["knn_distr_before"] = knn_distr_before
        returned_value["knn_distr_after"] = knn_distr_after

        # return the relative poses and the log-softmax from the first and second classifier
        return returned_value

class NS2RPR(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_dim = config.get("input_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation='gelu', batch_first=True)

        self.transformer_decoder_rot = nn.TransformerDecoder(transformer_decoder_layer,
                                                               num_layers=config.get("num_decoder_layers"),
                                                               norm=nn.LayerNorm(d_model))

        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.cls1 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.cls2 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))

        self.rel_regressor_x = PoseRegressor(d_model, 3)
        self.rel_regressor_q = PoseRegressor(d_model, 4)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, query, knn):

        knn = self.proj(knn)
        query = self.proj(query)

        # apply first classifier on knn
        knn_distr_before = self.cls1(knn)

        # apply decoder
        out = self.ln(self.transformer_decoder(knn, query.unsqueeze(1)))

        # apply second classifier on decoders outputs
        knn_distr_after = self.cls2(out)

        # apply regressors
        returned_value = {}
        num_neighbors = knn.shape[1]
        for i in range(num_neighbors):
            rel_x = self.rel_regressor_x(out[:, i, :])
            rel_q = self.rel_regressor_q(out[:, i, :])
            returned_value["rel_pose_{}".format(i)] = torch.cat((rel_x, rel_q), dim=1)
        returned_value["knn_distr_before"] = knn_distr_before
        returned_value["knn_distr_after"] = knn_distr_after

        # return the relative poses and the log-softmax from the first and second classifier
        return returned_value










