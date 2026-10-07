import torch
from torch import nn

from models.modules import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):
    def __init__(self, module_cfg):
        super().__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.max_positions = 2056
        self.embed_aatype = self._cfg.embed_aatype

        embed_size = self.c_pos_emb + self.c_timestep_emb * 2 + 2

        if self.embed_aatype:
            # [A, G, C, U, UNK, MASK]. UNK is observed missing identity;
            # MASK is the categorical flow-noise state.
            self.aatype_embedding = nn.Embedding(self._cfg.aatype_pred_num_tokens, self.c_s)
            embed_size += self.c_s + self._cfg.c_timestep_emb + self._cfg.aatype_pred_num_tokens

        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, seq_len, res_mask_3d):
        timestep_emb = get_time_embedding(timesteps[:, 0], self.c_timestep_emb, max_positions=self.max_positions)
        timestep_emb = timestep_emb[:, None, :].expand(-1, seq_len, -1)
        return timestep_emb * res_mask_3d

    def forward(
        self,
        *,
        so3_t,
        r3_t,
        cat_t,
        res_mask,
        diffuse_mask,
        frame_mask,
        pos,
        aatypes,
        aatypes_sc,
    ):
        seq_len = res_mask.shape[1]
        res_mask_3d = res_mask.unsqueeze(-1)

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=self.max_positions)
        pos_emb = pos_emb * res_mask_3d

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            frame_mask[..., None],
            self.embed_t(so3_t, seq_len, res_mask_3d),
            self.embed_t(r3_t, seq_len, res_mask_3d),
        ]
        if self.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, seq_len, res_mask_3d))
            input_feats.append(aatypes_sc)

        return self.linear(torch.cat(input_feats, dim=-1))
