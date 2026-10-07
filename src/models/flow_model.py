import torch
from torch import nn

from models import ipa_pytorch
from models.edge_feature_net import EdgeFeatureNet
from models.node_feature_net import NodeFeatureNet
from utils import data_utils as du


class FlowModel(nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        if self._model_conf.aatype_pred:
            node_embed_size = self._model_conf.node_embed_size
            self.aatype_pred_net = nn.Sequential(
                nn.LayerNorm(node_embed_size),
                nn.Linear(node_embed_size, node_embed_size),
                nn.SiLU(),
                nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens),
            )
            self.seq_update_norm = nn.LayerNorm(node_embed_size)
            self.seq_update_scale = nn.Parameter(torch.full((self._ipa_conf.num_blocks,), 0.1))

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=self._model_conf.transformer_dropout,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = nn.TransformerEncoder(
                tfmr_layer,
                self._ipa_conf.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f"bb_update_{b}"] = ipa_pytorch.BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.torsion_pred = ipa_pytorch.TorsionAngles(self._ipa_conf.c_s, 9)

    def forward(self, input_feats):
        node_mask = input_feats["res_mask"]
        frame_mask = input_feats.get("frame_mask", node_mask)
        edge_mask = frame_mask[:, None] * frame_mask[:, :, None]
        diffuse_mask = input_feats["diffuse_mask"]
        aatype_diffuse_mask = input_feats.get("aatype_diffuse_mask", diffuse_mask)
        res_index = input_feats["res_idx"]
        so3_t = input_feats["so3_t"]
        r3_t = input_feats["r3_t"]
        cat_t = input_feats["cat_t"]
        trans_t = input_feats["trans_t"]
        rotmats_t = input_feats["rotmats_t"]
        aatypes_t = input_feats["aatypes_t"].long()
        trans_sc = input_feats["trans_sc"]
        aatypes_sc = input_feats["aatypes_sc"]

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            res_mask=node_mask,
            diffuse_mask=diffuse_mask,
            frame_mask=frame_mask,
            pos=res_index,
            aatypes=aatypes_t,
            aatypes_sc=aatypes_sc,
        )

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # Initial rigids
        init_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        pred_logits = None
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, curr_rigids, frame_mask)
            ipa_embed *= frame_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]

            if self._model_conf.aatype_pred:
                pred_logits = self.aatype_pred_net(node_embed)
                pred_logits_clean = pred_logits.clone()
                pred_logits_clean[..., du.UNK_TOKEN_INDEX] = -1e9
                pred_logits_clean[..., du.MASK_TOKEN_INDEX] = -1e9
                pred_probs = nn.functional.softmax(pred_logits_clean, dim=-1)
                known_probs = nn.functional.one_hot(
                    aatypes_t,
                    num_classes=self._model_conf.aatype_pred_num_tokens,
                ).to(pred_probs.dtype)
                seq_probs = torch.where(
                    aatype_diffuse_mask[..., None].bool(),
                    pred_probs,
                    known_probs,
                )
                seq_embed = torch.matmul(
                    seq_probs,
                    self.node_feature_net.aatype_embedding.weight,
                )
                node_embed = self.seq_update_norm(node_embed + self.seq_update_scale[b] * seq_embed)
                node_embed = node_embed * node_mask[..., None]

            rigid_update = self.trunk[f"bb_update_{b}"](node_embed)

            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        _, pred_torsions = self.torsion_pred(node_embed, init_node_embed)

        if self._model_conf.aatype_pred:
            pred_logits_clean = pred_logits.clone()
            pred_logits_clean[:, :, du.UNK_TOKEN_INDEX] = -1e9
            pred_logits_clean[:, :, du.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes = torch.argmax(pred_logits_clean, dim=-1)
        else:
            pred_aatypes = aatypes_t
            pred_logits = nn.functional.one_hot(
                pred_aatypes, num_classes=self._model_conf.aatype_pred_num_tokens
            ).float()

        return {
            "pred_torsions": pred_torsions,
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
            "pred_logits": pred_logits,
            "pred_aatypes": pred_aatypes,
            "node_embed": node_embed,
        }
