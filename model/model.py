import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import dgl
import torchvision.transforms.functional as TF

from model.encoder.encoder import Encoder

from .graphsage.graph_sage_net import GraphSageNet
from .sfcn.sfcn_model import SpixelNet, SpixelNet1l_bn
from .decoder.decoder import Decoder

import util.util as util

class OursNet(nn.Module):

    def __init__(
            self,
            sfcn_model: nn.Module=None,
            encoder_model: nn.Module=None,
            gnn_model: nn.Module=None,
            decoder_model: nn.Module=None,
            # hard_gnn_output: bool=False,
            sfcn_cell_size: int=16,
            use_decoder: bool=True,
            random_drop_edges: float = 0.,
            n_classes: int=21,
            device: torch.device=None):
        super(OursNet, self).__init__()
        self.device = device if device is not None else torch.device('cuda')
        self.sfcn = sfcn_model if sfcn_model is not None else SpixelNet1l_bn()
        self.encoder = encoder_model if encoder_model is not None else Encoder()
        self.gnn = gnn_model if gnn_model is not None else GraphSageNet(device=self.device)
        self.decoder = decoder_model if decoder_model is not None else Decoder()
        # self.hard_gnn_output = hard_gnn_output
        self.sfcn_cell_size = sfcn_cell_size
        self.use_decoder = use_decoder
        self.random_drop_edges = random_drop_edges

        self.spixl_pred_map = nn.Linear(self.gnn.out_dim, n_classes, True, device)

    def forward(self, X: torch.Tensor, Xrgb: torch.Tensor):
        """
        Model with SFCN and GNN (curently GraphSage copied from Superpixel GCN)
        Args:
          X: [B, Channels, H, W] input images with channels (L, A, B, x, y) (or rgb)
        Returns:
          y_hat: [B, Classes, H, W] Classes for VOC2012_test would be 21
        """
        
        B, _, H, W = X.shape
        C = util.add_coordinate_channels(Xrgb) # [B, Channels+2, H, W], all channels in [0, 1]
        C_pos = C[:, 3:, :, :]
        C_pos = C_pos / C_pos.std()
        # C_lab = TF.normalize(
        #     C[:, :3, :, :],
        #     mean=[0.39523883, 0.50547228, 0.52167666],
        #     std=[0.29642959, 0.0373407,  0.05966596])
        # C_lab = TF.normalize(
        #     C[:, :3, :, :],
        #     mean=[0.3413, 0.4789, 0.5199],
        #     std=[0.1957, 0.0119, 0.0187])

        # # feature encoder
        # C_embed = self.encoder(C_lab) # [B, Feat, H, W]

        C_rgb = TF.normalize(
            C[:, :3, :, :],
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        # feature encoder
        C_embed = self.encoder(C_rgb) # [B, Feat, H, W]

        Q = self.sfcn(X) # [B, 9, H, W]

        # Convert to dgl batch
        C_concat = torch.cat([C_embed, C_pos], dim=1) # [B, Feat + 2, H, W]
        Graph, G = util.spixl_to_dgl_batch(
            Q, C_concat,
            cell_size=self.sfcn_cell_size,
            batch_unique_id=False,
            drop_edge_prob=self.random_drop_edges,
            device=self.device) # dgl Batch

        # call gnn
        Feat_node = self.gnn(Graph, Graph.ndata['feature']) # [N, C]
        Logit_node = self.spixl_pred_map(Feat_node) # [N, Classes]

        Feat_spixl = util.dgl_batch_to_pixel_pred(
            Graph, Feat_node, Q, self.sfcn_cell_size, False,
            device=self.device) # [B, C, H, W]
        Logit_spixl = util.dgl_batch_to_pixel_pred(
            Graph, Logit_node, Q, self.sfcn_cell_size, False,
            device=self.device) # [B, Classes, H, W]

        # call decoder
        # if self.use_decoder:
        Logit = self.decoder(C_embed, Feat_spixl) # [B, Classes, H, W]
        # else:
        #     Logit = Logit_spixl
        return Logit.softmax(1), Q, C, G, Logit_node, Logit_spixl, Logit, Graph

def get_model(
        config: object, 
        device: torch.device=torch.device('cuda')) -> tuple[OursNet, SpixelNet, GraphSageNet, Decoder]:
    image_size = [config['image_height'], config['image_width']]
    image_channel = config['image_channel']
    cell_size = config['cell_size']
    assert image_size[0] % cell_size == 0
    assert image_size[1] % cell_size == 0
    # hard_gnn_output = config['hard_gnn_output']
    use_decoder = config['use_decoder']
    n_classes = config['n_classes']

    # sfcn
    sfcn_use_pretrain_weight = config['sfcn']['use_pretrain_weight']
    sfcn_freeze_weight = config['sfcn']['freeze_weight']
    if sfcn_freeze_weight:
        assert sfcn_use_pretrain_weight == True

    # graph_sage_net
    gnn_in_dim = config['gnn']['in_dim']
    # gnn_hidden_dims = config['gnn']['hidden_dims']
    gnn_out_dim = config['gnn']['out_dim']
    # gnn_n_classes = config['gnn']['n_classes']
    gnn_in_feat_dropout = config['gnn']['in_feat_dropout']
    gnn_dropout = config['gnn']['dropout']
    gnn_aggregator_type = config['gnn']['aggregator_type']
    gnn_batch_norm = config['gnn']['batch_norm']
    gnn_residual = config['gnn']['residual']
    gnn_dgl_builtin = config['gnn']['dgl_builtin']
    gnn_tag_kernel = config['gnn']['tag_kernel']
    gnn_conv_type = config['gnn']['conv_type']
    gnn_readout = config['gnn']['readout']
    gnn_random_drop_edges = config['gnn'].get('random_drop_edges', 0.)

    # encoder
    try:
        encoder_out_dim = config['encoder']['out_dim']
    except:
        encoder_out_dim = 3
    assert gnn_in_dim == encoder_out_dim + 2 # currently no embedding

    # decoder
    decoder_fp_channels = config['decoder']['fp_channels']
    decoder_fsp_channels = config['decoder']['fsp_channels']
    decoder_hidden_channels = config['decoder']['hidden_channels']
    # decoder_use_se = config['decoder']['use_se']
    # decoder_se_reduction = config['decoder']['se_reduction']
    decoder_dropout = config['decoder']['dropout']
    decoder_use_gn = config['decoder']['use_gn']

    sfcn_model = None
    if sfcn_use_pretrain_weight:
        assert cell_size == 16
        data = torch.load("data/input/weight/SpixelNet_bsd_ckpt.tar", map_location=device, weights_only=True)
        sfcn_model = SpixelNet1l_bn(data)
    else:
        sfcn_model = SpixelNet1l_bn()

    if sfcn_freeze_weight:
        for param in sfcn_model.parameters():
            param.requires_grad = False

    encoder_model = Encoder(
        in_channels=image_channel,
        out_channels=encoder_out_dim,
    )
    
    gnn_model = GraphSageNet(
        gnn_in_dim,
        # gnn_hidden_dims,
        gnn_out_dim,
        # gnn_n_classes,
        gnn_in_feat_dropout,
        gnn_dropout,
        gnn_aggregator_type,
        gnn_batch_norm,
        gnn_residual,
        gnn_dgl_builtin,
        gnn_tag_kernel,
        gnn_conv_type,
        gnn_readout,
        device)
    
    decoder_model = Decoder(
        decoder_fp_channels,
        decoder_fsp_channels,
        decoder_hidden_channels,
        n_classes,
        # decoder_use_se,
        # decoder_se_reduction,
        decoder_dropout,
        decoder_use_gn)
    
    model = OursNet(
        sfcn_model,
        encoder_model,
        gnn_model,
        decoder_model,
        # hard_gnn_output,
        cell_size,
        use_decoder,
        gnn_random_drop_edges,
        n_classes,
        device)
    
    return model, sfcn_model, encoder_model, gnn_model, decoder_model
