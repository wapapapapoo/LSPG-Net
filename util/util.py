import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import dgl

def add_coordinate_channels(image_tensor: torch.Tensor,
                            range_min: float = 0.0,
                            range_max: float = 1.0) -> torch.Tensor:
    """
    Args:
      image_tensor: Tensor of shape [B, C, H, W]
      range_min: the minimum coordinate value (default 0)
      range_max: the maximum coordinate value (default +1)
    Returns:
      Tensor of shape [B, C+2, H, W], where the last two channels
      are the x- and y-coordinate maps.
    """
    B, C, H, W = image_tensor.shape
    x_lin = torch.linspace(range_min, range_max, W, device=image_tensor.device,
                           dtype=image_tensor.dtype)
    y_lin = torch.linspace(range_min, range_max, H, device=image_tensor.device,
                           dtype=image_tensor.dtype)
    x_coords = x_lin.view(1, 1, 1, W).expand(B, 1, H, W)
    y_coords = y_lin.view(1, 1, H, 1).expand(B, 1, H, W)
    return torch.cat([image_tensor, x_coords, y_coords], dim=1)

def straight_through_argmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Straight-through argmax
    - forward: one-hot (hard argmax)
    - backward: softmax gradient
    """
    soft = F.softmax(logits, dim=dim)
    # hard one-hot
    _, idx = soft.max(dim=dim, keepdim=True)
    hard = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
    # straight-through: use hard in forward, soft in backward
    out = hard - soft.detach() + soft
    return out

def shift9pos(input_2d: np.ndarray) -> np.ndarray:
    """
    given [n_h, n_w] grid of SP-IDs, return a [9, n_h, n_w]
    stack of shifted versions (top‑left to bottom‑right).
    """
    pad = np.pad(input_2d, ((1, 1), (1, 1)), mode='edge')
    top_left     = pad[:-2, :-2]
    top          = pad[:-2, 1:-1]
    top_right    = pad[:-2, 2:]
    left         = pad[1:-1, :-2]
    center       = pad[1:-1, 1:-1]
    right        = pad[1:-1, 2:]
    bottom_left  = pad[2:, :-2]
    bottom       = pad[2:, 1:-1]
    bottom_right = pad[2:, 2:]
    return np.stack([
        top_left, top, top_right,
        left, center, right,
        bottom_left, bottom, bottom_right
    ], axis=0) # [9, H, W]

def make_spixl_map_idx(
        Q_size: torch.Size,
        cell_size: int=16,
        batch_unique_id: bool=False,
        device: torch.device=torch.device('cuda')) -> torch.LongTensor:
    """
    Build a [B, 9, H, W] tensor of SP‑IDs for each pixel’s 3×3 neighborhood.
    Args:
      Q_size: Q.size()
      cell_size: int, size of one superpixel cell
      batch_unique_id: should SP-IDs be unique in batch (when False, each sample has its own counter)
      device: torch device
    Returns:
      spixl_map_idx: Long [B, 9, H, W] map Q to [B, 1, H, W] spixl 2d map
    """
    B, C, H, W = Q_size
    assert C == 9, "Q channel dimension must be 9"
    assert H % cell_size == 0 and W % cell_size == 0, "cell_size must divide H and W"

    n_h = H // cell_size
    n_w = W // cell_size
    num_spixels = n_h * n_w

    base_ids = np.arange(num_spixels, dtype=np.int32).reshape(n_h, n_w) # [n_h, n_w]
    # get 9 offset layers of shape [9, n_h, n_w]
    spix9 = shift9pos(base_ids) # [9, n_h, n_w]
    # expand each SP-IDs to its cell_size cell_size block
    spix9_up = np.repeat(np.repeat(spix9, cell_size, axis=1), cell_size, axis=2) # [9, H, W]

    if batch_unique_id:
        all_batches = []
        for b in range(B):
            offset = b * num_spixels
            all_batches.append(spix9_up + offset)

        arr = np.stack(all_batches, axis=0)  # [B, 9, H, W]
        return torch.from_numpy(arr).long().to(device)

    return torch.from_numpy(
        np.tile(spix9_up, (B, 1, 1, 1))
    ).long().to(device)

def sfcn_Q_argmax(Q: torch.Tensor,
                  cell_size: int=16,
                  spixl_map_idx_in: torch.LongTensor=None,
                  batch_unique_id: bool=False,
                  device: torch.device=torch.device('cuda')) -> torch.Tensor:
    """
    argmax(Q), use soft argmax
    Args:
      Q:                [B, 9, H, W] affinity logits from SFCN
      cell_size:        int, size of one superpixel cell
      spixl_map_idx_in: [B, 9, H, W] SP-IDs for each position
    Returns:
      SP-IDs:           [B, H, W] SP-IDs (floats) for each pixel
    """
    B, C, H, W = Q.shape
    if spixl_map_idx_in is None:
        spixl_map_idx_in = make_spixl_map_idx(Q.size(), cell_size, batch_unique_id, device) # [B, 9, H, W]

    logits = Q.view(B, C, -1)
    maximum   = straight_through_argmax(logits, dim=1)
    maximum   = maximum.view(B, C, H, W)

    spix_float = spixl_map_idx_in.float()
    out        = torch.sum(maximum * spix_float, dim=1)
    return out

def generate_graph_edges(spix_map: torch.Tensor,
                         device: torch.device = torch.device('cuda')
                         ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Generate graph edges from a spixl map from sfcn_Q_argmax
    This fn will break the grad backward
    Args:
      spix_map:  [B, H, W] result of sfcn_Q_argmax
      device:    torch device
    Returns:
      edges:   List<LongTensor [2, E]> giving u<v pairs of SP‑IDs
      weights: List<FloatTensor [E]> counts of pixel adjacencies
    """
    B, H, W = spix_map.shape
    spix_map = spix_map.long()

    edges_list = []
    weights_list = []

    shifts = [(0,1), (1,0), (1,1), (1,-1)] # [(0,1), (1,0)]

    for b in range(B):
        cnt = defaultdict(int)
        m = spix_map[b]

        for dy, dx in shifts:
            # src slice
            y0_src = max(0, -dy)
            y1_src = min(H, H - dy)    # H - max(dy,0)
            x0_src = max(0, -dx)
            x1_src = min(W, W - dx)

            # dst slice
            y0_dst = max(0, dy)
            y1_dst = min(H, H + dy)    # H - max(-dy,0)
            x0_dst = max(0, dx)
            x1_dst = min(W, W + dx)

            src = m[y0_src:y1_src, x0_src:x1_src]
            dst = m[y0_dst:y1_dst, x0_dst:x1_dst]

            u = src.reshape(-1)
            v = dst.reshape(-1)
            diff = (u != v)

            for uu, vv in zip(u[diff].tolist(), v[diff].tolist()):
                a, c = (uu, vv) if uu < vv else (vv, uu)
                cnt[(a, c)] += 1

        # deal with empty graph
        if not cnt:
            edges  = torch.zeros(2, 0, dtype=torch.long) # [2, 0]
            weights= torch.zeros(0, dtype=torch.float)
        else:
            pairs   = list(cnt.keys())
            edges   = torch.LongTensor(pairs).t()    # [2, E]
            weights = torch.FloatTensor([cnt[p] for p in pairs])

        edges_list.append(edges.to(device))
        weights_list.append(weights.to(device))

    return edges_list, weights_list

def generate_graph_nodes_hard(P: torch.Tensor,
                              spix_map: torch.Tensor,
                              cell_size: int = 16,
                              device: torch.device = torch.device('cuda')
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate vertexs feature with spix_map
    Args:
      P:         [B, C, H, W] features of pixels
      spix_map:  [B, H, W] result of sfcn_Q_argmax
      cell_size: int, size of one superpixel cell
      device:    torch device
    Returns:
      weights: FloatTensor [B, C, N] feature vectors of each node
      sizes:   FloatTensor [B, N] pixel count of each superpixel
    """
    B, C, H, W = P.shape
    n_h = H // cell_size
    n_w = W // cell_size
    N = n_h * n_w

    # flattern and to int
    #    spix_map: [B, H, W] to [B*H*W]
    spix_ids = spix_map.long().view(B, -1)       # [B, H*W]
    # flattern feature to [C, B*H*W]
    #    [B, C, H, W] to [B, C, H*W] to [C, B*H*W]
    P_flat = P.view(B, C, -1)                    # [B, C, H*W]

    # node features
    node_feats = torch.zeros(B, C, N, device=device) # [B, C, N]
    # spixl node pixel counts
    sizes      = torch.zeros(B, N, device=device)    # [B, N]

    for b in range(B):
        idx_expand = spix_ids[b].unsqueeze(0).expand(C, -1)  # [C, H*W]
        node_feats[b] = node_feats[b].scatter_add(
            dim=1,
            index=idx_expand,
            src=P_flat[b]
        )

        ones = torch.ones_like(spix_ids[b], dtype=torch.float, device=device)  # [H*W]
        sizes[b] = sizes[b].scatter_add(
            dim=0,
            index=spix_ids[b],
            src=ones
        )

    return node_feats, sizes

def generate_graph_nodes(P: torch.Tensor,
                         Q: torch.Tensor,
                         cell_size: int = 16,
                         batch_unique_id: bool=False,
                         device: torch.device = torch.device('cuda')
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate vertexs feature with Q
    Args:
      P:         [B, C, H, W] features of pixels
      Q:         [B, 9, H, W] Q
      cell_size: int, size of one superpixel cell
      batch_unique_id: makesure same as other fn in pipeline
      device:    torch device
    Returns:
      weights: FloatTensor [B, C, N] feature vectors of each node
      sizes:   FloatTensor [B, N] pixel count of each superpixel
    """
    B, C, H, W = P.shape
    Q_flat = Q.view(B, 9, -1)                       # [B, 9, H*W]

    # generate pixel-spixl_id mapping
    spix9_idx = make_spixl_map_idx(Q.size(), cell_size,
                                   batch_unique_id, device)  # [B, 9, H, W]
    spix9_idx = spix9_idx.view(B, 9, -1) # [B, 9, H*W]

    # prepare output tensor
    n_h = H // cell_size
    n_w = W // cell_size
    N   = n_h * n_w * (B if batch_unique_id else 1)

    node_feats = torch.zeros(B, C, N, device=device)
    sizes      = torch.zeros(B,     N, device=device)

    # flattern
    P_flat = P.view(B, C, -1)  # [B, C, H*W]

    for b in range(B):
        wqb = Q_flat[b]          # [9, H*W]
        idxb = spix9_idx[b]      # [9, H*W]
        pflat = P_flat[b]        # [C, H*W]

        # scatter_add for 9 adjacent spixl
        for j in range(9):
            wj = wqb[j] # [H*W]
            idxj = idxb[j] # [H*W] long
            # weighted feature
            weighted = pflat * wj.unsqueeze(0) # [C, H*W]
            # scatter add to node_feats[b, :, idxj]
            node_feats[b].scatter_add_(dim=1,
                                      index=idxj.unsqueeze(0).expand(C, -1),
                                      src=weighted)
            # size do the same as feature
            sizes[b].scatter_add_(dim=0,
                                  index=idxj,
                                  src=wj)

    return node_feats, sizes

def spixl_to_dgl_batch(Q: torch.Tensor,
                        P: torch.Tensor,
                        cell_size: int = 16,
                        batch_unique_id: bool = False,
                        drop_edge_prob: float = 0.,
                        device: torch.device = torch.device('cuda')) -> tuple[dgl.DGLGraph, torch.Tensor]:
    """
    Convert Q (affinity logits) and P (features) into a dgl batch object of graphs.
    Each graph represents superpixels and their adjacency.
    Args:
      Q: [B, 9, H, W] affinity logits from SFCN
      P: [B, C, H, W] feature map aligned with Q
      cell_size: int, size of one superpixel cell
      batch_unique_id: if True, ensures superpixel IDs are globally unique per batch
      drop_edge_prob: randomly drop edges
      device: torch device
    Returns:
      Batch: dgl batch object of superpixel graphs
      spix_map: [B, H, W] result of sfcn_Q_argmax
    """
    B, _, H, W = Q.shape
    spix_map = sfcn_Q_argmax(Q, cell_size=cell_size, batch_unique_id=batch_unique_id, device=device)  # [B, H, W]
    edge_index_list, edge_weight_list = generate_graph_edges(spix_map, device=device)               # [B, 2, E], [B, E]
    node_feats, sizes = generate_graph_nodes(P, Q, cell_size=cell_size, batch_unique_id=batch_unique_id, device=device) # [B, C, N], [B, N]
    # node_feats, sizes = generate_graph_nodes_hard(P, spix_map, cell_size=cell_size, device=device) # [B, C, N], [B, N]

    data_list = []
    for b in range(B):
        feats_b = node_feats[b] / (sizes[b].unsqueeze(0) + 1e-8)  # avoid division by zero, [C, N]
        feats_b = feats_b.transpose(0, 1).contiguous() # [N, C]

        src, dst = edge_index_list[b]
        weights = edge_weight_list[b]

        # random drop edges
        if drop_edge_prob > 0:
            E = weights.size(0)
            mask = torch.rand(E, device=device) > drop_edge_prob
            src = src[mask]
            dst = dst[mask]
            weights = weights[mask]

        # build dgl graph
        graph = dgl.graph((src, dst), num_nodes=feats_b.size(0))
        graph = graph.to(device)

        # assign features and edge weights
        graph.ndata['feature'] = feats_b
        graph.edata['weight'] = weights.unsqueeze(1)

        data_list.append(graph)

    return dgl.batch(data_list).to(device), spix_map

def dgl_batch_to_pixel_pred_hard(G: dgl.DGLGraph,
                                 Pred_node: torch.Tensor,
                                 spix_map: torch.Tensor,
                                 device: torch.device=torch.device('cuda')) -> torch.Tensor:
    """
    mapping prediction of dgl.Graph to tensor, with spix_map
    Args:
      G: Batched DGLGraph
      Pred_node: predict [N, Classes], make sure N = vertex nmber of G
      spix_map:  [B, H, W] result of sfcn_Q_argmax
    Returns:
      pixel_pred: FloatTensor [B, Classes, H, W]
    """
    B, H, W = spix_map.shape
    Classes = Pred_node.size(1)
    spix_map = spix_map.long().to(device)

    # vertex amount for each graph
    lengths = G.batch_num_nodes().tolist() # List<int> of length B

    # split F
    pred_split = list(torch.split(Pred_node, lengths, dim=0)) # List<[N_i, C]>

    # for each graph, indexed by spix_map
    pixel_preds = []
    for b, node_pred in enumerate(pred_split):
        idx = spix_map[b] # [H, W]
        # indexed pred with spix_map
        pix = node_pred[idx] # [H, W, C]
        pix = pix.permute(2, 0, 1).contiguous() # [C, H, W]
        pixel_preds.append(pix)

    return torch.stack(pixel_preds, dim=0) # [B, C, H, W]

def dgl_batch_to_pixel_pred(G: dgl.DGLGraph,
                            Pred_node: torch.Tensor,
                            Q: torch.Tensor,
                            cell_size: int = 16,
                            batch_unique_id: bool = False,
                            device: torch.device = torch.device('cuda')
                            ) -> torch.Tensor:
    """
    Soft mapping of GNN node predictions back to pixel grid via Q weights.

    Args:
      G:             Batched DGLGraph
      Pred_node:     FloatTensor [sum(N_i), C]  GNN output per node
      Q:             FloatTensor [B, 9, H, W]   SFCN affinities
      cell_size:     int, superpixel grid size
      batch_unique_id: bool, same as used in make_spixl_map_idx
      device:        torch device

    Returns:
      pixel_pred:    FloatTensor [B, C, H, W]
    """
    B, _, H, W = Q.shape

    # generate pixel-spixl_id mapping
    spix9_idx = make_spixl_map_idx(Q.size(), cell_size,
                                   batch_unique_id, device)  # LongTensor

    # split Pred_node into list per graph
    lengths = G.batch_num_nodes().tolist()                  # list of B ints
    pred_split = torch.split(Pred_node, lengths, dim=0)     # tuple of [N_i, C]

    C = Pred_node.size(1)
    pixel_preds = []

    for b in range(B):
        node_pred = pred_split[b]    # [N_i, C]
        wq = Q[b]                    # [9, H, W]
        idx = spix9_idx[b]           # [9, H, W]

        # accumulate weighted sum over the 9 neighbors
        # result shape: [H, W, C]
        pix = torch.zeros(H, W, C, device=device)
        for j in range(9):
            # gather node j's prediction for each pixel
            # node_pred[idx[j]]: [H, W, C]
            pj = node_pred[idx[j]]      # indexing is differentiable
            # weight and sum
            pix += pj * wq[j].unsqueeze(-1)

        # move channel dim forward: [C, H, W]
        pix = pix.permute(2, 0, 1).contiguous()
        pixel_preds.append(pix)

    # stack to [B, C, H, W]
    return torch.stack(pixel_preds, dim=0)
