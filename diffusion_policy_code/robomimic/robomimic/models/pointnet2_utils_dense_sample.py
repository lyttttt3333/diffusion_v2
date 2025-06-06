import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import ball_query, sample_farthest_points
from time import time
import numpy as np
from typing import Optional, Tuple, Union, List

def test_sample_farthest_points():
    B, N, C = 2, 100, 3
    xyz = torch.randn(B, N, C)
    
    old_idx = old_farthest_point_sample(xyz, 10)
    old_idx = old_idx.sort()[0]
    old_sampled_xyz = index_points(xyz, old_idx)
    
    sampled_xyz, idx = sample_farthest_points(xyz, K = 10, random_start_point=False)
    for b in range(B):
        sampled_xyz[b] = sampled_xyz[b][idx.sort()[1][b]]
    idx = idx.sort()[0]
    
    print("new sampled xyz: ", sampled_xyz)
    print("old sampled xyz: ", old_sampled_xyz)
    print("new idx: ", idx)
    print("old idx: ", old_idx)
    
    ### new version
    # new_xyz, fps_idx = sample_farthest_points(xyz, K = npoint) # [B, npoint, C]
    
    ### old version
    # fps_idx = old_farthest_point_sample(xyz, npoint) # [B, npoint, C]
    # new_xyz = index_points(xyz, fps_idx)

def test_ball_query():
    ### new version
    # _, idx, grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=True) # [B, npoint, nsample]
    
    ### old version
    # idx = old_query_ball_point(radius, nsample, xyz, new_xyz)
    # grouped_xyz = index_points(xyz, idx)
    
    B, N, C = 2, 100, 3
    M = 10
    xyz = torch.randn(B, N, C)
    new_xyz = torch.randn(B, M, C)
    radius = 2.0
    nsample = 16
    
    _, new_idx, new_grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=True)
    
    old_idx = old_query_ball_point(radius, nsample, xyz, new_xyz)
    old_grouped_xyz = index_points(xyz, old_idx)
    
    print("new idx: ", new_idx)
    print("old idx: ", old_idx)
    print('')

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def old_farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(1, npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def old_query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, attention, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    new_xyz, fps_idx = sample_farthest_points(xyz, K = npoint) # [B, npoint, C]
    _, idx, grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=True) # [B, npoint, nsample]
    grouped_xyz_norm = grouped_xyz  - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]

        grouped_attention = index_points(attention, idx)
        new_attention = torch.mean(grouped_attention, dim=-2).reshape(B,1,-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points, new_attention

def sample_and_group_attn(npoint, radius, nsample, xyz, points, attention, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """

    attn_mask = attention
    attn_xyz = attn_mask * xyz

    none_attn_mask = 1 - attention
    none_attn_xyz = none_attn_mask * xyz

    B, N, C = xyz.shape
    S_none_attn, S_attn = npoint
    S = S_none_attn + S_attn
    new_attn_xyz, fps_attn_idx = sample_farthest_points(attn_xyz, K = S_attn)
    # np.save("/media/yitong/932e6800-38b1-46b9-a874-381bb69f0e77/diff_llm/test_attn.npy",new_attn_xyz.detach().clone().cpu().numpy())
    new_none_attn_xyz, fps_none_attn_idx = sample_farthest_points(none_attn_xyz, K = S_none_attn) # [B, npoint, C]
    # np.save("/media/yitong/932e6800-38b1-46b9-a874-381bb69f0e77/diff_llm/test_none_attn.npy",new_none_attn_xyz.detach().clone().cpu().numpy())
    new_xyz = torch.cat([new_attn_xyz, new_none_attn_xyz], dim=-2)
    fps_idx = torch.cat([fps_attn_idx, fps_none_attn_idx], dim=-1)
    new_attention = index_points(attention, fps_idx[...,:])
    # np.save("/media/yitong/932e6800-38b1-46b9-a874-381bb69f0e77/diff_llm/test_all.npy",new_xyz.detach().clone().cpu().numpy())
    # np.save("/media/yitong/932e6800-38b1-46b9-a874-381bb69f0e77/diff_llm/attn.npy",new_attention.detach().clone().cpu().numpy())

    _, idx, grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=True) # [B, npoint, nsample]
    grouped_xyz_norm = grouped_xyz  - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points, new_attention


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstractionAttn(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, bn=False):
        super(PointNetSetAbstractionAttn, self).__init__()
        self.npoint_all, self.npoint_none_attn = npoint
        self.npoint_attn = self.npoint_all - self.npoint_none_attn
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if self.bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if nsample is not None:
            self.max_pool = nn.MaxPool2d((nsample, 1))
        else:
            self.max_pool = None
        self.group_all = group_all
        self.identity = nn.Identity() # hack to get new_xyz

    def forward(self, xyz, points, attention):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if attention is not None:
            attention = attention.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            new_attention = None
        else:
            new_xyz, new_points, new_attention = sample_and_group_attn((self.npoint_none_attn, self.npoint_attn), self.radius, self.nsample, xyz, points, attention)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  F.relu(bn(conv(new_points)))
            else:
                new_points = F.relu(conv(new_points)) # [B, C+D, nsample, npoint]

        if self.max_pool is not None:
            new_points = self.max_pool(new_points)[:, :, 0]
        else:
            new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        new_attention = new_attention.permute(0, 2, 1)
        new_xyz = self.identity(new_xyz)
        return new_xyz, new_points, new_attention


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, bn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if self.bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if nsample is not None:
            self.max_pool = nn.MaxPool2d((nsample, 1))
        else:
            self.max_pool = None
        self.group_all = group_all
        self.identity = nn.Identity() # hack to get new_xyz

    def forward(self, xyz, points, attention):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if attention is not None:
            attention = attention.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            new_attention = None
        else:
            new_xyz, new_points, new_attention = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, attention)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  F.relu(bn(conv(new_points)))
            else:
                new_points = F.relu(conv(new_points)) # [B, C+D, nsample, npoint]

        if self.max_pool is not None:
            new_points = self.max_pool(new_points)[:, :, 0]
        else:
            new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        new_xyz = self.identity(new_xyz)
        return new_xyz, new_points, new_attention


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz, fps_idx = sample_farthest_points(xyz, K = S) # [B, npoint, C]
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            _, group_idx, grouped_xyz = ball_query(new_xyz, xyz, K=K, radius=radius, return_nn=True) # [B, npoint, nsample]
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNetFeatureInter(nn.Module):
    def __init__(self):
        super(PointNetFeatureInter, self).__init__()

    def forward(self, xyz1, xyz2, points1):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S] (S < N)
            points1: input points data, [B, D, N]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        dists = square_distance(xyz2, xyz1)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, S, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm # [B, S, 3]
        interpolated_points = torch.sum(index_points(points1, idx) * weight.view(B, S, 3, 1), dim=2)
        interpolated_points = interpolated_points.permute(0, 2, 1) # [B, D, S]
        return interpolated_points


class PointNet2Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, use_bn=False):
        super(PointNet2Encoder, self).__init__()


        # pn2.1_r_0.04
        self.sa1 = PointNetSetAbstractionAttn(npoint=(512,256), radius=0.02, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstractionAttn(npoint=(128,64), radius=0.045, nsample=64, in_channel=128 + 3 + 1, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3 + 1, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn = use_bn
        if self.bn:
            self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, out_channels)
        if self.bn:
            self.bn2 = nn.BatchNorm1d(256)

        # copy variables
        self.in_channels = in_channels

    def forward(self, xyz):
        B, D, N = xyz.size()
        assert D == self.in_channels
        if D > 3:
            norm = xyz[:, 3:, :]
            norm[norm>0] = 1
            norm[norm<=0] = 0
            xyz = xyz[:, :3, :]
            # norm = 1 -norm
        else:
            norm = None
        l1_xyz, l1_points, l1_attention = self.sa1(xyz, norm, norm)
        l1_points = torch.cat([l1_points, l1_attention], dim = 1)
        l2_xyz, l2_points, l2_attention = self.sa2(l1_xyz, l1_points, l1_attention)
        l2_points = torch.cat([l2_points, l2_attention], dim=1)
        l3_xyz, l3_points, l3_attention = self.sa3(l2_xyz, l2_points, l2_attention)
        x = l3_points.view(B, 1024)
        if self.bn:
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
        else:
            x = self.drop1(F.relu(self.fc1(x)))
            x = self.fc2(x)

        return x

class PointNet2DenseEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, use_bn=False):
        super(PointNet2DenseEncoder, self).__init__()
        
        # self.sa1 = PointNetSetAbstractionMsg(50, [0.02, 0.04, 0.08], [8, 16, 64], in_channel - 3,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(10, [0.04, 0.08, 0.16], [16, 32, 64], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        # self.sa1 = PointNetSetAbstraction(npoint=64, radius=0.04, nsample=16, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=16, radius=0.08, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.04, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.08, nsample=64, in_channel=128 + 3, mlp=[128, 256, 512], group_all=False)

        # copy variables
        self.in_channels = in_channels
        
    def forward(self, xyz):
        B, D, N = xyz.size()
        assert D == self.in_channels
        if D > 3:
            norm = xyz[:, 3:, :]
            norm[norm>0] = 1
            norm[norm<=0] = 0
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        return l2_points

class PointNetAttnEncoder(nn.Module):
    def __init__(self, in_channels=3, use_bn=False):
        super(PointNetAttnEncoder, self).__init__()
        
        feat_copy = in_channels - 3
        
        # self.sa1 = PointNetSetAbstractionMsg(50, [0.02, 0.04, 0.08], [8, 16, 64], in_channel - 3,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(10, [0.04, 0.08, 0.16], [16, 32, 64], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        # self.sa1 = PointNetSetAbstraction(npoint=64, radius=0.04, nsample=16, in_channel=in_channels-3, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=16, radius=0.08, nsample=32, in_channel=128 * feat_copy + 3, mlp=[128 * feat_copy, 128 * feat_copy, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 * feat_copy + 3, mlp=[256 * feat_copy, 512 * feat_copy, 1024], group_all=True)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channels-3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 * feat_copy + 3, mlp=[128 * feat_copy, 128 * feat_copy, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 * feat_copy + 3, mlp=[256 * feat_copy, 512 * feat_copy, 1024], group_all=True)
        self.fi = PointNetFeatureInter()
        self.fc1 = nn.Linear(1024, 512)
        self.bn = use_bn
        if self.bn:
            self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        if self.bn:
            self.bn2 = nn.BatchNorm1d(256)

        # copy variables
        self.in_channels = in_channels

    def forward(self, xyz):
        B, D, N = xyz.size()
        assert D == self.in_channels
        if D > 3:
            sim = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            raise RuntimeError("Need similarity values")
        l1_xyz, l1_points = self.sa1(xyz, None) # (B, 3, N_sample), (B, 128, N_sample)
        
        # attn-like operation
        sim1 = self.fi(xyz, l1_xyz, sim) # (B, D-3, N_sample)
        l1_points = l1_points.unsqueeze(1) * sim1.unsqueeze(2) # (B, 1, 128, N_sample) * (B, D-3, 1, N_sample) = (B, D-3, 64, N)
        N_l1 = l1_points.shape[-1]
        l1_points = l1_points.reshape(B, -1, N_l1) # (B, (D-3)*64, N)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # attn-like operation
        sim2 = self.fi(xyz, l2_xyz, sim)
        l2_points = l2_points.unsqueeze(1) * sim2.unsqueeze(2) # (B, 1, 64, N) * (B, D-3, 1, N) = (B, D-3, 64, N)
        N_l2 = l2_points.shape[-1]
        l2_points = l2_points.reshape(B, -1, N_l2) # (B, (D-3)*64, N)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        if self.bn:
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
        else:
            x = self.drop1(F.relu(self.fc1(x)))
            x = self.fc2(x)

        return x

class CAMPointNetEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, use_bn=False, use_softmax=False):
        super(CAMPointNetEncoder, self).__init__()
        
        self.pn1 = PointNet2Encoder(in_channels=in_channels, use_bn=use_bn) # extract heatmap of heatmap
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels - 3),
            nn.ReLU(),
        )
        self.pn2 = PointNet2Encoder(in_channels=4, out_channels=out_channels, use_bn=use_bn) # extract feature
        
        # copy variables
        self.in_channels = in_channels
        self.use_softmax = use_softmax

    def forward(self, pcd):
        B, D, N = pcd.size()
        assert D == self.in_channels
        assert D > 3 # must have heatmap
        sim = pcd[:, 3:, :]
        xyz = pcd[:, :3, :]
        heatmap_feat = self.pn1(pcd) # (B, 256)
        heatmap = self.heatmap_mlp(heatmap_feat) # (B, D-3)
        if self.use_softmax:
            heatmap = F.softmax(heatmap, dim=1)
        sim_weighted = sim * heatmap.unsqueeze(-1) # (B, D-3, N)
        sim_weighted_sum = torch.sum(sim_weighted, dim=1, keepdim=True) # (B, 1, N)
        new_pcd = torch.cat([xyz, sim_weighted_sum], dim=1) # (B, 4, N)
        final_feat = self.pn2(new_pcd) # (B, 256)
        return final_feat, sim_weighted_sum

if __name__ == '__main__':
    # test_sample_farthest_points()
    test_ball_query()
