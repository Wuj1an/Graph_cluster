"""End-to-end demonstration of ST-Graph construction and ConGNN clustering.

This public demo mirrors the core workflow in ``Graph_cluster.ipynb`` while
using a small in-memory list of synthetic Weibo-style posts. It does not read
the study dataset or any local model path.

Workflow
--------
1. Encode post text with BERT (or an offline TF-IDF fallback).
2. Transform engagement and user-profile features with log1p and scaling.
3. Build a temporally directed ST-Graph from weighted similarities and an
   adaptive K-nearest-neighbor rule.
4. add PageRank to the node features and use it to enhance edge weights.
5. Train a three-layer GraphSAGE encoder with an InfoNCE objective.
6. Select the number of communities from K=2,...,8 using Silhouette and run
   K-means on the learned node representations.

Install the required packages before running the demo::

    pip install numpy pandas scikit-learn networkx matplotlib torch \
        torch-geometric transformers

Run::

    python Graph_cluster_demo.py

The script first tries ``google-bert/bert-base-chinese``. If the model cannot
be loaded, it uses a deterministic TF-IDF representation so that the complete
pipeline can still be demonstrated offline. Use ``--strict-bert`` to disable
this fallback.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Avoid the known small-dataset KMeans memory warning on Windows with MKL.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge


SEED = 42


# Synthetic examples preserve the schema used by the study without exposing
# any original Weibo text or account information.
SAMPLE_POSTS: List[Dict[str, object]] = [
    {
        "text": "新一代大模型提升了代码生成和办公写作效率，企业开始测试智能助手。",
        "timestamp": "2023-03-05 09:10:00",
        "likes": 420,
        "shares": 96,
        "comments": 83,
        "user_followers": 180000,
        "user_activity": 28,
        "actor_type": "Enterprise",
    },
    {
        "text": "生成式人工智能正在帮助设计团队快速形成创意草稿和营销文案。",
        "timestamp": "2023-03-18 14:20:00",
        "likes": 360,
        "shares": 75,
        "comments": 64,
        "user_followers": 92000,
        "user_activity": 21,
        "actor_type": "Regular User",
    },
    {
        "text": "智能办公工具可以完成摘要、翻译和文本改写，减少重复劳动。",
        "timestamp": "2023-04-02 11:35:00",
        "likes": 515,
        "shares": 118,
        "comments": 91,
        "user_followers": 240000,
        "user_activity": 34,
        "actor_type": "Media",
    },
    {
        "text": "AI辅助编程降低了原型开发成本，但复杂任务仍需要人工检查。",
        "timestamp": "2023-05-11 17:05:00",
        "likes": 275,
        "shares": 54,
        "comments": 72,
        "user_followers": 64000,
        "user_activity": 18,
        "actor_type": "Regular User",
    },
    {
        "text": "企业发布生成式AI服务，强调模型能力和商业应用前景。",
        "timestamp": "2023-06-09 08:45:00",
        "likes": 630,
        "shares": 146,
        "comments": 105,
        "user_followers": 410000,
        "user_activity": 31,
        "actor_type": "Enterprise",
    },
    {
        "text": "教育场景开始使用AI生成练习题和课程提纲，教师关注实际效果。",
        "timestamp": "2023-07-16 13:30:00",
        "likes": 198,
        "shares": 42,
        "comments": 57,
        "user_followers": 45000,
        "user_activity": 16,
        "actor_type": "Organization",
    },
    {
        "text": "人工智能绘画为内容创作者提供了新的表达工具和制作方式。",
        "timestamp": "2023-08-23 19:15:00",
        "likes": 712,
        "shares": 163,
        "comments": 132,
        "user_followers": 530000,
        "user_activity": 40,
        "actor_type": "Public Figure",
    },
    {
        "text": "开源模型推动开发者社区合作，也加快了生成式AI应用落地。",
        "timestamp": "2023-10-08 10:05:00",
        "likes": 488,
        "shares": 122,
        "comments": 88,
        "user_followers": 210000,
        "user_activity": 37,
        "actor_type": "Organization",
    },
    {
        "text": "AI换脸视频冒充演员传播，肖像权和身份盗用风险引发争议。",
        "timestamp": "2024-01-12 12:15:00",
        "likes": 845,
        "shares": 302,
        "comments": 276,
        "user_followers": 680000,
        "user_activity": 46,
        "actor_type": "Media",
    },
    {
        "text": "深度伪造语音被用于诈骗，平台需要加强检测和风险提示。",
        "timestamp": "2024-02-07 16:40:00",
        "likes": 920,
        "shares": 355,
        "comments": 301,
        "user_followers": 760000,
        "user_activity": 49,
        "actor_type": "Media",
    },
    {
        "text": "AI生成短剧快速增加，低成本制作也带来版权归属问题。",
        "timestamp": "2024-03-19 20:05:00",
        "likes": 675,
        "shares": 214,
        "comments": 187,
        "user_followers": 350000,
        "user_activity": 35,
        "actor_type": "Public Figure",
    },
    {
        "text": "视频生成模型展示了新的创作能力，同时需要防范虚假内容。",
        "timestamp": "2024-04-25 09:55:00",
        "likes": 1040,
        "shares": 401,
        "comments": 338,
        "user_followers": 890000,
        "user_activity": 52,
        "actor_type": "Enterprise",
    },
    {
        "text": "有人利用AI合成人脸制作虚假广告，消费者权益受到影响。",
        "timestamp": "2024-05-14 18:30:00",
        "likes": 590,
        "shares": 248,
        "comments": 225,
        "user_followers": 280000,
        "user_activity": 33,
        "actor_type": "Regular User",
    },
    {
        "text": "平台公布生成视频审核规范，重点识别换脸和伪造素材。",
        "timestamp": "2024-06-21 10:45:00",
        "likes": 530,
        "shares": 188,
        "comments": 143,
        "user_followers": 460000,
        "user_activity": 29,
        "actor_type": "Organization",
    },
    {
        "text": "未经授权模仿演员声线制作短剧，可能构成声音权益侵害。",
        "timestamp": "2024-08-03 15:20:00",
        "likes": 730,
        "shares": 269,
        "comments": 241,
        "user_followers": 510000,
        "user_activity": 38,
        "actor_type": "Public Figure",
    },
    {
        "text": "监管部门提醒公众识别AI换脸诈骗并妥善保护个人信息。",
        "timestamp": "2024-10-17 08:25:00",
        "likes": 810,
        "shares": 327,
        "comments": 196,
        "user_followers": 620000,
        "user_activity": 26,
        "actor_type": "Government",
    },
    {
        "text": "生成内容标识办法提出显式标识要求，帮助用户识别合成信息。",
        "timestamp": "2025-01-09 09:00:00",
        "likes": 710,
        "shares": 290,
        "comments": 174,
        "user_followers": 740000,
        "user_activity": 30,
        "actor_type": "Government",
    },
    {
        "text": "平台上线AI内容标签和检测工具，进一步明确审核责任。",
        "timestamp": "2025-02-15 13:15:00",
        "likes": 665,
        "shares": 252,
        "comments": 168,
        "user_followers": 580000,
        "user_activity": 32,
        "actor_type": "Enterprise",
    },
    {
        "text": "虚假视频未标明由AI生成，容易造成身份冒用和信息误导。",
        "timestamp": "2025-03-28 21:10:00",
        "likes": 960,
        "shares": 418,
        "comments": 362,
        "user_followers": 830000,
        "user_activity": 54,
        "actor_type": "Media",
    },
    {
        "text": "合成内容水印有助于追踪来源，但跨平台识别标准仍需统一。",
        "timestamp": "2025-04-20 11:40:00",
        "likes": 438,
        "shares": 157,
        "comments": 126,
        "user_followers": 310000,
        "user_activity": 27,
        "actor_type": "Organization",
    },
    {
        "text": "模型训练数据受到污染可能放大谣言，数据治理成为新的焦点。",
        "timestamp": "2025-05-31 17:50:00",
        "likes": 782,
        "shares": 311,
        "comments": 289,
        "user_followers": 570000,
        "user_activity": 43,
        "actor_type": "Media",
    },
    {
        "text": "AI应用应当清晰说明数据来源、内容标识和用户申诉渠道。",
        "timestamp": "2025-06-18 08:35:00",
        "likes": 502,
        "shares": 181,
        "comments": 147,
        "user_followers": 390000,
        "user_activity": 25,
        "actor_type": "Government",
    },
    {
        "text": "网民关注平台是否及时处置未标识的伪造图片和虚假账号。",
        "timestamp": "2025-07-22 19:25:00",
        "likes": 618,
        "shares": 236,
        "comments": 221,
        "user_followers": 260000,
        "user_activity": 39,
        "actor_type": "Regular User",
    },
    {
        "text": "完善生成内容标识制度能够提升信息透明度和平台治理能力。",
        "timestamp": "2025-09-04 14:05:00",
        "likes": 549,
        "shares": 205,
        "comments": 139,
        "user_followers": 430000,
        "user_activity": 28,
        "actor_type": "Organization",
    },
    {
        "text": "智能体进入日常办公流程，协助整理资料、制作视频和安排任务。",
        "timestamp": "2026-01-13 09:30:00",
        "likes": 742,
        "shares": 225,
        "comments": 193,
        "user_followers": 520000,
        "user_activity": 41,
        "actor_type": "Enterprise",
    },
    {
        "text": "开源智能体降低了自动化工具门槛，个人用户开始搭建工作助手。",
        "timestamp": "2026-02-24 16:15:00",
        "likes": 688,
        "shares": 198,
        "comments": 177,
        "user_followers": 330000,
        "user_activity": 36,
        "actor_type": "Regular User",
    },
    {
        "text": "AI创作工具提升视频剪辑效率，内容生产逐步形成稳定流程。",
        "timestamp": "2026-03-17 12:45:00",
        "likes": 805,
        "shares": 247,
        "comments": 204,
        "user_followers": 610000,
        "user_activity": 45,
        "actor_type": "Public Figure",
    },
    {
        "text": "平台服务条款中的免责边界需要明确，用户应获得合理申诉机制。",
        "timestamp": "2026-04-29 18:20:00",
        "likes": 576,
        "shares": 216,
        "comments": 238,
        "user_followers": 440000,
        "user_activity": 34,
        "actor_type": "Organization",
    },
    {
        "text": "生成内容涉及著作权争议，创作者要求平台明确责任和收益分配。",
        "timestamp": "2026-05-20 10:10:00",
        "likes": 891,
        "shares": 336,
        "comments": 315,
        "user_followers": 700000,
        "user_activity": 48,
        "actor_type": "Public Figure",
    },
    {
        "text": "训练数据来源和授权范围受到关注，企业需要完善合规记录。",
        "timestamp": "2026-06-08 08:50:00",
        "likes": 641,
        "shares": 229,
        "comments": 186,
        "user_followers": 490000,
        "user_activity": 30,
        "actor_type": "Enterprise",
    },
    {
        "text": "监管机构发布人工智能服务指引，强调平台责任与数据安全。",
        "timestamp": "2026-07-12 14:35:00",
        "likes": 754,
        "shares": 281,
        "comments": 211,
        "user_followers": 810000,
        "user_activity": 33,
        "actor_type": "Government",
    },
    {
        "text": "普通用户关心AI生成作品能否获得版权以及平台如何处理侵权投诉。",
        "timestamp": "2026-08-03 20:40:00",
        "likes": 607,
        "shares": 194,
        "comments": 259,
        "user_followers": 290000,
        "user_activity": 42,
        "actor_type": "Regular User",
    },
]


@dataclass(frozen=True)
class GraphConfig:
    text_weight: float = 0.5
    engagement_weight: float = 0.3
    user_weight: float = 0.2
    similarity_threshold: float = 0.45
    k_neighbors: int = 5
    pagerank_alpha: float = 0.85
    pagerank_boost: float = 5.0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_records(records: Sequence[Dict[str, object]]) -> pd.DataFrame:
    required = {
        "text",
        "timestamp",
        "likes",
        "shares",
        "comments",
        "user_followers",
        "user_activity",
        "actor_type",
    }
    frame = pd.DataFrame(records)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    numeric_cols = [
        "likes",
        "shares",
        "comments",
        "user_followers",
        "user_activity",
    ]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="raise")
    if frame.empty or len(frame) < 4:
        raise ValueError("At least four records are required for graph clustering.")
    return frame.reset_index(drop=True)


def encode_with_bert(
    texts: Sequence[str], model_name: str, device: torch.device, batch_size: int = 8
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    chunks: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            chunks.append(output.last_hidden_state[:, 0, :].cpu().numpy())

    return np.vstack(chunks).astype(np.float32)


def encode_with_tfidf(texts: Sequence[str], max_features: int = 256) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )
    return vectorizer.fit_transform(texts).toarray().astype(np.float32)


def encode_texts(
    texts: Sequence[str],
    encoder: str,
    model_name: str,
    device: torch.device,
    strict_bert: bool,
) -> Tuple[np.ndarray, str]:
    if encoder == "tfidf":
        return encode_with_tfidf(texts), "TF-IDF"

    try:
        return encode_with_bert(texts, model_name, device), model_name
    except Exception as exc:
        if strict_bert:
            raise RuntimeError(f"Unable to load BERT encoder {model_name!r}") from exc
        warnings.warn(
            f"BERT could not be loaded ({exc}). Falling back to offline TF-IDF.",
            RuntimeWarning,
        )
        return encode_with_tfidf(texts), "TF-IDF fallback"


def scaled_log_features(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
    return StandardScaler().fit_transform(values).astype(np.float32)


def build_st_graph(
    frame: pd.DataFrame,
    text_features: np.ndarray,
    config: GraphConfig,
) -> Tuple[Data, nx.DiGraph, Dict[str, np.ndarray]]:
    engagement_features = scaled_log_features(frame, ["likes", "shares", "comments"])
    user_features = scaled_log_features(frame, ["user_followers", "user_activity"])

    sim_text = cosine_similarity(text_features)
    sim_engagement = cosine_similarity(engagement_features)
    sim_user = cosine_similarity(user_features)
    combined_similarity = (
        config.text_weight * sim_text
        + config.engagement_weight * sim_engagement
        + config.user_weight * sim_user
    )
    combined_similarity = (combined_similarity + combined_similarity.T) / 2.0
    combined_similarity = np.clip(combined_similarity, 0.0, 1.0)
    np.fill_diagonal(combined_similarity, 0.0)

    timestamps = frame["timestamp"].array.asi8.astype(np.float64) / 1e9
    local_density = (combined_similarity > config.similarity_threshold).sum(axis=1)
    edges: List[Tuple[int, int]] = []

    for source in range(len(frame)):
        if local_density[source] > 20:
            adaptive_k = max(3, config.k_neighbors // 2)
        elif local_density[source] > 10:
            adaptive_k = config.k_neighbors
        else:
            adaptive_k = min(len(frame) - 1, config.k_neighbors * 2)

        candidates = np.flatnonzero(
            combined_similarity[source] > config.similarity_threshold
        )
        candidates = candidates[timestamps[source] > timestamps[candidates]]
        if candidates.size:
            order = np.argsort(combined_similarity[source, candidates])[::-1]
            for target in candidates[order[:adaptive_k]]:
                edges.append((source, int(target)))

    if not edges:
        raise RuntimeError(
            "No edges were generated. Reduce similarity_threshold or add more related records."
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(frame)))
    graph.add_weighted_edges_from(
        (source, target, float(combined_similarity[source, target]))
        for source, target in edges
    )

    pagerank = nx.pagerank(graph, alpha=config.pagerank_alpha, weight="weight")
    pagerank_values = np.array(
        [pagerank[node] for node in range(len(frame))], dtype=np.float32
    )
    pr_min, pr_max = float(pagerank_values.min()), float(pagerank_values.max())
    if pr_max > pr_min:
        pagerank_scaled = (pagerank_values - pr_min) / (pr_max - pr_min)
    else:
        pagerank_scaled = np.zeros_like(pagerank_values)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    source_nodes = edge_index[0].numpy()
    target_nodes = edge_index[1].numpy()
    base_weights = combined_similarity[source_nodes, target_nodes]
    enhanced_weights = base_weights * (
        1.0 + config.pagerank_boost * pagerank_scaled[target_nodes]
    )
    if enhanced_weights.max() > 0:
        enhanced_weights = enhanced_weights / enhanced_weights.max()

    time_gap_days = np.abs(timestamps[source_nodes] - timestamps[target_nodes]) / 86400.0
    time_decay = np.exp(-(np.log(2.0) / 7.0) * time_gap_days)
    edge_attributes = np.column_stack(
        [
            enhanced_weights,
            sim_text[source_nodes, target_nodes],
            sim_engagement[source_nodes, target_nodes],
            sim_user[source_nodes, target_nodes],
            time_gap_days,
            time_decay,
            pagerank_scaled[source_nodes],
            pagerank_scaled[target_nodes],
        ]
    ).astype(np.float32)

    node_features = np.column_stack(
        [text_features, engagement_features, user_features, pagerank_scaled]
    ).astype(np.float32)
    data = Data(
        x=torch.from_numpy(node_features),
        edge_index=edge_index,
        edge_attr=torch.from_numpy(edge_attributes),
    )
    data.pagerank_scores = torch.from_numpy(pagerank_scaled)

    for edge_id, (source, target) in enumerate(edges):
        graph[source][target]["weight"] = float(enhanced_weights[edge_id])

    components = {
        "text": sim_text,
        "engagement": sim_engagement,
        "user": sim_user,
        "combined": combined_similarity,
        "pagerank": pagerank_scaled,
    }
    return data, graph, components


class ConGNN(nn.Module):
    """Three-layer GraphSAGE encoder with a contrastive projection head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (128, 64),
        output_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dims[0])
        self.conv2 = SAGEConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = SAGEConv(hidden_dims[1], output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout = nn.Dropout(dropout)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, project: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        representation = self.conv3(x, edge_index)
        if project:
            return representation, self.projection_head(representation)
        return representation


def info_nce_loss(
    view_a: torch.Tensor, view_b: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    view_a = F.normalize(view_a, dim=1)
    view_b = F.normalize(view_b, dim=1)
    batch_size = view_a.shape[0]
    mask = torch.eye(batch_size, dtype=torch.bool, device=view_a.device)

    similarity_aa = (view_a @ view_a.T / temperature).masked_fill(mask, -1e9)
    similarity_bb = (view_b @ view_b.T / temperature).masked_fill(mask, -1e9)
    similarity_ab = view_a @ view_b.T / temperature
    positive = torch.diag(similarity_ab).unsqueeze(1)
    negative_ab = similarity_ab.masked_fill(mask, -1e9)
    negative_ba = similarity_ab.T.masked_fill(mask, -1e9)

    logits_a = torch.cat([positive, similarity_aa, negative_ab], dim=1)
    logits_b = torch.cat([positive, similarity_bb, negative_ba], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=view_a.device)
    return (
        F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels)
    ) / 2.0


def augment_graph(
    data: Data,
    feature_mask_rate: float,
    edge_drop_rate: float,
    noise_std: float,
    use_noise: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = data.x.clone()
    if use_noise:
        features = features + torch.randn_like(features) * noise_std
    else:
        mask = torch.rand_like(features) > feature_mask_rate
        features = features * mask
    augmented_edges, _ = dropout_edge(data.edge_index, p=edge_drop_rate)
    return features, augmented_edges


def train_congnn(
    data: Data,
    device: torch.device,
    epochs: int,
    learning_rate: float = 0.001,
) -> Tuple[ConGNN, List[float]]:
    data = data.to(device)
    model = ConGNN(input_dim=data.num_node_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    losses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        features_a, edges_a = augment_graph(data, 0.2, 0.2, 0.1, use_noise=False)
        features_b, edges_b = augment_graph(data, 0.2, 0.2, 0.1, use_noise=True)
        _, projection_a = model(features_a, edges_a)
        _, projection_b = model(features_b, edges_b)
        loss = info_nce_loss(projection_a, projection_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:>3}/{epochs}: contrastive loss = {loss_value:.4f}")

    model.load_state_dict(best_state)
    return model, losses


def extract_embeddings(model: ConGNN, data: Data, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        embeddings = model(
            data.x.to(device), data.edge_index.to(device), project=False
        )
    return embeddings.cpu().numpy()


def select_communities(
    embeddings: np.ndarray, min_k: int = 2, max_k: int = 8
) -> Tuple[int, np.ndarray, Dict[int, float]]:
    upper = min(max_k, len(embeddings) - 1)
    scores: Dict[int, float] = {}
    labels_by_k: Dict[int, np.ndarray] = {}
    for k in range(min_k, upper + 1):
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(
            embeddings
        )
        scores[k] = float(silhouette_score(embeddings, labels))
        labels_by_k[k] = labels
        print(f"K={k}: Silhouette = {scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    return best_k, labels_by_k[best_k], scores


def partition_stability(
    embeddings: np.ndarray, n_clusters: int, seeds: Sequence[int] = range(10)
) -> Tuple[float, float]:
    partitions = [
        KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(
            embeddings
        )
        for seed in seeds
    ]
    ari_values: List[float] = []
    nmi_values: List[float] = []
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            ari_values.append(adjusted_rand_score(partitions[left], partitions[right]))
            nmi_values.append(
                normalized_mutual_info_score(partitions[left], partitions[right])
            )
    return float(np.mean(ari_values)), float(np.mean(nmi_values))


def calculate_network_metrics(
    graph: nx.DiGraph, labels: np.ndarray
) -> Dict[str, float]:
    undirected = graph.to_undirected()
    communities = [
        set(np.flatnonzero(labels == cluster_id)) for cluster_id in np.unique(labels)
    ]
    return {
        "density": float(nx.density(graph)),
        "average_clustering": float(nx.average_clustering(undirected, weight="weight")),
        "modularity": float(
            nx.algorithms.community.quality.modularity(
                undirected, communities, weight="weight"
            )
        ),
    }


def save_outputs(
    frame: pd.DataFrame,
    graph: nx.DiGraph,
    pagerank: np.ndarray,
    labels: np.ndarray,
    losses: Sequence[float],
    silhouette_scores: Dict[int, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    result["pagerank"] = pagerank
    result["community"] = labels
    result.to_csv(output_dir / "community_assignments.csv", index=False, encoding="utf-8-sig")

    actor_types = sorted(frame["actor_type"].unique())
    palette = plt.get_cmap("tab10")
    actor_colors = {
        actor: palette(index % 10) for index, actor in enumerate(actor_types)
    }
    positions = nx.spring_layout(
        graph.to_undirected(), seed=SEED, weight="weight", iterations=200
    )
    node_colors = [actor_colors[frame.loc[node, "actor_type"]] for node in graph.nodes]
    node_sizes = 80.0 + 420.0 * pagerank[list(graph.nodes)]

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axes[0],
        width=0.7,
        alpha=0.25,
        arrows=False,
        edge_color="#777777",
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axes[0],
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.4,
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label=actor,
            markerfacecolor=actor_colors[actor],
            markeredgecolor="black",
            markersize=7,
        )
        for actor in actor_types
    ]
    axes[0].legend(handles=legend, fontsize=7, frameon=False, loc="best")
    axes[0].set_title("ST-Graph (size: PageRank)")
    axes[0].axis("off")

    axes[1].plot(range(1, len(losses) + 1), losses, color="#2878B5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("InfoNCE loss")
    axes[1].set_title("ConGNN training")
    axes[1].grid(alpha=0.25)

    k_values = list(silhouette_scores)
    axes[2].plot(
        k_values,
        [silhouette_scores[k] for k in k_values],
        marker="o",
        color="#C82423",
    )
    axes[2].set_xticks(k_values)
    axes[2].set_xlabel("Number of communities (K)")
    axes[2].set_ylabel("Silhouette")
    axes[2].set_title("Adaptive community selection")
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "graph_cluster_demo.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-encoder",
        choices=("bert", "tfidf"),
        default="bert",
        help="Use BERT as in the study or the lightweight offline TF-IDF option.",
    )
    parser.add_argument(
        "--bert-model",
        default="google-bert/bert-base-chinese",
        help="Public Hugging Face model identifier used for CLS embeddings.",
    )
    parser.add_argument(
        "--strict-bert",
        action="store_true",
        help="Stop if BERT cannot be loaded instead of using the TF-IDF fallback.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    frame = validate_records(SAMPLE_POSTS)
    text_features, encoder_name = encode_texts(
        frame["text"].tolist(),
        encoder=args.text_encoder,
        model_name=args.bert_model,
        device=device,
        strict_bert=args.strict_bert,
    )
    print(f"Text representation: {encoder_name}; shape={text_features.shape}")

    data, graph, components = build_st_graph(frame, text_features, GraphConfig())
    print(
        f"ST-Graph: {data.num_nodes} nodes, {data.num_edges} directed edges, "
        f"{data.num_node_features} node features"
    )

    model, losses = train_congnn(data, device=device, epochs=args.epochs)
    embeddings = extract_embeddings(model, data, device)
    best_k, labels, silhouette_scores = select_communities(embeddings, 2, 8)
    ari, nmi = partition_stability(embeddings, best_k)
    metrics = calculate_network_metrics(graph, labels)

    print(f"Selected communities: K={best_k}")
    print(f"Best Silhouette: {silhouette_scores[best_k]:.4f}")
    print(f"Mean ARI across K-means seeds: {ari:.4f}")
    print(f"Mean NMI across K-means seeds: {nmi:.4f}")
    print(f"Modularity Q: {metrics['modularity']:.4f}")
    print(f"Graph density: {metrics['density']:.4f}")
    print(f"Average clustering: {metrics['average_clustering']:.4f}")

    save_outputs(
        frame,
        graph,
        components["pagerank"],
        labels,
        losses,
        silhouette_scores,
        args.output_dir,
    )
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
