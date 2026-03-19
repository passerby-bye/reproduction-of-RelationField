# Copyright (c) 2026 - Lightweight 3D Scene Graph extraction from RelationField
"""
Lightweight 3D Scene Graph extraction from a trained RelationField model.

Pipeline:
  1. Load trained RelationField model from config.yml
  2. Sample points from the scene mesh / point cloud
  3. Query instance embeddings at sampled points → DBSCAN clustering → instances
  4. For each instance, pick an object label via OpenSeg cosine similarity
  5. Select up to MAX_PAIRS instance pairs; predict relations via relation field + Jina
  6. Draw and save a scene graph visualization

Usage:
    python extract_scene_graph.py \\
        --config outputs/replica_office0/relationfield/2026-03-09_231057/config.yml \\
        --out_dir sg_results/replica_office0

    python extract_scene_graph.py \\
        --config outputs/seq01_02/relationfield/2026-03-14_015920/config.yml \\
        --out_dir sg_results/seq01_02
"""

import argparse
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from sklearn.cluster import DBSCAN

from nerfstudio.utils.eval_utils import eval_setup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Object label candidates (covers Replica + RIO scenes)
OBJECT_LABELS = [
    "chair", "table", "sofa", "couch", "floor", "wall", "ceiling",
    "screen", "monitor", "carpet", "cushion", "pillow", "painting",
    "picture", "window", "curtain", "door", "trash can", "plant",
    "basket", "tile", "bed", "desk", "shelf", "lamp", "book",
    "mat", "box", "bag",
]

# Relation predicate candidates
RELATION_CANDIDATES = [
    "standing on",
    "lying on",
    "attached to",
    "hanging on",
    "next to",
    "above",
    "below",
    "part of",
    "supporting",
    "same type as",
]

# DBSCAN params for clustering instance embeddings
DBSCAN_EPS = 0.25         # cosine-space distance threshold
DBSCAN_MIN_SAMPLES = 10   # min points per cluster

INSTANCE_SCALE = 0.5      # scale param for instance field
N_SAMPLE_POINTS = 6000    # points to sample from mesh
CHUNK_SIZE = 800           # points per forward-pass chunk (memory safety)
MAX_CLUSTERS = 20          # keep at most this many clusters for labeling
MAX_PAIRS = 10             # max (subject, object) pairs for relation prediction
MIN_CLUSTER_SIZE = 30      # ignore tiny clusters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mesh_and_sample(data_dir: Path, n: int) -> np.ndarray:
    """Load the scene mesh from transforms.json and uniformly sample n points."""
    tf_path = data_dir / "transforms.json"
    if not tf_path.exists():
        raise FileNotFoundError(f"transforms.json not found in {data_dir}")
    with open(tf_path) as f:
        tf = json.load(f)
    ply_rel = tf.get("ply_file_path")
    if ply_rel is None:
        raise ValueError("transforms.json has no ply_file_path key")
    ply_path = data_dir / ply_rel
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    print(f"  Loading mesh: {ply_path}")
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float32)


def query_instance_embeddings(model, points: np.ndarray) -> np.ndarray:
    """Query instance embeddings at points in chunks. Returns [N, 256] numpy."""
    all_inst = []
    for i in range(0, len(points), CHUNK_SIZE):
        chunk = points[i : i + CHUNK_SIZE]
        # get_outputs_for_points expects [B, N, 3]
        out = model.get_outputs_for_points(chunk[np.newaxis], scale=INSTANCE_SCALE)
        inst = out["instance"][0].float().cpu().numpy()  # [chunk, 256]
        all_inst.append(inst)
        torch.cuda.empty_cache()
    return np.concatenate(all_inst, axis=0)


def query_semantic_embeddings(model, points: np.ndarray) -> np.ndarray:
    """Query openseg embeddings at points in chunks. Returns [N, D] numpy."""
    all_sem = []
    for i in range(0, len(points), CHUNK_SIZE):
        chunk = points[i : i + CHUNK_SIZE]
        out = model.get_outputs_for_points(chunk[np.newaxis], scale=INSTANCE_SCALE)
        sem = out["openseg"][0].float().cpu().numpy()
        all_sem.append(sem)
        torch.cuda.empty_cache()
    return np.concatenate(all_sem, axis=0)


def cluster_instances(inst_emb: np.ndarray):
    """DBSCAN on L2 distance of normalized instance embeddings."""
    normed = inst_emb / (np.linalg.norm(inst_emb, axis=1, keepdims=True) + 1e-8)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(normed)
    return labels  # -1 = noise


def label_cluster(cluster_sem_emb: np.ndarray, click_scene, label_list) -> str:
    """Pick the best-matching object label for a cluster's average openseg embedding."""
    avg_emb = cluster_sem_emb.mean(axis=0)
    avg_emb_t = torch.from_numpy(avg_emb).float().cuda().unsqueeze(0)
    avg_emb_t = F.normalize(avg_emb_t, dim=-1)

    click_scene.set_positives(label_list)
    # pos_embeds: [num_labels, D] on CUDA
    sims = (avg_emb_t @ click_scene.pos_embeds.cuda().T)[0].cpu().numpy()
    best_idx = int(np.argmax(sims))
    return label_list[best_idx], float(sims[best_idx])


def predict_relation(model, subj_pts: np.ndarray, obj_centroid: np.ndarray,
                     click_scene, relation_list) -> tuple:
    """
    Predict the most likely predicate between a subject cluster and an object centroid.
    Returns (best_predicate, score).
    """
    query = obj_centroid[np.newaxis]  # [1, 3]
    rel_feats = []
    for i in range(0, len(subj_pts), CHUNK_SIZE):
        chunk = subj_pts[i : i + CHUNK_SIZE]
        out = model.get_outputs_for_points_with_query_batch(chunk, query)
        rel_feats.append(out["relation"].float().cpu())
        torch.cuda.empty_cache()
    rel_feat = torch.cat(rel_feats, dim=0)   # [N, 512]
    avg_rel = rel_feat.mean(dim=0, keepdim=True)  # [1, 512]
    avg_rel = F.normalize(avg_rel, dim=-1).cuda()

    scores = {}
    for pred in relation_list:
        click_scene.set_relation_positives([pred])
        pos_emb = click_scene.bert_pos_embds  # [1, 512]
        scores[pred] = float((avg_rel @ pos_emb.T)[0, 0].item())

    best_pred = max(scores, key=scores.__getitem__)
    return best_pred, scores[best_pred]


def draw_scene_graph(nodes: list, edges: list, out_path: Path):
    """
    Draw a scene graph using matplotlib only (no networkx dependency).
    nodes: list of (id, label)
    edges: list of (subj_id, pred, obj_id, score)
    """
    import math

    node_ids = [n[0] for n in nodes]
    node_labels = {n[0]: n[1] for n in nodes}
    n = len(node_ids)

    # Circular layout
    angles = [2 * math.pi * i / n for i in range(n)]
    pos = {nid: (math.cos(a), math.sin(a)) for nid, a in zip(node_ids, angles)}

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Lightweight 3D Scene Graph (RelationField)", fontsize=14, pad=20)

    # Draw edges with arrows
    edge_colors = plt.cm.tab20(np.linspace(0, 1, max(len(edges), 1)))
    for idx, (s_id, pred, o_id, score) in enumerate(edges):
        sx, sy = pos[s_id]
        ox, oy = pos[o_id]
        ax.annotate(
            "",
            xy=(ox, oy), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="->",
                color=edge_colors[idx % len(edge_colors)],
                lw=1.5,
                connectionstyle="arc3,rad=0.15",
            ),
        )
        mid_x = (sx + ox) / 2 + 0.05
        mid_y = (sy + oy) / 2 + 0.05
        ax.text(mid_x, mid_y, f"{pred}\n({score:.2f})",
                fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray", alpha=0.8))

    # Draw nodes
    for nid, (x, y) in pos.items():
        ax.plot(x, y, "o", markersize=20, color="steelblue", zorder=5)
        ax.text(x, y + 0.13, f"[{nid}]\n{node_labels[nid]}",
                fontsize=9, ha="center", va="bottom",
                fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scene graph saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str, out_dir: str, ply_override: str = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load model ---
    print("Loading model ...")
    _, pipeline, _, _ = eval_setup(Path(config_path), test_mode="inference")
    model = pipeline.model
    model.eval()
    click_scene = model.click_scene
    click_scene.clip_device = "cpu"   # keep CLIP on CPU to save VRAM
    device = model.device

    # --- 2. Find data dir and load mesh ---
    if ply_override:
        data_dir = Path(ply_override).parent
    else:
        # nerfstudio stores the data path in datamanager config
        dm_cfg = pipeline.datamanager.config
        data_dir = Path(dm_cfg.data)
    print(f"Data dir: {data_dir}")

    print(f"Sampling {N_SAMPLE_POINTS} surface points ...")
    points = load_mesh_and_sample(data_dir, N_SAMPLE_POINTS)  # [N, 3]
    print(f"  Got {len(points)} points")

    # Save point cloud preview
    np.save(out_dir / "sampled_points.npy", points)

    # --- 3. Instance embeddings → DBSCAN clustering ---
    print("Querying instance embeddings ...")
    inst_emb = query_instance_embeddings(model, points)   # [N, 256]
    print(f"  Instance embeddings shape: {inst_emb.shape}")

    print("Clustering with DBSCAN ...")
    cluster_labels = cluster_instances(inst_emb)           # [N], -1=noise
    unique_ids = [c for c in np.unique(cluster_labels) if c != -1]
    print(f"  Found {len(unique_ids)} clusters (noise points: {(cluster_labels==-1).sum()})")

    # Filter small clusters
    unique_ids = [c for c in unique_ids
                  if (cluster_labels == c).sum() >= MIN_CLUSTER_SIZE]
    print(f"  After size filter (>={MIN_CLUSTER_SIZE}): {len(unique_ids)} clusters")

    # Sort by cluster size (largest first), keep at most MAX_CLUSTERS
    unique_ids.sort(key=lambda c: -(cluster_labels == c).sum())
    unique_ids = unique_ids[:MAX_CLUSTERS]

    cluster_info = {}  # id → {pts, centroid}
    for cid in unique_ids:
        mask = cluster_labels == cid
        cluster_info[cid] = {
            "pts": points[mask],
            "centroid": points[mask].mean(axis=0),
            "size": int(mask.sum()),
        }

    # --- 4. Query semantic embeddings and label clusters ---
    print("Querying semantic embeddings for each cluster ...")
    sem_emb = query_semantic_embeddings(model, points)     # [N, D]

    print("Labeling clusters with OpenSeg/CLIP ...")
    nodes = []   # (cid, label_str)
    for cid in unique_ids:
        mask = cluster_labels == cid
        cluster_sem = sem_emb[mask]
        label, score = label_cluster(cluster_sem, click_scene, OBJECT_LABELS)
        cluster_info[cid]["label"] = label
        cluster_info[cid]["label_score"] = score
        nodes.append((cid, label))
        c = cluster_info[cid]["centroid"]
        print(f"  cluster {cid:3d}  size={cluster_info[cid]['size']:4d}  "
              f"label='{label}'  score={score:.3f}  "
              f"centroid=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")

    # Save cluster info JSON
    cluster_json = {
        str(cid): {
            "label": info["label"],
            "label_score": info["label_score"],
            "size": info["size"],
            "centroid": info["centroid"].tolist(),
        }
        for cid, info in cluster_info.items()
    }
    with open(out_dir / "clusters.json", "w") as f:
        json.dump(cluster_json, f, indent=2)

    # --- 5. Select instance pairs and predict relations ---
    print(f"\nSelecting up to {MAX_PAIRS} instance pairs for relation prediction ...")

    # Pair selection: prefer pairs that are spatially close and have different labels
    centroids = np.stack([cluster_info[c]["centroid"] for c in unique_ids])
    dists = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)

    pair_candidates = []
    for i, ci in enumerate(unique_ids):
        for j, cj in enumerate(unique_ids):
            if ci >= cj:
                continue
            li = cluster_info[ci]["label"]
            lj = cluster_info[cj]["label"]
            spatial_dist = float(dists[i, j])
            # prefer pairs within 2m and with different labels
            if spatial_dist < 2.0:
                bonus = 0.0 if li == lj else 0.5
                pair_candidates.append((spatial_dist - bonus, ci, cj))
    pair_candidates.sort(key=lambda x: x[0])
    selected_pairs = [(ci, cj) for _, ci, cj in pair_candidates[:MAX_PAIRS]]
    print(f"  Selected {len(selected_pairs)} pairs")

    edges = []   # (subj_id, predicate, obj_id, score)
    for subj_id, obj_id in selected_pairs:
        subj_pts = cluster_info[subj_id]["pts"]
        obj_centroid = cluster_info[obj_id]["centroid"]
        subj_label = cluster_info[subj_id]["label"]
        obj_label = cluster_info[obj_id]["label"]

        print(f"  Predicting: '{subj_label}' → ? → '{obj_label}'", end="  ")
        pred, score = predict_relation(model, subj_pts, obj_centroid,
                                       click_scene, RELATION_CANDIDATES)
        print(f"  → '{pred}' ({score:.3f})")
        edges.append((subj_id, pred, obj_id, score))

    # Save edges JSON
    edges_json = [
        {
            "subject_id": int(s),
            "subject_label": cluster_info[s]["label"],
            "predicate": p,
            "object_id": int(o),
            "object_label": cluster_info[o]["label"],
            "score": sc,
        }
        for s, p, o, sc in edges
    ]
    with open(out_dir / "scene_graph_edges.json", "w") as f:
        json.dump(edges_json, f, indent=2)
    print(f"\n  Edges saved: {out_dir}/scene_graph_edges.json")

    # Print readable summary
    print("\n=== Scene Graph Summary ===")
    for e in edges_json:
        print(f"  [{e['subject_id']}] {e['subject_label']:15s}  --[{e['predicate']:15s}]-->  "
              f"[{e['object_id']}] {e['object_label']:15s}  (score={e['score']:.3f})")

    # --- 6. Visualize ---
    print("\nDrawing scene graph ...")
    # Only include nodes that appear in at least one edge
    edge_node_ids = set()
    for s, _, o, _ in edges:
        edge_node_ids.add(s)
        edge_node_ids.add(o)
    vis_nodes = [(cid, cluster_info[cid]["label"])
                 for cid in unique_ids if cid in edge_node_ids]

    draw_scene_graph(vis_nodes, edges, out_dir / "scene_graph.png")

    # Also draw a 3D scatter of cluster centroids colored by cluster id
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    for idx, cid in enumerate(unique_ids):
        c = cluster_info[cid]["centroid"]
        ax3.scatter(*c, color=colors[idx], s=80, zorder=5)
        ax3.text(c[0], c[1], c[2] + 0.05,
                 f"[{cid}]{cluster_info[cid]['label']}", fontsize=6)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("Cluster Centroids (3D)")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_centroids_3d.png", dpi=100)
    plt.close(fig)
    print(f"  3D plot saved: {out_dir}/cluster_centroids_3d.png")

    print(f"\nDone. Results in: {out_dir}/")


def get_args():
    parser = argparse.ArgumentParser(description="Lightweight 3D Scene Graph extraction")
    parser.add_argument("--config", required=True,
                        help="Path to config.yml of a trained RelationField model")
    parser.add_argument("--out_dir", default="sg_results",
                        help="Output directory (default: sg_results/)")
    parser.add_argument("--ply", default=None,
                        help="Override path to the scene .ply mesh file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args.config, args.out_dir, args.ply)
