# Copyright (c) 2025 - Evaluation script for relationship query analysis
"""
Offline evaluation script for RelationField relationship queries.
Evaluates 20 queries (Comparison A: noun vs relation, Comparison B: simple vs abstract)
on replica_office0 without requiring the interactive viewer.

Usage:
    python eval_relation_queries.py --config outputs/replica_office0/relationfield/2026-03-09_231057/config.yml
"""

import argparse
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.eval_utils import eval_setup

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------

# Approximate 3D positions for reference objects in replica_office0
# (x, y, z) in scene coordinates. Adjust if results look off.
POSITIONS = {
    "chair":     np.array([ 0.3,  -1.5,  -0.3]),  # chair near table
    "table":     np.array([ 0.5,  -1.0,  -0.2]),  # main table
    "couch":     np.array([-1.0,   0.2,  -0.2]),  # couch area
    "floor":     np.array([ 0.3,  -0.8,  -0.9]),  # floor center
    "wall":      np.array([ 2.1,  -0.5,   0.3]),  # wall (for paintings)
    "trash_can": np.array([ 1.0,  -2.5,  -0.5]),  # trash can area
    "screen":    np.array([ 1.5,  -0.5,   0.3]),  # screen on wall
    "window":    np.array([ 0.5,   1.5,   0.5]),  # window/curtain
    "carpet":    np.array([ 0.0,  -1.0,  -0.85]), # carpet on floor
    "cushion":   np.array([-0.8,   0.3,  -0.1]),  # cushion on couch
}

# 20 test queries split into two comparisons
QUERIES = {
    # --- Comparison A: Noun query vs Relationship query ---
    # Same anchor object, check if relation helps disambiguate
    "A1_noun":     {"type": "object",   "text": "chair",       "ref": "chair"},
    "A1_relation": {"type": "relation", "text": "chair standing on floor", "ref": "floor"},
    "A2_noun":     {"type": "object",   "text": "trash can",   "ref": "trash_can"},
    "A2_relation": {"type": "relation", "text": "trash can next to trash can", "ref": "trash_can"},
    "A3_noun":     {"type": "object",   "text": "screen",      "ref": "screen"},
    "A3_relation": {"type": "relation", "text": "screen attached to wall", "ref": "wall"},
    "A4_noun":     {"type": "object",   "text": "cushion",     "ref": "cushion"},
    "A4_relation": {"type": "relation", "text": "cushion lying on couch", "ref": "couch"},
    "A5_noun":     {"type": "object",   "text": "table",       "ref": "table"},
    "A5_relation": {"type": "relation", "text": "table standing on floor", "ref": "floor"},

    # --- Comparison B: Simple spatial vs Abstract/complex ---
    # Simple spatial (should work well)
    "B1_simple":   {"type": "relation", "text": "painting attached to wall",  "ref": "wall"},
    "B2_simple":   {"type": "relation", "text": "curtain hanging on window",  "ref": "window"},
    "B3_simple":   {"type": "relation", "text": "carpet lying on floor",      "ref": "floor"},
    "B4_simple":   {"type": "relation", "text": "cushion above couch",        "ref": "couch"},
    "B5_simple":   {"type": "relation", "text": "door part of wall",          "ref": "wall"},

    # Abstract (may be less stable)
    "B6_abstract": {"type": "relation", "text": "screen same type screen",    "ref": "screen"},
    "B7_abstract": {"type": "relation", "text": "chair same type chair",      "ref": "chair"},
    "B8_abstract": {"type": "relation", "text": "trash can beside trash can", "ref": "trash_can"},
    "B9_abstract": {"type": "relation", "text": "floor supporting carpet",    "ref": "carpet"},
    "B10_reverse": {"type": "relation", "text": "couch lying on cushion",     "ref": "cushion"},  # reversed (should fail)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_reference_position(model, pipeline, approx_pos: np.ndarray) -> tuple:
    """Mirror viewer's _on_rayclick_relation: shoot a ray from the nearest training
    camera toward approx_pos, use proposal_sampler to find the actual surface, and
    return (ref_position_nerf, top_k_ray_samples) — both in NeRF coordinate space."""
    device = model.device

    # Find the training camera whose origin is closest to approx_pos
    cam_to_worlds = pipeline.datamanager.train_dataset.cameras.camera_to_worlds  # (N,3,4)
    cam_origins = cam_to_worlds[:, :3, 3].numpy()          # (N, 3) in NeRF coords
    dists = np.linalg.norm(cam_origins - approx_pos[None], axis=-1)
    best_idx = int(np.argmin(dists))
    cam_origin = cam_origins[best_idx]                     # (3,)

    direction = approx_pos - cam_origin
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    origin_t = torch.tensor(cam_origin, dtype=torch.float32).view(1, 3).to(device)
    dir_t    = torch.tensor(direction,  dtype=torch.float32).view(1, 3).to(device)

    bundle = RayBundle(
        origins=origin_t,
        directions=dir_t,
        pixel_area=torch.full((1, 1), 1e-3, device=device),
        nears=torch.full((1, 1), 0.05,  device=device),
        fars=torch.full((1, 1), 100.0,  device=device),
        camera_indices=torch.zeros((1, 1), dtype=torch.long, device=device),
    )

    with torch.no_grad():
        ray_samples, _, _ = model.proposal_sampler(bundle, density_fns=model.density_fns)
        field_out = model.field.forward(ray_samples, compute_normals=False)
        weights = ray_samples.get_weights(field_out[FieldHeadNames.DENSITY])

        n_feat = model.config.num_feat_samples
        _, best_ids = torch.topk(weights, n_feat, dim=-2, sorted=False)

        def gather_fn(t):
            return torch.gather(t, -2, best_ids.expand(*best_ids.shape[:-1], t.shape[-1]))
        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)

        top_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)
        top_weights = torch.gather(weights, -2, best_ids)
        depth = model.renderer_depth(weights=top_weights, ray_samples=top_samples)

    dist_val = depth[0, 0].item()
    ref_pos = cam_origin + direction * dist_val
    print(f"    approx={np.round(approx_pos,3)}  →  surface={np.round(ref_pos,3)}  (depth={dist_val:.3f})")
    return ref_pos, top_samples


def get_camera(pipeline, frame_idx: int = 80, downscale: float = 8.0) -> Cameras:
    """Return a single training camera, downscaled to save VRAM."""
    cameras = pipeline.datamanager.train_dataset.cameras
    camera = cameras[frame_idx : frame_idx + 1].to("cpu")
    camera.rescale_output_resolution(1.0 / downscale)
    return camera


def render_camera(pipeline, camera: Cameras, chunk_size: int = 512) -> dict:
    """Render a single camera in small chunks, keeping only rgb + relevancy scalars.
    Discards high-dim feature tensors (openseg, clip, relation_map) immediately after
    each chunk to avoid OOM."""
    model = pipeline.model
    model.eval()
    with torch.no_grad():
        ray_bundle = camera.generate_rays(
            camera_indices=0,
            aabb_box=model.scene_box if hasattr(model, "scene_box") else None,
        )
        # Flatten (H, W) → (H*W,) so chunks are individual rays, not rows
        ray_bundle = ray_bundle.flatten()
        n_rays = ray_bundle.origins.shape[0]
        rgb_list, relevancy_list, raw_list = [], [], []

        for i in range(0, n_rays, chunk_size):
            # slice and move chunk to GPU
            chunk = ray_bundle[i : i + chunk_size].to(model.device)
            out = model.get_outputs(chunk)

            rgb_list.append(out["rgb"].detach().cpu())
            # relation relevancy (scalar per ray) — only exists when relation query active
            if "relation_relevancy_scaled" in out:
                relevancy_list.append(out["relation_relevancy_scaled"].detach().cpu())
            # object query: compute relevancy from openseg in-chunk then discard openseg
            click_scene = pipeline.model.click_scene
            feat_key = click_scene._dropdown_value  # "openseg" or "clip"
            if feat_key in out and click_scene.pos_embeds is not None:
                import torch.nn.functional as F_
                feat = F_.normalize(out[feat_key].view(-1, out[feat_key].shape[-1]), dim=-1)
                rel = click_scene.get_relevancy(feat, 0)
                if rel is not None:
                    raw_list.append(rel[:, 0].detach().cpu())

            del out, chunk
            torch.cuda.empty_cache()

    result = {"rgb": torch.cat(rgb_list, dim=0)}
    if relevancy_list:
        result["relation_relevancy_scaled"] = torch.cat(relevancy_list, dim=0)
    if raw_list:
        result["relevancy"] = torch.cat(raw_list, dim=0)
    return result


def overlay_heatmap(rgb: torch.Tensor, activation: torch.Tensor,
                    thresh: float = 0.3, normalize: bool = False) -> np.ndarray:
    """Overlay turbo heatmap on RGB image. Returns HxWx3 numpy array in [0,1]."""
    act = activation.detach().cpu().float()
    rgb_np = rgb.detach().cpu().float()

    if normalize:
        # stretch to full [0,1] range regardless of absolute values
        a_min, a_max = act.min(), act.max()
        if (a_max - a_min) > 1e-8:
            act = (act - a_min) / (a_max - a_min)
        thresh = 0.0  # threshold is meaningless after normalization

    act = torch.clamp(act - thresh, 0, 1)
    if act.max() > 1e-6:
        act = act / act.max()
    act = torch.clamp(act, 0, 0.85)

    cmap = torch.tensor(matplotlib.colormaps["turbo"].colors, dtype=torch.float32)
    overlay = cmap[(act * 255).long()]  # HxWx3
    mask = (act <= 0)
    alpha = 0.35
    result = 0.9 * overlay + 0.1 * rgb_np
    result[mask] = (1 - alpha) * rgb_np[mask]
    return result.numpy()


def save_result(rgb_np: np.ndarray, heatmap_np: np.ndarray, query_name: str,
                query_text: str, ref_name: str, out_dir: Path):
    """Save side-by-side comparison: RGB | heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb_np)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(heatmap_np)
    axes[1].set_title(f"Query: \"{query_text}\"\nRef: {ref_name}")
    axes[1].axis("off")
    fig.suptitle(query_name, fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f"{query_name}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_eval(config_path: str, out_dir: str, frame_idx: int = 80, thresh: float = 0.3, downscale: float = 8.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    _, pipeline, _, _ = eval_setup(Path(config_path), test_mode="inference")
    model = pipeline.model
    model.eval()
    click_scene = model.click_scene
    # Keep CLIP model on CPU to save VRAM (only embeddings are moved to GPU)
    click_scene.clip_device = "cpu"

    print(f"Rendering camera frame {frame_idx}...")
    camera = get_camera(pipeline, frame_idx, downscale=downscale)
    # Keep camera on CPU — rays are generated on CPU and moved chunk by chunk to GPU
    H = int(camera.height[0].item())
    W = int(camera.width[0].item())
    print(f"  Render resolution: {W}x{H} ({H*W} rays)")

    # Render base RGB once (no query active)
    torch.cuda.empty_cache()
    base_outputs = render_camera(pipeline, camera)
    rgb_base = base_outputs["rgb"].detach().cpu().view(H, W, 3)
    rgb_np = rgb_base.numpy()
    del base_outputs
    torch.cuda.empty_cache()

    # Save base RGB
    plt.imsave(out_dir / "00_base_rgb.png", rgb_np)
    print(f"  Saved base RGB: {out_dir}/00_base_rgb.png  ({W}x{H})")

    # Pre-compute reference positions by shooting rays from training cameras,
    # mirroring the viewer's _on_rayclick_relation logic.
    print("\nComputing reference positions via proposal sampler...")
    ref_position_cache: dict = {}   # ref_name -> (position_np, ray_samples_topk)
    needed_refs = {q["ref"] for q in QUERIES.values() if q["type"] == "relation"}
    for ref_name in sorted(needed_refs):
        print(f"  [{ref_name}]")
        approx = POSITIONS[ref_name]
        ref_pos, ref_samples = compute_reference_position(model, pipeline, approx)
        ref_position_cache[ref_name] = (ref_pos, ref_samples)
    torch.cuda.empty_cache()

    results_summary = []

    for query_name, q in QUERIES.items():
        print(f"\n[{query_name}] '{q['text']}' | ref: {q['ref']}")

        # Reset click scene state
        click_scene.selected_relation_position = None
        click_scene.selected_relation_samples = None
        click_scene.pos_embeds = None
        click_scene.neg_embeds = None
        click_scene.bert_pos_embds = None
        click_scene.bert_neg_embds = None
        click_scene.positives = []
        click_scene.relation_positives = []

        torch.cuda.empty_cache()
        if q["type"] == "object":
            # --- Object query (CLIP-based) ---
            click_scene.set_positives([q["text"]])
            outputs = render_camera(pipeline, camera)

            if "relevancy" not in outputs:
                print("  [SKIP] No relevancy output for object query")
                del outputs
                torch.cuda.empty_cache()
                continue
            activation = outputs["relevancy"].view(H, W).detach().cpu()

        else:
            # --- Relation query (Jina-based) ---
            ref_pos, ref_samples = ref_position_cache[q["ref"]]
            click_scene.set_relation_positives([q["text"]])
            click_scene.selected_relation_position = ref_pos
            click_scene.selected_relation_samples = ref_samples

            outputs = render_camera(pipeline, camera)

            if "relation_relevancy_scaled" not in outputs:
                print("  [SKIP] No relation_relevancy_scaled in outputs (relation field not active?)")
                del outputs
                torch.cuda.empty_cache()
                continue
            activation = outputs["relation_relevancy_scaled"].view(H, W).detach().cpu()

        del outputs
        torch.cuda.empty_cache()

        # Normalize relation queries so the heatmap spans the full dynamic range.
        # Absolute magnitude depends on distance-scaling; relative pattern matters more.
        use_norm = (q["type"] == "relation")
        heatmap_np = overlay_heatmap(rgb_base, activation, thresh=thresh, normalize=use_norm)
        save_result(rgb_np, heatmap_np, query_name, q["text"], q["ref"], out_dir)

        # Record max/mean relevancy as semi-quantitative metric
        act_vals = activation.detach().cpu().float()
        results_summary.append({
            "query": query_name,
            "text": q["text"],
            "type": q["type"],
            "ref": q["ref"],
            "max_relevancy": float(act_vals.max()),
            "mean_relevancy": float(act_vals.mean()),
            "top10pct_mean": float(act_vals.flatten().topk(max(1, int(0.1 * H * W)))[0].mean()),
        })

    # Save summary CSV
    import csv
    csv_path = out_dir / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "text", "type", "ref",
                                                "max_relevancy", "mean_relevancy", "top10pct_mean"])
        writer.writeheader()
        writer.writerows(results_summary)
    print(f"\nSaved summary: {csv_path}")

    # Print summary table
    print("\n=== Results Summary ===")
    print(f"{'Query':<20} {'Type':<10} {'Max':>8} {'Mean':>8} {'Top10%':>8}  Text")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['query']:<20} {r['type']:<10} {r['max_relevancy']:>8.4f} "
              f"{r['mean_relevancy']:>8.4f} {r['top10pct_mean']:>8.4f}  {r['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to config.yml from training output")
    parser.add_argument("--out_dir", default="eval_results",
                        help="Directory to save result images and CSV")
    parser.add_argument("--frame", type=int, default=80,
                        help="Camera frame index to render (0-199)")
    parser.add_argument("--thresh", type=float, default=0.3,
                        help="Relevancy threshold for heatmap overlay")
    parser.add_argument("--downscale", type=float, default=8.0,
                        help="Downscale factor for rendering resolution (8=1/8 width&height)")
    args = parser.parse_args()

    run_eval(args.config, args.out_dir, args.frame, args.thresh, args.downscale)
