import os
import heapq
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import argparse


def build_edge_graph(mesh: trimesh.Trimesh):

    edges = mesh.edges_unique
    V = mesh.vertices
    neighbors = [[] for _ in range(len(V))]
    weights   = [[] for _ in range(len(V))]
    e_len = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    for (u, v), w in zip(edges, e_len):
        neighbors[u].append(v); weights[u].append(w)
        neighbors[v].append(u); weights[v].append(w)
    neighbors = [np.asarray(n, dtype=np.int64) for n in neighbors]
    weights   = [np.asarray(w, dtype=np.float64) for w in weights]
    return neighbors, weights


def nearest_label_all_vertices(vertices, label_to_points):

    trees = {}
    labels_sorted = sorted(label_to_points.keys(), key=lambda x: int(x))
    for lab in labels_sorted:
        P = np.asarray(label_to_points[lab], dtype=np.float64)
        trees[lab] = cKDTree(P) if len(P) > 0 else None

    V = vertices.shape[0]
    nearest_label = np.zeros(V, dtype=np.int64)
    dmin_per_v = np.full(V, np.inf, dtype=np.float64)

    for lab in labels_sorted:
        tree = trees[lab]
        if tree is None:
            continue
        d, _ = tree.query(vertices, k=1, workers=-1)
        mask = d < dmin_per_v
        dmin_per_v[mask] = d[mask]
        nearest_label[mask] = int(lab)

    return nearest_label, dmin_per_v, trees


def multisource_geodesic_propagation_with_fallback(
    neighbors, weights, seed_mask, seed_labels, fallback_labels
):

    V = len(neighbors)
    labels = np.full(V, -1, dtype=np.int64)
    dist   = np.full(V, np.inf, dtype=np.float64)
    pq = []


    for v in range(V):
        if seed_mask[v]:
            labels[v] = seed_labels[v]
            dist[v] = 0.0
            heapq.heappush(pq, (0.0, v))

    if len(pq) == 0:
        return fallback_labels.copy(), np.zeros(V, dtype=np.float64)

    # Dijkstra
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue
        lab_u = labels[u]
        for nv, w in zip(neighbors[u], weights[u]):
            nd = d_u + w
            if nd < dist[nv]:
                dist[nv] = nd
                labels[nv] = lab_u
                heapq.heappush(pq, (nd, nv))


    miss = (labels == -1)
    if np.any(miss):
        labels[miss] = fallback_labels[miss]
        dist[miss] = 0.0  

    return labels, dist


def face_majority_label(mesh: trimesh.Trimesh, vlabels, vdist):

    F = mesh.faces.shape[0]
    flabels = np.zeros(F, dtype=np.int64)
    for i in range(F):
        vs = mesh.faces[i]
        labs = vlabels[vs]
        vals, counts = np.unique(labs, return_counts=True)
        if len(vals) == 1:
            flabels[i] = vals[0]
        else:
            idx = np.argmax(counts)
            if np.sum(counts == counts[idx]) == 1:
                flabels[i] = vals[idx]
            else:
                best_lab, best_sum = None, np.inf
                for lab in vals:
                    s = vdist[vs][labs == lab].sum()
                    if s < best_sum:
                        best_sum, best_lab = s, lab
                flabels[i] = best_lab
    return flabels


def ensure_nonempty_per_label(mesh, flabels, label_to_points, min_faces=10):

    labels_sorted = sorted(label_to_points.keys(), key=lambda x: int(x))
    F = mesh.faces.shape[0]
    adj = mesh.face_adjacency  # (M,2)
    face_nbrs = [[] for _ in range(F)]
    for a, b in adj:
        face_nbrs[a].append(b)
        face_nbrs[b].append(a)

    tri_centers = mesh.triangles_center

    for lab in labels_sorted:
        lab_i = int(lab)
        if np.any(flabels == lab_i):
            continue

        P = np.asarray(label_to_points[lab], dtype=np.float64)
        if len(P) == 0:
            continue

        c = P.mean(axis=0)
        idx0 = np.argmin(np.linalg.norm(tri_centers - c[None, :], axis=1))

        picked = set([idx0])
        frontier = [idx0]
        while len(picked) < min_faces and frontier:
            new_frontier = []
            for f in frontier:
                for g in face_nbrs[f]:
                    if g not in picked:
                        picked.add(g)
                        new_frontier.append(g)
            frontier = new_frontier
        flabels[list(picked)] = lab_i

    return flabels


def export_label_submeshes(mesh: trimesh.Trimesh, flabels, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    unique_labs = np.unique(flabels)
    for lab in unique_labs:
        mask = (flabels == lab)
        if not np.any(mask):
            continue
        sub = mesh.submesh([np.nonzero(mask)[0]], append=True, repair=True)
        if sub.vertices.shape[0] == 0 or sub.faces.shape[0] == 0:
            continue

        os.makedirs(os.path.join(out_dir, f"{lab}"), exist_ok=True)
        export_path = os.path.join(out_dir,f"{lab}", f"{lab}.obj")
        sub.export(export_path)
        print(f"[+] Saved: {export_path}  (V={len(sub.vertices)}, F={len(sub.faces)})")



def segment_mesh_by_wrapped_pcd_no_minus1(
    mesh,
    label_to_points: dict,
    out_dir: str = "out_submeshes",
    seed_tau_ratio: float = 0.02,
    min_seed_faces: int = 20
):

    
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    V = mesh.vertices
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    tau_seed = bbox_diag * seed_tau_ratio

    nearest_lab, dmin, _ = nearest_label_all_vertices(mesh.vertices, label_to_points)


    neighbors, weights = build_edge_graph(mesh)
    seed_mask = (dmin <= tau_seed)
    vlabels, vdist = multisource_geodesic_propagation_with_fallback(
        neighbors, weights,
        seed_mask=seed_mask,
        seed_labels=nearest_lab,     
        fallback_labels=nearest_lab  
    )

    flabels = face_majority_label(mesh, vlabels, vdist)

    flabels = ensure_nonempty_per_label(mesh, flabels, label_to_points, min_faces=min_seed_faces)

    export_label_submeshes(mesh, flabels, out_dir)


import logging


LOGGER = logging.getLogger("physx_cot.split")


def split_directory(basepath: str, voxel_resolution: int = 32) -> None:
    for name in sorted(os.listdir(basepath)):
        sample_dir = os.path.join(basepath, name)
        mesh_path = os.path.join(sample_dir, "sample.glb")
        if not os.path.isfile(mesh_path):
            continue

        output_dir = os.path.join(sample_dir, "objs")
        os.makedirs(output_dir, exist_ok=True)
        mesh = trimesh.load(mesh_path, force="mesh")
        rotation = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
        mesh.apply_transform(rotation)

        part_points = {}
        part_index = 0
        while True:
            points_path = os.path.join(sample_dir, f"ind_{part_index}.npy")
            if not os.path.isfile(points_path):
                break
            part_points[str(part_index)] = np.load(points_path) / voxel_resolution - 0.5
            part_index += 1

        if not part_points:
            LOGGER.warning("Skipping %s: no per-part voxel files found", name)
            continue

        segment_mesh_by_wrapped_pcd_no_minus1(
            mesh=mesh,
            label_to_points=part_points,
            out_dir=output_dir,
            seed_tau_ratio=0.02,
            min_seed_faces=20,
        )
        LOGGER.info("Completed %s", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split decoded meshes into PhysX-CoT parts.")
    parser.add_argument("--basepath", required=True, help="Directory containing per-sample outputs.")
    parser.add_argument("--voxel_resolution", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not os.path.isdir(args.basepath):
        parser.error(f"output directory not found: {args.basepath}")
    split_directory(args.basepath, voxel_resolution=args.voxel_resolution)


if __name__ == "__main__":
    main()
                
