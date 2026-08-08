import os
import numpy as np
import re
import json
import xml.etree.ElementTree as ET
import argparse
from scipy.spatial import cKDTree as KDTree
import trimesh
from typing import List, Dict, Optional
from collections import defaultdict, deque

import logging
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
def _pairwise_nn(a: np.ndarray, b: np.ndarray):


    ta = KDTree(a)
    tb = KDTree(b)
    dist_ab, idx_ab = tb.query(a, k=1, workers=-1)
    dist_ba, idx_ba = ta.query(b, k=1, workers=-1)
    
    return idx_ab, dist_ab, idx_ba, dist_ba

def _robust_threshold(d, method="mad", q=0.2, k=2.5):

    d = np.asarray(d)
    if method == "quantile":
        return np.quantile(d, q)
    # MAD
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-12
    return med + k * 1.4826 * mad

def find_adjacent_region(
    a: np.ndarray,
    b: np.ndarray,
    thr: float | None = None,
    thr_mode: str = "mad",  
    q: float = 0.2,
    expand_radius: float | None = None,
):

    assert a.ndim == 2 and a.shape[1] == 3
    assert b.ndim == 2 and b.shape[1] == 3

    idx_ab, dist_ab, idx_ba, dist_ba = _pairwise_nn(a, b)

   
    mutual = np.arange(len(a)) == idx_ba[idx_ab]
    d_mutual = dist_ab[mutual]
    i_a = np.nonzero(mutual)[0]
    j_b = idx_ab[mutual]

    if len(i_a) == 0:
        return dict(a_idx=np.array([], dtype=int),
                    b_idx=np.array([], dtype=int),
                    pairs=np.empty((0,2), dtype=int),
                    midpoints=np.empty((0,3), dtype=a.dtype),
                    plane=None,
                    thr=0.0)

    used_thr = _robust_threshold(d_mutual, thr_mode, q=q) if thr is None else thr

    keep = d_mutual <= used_thr
    i_a = i_a[keep]
    j_b = j_b[keep]
    d_kept = d_mutual[keep]

    if len(i_a) == 0 and len(d_mutual) > 0 and thr is None:
        used_thr = _robust_threshold(d_mutual, "quantile", q=max(0.4, q))
        keep = d_mutual <= used_thr
        i_a = np.nonzero(mutual)[0][keep]
        j_b = idx_ab[mutual][keep]

    pairs = np.stack([i_a, j_b], axis=1) if len(i_a) else np.empty((0,2), dtype=int)
    midpoints = (a[i_a] + b[j_b]) * 0.5 if len(i_a) else np.empty((0,3), dtype=a.dtype)

    def _expand_within_cloud(points, seeds, radius):
        if radius is None or len(seeds) == 0:
            return np.unique(seeds)
        
        t = KDTree(points)
        idxs = set(seeds.tolist())
        for p in points[seeds]:
            hits = t.query_ball_point(p, r=radius)
            idxs.update(hits)
        return np.fromiter(idxs, dtype=int)


    a_idx = np.unique(i_a)
    b_idx = np.unique(j_b)
    a_idx = _expand_within_cloud(a, a_idx, expand_radius)
    b_idx = _expand_within_cloud(b, b_idx, expand_radius)

    plane = None
    if len(midpoints) >= 3:
        c = midpoints.mean(axis=0)
        X = midpoints - c
        # SVD
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        n = vh[-1]  

        n = n / (np.linalg.norm(n) + 1e-12)
        plane = (c, n)

    return dict(
        a_idx=a_idx,
        b_idx=b_idx,
        pairs=pairs,
        midpoints=midpoints,
        plane=plane,
        thr=float(used_thr),
    )




######################################


#refine_voxel
GRID = 32
NEI6 = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int8)

def rasterize(points, grid=GRID):
    occ = np.zeros((grid,grid,grid), dtype=bool)
    pts = np.asarray(points, dtype=np.int16)
    mask = ((pts>=0)&(pts<grid)).all(1)
    x,y,z = pts[mask].T
    occ[x,y,z] = True
    return occ

def boundary_mask(occ):
    bnd = np.zeros_like(occ)
    xs,ys,zs = np.where(occ)
    for dx,dy,dz in NEI6:
        x2 = np.clip(xs+dx, 0, occ.shape[0]-1)
        y2 = np.clip(ys+dy, 0, occ.shape[1]-1)
        z2 = np.clip(zs+dz, 0, occ.shape[2]-1)
        bnd[xs,ys,zs] |= ~occ[x2,y2,z2]
    return bnd

def idx_to_xyz(idx):
    return np.stack(np.where(idx), axis=1).astype(np.int16)

def most_adjacent_shell_6n(A_xyz, B_xyz, grid=GRID):
    A = rasterize(A_xyz, grid)
    B = rasterize(B_xyz, grid)
    A_front = boundary_mask(A) & A
    B_front = boundary_mask(B) & B

    touch_pairs = []
    if A_front.any() and B_front.any():
        Ax,Ay,Az = np.where(A_front)
        Aset = set(zip(Ax,Ay,Az))
        B_occ = B
        for dx,dy,dz in NEI6:
            nb = (np.clip(Ax+dx,0,grid-1),
                  np.clip(Ay+dy,0,grid-1),
                  np.clip(Az+dz,0,grid-1))
            hit = B_occ[nb]
            if hit.any():
                for (x,y,z),(x2,y2,z2),h in zip(zip(Ax,Ay,Az),
                                                zip(*nb), hit):
                    if h:
                        touch_pairs.append(((x,y,z),(x2,y2,z2)))
        if touch_pairs:
            mid = np.array([(np.array(a)+np.array(b))/2.0 for a,b in touch_pairs], dtype=np.float32)
            return {
                "metric": "6-neighbor steps",
                "min_grid_distance": 1,
                "pairs": np.array(touch_pairs, dtype=np.int16),
                "midpoints": mid,  # (M,3), 
            }

    A_wave = A_front.copy()
    B_wave = B_front.copy()
    visitedA = A_front.copy()
    visitedB = B_front.copy()
    dist = 1  

    while True:
        def dilate_once(wave, solid):
            xs,ys,zs = np.where(wave)
            nxt = np.zeros_like(wave)
            for dx,dy,dz in NEI6:
                x2 = np.clip(xs+dx, 0, grid-1)
                y2 = np.clip(ys+dy, 0, grid-1)
                z2 = np.clip(zs+dz, 0, grid-1)
                nxt[x2,y2,z2] = True
            nxt &= ~solid
            return nxt

        A_next = dilate_once(A_wave, A)
        B_next = dilate_once(B_wave, B)

        A_next &= ~visitedA
        B_next &= ~visitedB

        visitedA |= A_next
        visitedB |= B_next


        meet = A_next & visitedB
        if meet.any():
            meet_xyz = idx_to_xyz(meet)
 
            B_prev = B_wave  
            pairs = []
            for x,y,z in meet_xyz:
                for dx,dy,dz in NEI6:
                    x2 = np.clip(x+dx,0,grid-1); y2 = np.clip(y+dy,0,grid-1); z2 = np.clip(z+dz,0,grid-1)
                    if B_prev[x2,y2,z2]:
                        pairs.append(((x,y,z),(x2,y2,z2)))
            pairs = np.unique(np.array(pairs, dtype=np.int16), axis=0)
            mid = (pairs[:,0,:].astype(np.float32)+pairs[:,1,:].astype(np.float32))/2.0
            return {
                "metric": "6-neighbor steps",
                "min_grid_distance": dist+1,  
                "pairs": pairs,               # (M,2,3) 
                "midpoints": mid,             # (M,3)
            }

        if not (A_next.any() or B_next.any()):
            return {"metric":"6-neighbor steps","min_grid_distance":None,"pairs":np.zeros((0,2,3),np.int16),"midpoints":np.zeros((0,3),np.float32)}
        A_wave, B_wave = A_next, B_next
        dist += 1


def bbox_corners_and_edge_midpoints(pts: np.ndarray):

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs

    corners = np.array([[x, y, z]
                        for x in [x0, x1]
                        for y in [y0, y1]
                        for z in [z0, z1]], dtype=float)
    corners = np.unique(corners, axis=0)


    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    edge_mids = []
    for y in [y0, y1]:
        for z in [z0, z1]:
            edge_mids.append([xm, y, z])
    for x in [x0, x1]:
        for z in [z0, z1]:
            edge_mids.append([x, ym, z])
    for x in [x0, x1]:
        for y in [y0, y1]:
            edge_mids.append([x, y, zm])
    edge_mids = np.unique(np.array(edge_mids, dtype=float), axis=0)

    center = np.array([xm, ym, zm], dtype=float)
    return corners, edge_mids, center

def generate_allcandidate(ind_a_index,ind_b_index,datapath):

    ind_a=[]
    ind_b=[]
    for ind in ind_a_index:
        ind_a.append(np.load(os.path.join(datapath,'ind_'+str(ind)+'.npy')))

    for ind in ind_b_index:
        ind_b.append(np.load(os.path.join(datapath,'ind_'+str(ind)+'.npy')))

    ind_a=np.concatenate(ind_a)
    ind_b=np.concatenate(ind_b)

    results=most_adjacent_shell_6n(ind_a,ind_b)
    ind_a_nei=results['pairs'][:,0]
    ind_b_nei=results['pairs'][:,1]




    corners, edge_mids, center=bbox_corners_and_edge_midpoints(ind_a_nei)
    bbox_corners_a=np.concatenate([center[None]])
    

    corners, edge_mids, center=bbox_corners_and_edge_midpoints(ind_b_nei)
    bbox_corners_b=np.concatenate([center[None]])

    allcandidate=np.concatenate([bbox_corners_a,bbox_corners_b])
    allcandidate=allcandidate/32-0.5
    return allcandidate

def generate_allcandidate_center(ind_a_index,ind_b_index,datapath):

    ind_a=[]
    ind_b=[]
    for ind in ind_a_index:
        ind_a.append(np.load(os.path.join(datapath,'ind_'+str(ind)+'.npy')))

    for ind in ind_b_index:
        ind_b.append(np.load(os.path.join(datapath,'ind_'+str(ind)+'.npy')))

    ind_a=np.concatenate(ind_a)
    ind_b=np.concatenate(ind_b)

    results=most_adjacent_shell_6n(ind_a,ind_b)
    ind_a_nei=results['pairs'][:,0]
    ind_b_nei=results['pairs'][:,1]

    corners, edge_mids, center=bbox_corners_and_edge_midpoints(ind_a_nei)
    bbox_corners_a=np.concatenate([corners, edge_mids, center[None]]).mean(0)
    

    corners, edge_mids, center=bbox_corners_and_edge_midpoints(ind_b_nei)
    bbox_corners_b=np.concatenate([corners, edge_mids, center[None]]).mean(0)

    allcandidate=(bbox_corners_a+bbox_corners_b)/2
    allcandidate=allcandidate/32-0.5
    return allcandidate
#############################################
def make_origin_element(xyz, rpy):
    origin = ET.Element('origin')
    origin.set('xyz', ' '.join(xyz))
    origin.set('rpy', ' '.join(rpy))
    return origin

def add_inertial(link_element,xyz="0 0 0"):
    inertial = ET.SubElement(link_element, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=xyz, rpy="0 0 0")
    ET.SubElement(inertial, 'mass', value="1.0")
    ET.SubElement(inertial, 'inertia', ixx="1.0", ixy="0.0", ixz="0.0",
                  iyy="1.0", iyz="0.0", izz="1.0")

def add_fixed_joint(robot, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    return joint


def _to_nums(lst, expect_len):
    out = []
    for s in lst:
        s = s.strip()
        if s:
            try:
                v = float(int(s))
            except ValueError:
                try:
                    v = float(s)
                except ValueError:
                    v = 0.0
        else:
            v = 0.0
        out.append(v)
    if len(out) < expect_len:
        out += [0.0] * (expect_len - len(out))
    elif len(out) > expect_len:
        out = out[:expect_len]
    return out

def clean_npfloat64(values):
    cleaned = []
    for s in values:
        s = s.strip()
        if s.startswith('np.float64('): 
            num_str = re.sub(r'.*?\((.*?)\)', r'\1', s)
            cleaned.append((num_str))
        else:
            cleaned.append((s))
    return cleaned

def _extract_bracket_list(block, key, expect_len):

    pattern = rf'{re.escape(key)}[^:\[]*:\s*\[([^\]]*)\]'
    m = re.search(pattern, block, flags=re.IGNORECASE)
    if not m:
        return [0.0] * expect_len
    raw = m.group(1)
    items = [x for x in raw.split(',')]
    items=clean_npfloat64(items)
    return _to_nums(items, expect_len)

#mujuco


def find_body_by_name(root: ET.Element, name: str) -> ET.Element:
    for elem in root.iter("body"):
        if elem.get("name") == name:
            return elem
    return None

def move_element(child: ET.Element, new_parent: ET.Element):
    old_parent = child.getparent() if hasattr(child, "getparent") else None
    if old_parent is None:
        for elem in new_parent.iter():
            pass
    def _find_parent(root, node):
        for e in root.iter():
            for c in list(e):
                if c is node:
                    return e
        return None

    root = new_parent
    while root.getparent() is not None if hasattr(root, "getparent") else False:
        root = root.getparent()

    parent = _find_parent(root, child)
    if parent is not None:
        parent.remove(child)
    new_parent.append(child)

def reparent_by_group_info(mjcf_root: ET.Element, group_info: dict,
                           base_body_name: str = "base",
                           group_body_prefix: str = "grouppart_"):

    parent_of = {}
    for gkey, gval in group_info.items():
        if str(gkey) == "0":
            continue

        try:
            parent_str = str(gval[1])
        except Exception as e:
            raise ValueError(f"group_info['{gkey}'] lack parent group: {gval}") from e
        parent_of[str(gkey)] = parent_str


    base_body = find_body_by_name(mjcf_root, base_body_name)
    if base_body is None:
        raise ValueError(f"cannot find base body: name='{base_body_name}'")

    def body_name_for_group(gid: str) -> str:
        if gid == "0":
            return base_body_name
        return f"{group_body_prefix}{gid}"

    children_of = defaultdict(list)
    indeg = defaultdict(int)
    nodes = set(["0"])  
    for c, p in parent_of.items():
        nodes.add(c); nodes.add(p)
        children_of[p].append(c)
        indeg[c] += 1
        indeg.setdefault(p, 0)

    q = deque([n for n in nodes if indeg[n] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in children_of[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != len(nodes):
        raise ValueError("Detect loop in group_info")

    
    for gid in topo:
        if gid == "0":
            continue
        child_name = body_name_for_group(gid)
        parent_name = body_name_for_group(parent_of[gid])

        child_body = find_body_by_name(mjcf_root, child_name)
        parent_body = find_body_by_name(mjcf_root, parent_name)

        if child_body is None:
            print(f"skip: {child_name}")
            continue
        if parent_body is None:
            raise ValueError(f"cannot find parent body: {parent_name} (child group: {gid} parent group: {parent_of[gid]} ")

        already_child = False
        for c in list(parent_body):
            if c is child_body:
                already_child = True
                break
        if already_child:
            continue


        def find_parent(root, node):
            for e in mjcf_root.iter():
                for c in list(e):
                    if c is node:
                        return e
            return None

        old_parent = find_parent(mjcf_root, child_body)
        if old_parent is not None:
            old_parent.remove(child_body)
        parent_body.append(child_body)


def _indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for e in elem:
            _indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def generate_mjcf(
    jsondata: dict={},
    fixed_base: int = 0,
    out_path: str = "test.xml",
    model_name: str = "test",
    # physics / options
    angle_unit: str = "radian",
    timestep: float = 0.002,
    gravity: str = "0 0 -9.81",
    wind: str = "0 0 0",
    integrator: str = "implicitfast",
    density: float = 1.225,
    viscosity: float = 1.8e-5,
    # visual
    realtime: int = 1,
    shadowsize: int = 16384,
    numslices: int = 28,
    offsamples: int = 4,
    headlight_diffuse: str = "2 2 2",
    headlight_specular: str = "0.5 0.5 0.5",
    headlight_active: int = 1,
    rgba_fog: str = "0 1 0 1",
    rgba_haze: str = "1 0 0 1",
    # skybox
    skybox_file: Optional[str] = None,
    skybox_gridsize: str = "3 4",
    skybox_gridlayout: str = ".U..LFRB.D..",
    # plane checker
    plane_texture_name: str = "plane",
    plane_material_name: str = "plane",
    plane_rgb1: str = ".1 .1 .1",
    plane_rgb2: str = ".5 .5 .5",
    plane_width: int = 512,
    plane_height: int = 512,
    plane_mark: str = "cross",
    plane_markrgb: str = ".8 .8 .8",
    plane_reflectance: float = 0.3,
    plane_texrepeat: str = "1 1",
    plane_texuniform: str = "true",
    # contact / fluid defaults
    geom_solref: str = ".5e-4",
    geom_solimp: str = "0.9 0.99 1e-4",
    geom_fluidcoef: str = "0.5 0.25 0.5 2.0 1.0",
    # parts (each part creates: mesh, texture, material, default class, and sample body usage)
    parts: List[Dict] = None,
    # world items
    floor_condim: int = 6,
    floor_size: str = "0 0 .25",
    light_pos: str = "30 30 30",
    light_dir: str = "0 -2 -1",
    light_ambient: str = ".3 .3 .3",
    light_diffuse: str = ".5 .5 .5",
    light_specular: str = ".5 .5 .5",
    # demo body placements
    base_pos: str = "0 0 1",
    base_euler: str = "0 0 0",
    part_pos: str = "0 0 1.2",
    part_euler: str = "1.5 0 0",
    deformable: int = 0,
):
    
    if parts is None or len(parts) == 0:
        raise ValueError("at least one part")

    # ---- root
    mujoco = ET.Element("mujoco", attrib={"model": model_name})
    ET.SubElement(mujoco, "compiler", attrib={"angle": angle_unit, "autolimits": "true"})
    ET.SubElement(
        mujoco,
        "option",
        attrib={
            "timestep": f"{timestep}",
            "gravity": gravity,
            "wind": wind,
            "integrator": integrator,
            "density": f"{density}",
            "viscosity": f"{viscosity}",
        },
    )

    # ---- visual
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "global", attrib={"realtime": str(realtime)})
    ET.SubElement(
        visual,
        "quality",
        attrib={"shadowsize": str(shadowsize), "numslices": str(numslices), "offsamples": str(offsamples)},
    )
    ET.SubElement(
        visual,
        "headlight",
        attrib={"diffuse": headlight_diffuse, "specular": headlight_specular, "active": str(headlight_active)},
    )
    ET.SubElement(visual, "rgba", attrib={"fog": rgba_fog, "haze": rgba_haze})

    # ---- asset
    asset = ET.SubElement(mujoco, "asset")

    # parts assets
    for p in parts:
        pname = p["name"]
        # mesh
        ET.SubElement(
            asset,
            "mesh",
            attrib={"name": pname, "file": p["mesh_file"], "scale": p.get("scale", "1 1 1")},
        )
        # texture
        tex_name = f"{pname}_tex"
        ET.SubElement(asset, "texture", attrib={"type": "2d", "name": tex_name, "file": p["tex_file"]})
        # material
        mat_name = f"{pname}_img"
        ET.SubElement(asset, "material", attrib={"name": mat_name, "texture": tex_name})

    # skybox
    if skybox_file:
        ET.SubElement(
            asset,
            "texture",
            attrib={"type": "skybox", "file": skybox_file, "gridsize": skybox_gridsize, "gridlayout": skybox_gridlayout},
        )

    # plane checker texture + material
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "name": plane_texture_name,
            "type": "2d",
            "builtin": "checker",
            "rgb1": plane_rgb1,
            "rgb2": plane_rgb2,
            "width": str(plane_width),
            "height": str(plane_height),
            "mark": plane_mark,
            "markrgb": plane_markrgb,
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": plane_material_name,
            "reflectance": str(plane_reflectance),
            "texture": plane_texture_name,
            "texrepeat": plane_texrepeat,
            "texuniform": plane_texuniform,
        },
    )

    # ---- default
    default = ET.SubElement(mujoco, "default")
    ET.SubElement(
        default,
        "geom",
        attrib={"solref": geom_solref, "solimp": geom_solimp, "fluidcoef": geom_fluidcoef},
    )

    # per-part default class
    for p in parts:
        pname = p["name"]
        dclass = ET.SubElement(default, "default", attrib={"class": pname})
        attrib = {
            "type": "mesh",
            "mesh": pname,
            "contype": p.get("contype", "1"),
            "conaffinity": p.get("conaffinity", "1"),
        }
        if "density" in p:
            attrib["density"] = str(p["density"])
        if "fluidshape" in p:
            attrib["fluidshape"] = p["fluidshape"]
        ET.SubElement(dclass, "geom", attrib=attrib)

    # ---- worldbody
    world = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        world,
        "geom",
        attrib={
            "name": "floor",
            "pos": "0 0 0",
            "size": floor_size,
            "type": "plane",
            "material": plane_material_name,
            "condim": str(floor_condim),
        },
    )
    ET.SubElement(
        world,
        "light",
        attrib={
            "directional": "true",
            "ambient": light_ambient,
            "pos": light_pos,
            "dir": light_dir,
            "diffuse": light_diffuse,
            "specular": light_specular,
        },
    )

    base_body = ET.SubElement(world, "body", attrib={"name": "base", "pos": base_pos, "euler": base_euler})
    if fixed_base==0:
        ET.SubElement(base_body, "freejoint")
    for idx in jsondata['group_info']['0']:
        part = parts_cfg[idx]
        ET.SubElement(base_body, "geom", attrib={
            "class": part["name"],
            "material": f'{part["name"]}_img'
        })
    have_free=0
    dimscale=float(p.get("scale", "1 1 1").split(' ')[0])
    for group_idx in range(1,len(jsondata['group_info'])):
        if jsondata['group_info'][str(group_idx)][-1]=='A':
            have_free+=1
        elif jsondata['group_info'][str(group_idx)][-1]=='B':
            movable_body = ET.SubElement(world, "body", attrib={"name": "grouppart_"+str(group_idx), "pos": "0 0 0"})

            ET.SubElement(
                movable_body, "joint",
                attrib={
                    "type": "slide",
                    "name": "slide_"+str(group_idx),
                    "axis": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][:3])),
                    "range": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][6:8])),
                    "damping": "0.001",
                    "frictionloss": "0.0",
                    "stiffness": "0"
                }
            )
            
            for idx in jsondata['group_info'][str(group_idx)][0]:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
        elif jsondata['group_info'][str(group_idx)][-1]=='C':
            movable_body = ET.SubElement(world, "body", attrib={"name": "grouppart_"+str(group_idx), "pos": "0 0 0"})
            if jsondata['group_info'][str(group_idx)][2][6]==-1 and jsondata['group_info'][str(group_idx)][2][7]==1:
                ET.SubElement(
                    movable_body, "joint",
                    attrib={
                        "type": "hinge",
                        "name": "pivot_"+str(group_idx),
                        "axis": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][:3])),
                        "pos": " ".join(map(str, (np.array(jsondata['group_info'][str(group_idx)][2][3:6])*dimscale).tolist())),
                        "range": " ".join(map(str, (np.array([-3000,3000])*np.pi).tolist())),
                        "damping": "0.001",
                        "frictionloss": "0.0",
                        "stiffness": "0"
                    }
                )
            
            else:
                ET.SubElement(
                    movable_body, "joint",
                    attrib={
                        "type": "hinge",
                        "name": "pivot_"+str(group_idx),
                        "axis": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][:3])),
                        "pos": " ".join(map(str, (np.array(jsondata['group_info'][str(group_idx)][2][3:6])*dimscale).tolist())),
                        "range": " ".join(map(str, (np.array(jsondata['group_info'][str(group_idx)][2][6:8])*np.pi).tolist())),
                        "damping": "0.001",
                        "frictionloss": "0.0",
                        "stiffness": "0"
                    }
                )
            
            for idx in jsondata['group_info'][str(group_idx)][0]:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
        elif jsondata['group_info'][str(group_idx)][-1]=='D':
            movable_body = ET.SubElement(world, "body", attrib={"name": "grouppart_"+str(group_idx), "pos": "0 0 0"})
            ET.SubElement(
                movable_body, "joint",
                attrib={
                    "type": "ball",
                    "name": "ball_"+str(group_idx),
                    "pos": " ".join(map(str, (np.array(jsondata['group_info'][str(group_idx)][2][3:6])*dimscale).tolist())),
                    "damping": "0.001",
                    "frictionloss": "0.0",
                    "stiffness": "0"
                }
            )
            
            for idx in jsondata['group_info'][str(group_idx)][0]:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
        elif jsondata['group_info'][str(group_idx)][-1]=='CB':
            movable_body = ET.SubElement(world, "body", attrib={"name": "grouppart_"+str(group_idx), "pos": "0 0 0"})

            if jsondata['group_info'][str(group_idx)][2][6]==-1 and jsondata['group_info'][str(group_idx)][2][7]==1:
                ET.SubElement(
                    movable_body, "joint",
                    attrib={
                        "type": "hinge",
                        "name": "pivot_"+str(group_idx),
                        "axis": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][:3])),
                        "pos": " ".join(map(str, (np.array(jsondata['group_info'][str(group_idx)][2][3:6])*dimscale).tolist())),
                        "range": " ".join(map(str, (np.array([-3000,3000])*np.pi).tolist())),
                        "damping": "0.001",
                        "frictionloss": "0.0",
                        "stiffness": "0"
                    }
                )

            else:
                ET.SubElement(
                    movable_body, "joint",
                    attrib={
                        "type": "hinge",
                        "name": "pivot_"+str(group_idx),
                        "axis": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][:3])),
                        "pos": " ".join(map(str, (np.array(jsondata['group_info'][str(group_idx)][2][3:6])*dimscale).tolist())),
                        "range": " ".join(map(str, (np.array(jsondata['group_info'][str(group_idx)][2][6:8])*np.pi).tolist())),
                        "damping": "0.001",
                        "frictionloss": "0.0",
                        "stiffness": "0"
                    }
                )
            ET.SubElement(
                movable_body, "joint",
                attrib={
                    "type": "slide",
                    "name": "slide_"+str(group_idx),
                    "axis": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][8:11])),
                    "range": " ".join(map(str, jsondata['group_info'][str(group_idx)][2][14:])),
                    "damping": "0.001",
                    "frictionloss": "0.0",
                    "stiffness": "0"
                }
            )
            
            for idx in jsondata['group_info'][str(group_idx)][0]:
                part = parts_cfg[idx]
                ET.SubElement(movable_body, "geom", attrib={
                    "class": part["name"],
                    "material": f'{part["name"]}_img'
                })
    if have_free>0:
        for group_idx in range(1,len(jsondata['group_info'])):
            if jsondata['group_info'][str(group_idx)][-1]=='A':
                movable_body = ET.SubElement(world, "body", attrib={"name": "grouppart_"+str(group_idx), "pos": "0 0 1", "euler": base_euler})
                ET.SubElement(movable_body, "freejoint")
                
                for idx in jsondata['group_info'][str(group_idx)][0]:
                    part = parts_cfg[idx]
                    ET.SubElement(movable_body, "geom", attrib={
                        "class": part["name"],
                        "material": f'{part["name"]}_img'
                    })

    reparent_by_group_info(mujoco, jsondata['group_info'], base_body_name="base", group_body_prefix="grouppart_")
    
    if have_free>0:
        for group_idx in range(1,len(jsondata['group_info'])):
            if jsondata['group_info'][str(group_idx)][-1]=='A' and deformable==0:
                extract_body_to_world(mujoco, "grouppart_"+str(group_idx))
            elif jsondata['group_info'][str(group_idx)][-1]=='A' and deformable==1:

                world = find_worldbody(mujoco)
                target = find_body(mujoco, "grouppart_"+str(group_idx))
                parent = find_parent(mujoco, target)
                parent.remove(target)

                filename=target.findall('geom')[0].get('class')
                meshid=filename.split('l_')[1].split('_')[0]



                str_list=jsondata['dimension'].split(' ')[0].split('*')
                sorted_list = sorted(str_list, key=float, reverse=True)
                scaling=float(sorted_list[0])/100



                mesh=trimesh.load(os.path.join(out_path.split('basic.xml')[0],"./objs",str(meshid),str(meshid)+'.obj'))
                voxel_size=0.01
                voxel_grid=mesh.voxelized(pitch=voxel_size)
                occupied = voxel_grid.matrix
                volume = np.sum(occupied) * (voxel_size ** 3)
                mass=volume*(scaling**3)*jsondata['parts'][int(meshid)]['density']



                flex=ET.SubElement(
                    world, "flexcomp",
                    attrib={
                        "type": "mesh",
                        "file": os.path.join("./objs",str(meshid),str(meshid)+'.obj'),
                        "pos": "0 0 1",
                        "scale": p.get("scale", "1 1 1"),
                        "dim": "2",
                        "euler": "0 0 0",
                        "radius": "0.001",
                        "name": filename,
                        "dof": "trilinear",
                        "mass": str(mass),
                    }
                )
                ET.SubElement(
                    flex, "elasticity",
                    attrib={
                        "young": str(float(jsondata['parts'][int(meshid)]["Young's Modulus (GPa)"])*1e9),
                        "poisson": str(jsondata['parts'][int(meshid)]["Poisson's Ratio"]),
                        "damping": "0.001"
                    }
                )
                ET.SubElement(
                    flex, "contact",
                    attrib={
                        "selfcollide": "none",
                        "internal": "false",
                    }
                )


    _indent(mujoco)
    tree = ET.ElementTree(mujoco)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


####################
def find_worldbody(root: ET.Element) -> ET.Element:
    for e in root.iter("worldbody"):
        return e
    raise ValueError("cannot find <worldbody>")

def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None

def find_parent(root: ET.Element, node: ET.Element) -> ET.Element | None:
    for e in root.iter():
        for c in list(e):
            if c is node:
                return e
    return None

def is_direct_child_of_world(root: ET.Element, node: ET.Element) -> bool:
    parent = find_parent(root, node)
    return parent is not None and parent.tag == "worldbody"

def extract_body_to_world(root: ET.Element, body_name: str) -> bool:

    world = find_worldbody(root)
    target = find_body(root, body_name)

    if target is None:
        raise ValueError(f"  cannot find body: {body_name}")

    if is_direct_child_of_world(root, target):
        print(f" '{body_name}' skip")
        return False

    parent = find_parent(root, target)
    if parent is None:
        raise RuntimeError("  cannot find the parent node")

    parent.remove(target)
    world.append(target)
    print(f"  Move '{body_name}' from parent body '{parent.get('name')}' to <worldbody> ")
    return True


####################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert urdf format to simplified format")
    parser.add_argument('--voxel_define', type=int, default=32, help='Resolution of the voxel.')
    parser.add_argument('--basepath', type=str, default='./test_demo', help='Path of the voxel.')
    parser.add_argument('--process', type=int, default=0, help='whether use postprocess.')
    parser.add_argument('--fixed_base', type=int, default=0, help='whether fix the basement of object in mjcf.')
    parser.add_argument('--deformable', type=int, default=0, help='whether introduce deformable objects in mjcf.')
    args = parser.parse_args()
    logger = get_logger(os.path.join('exp_urdf.log'),verbosity=1)
    logger.info('start')

    voxel_define=args.voxel_define
    
    basepath=args.basepath
    namelist=os.listdir(basepath)

    for filename in namelist:
        logger.info('begin: '+filename)
        if os.path.exists(os.path.join(basepath,filename,'objs')):

            with open(os.path.join(basepath,filename,'basic_info.txt'), "r", encoding="utf-8") as f:
                basicqu = f.read()

            lines = [line.strip() for line in basicqu.strip().split('\n') if line.strip()]

            data = {}


            data['object_name'] = re.search(r'Name:\s*(.*)', lines[0]).group(1)
            data['category'] = re.search(r'Category:\s*(.*)', lines[1]).group(1)
            data['dimension'] = re.search(r'Dimension:\s*(.*)', lines[2]).group(1)

            parts = []
            for line in lines:
                if line.startswith("l_"):
                    match = re.match(
                        r'l_(\d+):\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*(.*)',
                        line
                    )
                    if match:
                        label = int(match.group(1))
                        name = match.group(2).strip()
                        priority_rank = int(match.group(3))
                        material = match.group(4).strip()
                        density = match.group(5).strip()
   
                        young = (match.group(6).strip())
                        poisson = (match.group(7).strip())
                        basic_desc = match.group(8).strip()

                        

                        parts.append({
                            "label": label,
                            "name": name,
                            "material": material,
                            "density": density,
                            "priority_rank": priority_rank,
                            "Basic_description": basic_desc,
                            "Young's Modulus (GPa)": young,
                            "Poisson's Ratio": poisson
                        })

            data['parts'] = parts


            group_info = {}
            for i, line in enumerate(lines):
                if re.match(r'^group_\d+\s*:', line.strip(), flags=re.IGNORECASE):

                    gm = re.search(r'group_(\d+):\s*\[(.*?)\]', line, flags=re.IGNORECASE)
                    if not gm:
                        continue
                    gid = gm.group(1)
                    members_raw = gm.group(2)

                    members = []
                    for tok in members_raw.split(','):
                        tok = tok.strip().strip("'").strip('"')
                        nm = re.search(r'l_(\d+)', tok, flags=re.IGNORECASE)
                        if nm:
                            members.append(int(nm.group(1)))


                    tm = re.search(r'Type:\s*([A-Za-z])', line, flags=re.IGNORECASE)
                    gtype = tm.group(1).upper() if tm else "E"  
                    if ': CB' in line:
                        gtype='CB'


                    rel_idx = None
                    

                    rel_matches = re.findall(r'(?:relative\s*to\s*)+group_(\d+)', line, flags=re.IGNORECASE)
                    if rel_matches:
                        rel_idx = int(rel_matches[-1])

                    param_vec = [0.0] * 8

                    if gtype not in ("E", "A", "CB"):

                        scan_block = line


                        dir_v = _extract_bracket_list(scan_block, 'direction', 3)
                        pos_v = _extract_bracket_list(scan_block, 'position', 3)


                        pos_v=((np.array(pos_v)) / voxel_define - 0.5).tolist()
                        
                        if gtype in ("C"):
                            rng_v = _extract_bracket_list(scan_block, 'range', 2)
                            rng_v=(np.array(rng_v)/180).tolist()

                        if gtype in ("B"):
                            rng_v = _extract_bracket_list(scan_block, 'range', 2)
                            rng_v=(np.array(rng_v)/voxel_define).tolist()

                        param_vec = dir_v + pos_v + rng_v 

                    elif gtype in ("CB"):
                        scan_block = line


                        dir_v = _extract_bracket_list(scan_block, 'axis direction', 3)
                        pos_v = _extract_bracket_list(scan_block, 'axis position', 3)

                        pos_v=((np.array(pos_v)) / voxel_define - 0.5).tolist()
                        rng_v = _extract_bracket_list(scan_block, 'revolute range', 2)
                        rng_v=(np.array(rng_v)/180).tolist()

                        dir_v1 = _extract_bracket_list(scan_block, 'slide direction', 3)
                        
                        rng_v1 = _extract_bracket_list(scan_block, 'slide range', 2)
                        rng_v1=(np.array(rng_v1)/voxel_define).tolist()

                        param_vec = dir_v + pos_v + rng_v+dir_v1+[0,0,0]+rng_v1 


  
                    if gid==str(0):
                        group_info[gid] = members
                    else:
                        group_info[gid] = [members,str(rel_idx),param_vec,gtype]

            data['group_info'] = group_info

           

            if args.process:
                for group_id in range(1,len(group_info)):
                    if group_info[str(group_id)][-1]=='C' or group_info[str(group_id)][-1]=='CB':
                        if group_info[str(group_id)][1]=='0':
                            group_b=group_info['0']
                        else:
                            if group_info[str(group_id)][1]=='0':
                                group_b=group_info[group_info[str(group_id)][1]]
                            else:
                                group_b=group_info[group_info[str(group_id)][1]][0]
  
                        allcandidate=generate_allcandidate(group_info[str(group_id)][0],group_b,os.path.join(basepath,filename))

                        axisdir=np.array(group_info[str(group_id)][2][:3])
                        axisdir = np.int32(axisdir / np.linalg.norm(axisdir))
                        weights=np.array([1,1,1])
                        weights[np.where(axisdir==1)]=0
                        error=(allcandidate - np.array(group_info[str(group_id)][2][3:6]))*weights

                        dist = np.linalg.norm(error, axis=1)
                        idx = np.argmin(dist)
                        nearest_point = allcandidate[idx]
    
                        if np.linalg.norm((nearest_point-np.array(group_info[str(group_id)][2][3:6]))*weights)<0.03:
                            group_info[str(group_id)][2][3:6]=nearest_point.tolist()

                    if group_info[str(group_id)][-1]=='D':
                        if group_info[str(group_id)][1]=='0':
                            group_b=group_info['0']
                        else:
                            if group_info[str(group_id)][1]=='0':
                                group_b=group_info[group_info[str(group_id)][1]]
                            else:
                                group_b=group_info[group_info[str(group_id)][1]][0]



                        allcandidate=generate_allcandidate_center(group_info[str(group_id)][0],group_b,os.path.join(basepath,filename))

                        weights=np.array([1,1,1])
                        error=(allcandidate - np.array(group_info[str(group_id)][2][3:6]))*weights
                        dist = np.linalg.norm(error)
                        idx = np.argmin(dist)
                        nearest_point = allcandidate[idx]

                        if np.linalg.norm((nearest_point-np.array(group_info[str(group_id)][2][3:6]))*weights)<0.03:

                            group_info[str(group_id)][2][3:6]=nearest_point.tolist()


                




            with open(os.path.join(basepath,filename,'basic_info.json'), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)


            
            jsonfile=os.path.join(basepath,filename,'basic_info.json')
            geofile=os.path.join(basepath,filename,'objs')

            with open(jsonfile,'r') as fp:
                jsondata=json.load(fp)

            mov=jsondata['group_info']

            robot = ET.Element('robot', name='scene')
            link = ET.SubElement(robot, 'link', name='l_world')
            add_inertial(link)

            save=1


            if len(mov)==1:
                fixlist=mov['0']
                for fixindex in fixlist:
                    link = ET.SubElement(robot, 'link', name='l_'+str(fixindex))
                    add_inertial(link)
                    if os.path.exists(os.path.join(geofile,str(fixindex),str(fixindex)+'.obj')):
                        visual = ET.SubElement(link, 'visual')
                        geometry = ET.SubElement(visual, "geometry")
                        ET.SubElement(geometry, "mesh", filename=os.path.join('./objs',str(fixindex),str(fixindex)+'.obj'), scale="1 1 1")
                        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

                for i in range(len(fixlist)-1):
                    parentname='l_'+str(fixlist[i])
                    childname='l_'+str(fixlist[i+1])
                    add_fixed_joint(robot, 'joint_fixed_'+str(fixlist[i])+'_'+str(fixlist[i+1]), parentname, childname, xyz="0 0 0", rpy="0 0 0")
                
                add_fixed_joint(robot, 'joint_fixed_world'+str(fixlist[0]), 'l_world', 'l_'+str(fixlist[0]), xyz="0 0 0", rpy="0 0 0")

            else:

                offset=False

                fixlist=mov['0']
                for fixindex in fixlist:
                    link = ET.SubElement(robot, 'link', name='l_'+str(fixindex))
                    add_inertial(link)
                    if os.path.exists(os.path.join(geofile,str(fixindex),str(fixindex)+'.obj')):
                        visual = ET.SubElement(link, 'visual')
                        geometry = ET.SubElement(visual, "geometry")
                        ET.SubElement(geometry, "mesh", filename=os.path.join('./objs',str(fixindex),str(fixindex)+'.obj'), scale="1 1 1")
                        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

                for i in range(len(fixlist)-1):
                    parentname='l_'+str(fixlist[i])
                    childname='l_'+str(fixlist[i+1])
                    add_fixed_joint(robot, 'joint_fixed_'+str(fixlist[i])+'_'+str(fixlist[i+1]), parentname, childname, xyz="0 0 0", rpy="0 0 0")
                add_fixed_joint(robot, 'joint_fixed_world'+str(fixlist[0]), 'l_world', 'l_'+str(fixlist[0]), xyz="0 0 0", rpy="0 0 0")


                groupnum=len(mov)
                for groupindex in range(1,groupnum):
                    fixlist=mov[str(groupindex)][0]
                    for fixindex in fixlist:
                        link = ET.SubElement(robot, 'link', name='l_'+str(fixindex))
                        add_inertial(link)
                        if os.path.exists(os.path.join(geofile,str(fixindex),str(fixindex)+'.obj')):
                            visual = ET.SubElement(link, 'visual')
                            geometry = ET.SubElement(visual, "geometry")
                            ET.SubElement(geometry, "mesh", filename=os.path.join('./objs',str(fixindex),str(fixindex)+'.obj'), scale="1 1 1")
                            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

                    for i in range(len(fixlist)-1):
                        parentname='l_'+str(fixlist[i])
                        childname='l_'+str(fixlist[i+1])
                        add_fixed_joint(robot, 'joint_fixed_'+str(fixlist[i])+'_'+str(fixlist[i+1]), parentname, childname, xyz="0 0 0", rpy="0 0 0")
                    if isinstance(mov[mov[str(groupindex)][1]][0], int):
                        parentgroupindex=str(mov[mov[str(groupindex)][1]][0])
                    else:
                        parentgroupindex=str(mov[mov[str(groupindex)][1]][0][0])

                    childgroupindex=fixlist[0]
                    parentgroupname='l_'+str(parentgroupindex)
                    childgroupname='l_'+str(childgroupindex)

                    abs_link = ET.SubElement(robot, 'link', name='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))
                    add_inertial(abs_link)
                    

                    if mov[str(groupindex)][-1]=='A':
                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz="0 0 0", rpy="0 0 0")

                        joint = ET.SubElement(robot, "joint", name='joint_free_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="floating")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

                    elif mov[str(groupindex)][-1]=='B':    
                        save+=1
                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz="0 0 0", rpy="0 0 0")

                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])

                        joint = ET.SubElement(robot, "joint", name='joint_prismatic_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="prismatic")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz)  
                        ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][-2]), upper=str(mov[str(groupindex)][-2][-1]), effort="2000.0", velocity="2.0")

                    elif mov[str(groupindex)][-1]=='C':
                        save+=1
                        point=str(mov[str(groupindex)][-2][3])+' '+str(mov[str(groupindex)][-2][4])+' '+str(mov[str(groupindex)][-2][5])  
                        pointrev=str(-mov[str(groupindex)][-2][3])+' '+str(-mov[str(groupindex)][-2][4])+' '+str(-mov[str(groupindex)][-2][5])    
                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])   

                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz=pointrev, rpy="0 0 0")

                        
                        if mov[str(groupindex)][-2][-2]==-1 and mov[str(groupindex)][-2][-1]==1:
                            joint = ET.SubElement(robot, "joint", name='joint_revolute_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="continuous")
                        else:
                            joint = ET.SubElement(robot, "joint", name='joint_revolute_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz)  

                        if mov[str(groupindex)][-2][-2]==-1 and mov[str(groupindex)][-2][-1]==1:
                            ET.SubElement(joint, "limit", effort="2000.0", velocity="2.0")
                        else:

                            ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][-2]*np.pi), upper=str(mov[str(groupindex)][-2][-1]*np.pi), effort="2000.0", velocity="2.0")

                    elif mov[str(groupindex)][-1]=='D': 
                        save+=1

                        point=str(mov[str(groupindex)][-2][3])+' '+str(mov[str(groupindex)][-2][4])+' '+str(mov[str(groupindex)][-2][5])  
                        pointrev=str(-mov[str(groupindex)][-2][3])+' '+str(-mov[str(groupindex)][-2][4])+' '+str(-mov[str(groupindex)][-2][5])    
                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])   

                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz=pointrev, rpy="0 0 0")
                        
                        abs_linkx = ET.SubElement(robot, 'link', name='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        add_inertial(abs_linkx,pointrev)
                        abs_linkz = ET.SubElement(robot, 'link', name='abstract_z_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        add_inertial(abs_linkz,pointrev)

                        joint = ET.SubElement(robot, "joint", name='joint_hinge_y_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_z_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz="0 0 1")  
                        ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

                        joint = ET.SubElement(robot, "joint", name='joint_hinge_z_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link='abstract_z_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "child", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz="1 0 0")  
                        ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

                        joint = ET.SubElement(robot, "joint", name='joint_hinge_x_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz="0 1 0")  
                        ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

                    elif mov[str(groupindex)][-1]=='CB': 
                        save+=1

                        point=str(mov[str(groupindex)][-2][3])+' '+str(mov[str(groupindex)][-2][4])+' '+str(mov[str(groupindex)][-2][5])  
                        pointrev=str(-mov[str(groupindex)][-2][3])+' '+str(-mov[str(groupindex)][-2][4])+' '+str(-mov[str(groupindex)][-2][5])    
                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])
                        xyz1=str(mov[str(groupindex)][-2][8])+' '+str(mov[str(groupindex)][-2][9])+' '+str(mov[str(groupindex)][-2][10])   

                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz=pointrev, rpy="0 0 0")
                        
                        
                        abs_linkx = ET.SubElement(robot, 'link', name='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        add_inertial(abs_linkx)

                        joint = ET.SubElement(robot, "joint", name='joint_prim_y_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="prismatic")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz1)  
                        ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][-2]), upper=str(mov[str(groupindex)][-2][-1]), effort="2000.0", velocity="2.0")

                        if mov[str(groupindex)][-2][6]==-1 and mov[str(groupindex)][-2][7]==1:
                            joint = ET.SubElement(robot, "joint", name='joint_revo_x_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="continuous")
                        else:
                            joint = ET.SubElement(robot, "joint", name='joint_revo_x_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz)  

                        if mov[str(groupindex)][-2][6]==-1 and mov[str(groupindex)][-2][7]==1:
                            ET.SubElement(joint, "limit", effort="2000.0", velocity="2.0")
                        else:
                            ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][6]*np.pi), upper=str(mov[str(groupindex)][-2][7]*np.pi), effort="2000.0", velocity="2.0")


                    else:
                        print('error type') 


                
            tree = ET.ElementTree(robot)
            ET.indent(tree, space="  ", level=0)
            tree.write(os.path.join(basepath,filename,'basic.urdf'), encoding="utf-8", xml_declaration=True)


            parts_cfg = jsondata['parts']

            
            nums = [int(x) for x in re.findall(r'\d+', jsondata['dimension'])]
            max_num = max(nums)/100
            for partind in range(len(parts_cfg)):
                parts_cfg[partind]['name']='l_'+str(parts_cfg[partind]['label'])+'_'+parts_cfg[partind]['name']
                parts_cfg[partind]['mesh_file']=os.path.join('./objs',str(partind),str(partind)+'.obj')
                parts_cfg[partind]['scale']=str(max_num)+' '+str(max_num)+' '+str(max_num)
                parts_cfg[partind]['tex_file']=os.path.join('./objs',str(partind),'material_0.png')
                parts_cfg[partind]['density']=float(parts_cfg[partind]['density'].split('g/cm')[0])*1000
                parts_cfg[partind]['fluidshape']='ellipsoid'
                parts_cfg[partind]['contype']='1'
                parts_cfg[partind]['conaffinity']='1'
            out = generate_mjcf(jsondata=jsondata,out_path=os.path.join(basepath,filename,'basic.xml'), parts=parts_cfg,fixed_base=args.fixed_base,deformable=args.deformable)
            
            logger.info('complete: '+filename)
        else:
            logger.info('skip: '+filename)




    
