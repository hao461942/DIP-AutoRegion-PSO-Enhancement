"""Single-image DINOv2/SLIC + classical enhancement + global-best PSO experiment."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / 'models_cache')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
import argparse
import csv
import json
import time
import warnings
import numpy as np
import cv2
import torch
from transformers import AutoModel
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import blur_effect
from pyswarms.single import GlobalBestPSO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import glare

CFG = dict(seed=42, max_side=1008, clusters=5, superpixels=650,
           min_region_fraction=0.02, max_regions=7,
           dark_mean=0.32, dark_median=0.30,
           bright_mean=0.70, bright_fraction=0.30, bright_pixel=0.85,
           saturated=0.995, texture_std=0.025, edge_threshold=0.04,
           edge_fraction=0.06, blur_threshold=0.55, patch_size=48,
           patch_coverage=0.95, min_blur_patches=3,
           shadow_anchor=0.42, shadow_target_fraction=0.45, target_lift_cap=0.10,
           clipping_weight=2.0, naturalness_weight=0.20,
           contrast_weight=0.05, contrast_tolerance=0.25,
           excessive_lift_weight=0.50, excessive_lift_threshold=0.12,
           high_valid_min=0.70, high_valid_max=0.98,
           min_high_pixels=128, min_high_fraction=0.01,
           high_floor=0.35, high_dark_weight=2.0, color_weight=0.2,
           sharpen_clip_weight=2.0, overshoot_weight=0.5,
           particles=12, iterations=15, w=0.7, c1=1.5, c2=1.5,
           blend_sigma=10.0, gain_sigma=8.0, guided_radius=24, guided_eps=0.01,
           max_gain=2.0, min_gain=0.75,
           shadow_fade_start=0.55, shadow_fade_end=0.85,
           highlight_start=0.55, highlight_end=0.85,
           saturation_protect_start=0.95, saturation_protect_end=0.995)
BOUNDS = {'dark': ([0.45, 1.0], [1.0, 3.0]),
          'bright': ([1.0], [2.5]), 'blur': ([0.5, 0.0], [2.0, 1.5])}
FIXED = {'dark': [0.725, 2.0], 'bright': [1.75], 'blur': [1.25, 0.75], 'normal': []}
ALGORITHM = {'dark': 'Gamma + CLAHE', 'bright': 'Gamma compression',
             'blur': 'Unsharp mask', 'normal': 'Identity'}
OUT = ROOT / 'results'
MODEL_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def table(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def read_rgb(path):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def save_rgb(path, rgb):
    rgb = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tofile(path)

def luminance(rgb):
    return np.einsum('...c,c->...', rgb, np.array([0.299, 0.587, 0.114], np.float32))

def local_std(y):
    return np.sqrt(np.maximum(0, ndi.uniform_filter(y*y, 9) - ndi.uniform_filter(y, 9)**2))

def gradients(y):
    return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)/8,
                    cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)/8)

def extract_and_segment(rgb):
    """One frozen model forward; input is aspect preserving and divisible by 14."""
    t = time.perf_counter()
    h, w = rgb.shape[:2]
    model = AutoModel.from_pretrained('facebook/dinov2-small',
                                     revision=MODEL_REVISION,
                                     cache_dir=str(ROOT/'models_cache'),
                                     attn_implementation='sdpa').eval().to('cuda')
    model.requires_grad_(False)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to('cuda')
    x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    with torch.inference_mode():
        tokens = model(pixel_values=x).last_hidden_state[0, 1:].float().cpu().numpy()
    assert len(tokens) == (h//14)*(w//14), 'Expected patch tokens only'
    np.save(OUT/'patch_features.npy', tokens)
    del model, x
    torch.cuda.empty_cache()
    feature_seconds = time.perf_counter() - t
    lowdim = PCA(n_components=16, random_state=CFG['seed']).fit_transform(tokens)
    feat = cv2.resize(lowdim.reshape(h//14, w//14, 16), (w, h), interpolation=cv2.INTER_LINEAR)
    sp = slic(rgb, n_segments=CFG['superpixels'], compactness=12, sigma=1,
              start_label=0, enforce_connectivity=True, channel_axis=-1)
    count = int(sp.max()+1)
    sizes = np.bincount(sp.ravel(), minlength=count)
    y = luminance(rgb)
    low = np.stack([y, local_std(y), *cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1:].transpose(2,0,1)], axis=-1)
    def pool(a):
        return np.stack([np.bincount(sp.ravel(), weights=a[:,:,k].ravel(), minlength=count)/sizes
                         for k in range(a.shape[-1])], axis=1)
    deep_sp, low_sp = pool(feat), pool(low)
    z = np.concatenate([StandardScaler().fit_transform(deep_sp)/np.sqrt(16),
                        StandardScaler().fit_transform(low_sp)/np.sqrt(4)], axis=1)
    km = KMeans(n_clusters=CFG['clusters'], random_state=CFG['seed'], n_init=10).fit(z, sample_weight=sizes)
    clustered = km.labels_[sp]
    # Split disconnected islands; only merge regions sharing a 4-connected boundary.
    labels = np.zeros((h,w), np.int32)
    next_id = 1
    for k in range(CFG['clusters']):
        cc, n = ndi.label(clustered == k)
        labels[cc>0] = cc[cc>0] + next_id - 1
        next_id += n
    pixel_z = z[sp]
    while True:
        ids, areas = np.unique(labels, return_counts=True)
        i = int(np.argmin(areas))
        if len(ids) <= CFG['max_regions'] and areas[i] >= h*w*CFG['min_region_fraction']:
            break
        small = int(ids[i])
        mask = labels == small
        neighbors = np.unique(labels[ndi.binary_dilation(mask) & ~mask])
        center = pixel_z[mask].mean(axis=0)
        target = min(neighbors, key=lambda n: np.linalg.norm(pixel_z[labels==n].mean(axis=0)-center))
        labels[mask] = target
    for i, old in enumerate(np.unique(labels), 1):
        labels[labels==old] = -i
    labels = -labels
    np.save(OUT/'labels_work.npy', labels)
    np.save(OUT/'superpixels.npy', sp)
    return labels, feature_seconds, time.perf_counter()-t

def valid_patches(y, mask):
    """Fixed original-image patches; no zero-filled region background is scored."""
    p = CFG['patch_size']
    patches = []
    for top in range(0, y.shape[0]-p+1, p//2):
        for left in range(0, y.shape[1]-p+1, p//2):
            sl = (slice(top, top+p), slice(left, left+p))
            block = y[sl]
            if (mask[sl].mean() >= CFG['patch_coverage'] and
                block.std() >= CFG['texture_std'] and
                (gradients(block)>CFG['edge_threshold']).mean() >= CFG['edge_fraction']):
                patches.append(sl)
    return patches

def blur_score(y, patches):
    if len(patches) < CFG['min_blur_patches']:
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        values = [float(blur_effect(y[sl], h_size=11, reduce_func=np.max)) for sl in patches]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if len(values) >= CFG['min_blur_patches'] else float('nan')

def identify(rgb, labels):
    y = luminance(rgb)
    contrast = local_std(y)
    regions = []
    for r in np.unique(labels):
        mask = labels==r
        v = y[mask]
        patches = valid_patches(y, mask)
        blur = blur_score(y, patches)
        high = float((v >= CFG['bright_pixel']).mean())
        if v.mean() < CFG['dark_mean'] and np.median(v) < CFG['dark_median']:
            kind = 'dark'
        elif v.mean() > CFG['bright_mean'] or high > CFG['bright_fraction']:
            kind = 'bright'
        elif np.isfinite(blur) and blur > CFG['blur_threshold']:
            kind = 'blur'
        else:
            kind = 'normal'
        regions.append(dict(region=int(r), area_work=int(mask.sum()), area_fraction=float(mask.mean()),
                            mean_luminance=float(v.mean()), median_luminance=float(np.median(v)),
                            local_contrast=float(contrast[mask].mean()), high_fraction=high,
                            saturated_luminance_fraction=float((v>=CFG['saturated']).mean()),
                            channel_clip_fraction=float((rgb[mask].max(axis=1)>=CFG['saturated']).mean()),
                            blur=blur, valid_blur_patches=len(patches), degradation=kind,
                            algorithm=ALGORITHM[kind]))
    return regions

def enhance(rgb, kind, params, scale=1.0):
    if kind == 'normal':
        return rgb.copy()
    y = luminance(rgb)
    if kind == 'dark':
        gamma, clip = params
        corrected = np.clip(np.rint(y**gamma * 255), 0, 255).astype(np.uint8)
        new_y = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8,8)).apply(corrected).astype(np.float32)/255
    elif kind == 'bright':
        new_y = y**params[0]
    else:
        sigma, amount = params
        new_y = y + amount*(y-cv2.GaussianBlur(y, (0,0), float(sigma*scale)))
    # Multiplicative luminance replacement preserves RGB ratios before gamut clipping.
    return np.clip(rgb*(new_y/np.maximum(y, 1e-6))[:,:,None], 0, 1).astype(np.float32)

def smoothstep(x, start, end):
    t=np.clip((x-start)/(end-start),0,1)
    return t*t*(3-2*t)

def regional_candidate(rgb, kind, params, scale=1.0):
    """Protect dark objects inside bright regions and highlights inside dark ones."""
    candidate=enhance(rgb,kind,params,scale)
    y=luminance(rgb)
    if kind=='dark':
        gate=1-smoothstep(y,CFG['shadow_fade_start'],CFG['shadow_fade_end'])
        candidate=rgb+(candidate-rgb)*gate[:,:,None]
    elif kind=='bright':
        gate=smoothstep(y,CFG['highlight_start'],CFG['highlight_end'])
        gate*=1-smoothstep(y,CFG['saturation_protect_start'],CFG['saturation_protect_end'])
        candidate=rgb+(candidate-rgb)*gate[:,:,None]
    gain=luminance(candidate)/np.maximum(y,1e-5)
    gain=np.clip(gain,CFG['min_gain'],CFG['max_gain'])
    return np.clip(rgb*gain[:,:,None],0,1).astype(np.float32)

def guided_filter(guide, value, radius, eps):
    """Scalar guided filter; filter the correction field, never the photograph."""
    def mean(a):
        return cv2.boxFilter(a,-1,(2*radius+1,2*radius+1),borderType=cv2.BORDER_REFLECT)
    mi,mp=mean(guide),mean(value)
    a=(mean(guide*value)-mi*mp)/(np.maximum(mean(guide*guide)-mi*mi,0)+eps)
    b=mp-a*mi
    return mean(a)*guide+mean(b)

def objective_factory(rgb, mask, kind):
    original_y = luminance(rgb)
    unclipped = rgb.max(axis=2) < CFG['saturated']
    patches = valid_patches(original_y, mask)
    high = mask & (original_y>=CFG['high_valid_min']) & (original_y<CFG['high_valid_max']) & unclipped
    # All masks/patches are selected from the original once, shared by all particles.
    blocks = []
    for top in range(0, mask.shape[0], 16):
        for left in range(0, mask.shape[1], 16):
            sl = (slice(top,top+16), slice(left,left+16))
            if mask[sl].mean() >= 0.5:
                blocks.append((sl, mask[sl]))
    original_means=np.array([float(original_y[sl][m].mean()) for sl,m in blocks])
    # Adapt the target to each original block instead of forcing every block to mid-gray.
    targets=original_means+np.minimum(CFG['target_lift_cap'],
                CFG['shadow_target_fraction']*np.maximum(CFG['shadow_anchor']-original_means,0))
    original_contrast=local_std(original_y)
    texture=mask&(original_contrast>.015)
    supported = ((kind!='bright' or high.sum() >= max(CFG['min_high_pixels'], mask.sum()*CFG['min_high_fraction']))
                 and (kind!='blur' or len(patches)>=CFG['min_blur_patches']))
    def score(params):
        result = regional_candidate(rgb, kind, params)
        y = luminance(result)
        newly_clipped = float(((result.max(axis=2)>=CFG['saturated']) & unclipped)[mask].mean())
        if kind == 'dark':
            means=np.array([float(y[sl][m].mean()) for sl,m in blocks])
            exposure=float(np.abs(means-targets).mean())
            change=float(np.abs(y-original_y)[mask].mean())
            ratio=local_std(y)[texture]/original_contrast[texture]
            contrast=float(np.maximum(np.abs(ratio-1)-CFG['contrast_tolerance'],0).mean()) if texture.any() else 0.
            excessive=float(np.maximum(y-original_y-CFG['excessive_lift_threshold'],0)[mask].mean())
            return float(exposure + CFG['clipping_weight']*newly_clipped
                         + CFG['naturalness_weight']*change + CFG['contrast_weight']*contrast
                         + CFG['excessive_lift_weight']*excessive)
        if kind == 'bright':
            if not supported:
                return float('nan')
            well = np.exp(-(y[high]-.5)**2/(2*.2**2)).mean()
            dark = np.maximum(0, CFG['high_floor']-y[mask]).mean()
            def chroma(a):
                return a/np.maximum(a.sum(axis=1,keepdims=True), 1e-6)
            color = np.abs(chroma(result[mask])-chroma(rgb[mask])).mean()
            return float(1-well+CFG['high_dark_weight']*dark+CFG['color_weight']*color)
        b = blur_score(y, patches)
        overshoot = np.maximum(0, np.abs(y-original_y)-.10)[mask].mean()
        return float(b+CFG['sharpen_clip_weight']*newly_clipped+CFG['overshoot_weight']*overshoot)
    return score, supported, int(high.sum())

def optimize(rgb, labels, regions):
    parameters, histories, chosen = [], {}, {}
    for reg in regions:
        r, kind = reg['region'], reg['degradation']
        fixed = FIXED[kind]
        if kind == 'normal':
            chosen[r] = []
            parameters.append(dict(region=r, degradation=kind, parameter_names='', fixed_parameters='[]',
                                   pso_parameters='[]', fixed_fitness='', best_fitness='',
                                   status='identity', valid_high_pixels=0, pso_seconds=0))
            continue
        score, supported, high_count = objective_factory(rgb, labels==r, kind)
        start = time.perf_counter()
        baseline = score(fixed) if supported else float('nan')
        if supported:
            np.random.seed(CFG['seed'])
            lo, hi = [np.array(v, dtype=float) for v in BOUNDS[kind]]
            init = np.random.uniform(lo, hi, (CFG['particles'],len(lo)))
            init[0] = fixed  # includes C as a known candidate, no extra PSO run
            pso = GlobalBestPSO(n_particles=CFG['particles'], dimensions=len(lo),
                                options={k:CFG[k] for k in ('w','c1','c2')}, bounds=(lo,hi),
                                init_pos=init, bh_strategy='reflective',
                                velocity_clamp=(-0.2,0.2))
            cost, params = pso.optimize(lambda swarm: np.array([score(p) for p in swarm]),
                                        iters=CFG['iterations'], verbose=False)
            params = params.tolist()
            histories[r] = [float(v) for v in pso.cost_history]
            status = 'optimized'
        else:
            params = [1.0] if kind=='bright' else [1.25,0.0]
            cost, status = float('nan'), 'skipped_insufficient_valid_pixels'
        chosen[r] = params
        names = {'dark':'gamma;clip_limit','bright':'gamma','blur':'sigma;amount'}[kind]
        parameters.append(dict(region=r, degradation=kind, parameter_names=names,
                               fixed_parameters=json.dumps(fixed), pso_parameters=json.dumps(params),
                               fixed_fitness=baseline, best_fitness=float(cost), status=status,
                               valid_high_pixels=high_count, pso_seconds=time.perf_counter()-start))
        print(f'Region {r}: {kind}, {params}, fitness {baseline:.5f} -> {cost:.5f}', flush=True)
    return chosen, parameters, histories

def fuse_full(rgb8, labels, regions, params):
    """Blend bounded log-luminance gains, smooth them, then retain native details."""
    h,w = rgb8.shape[:2]
    scale = w/labels.shape[1]
    rgb = rgb8.astype(np.float32)/255
    y=luminance(rgb)
    guide=cv2.resize(y,(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_AREA)
    numerator=np.zeros_like(guide);denominator=np.zeros_like(guide)
    detail=np.zeros_like(guide)
    normal_weight=np.zeros_like(guide)
    for reg in regions:
        r, kind = reg['region'], reg['degradation']
        weight=cv2.GaussianBlur((labels==r).astype(np.float32),(0,0),CFG['blend_sigma'])
        candidate=regional_candidate(rgb,kind,params[r],scale)
        log_gain=np.log(np.maximum(luminance(candidate),1e-5)/np.maximum(y,1e-5))
        small=cv2.resize(log_gain,(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_AREA)
        if kind=='blur': detail+=small*weight
        else: numerator+=small*weight
        denominator += weight
        if kind=='normal':normal_weight+=weight
        del candidate,log_gain,small,weight
    field=numerator/np.maximum(denominator,1e-6)
    field=cv2.GaussianBlur(field,(0,0),CFG['gain_sigma'])
    field=guided_filter(guide,field,CFG['guided_radius'],CFG['guided_eps'])
    field*=1-normal_weight
    field+=detail/np.maximum(denominator,1e-6)
    field=cv2.resize(field,(w,h),interpolation=cv2.INTER_LINEAR)
    # Negative correction must not darken trunks; positive correction must not lift sky.
    gate=np.where(field>=0,
                  1-smoothstep(y,CFG['shadow_fade_start'],CFG['shadow_fade_end']),
                  smoothstep(y,CFG['highlight_start'],CFG['highlight_end']))
    gate*=1-smoothstep(y,CFG['saturation_protect_start'],CFG['saturation_protect_end'])
    field=np.clip(field*gate,np.log(CFG['min_gain']),np.log(CFG['max_gain']))
    return np.clip(np.rint(rgb*np.exp(field)[:,:,None]*255),0,255).astype(np.uint8)

def evaluate(rgb, original, labels, regions, patches):
    y = luminance(rgb)
    original_y = luminance(original)
    new_clip = (rgb.max(axis=2)>=CFG['saturated']) & (original.max(axis=2)<CFG['saturated'])
    row = dict(mean_luminance=float(y.mean()), local_contrast=float(local_std(y).mean()),
               saturated_luminance_fraction=float((y>=CFG['saturated']).mean()),
               channel_clip_fraction=float((rgb.max(axis=2)>=CFG['saturated']).mean()),
               new_channel_clip_fraction=float(new_clip.mean()), blur=blur_score(y,patches),
               valid_blur_patches=len(patches), mean_abs_luminance_change=float(np.abs(y-original_y).mean()))
    row.update(boundary_metrics(rgb,original,labels))
    rows = []
    contrast = local_std(y)
    for reg in regions:
        r = reg['region']; m = labels==r
        pp = valid_patches(original_y,m)
        rows.append(dict(region=r, degradation=reg['degradation'], mean_luminance=float(y[m].mean()),
                         local_contrast=float(contrast[m].mean()), blur=blur_score(y,pp),
                         valid_blur_patches=len(pp), channel_clip_fraction=float((rgb[m].max(axis=1)>=CFG['saturated']).mean()),
                         new_channel_clip_fraction=float(new_clip[m].mean())))
    return row, rows

def boundary_metrics(rgb, original, labels):
    """Independent seam diagnostic on segmentation borders without true strong edges."""
    y,y0=luminance(rgb),luminance(original)
    loggain=np.log(np.maximum(y,1e-4)/np.maximum(y0,1e-4))
    jumps=[];excess=[]
    for axis in (0,1):
        l0,l1=(labels[:-1],labels[1:]) if axis==0 else (labels[:,:-1],labels[:,1:])
        v0,v1=(y0[:-1],y0[1:]) if axis==0 else (y0[:,:-1],y0[:,1:])
        g0,g1=(loggain[:-1],loggain[1:]) if axis==0 else (loggain[:,:-1],loggain[:,1:])
        j0,j1=(y[:-1],y[1:]) if axis==0 else (y[:,:-1],y[:,1:])
        valid=(l0!=l1)&(np.abs(v0-v1)<0.03)&(np.minimum(v0,v1)>.03)&(np.maximum(v0,v1)<.98)
        jumps.extend(np.abs(g0-g1)[valid].tolist())
        excess.extend(np.maximum(0,np.abs(j0-j1)-np.abs(v0-v1))[valid].tolist())
    return dict(boundary_gain_jump_p95=float(np.percentile(jumps,95)),
                boundary_luminance_excess_p95=float(np.percentile(excess,95)),boundary_pairs=len(jumps))

def plot_results(images, labels, regions, histories):
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':12})
    colors = plt.get_cmap('tab10')((labels-1)%10)[:,:,:3]
    fig, ax = plt.subplots(figsize=(10,7.5))
    ax.imshow(colors)
    for reg in regions:
        m = labels==reg['region']
        # Place number at the most interior pixel, never at a centroid outside the mask.
        distance = ndi.distance_transform_edt(np.pad(m, 1, constant_values=False))[1:-1,1:-1]
        cy,cx = np.unravel_index(np.argmax(distance),m.shape)
        ax.text(cx,cy,str(reg['region']),ha='center',va='center',color='white',fontsize=16,
                bbox=dict(facecolor='black',alpha=.7,edgecolor='none'))
    ax.set_title('Automatic regions (DINOv2 + SLIC + K-means)'); ax.axis('off')
    fig.savefig(OUT/'segmentation.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    save_rgb(OUT/'region_labels_color.png', colors.astype(np.float32))
    boundary = mark_boundaries(images['A'],labels,color=(1,0.2,0),mode='thick')
    save_rgb(OUT/'boundaries.png',boundary)
    titles = {'A':'Original','B':'B: Global fixed','C':'C: Regional fixed',
              'D':'Final: Enhancement + glare and sky balancing'}
    fig, axes = plt.subplots(1,len(images),figsize=(7*len(images),5.5),squeeze=False)
    for ax,(key,img) in zip(axes.flat,images.items()):
        ax.imshow(img); ax.set_title(titles[key]); ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    # Automatic representative crops: choose a large dark region, a different large
    # region, and the brightest detected region. No scene coordinates or labels.
    ordered = sorted(regions,key=lambda r:r['area_fraction'],reverse=True)
    selected = []
    for reg in [next((r for r in ordered if r['degradation']=='dark'),ordered[0]),
                max(regions,key=lambda r:r['mean_luminance']), *ordered]:
        if reg['region'] not in selected:
            selected.append(reg['region'])
        if len(selected)==min(3,len(regions)): break
    crops = []
    fig, axes = plt.subplots(len(selected),len(images),figsize=(4*len(images),3.4*len(selected)),squeeze=False)
    h,w = labels.shape
    for row,r in enumerate(selected):
        mask = labels==r
        cy,cx = np.unravel_index(np.argmax(ndi.distance_transform_edt(mask)),mask.shape)
        side = min(220,h,w)
        x = int(np.clip(cx-side//2,0,w-side)); yy = int(np.clip(cy-side//2,0,h-side))
        crops.append(dict(region=r,x=x,y=yy,width=side,height=side,coordinate_system='analysis image'))
        for ax,(key,img) in zip(axes[row],images.items()):
            with Image.open(OUT/f'{key}.png') as native:
                sx,sy=native.width/w,native.height/h
                box=(round(x*sx),round(yy*sy),round((x+side)*sx),round((yy+side)*sy))
                native_crop=np.asarray(native.crop(box))
            ax.imshow(native_crop); ax.set_title(f'{key}: Region {r}'); ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT/'crops.png',dpi=240,bbox_inches='tight'); plt.close(fig)
    save_json(OUT/'crop_locations.json',crops)
    fig,ax = plt.subplots(figsize=(8,4.5))
    for r,values in histories.items():
        ax.plot(range(1,len(values)+1),values,label=f'Region {r}',marker='.',markersize=4)
    ax.set(xlabel='Iteration',ylabel='Best fitness (minimize)',title='Global-best PSO convergence')
    if histories: ax.legend()
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/'pso_convergence.png',dpi=250); plt.close(fig)
    table('convergence.csv',[dict(region=r,iteration=i+1,fitness=v) for r,vs in histories.items() for i,v in enumerate(vs)]
          or [dict(region='',iteration='',fitness='')])

def apply_glare(rgb8,labels,regions):
    """Final auxiliary degradation branch; fixed classical settings, no extra PSO."""
    start=time.perf_counter()
    result,info=glare.suppress(rgb8,labels.shape)
    save_rgb(OUT/'before_sky_balance.png',info.pop('before_sky_balance'))
    h,w=labels.shape
    before=cv2.resize(rgb8,(w,h),interpolation=cv2.INTER_AREA)
    after=cv2.resize(result,(w,h),interpolation=cv2.INTER_AREA)
    fade_after,map_after=glare.fade(after)
    map_after=cv2.resize(np.nan_to_num(map_after,nan=0).astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    m=info['confidence']>.25
    y0=luminance(before.astype(np.float32)/255);y1=luminance(after.astype(np.float32)/255)
    def cv_map(y):
        mu=cv2.GaussianBlur(y,(0,0),7)
        std=np.sqrt(np.maximum(cv2.GaussianBlur(y*y,(0,0),7)-mu*mu,0))
        return std/np.maximum(mu,.03)
    cv0,cv1=cv_map(y0),cv_map(y1)
    g0,g1=gradients(y0),gradients(y1)
    valid=m&(g0>.01)
    ratio=float(np.exp(np.mean(np.log(np.maximum(g1[valid],1e-6)/g0[valid])))) if valid.any() else float('nan')
    score=dict(candidate_area_fraction=float(m.mean()),mean_confidence=float(info['confidence'].mean()),
               FADE_global_before=info['fade'],FADE_global_after=fade_after,
               FADE_patch_density_before=float(info['fade_map'][m].mean()) if m.any() else float('nan'),
               FADE_patch_density_after=float(map_after[m].mean()) if m.any() else float('nan'),
               local_rms_contrast_before=float(cv0[m].mean()) if m.any() else float('nan'),
               local_rms_contrast_after=float(cv1[m].mean()) if m.any() else float('nan'),
               visible_gradient_ratio=ratio,valid_gradient_pixels=int(valid.sum()),
               near_black_before=float((y0[m]<.01).mean()) if m.any() else 0,
               near_black_after=float((y1[m]<.01).mean()) if m.any() else 0,
               processing_seconds=time.perf_counter()-start)
    table('glare_metrics.csv',[score])
    rows=[]
    for r in regions:
        mask=labels==r['region'];selected=mask&m
        coverage=float(selected.sum()/mask.sum())
        r.update(glare_candidate_fraction=coverage,
                 glare_processing='Selective DCP' if coverage>.01 else 'Negligible / none')
        rows.append(dict(region=r['region'],primary_degradation=r['degradation'],
                         glare_candidate_fraction=coverage,
                         selected_pixels=int(selected.sum()),
                         FADE_patch_density_before=float(info['fade_map'][selected].mean()) if selected.any() else '',
                         FADE_patch_density_after=float(map_after[selected].mean()) if selected.any() else '',
                         auxiliary_processing='Selective DCP' if coverage>.01 else 'Negligible / none'))
    table('glare_region_info.csv',rows)
    table('region_info.csv',regions);save_json(OUT/'regions.json',regions)
    save_json(OUT/'glare_config.json',glare.CONFIG)
    save_json(OUT/'sky_balance.json',info['sky_matching'])
    if info['sky_matching']['components']:
        table('sky_components.csv',info['sky_matching']['components'])
    save_json(OUT/'glare_runtime.json',dict(**score,airlight=info['airlight'],
                input='enhanced_before_glare.png',additional_DINO_forwards=0,additional_PSO_runs=0,
                metric='Python adaptation of original LIVE FADE (12 features and released MVG parameters)',
                interpretation='Fog perception proxy; not a physical glare measurement',
                algorithm='Selective dark-channel dehazing followed by reference-based sky tone balancing'))
    np.save(OUT/'glare_confidence.npy',info['confidence'])
    np.save(OUT/'fade_density_before.npy',info['fade_map'])
    save_rgb(OUT/'glare_mask.png',np.repeat(info['confidence'][:,:,None],3,axis=2))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,data,title,cmap in zip(axes,(info['fade_map'],info['confidence'],info['transmission']),
                                 ('FADE patch density','Glare confidence','Refined transmission'),
                                 ('magma','inferno','viridis')):
        high=np.percentile(data,95) if title.startswith('FADE') else 1
        im=ax.imshow(data,cmap=cmap,vmin=0,vmax=high);ax.set_title(title);ax.axis('off')
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(OUT/'glare_analysis.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # An automatically located crop from the largest confident component.
    components,n=ndi.label(m)
    if n:
        areas=np.bincount(components.ravel());areas[0]=0
        cc=components==areas.argmax();yy,xx=np.where(cc)
        x0,x1=max(0,int(xx.min())-30),min(w,int(xx.max())+31)
        y0c,y1c=max(0,int(yy.min())-30),min(h,int(yy.max())+31)
        bounds=(x0/w,y0c/h,x1/w,y1c/h)
        fig,axes=plt.subplots(1,2,figsize=(9,6))
        for ax,image8,title in zip(axes,(rgb8,result),('Before glare suppression','Final result')):
            hh,ww=image8.shape[:2]
            ax.imshow(image8[round(bounds[1]*hh):round(bounds[3]*hh),round(bounds[0]*ww):round(bounds[2]*ww)])
            ax.set_title(title);ax.axis('off')
        fig.tight_layout();fig.savefig(OUT/'glare_detail.png',dpi=240,bbox_inches='tight');plt.close(fig)
        save_json(OUT/'glare_crop.json',dict(normalized_box=bounds,selection='Largest connected confidence > 0.25 component; display only'))
    print('Glare:',json.dumps(score),flush=True)
    return result,score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='Required runtime check, no model forward or experiment')
    parser.add_argument('--report-only',action='store_true',help='Regenerate figures/tables from saved outputs, no model/PSO')
    parser.add_argument('--reuse-segmentation',action='store_true',help='Reuse saved DINOv2/SLIC partition; rerun only enhancement')
    parser.add_argument('--glare-only',action='store_true',help='Apply only the new glare branch to saved enhanced_before_glare.png')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    cv2.setNumThreads(4); torch.set_num_threads(4)
    np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
    if args.glare_only:
        labels=np.load(OUT/'labels_work.npy')
        regions=json.loads((OUT/'regions.json').read_text(encoding='utf-8'))
        result,_=apply_glare(read_rgb(OUT/'enhanced_before_glare.png'),labels,regions)
        save_rgb(OUT/'D.png',result);save_rgb(OUT/'final.png',result)
        images={k:cv2.resize(read_rgb(OUT/f'{k}.png'),(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/255 for k in ('A','D')}
        histories=json.loads((OUT/'histories.json').read_text())
        plot_results(images,labels,regions,histories)
        original_y=luminance(images['A'])
        patches=[sl for r in regions for sl in valid_patches(original_y,labels==r['region'])]
        metrics,regional=[],[]
        for key,img in images.items():
            row,rr=evaluate(img,images['A'],labels,regions,patches)
            metrics.append(dict(method=key,**row));regional.extend([dict(method=key,**v) for v in rr])
        table('metrics.csv',metrics);table('region_metrics.csv',regional)
        runtime=json.loads((OUT/'runtime.json').read_text())
        runtime.update(final_pipeline='Regional PSO + smooth gain fusion + selective DCP + reference sky tone matching',
                       glare_processing_seconds=json.loads((OUT/'glare_runtime.json').read_text())['processing_seconds'],
                       glare_stage_reused_enhancement=True)
        save_json(OUT/'runtime.json',runtime)
        print('Final course-report result:',OUT/'final.png',flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: this experiment requires CUDA feature extraction')
    if args.check:
        model = AutoModel.from_pretrained('facebook/dinov2-small',revision=MODEL_REVISION,cache_dir=str(ROOT/'models_cache'),attn_implementation='sdpa').eval().to('cuda')
        print(json.dumps(dict(python=__import__('sys').version,torch=torch.__version__,cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(),model=model.config.model_type,
                             input_shape=read_rgb(ROOT/'input.jpg').shape),indent=2))
        return
    if args.report_only:
        labels=np.load(OUT/'labels_work.npy')
        regions=json.loads((OUT/'regions.json').read_text(encoding='utf-8'))
        histories=json.loads((OUT/'histories.json').read_text())
        images={k:cv2.resize(read_rgb(OUT/f'{k}.png'),(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/255 for k in ('A','D')}
        plot_results(images,labels,regions,histories)
        return
    experiment_start = time.perf_counter()
    rgb8 = read_rgb(ROOT/'input.jpg')
    h,w = rgb8.shape[:2]
    # Uniform scale, then pad right/bottom with reflection to avoid aspect distortion.
    scale = min(1.,CFG['max_side']/max(h,w))
    nw,nh = round(w*scale),round(h*scale)
    resized = cv2.resize(rgb8,(nw,nh),interpolation=cv2.INTER_AREA)
    pw,ph = (-nw)%14,(-nh)%14
    work8 = cv2.copyMakeBorder(resized,0,ph,0,pw,cv2.BORDER_REFLECT_101)
    work = work8.astype(np.float32)/255
    if args.reuse_segmentation:
        labels=np.load(OUT/'labels_work.npy')
        previous=json.loads((OUT/'runtime.json').read_text())
        feature_s,preprocess_s=previous['feature_seconds'],previous['segmentation_seconds']
    else:
        labels,feature_s,preprocess_s = extract_and_segment(work)
    # For this input no padding is needed. Store general padding metadata explicitly.
    if ph or pw:
        labels=labels[:nh,:nw]; work=work[:nh,:nw]
    regions = identify(work,labels)
    full_labels = cv2.resize(labels.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    for reg in regions:
        reg['area_native']=int((full_labels==reg['region']).sum())
    save_json(OUT/'regions.json',regions); table('region_info.csv',regions)
    cv2.imencode('.png',full_labels)[1].tofile(OUT/'labels_native.png')
    save_json(OUT/'config.json',CFG)
    print('Regions:',[(r['region'],r['degradation'],round(r['area_fraction'],3)) for r in regions],flush=True)
    t=time.perf_counter()
    chosen,parameters,histories=optimize(work,labels,regions)
    pso_s=time.perf_counter()-t
    table('parameters.csv',parameters); save_json(OUT/'histories.json',histories)
    save_json(OUT/'chosen_parameters.json',chosen)
    # Release 20MP label array before full-resolution float processing.
    del full_labels
    save_rgb(OUT/'A.png',rgb8)
    images={'A':work}
    render_times={'A':0.0}
    for key in 'D':
        t=time.perf_counter()
        if key=='B':
            out = np.rint(enhance(rgb8.astype(np.float32)/255,'dark',FIXED['dark'])*255).astype(np.uint8)
        else:
            params=chosen if key=='D' else {r['region']:FIXED[r['degradation']] for r in regions}
            out=fuse_full(rgb8,labels,regions,params)
        save_rgb(OUT/'enhanced_before_glare.png',out)
        out,glare_metrics=apply_glare(out,labels,regions)
        render_times[key]=time.perf_counter()-t
        save_rgb(OUT/f'{key}.png',out)
        if key=='D': save_rgb(OUT/'final.png',out)
        images[key]=cv2.resize(out,(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        del out
        print(f'{key} native render: {render_times[key]:.2f}s',flush=True)
    original_y=luminance(work)
    # Texture eligibility fixed from A; pool patches wholly inside individual regions.
    patches=[sl for r in regions for sl in valid_patches(original_y,labels==r['region'])]
    metrics,regional=[],[]
    for key,img in images.items():
        row,rr=evaluate(img,work,labels,regions,patches)
        row.update(method=key,render_seconds=render_times[key],
                   shared_segmentation_seconds=preprocess_s if key in 'CD' else 0,
                   optimization_seconds=pso_s if key=='D' else 0,
                   total_compute_seconds=render_times[key]+(preprocess_s if key in 'CD' else 0)+(pso_s if key=='D' else 0))
        metrics.append(row)
        regional.extend([dict(method=key,**v) for v in rr])
    table('metrics.csv',metrics); table('region_metrics.csv',regional)
    plot_results(images,labels,regions,histories)
    import importlib.metadata as md
    runtime=dict(python=__import__('sys').version,torch=torch.__version__,cuda=torch.version.cuda,
                 gpu=torch.cuda.get_device_name(),gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory/2**30,
                 model='facebook/dinov2-small',model_forward_count=0 if args.reuse_segmentation else 1,
                 reused_segmentation=args.reuse_segmentation,
                 model_revision=MODEL_REVISION,
                 original_size=[w,h],analysis_size=[nw,nh],padded_size=[nw+pw,nh+ph],
                 feature_seconds=feature_s,segmentation_seconds=preprocess_s,pso_seconds=pso_s,
                 final_pipeline='Regional PSO + smooth gain fusion + selective DCP + reference sky tone matching',
                 glare_processing_seconds=glare_metrics['processing_seconds'],
                 experiment_wall_seconds=time.perf_counter()-experiment_start,
                 metric_resolution=[nw,nh],timing_excludes='file output, plots and metrics; model cache is warm',
                 dependencies={p:md.version(p) for p in ['torch','transformers','numpy','opencv-python-headless','scikit-image','scikit-learn','scipy','pyswarms','matplotlib']})
    save_json(OUT/'runtime.json',runtime)
    print('Experiment complete:',OUT,flush=True)

if __name__=='__main__':
    main()
