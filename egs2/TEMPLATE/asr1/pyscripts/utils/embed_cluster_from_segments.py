#!/usr/bin/env python3
import argparse, os, json, torch, soundfile, humanfriendly
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from speechbrain.pretrained import EncoderClassifier

from tqdm.auto import tqdm

def parse_fs(val):
    # Parse values like "8k", "16000", "44.1k", etc.
    s = str(val).lower().strip()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(humanfriendly.parse_size(s))

def read_wavscp(path):
    tbl={}
    with open(path) as f:
        for L in f:
            k,v=L.strip().split(None,1); tbl[k]=v
    return tbl

def read_segments(path):
    # <utt-id> <rec-id> <start> <end>
    by_rec=defaultdict(list)
    with open(path) as f:
        for L in f:
            a=L.strip().split()
            if len(a)<4: continue
            uid, rec, st, ed = a[0], a[1], float(a[2]), float(a[3])
            by_rec[rec].append((uid, st, ed))
    for rec in by_rec: by_rec[rec].sort(key=lambda x:x[1])
    return by_rec

def read_rttm(path):
    # SPEAKER rec 1 start dur ... spk ...
    by_rec=defaultdict(list)
    with open(path) as f:
        for L in f:
            if not L or L[0]=="#": continue
            a=L.strip().split()
            if len(a)<9 or a[0]!="SPEAKER": continue
            rec=a[1]; st=float(a[3]); du=float(a[4]); spk=a[7]
            by_rec[rec].append((st, du, spk))
    for rec in by_rec: by_rec[rec].sort(key=lambda x:x[0])
    # Convert to segments-like tuples with local spk in uid
    seg_by_rec=defaultdict(list)
    cnt=defaultdict(int)
    for rec, turns in by_rec.items():
        for st,du,spk in turns:
            cnt[(rec,spk)] += 1
            uid=f"{rec}_{spk}_{cnt[(rec,spk)]:06d}"
            seg_by_rec[rec].append((uid, st, st+du))
    return seg_by_rec

def write_rttm(path, rec, turns):
    with open(path, "w") as f:
        for (st, du, spk) in turns:
            f.write(f"SPEAKER {rec} 1 {st:.2f} {du:.2f} <NA> <NA> {spk} <NA> <NA>\n")

# def extract_embeds(backend, model_id, wavscp, segs, fs, ngpu, work_dir):
#     # Stub: unit-norm random vectors (replace with real extractor)
#     rng = np.random.default_rng(0); D=512
#     embs={}
#     for (uid, st, ed) in segs:
#         v=rng.standard_normal(D); v/= (np.linalg.norm(v)+1e-9)
#         embs[uid]=v.astype(np.float32)
#     # Optional debug dump
#     os.makedirs(work_dir, exist_ok=True)
#     with open(os.path.join(work_dir, "utt_order.txt"), "w") as w:
#         for uid,_,_ in segs: w.write(uid+"\n")
#     return embs

# def cluster_ahc(embs_dict, stop_thr):
#     utts=list(embs_dict.keys())
#     if not utts:
#         return [], np.array([], dtype=int)
#     X=np.stack([embs_dict[u] for u in utts],0)
#     D=cosine_distances(X)
#     A=(D < float(stop_thr)).astype(np.int32)
#     seen=set(); comps=[]
#     for i in range(len(utts)):
#         if i in seen: continue
#         stack=[i]; comp=set([i]); seen.add(i)
#         while stack:
#             j=stack.pop()
#             for k in np.where(A[j])[0]:
#                 if k not in seen:
#                     seen.add(k); comp.add(k); stack.append(k)
#         comps.append(sorted(comp))
#     labels=np.zeros(len(utts),dtype=int)
#     for cid,comp in enumerate(comps):
#         for idx in comp: labels[idx]=cid
#     return utts, labels

_SB_MODEL = None
def _load_sb_model(device="cpu", source=None, savedir=None):
    """
    source:
      - default HF id: "speechbrain/spkrec-ecapa-voxceleb"
      - or a local folder with hparams + checkpoint
    savedir:
      - cache dir to store the downloaded model (optional)
    """
    global _SB_MODEL
    if _SB_MODEL is None:
        if source is None:
            source = "speechbrain/spkrec-ecapa-voxceleb"
        _SB_MODEL = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,                    # e.g., f"{work_dir}/ecapa_ckpt"
            run_opts={"device": device},
        )
    return _SB_MODEL

def extract_embeds(backend, embed_model, wavscp, segs, fs=16000, ngpu=0, work_dir=None, rec_id=None, show_progress=True):
    """
    segs: list of (utt, st, ed)
    """
    assert backend in ("speechbrain_ecapa",), "Only 'speechbrain_ecapa' supported now"
    assert rec_id is not None, "extract_embeds requires rec_id when segs are 3-tuples"

    device = "cuda" if (ngpu and torch.cuda.is_available()) else "cpu"
    sb = _load_sb_model(device=device, savedir=os.path.join(work_dir or ".", "ecapa_ckpt"))

    wav_path = wavscp[rec_id]
    wav, sr = soundfile.read(wav_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != fs:
        raise ValueError(f"Expected {fs} Hz, got {sr} for {rec_id}")

    embs = {}
    it = tqdm(segs, desc=f"Embeddings [{rec_id}]", unit="seg") if show_progress else segs

    min_len = int(0.5 * fs)
    for utt, st, ed in it:
        s_i, e_i = int(float(st) * fs), int(float(ed) * fs)
        if e_i <= s_i:
            continue
        seg = wav[s_i:e_i]

        if seg.size < min_len:
            mid = (s_i + e_i) // 2
            half = min_len // 2
            s_i = max(0, mid - half)
            e_i = min(len(wav), mid + half)
            seg = wav[s_i:e_i]
            if seg.size < int(0.3 * fs):
                continue

        x = torch.from_numpy(seg).float().unsqueeze(0).to(device)
        with torch.no_grad():
            emb = sb.encode_batch(x).squeeze(0).squeeze(0)
        emb = emb / (emb.norm(p=2) + 1e-12)
        embs[utt] = emb.cpu().numpy()

    return embs

# ---------- CLUSTERING (AHC, cosine) ----------
def cluster_ahc(embs_dict, stop_thr):
    """
    Args:
      embs_dict: {utt_id: np.ndarray [D]}
      stop_thr: float, distance threshold in COSINE DISTANCE (1 - cosine sim).
                e.g., 0.5 ~ sim 0.5 ; smaller merges more.
    Returns:
      utts_order: [utt_id, ...] in input order
      labels: np.ndarray [N] cluster id per utt
    """
    utts = list(embs_dict.keys())
    if not utts:
        return [], np.array([], dtype=int)
    X = np.stack([embs_dict[u] for u in utts], axis=0)  # [N, D]

    # Sklearn AHC with cosine; single pass, auto-stops at distance_threshold
    # linkage='average' works well with cosine
    ahc = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=float(stop_thr),
    )
    labels = ahc.fit_predict(X)
    return utts, labels

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wav_scp", required=True)
    # Accept either segments OR RTTM
    ap.add_argument("--segments", default="")
    ap.add_argument("--rttm", default="")
    ap.add_argument("--utt2spk_local", default="")
    ap.add_argument("--backend", default="xvector")
    ap.add_argument("--embed_model", default="pretrained/xvec")
    ap.add_argument("--method", choices=["AHC","VBx"], default="AHC")
    ap.add_argument("--ahc_stop_thr", type=float, default=0.70)
    ap.add_argument("--vb_plda_dir", default="")
    ap.add_argument("--vb_lda_dim", type=int, default=128)
    ap.add_argument("--vb_max_iters", type=int, default=20)
    ap.add_argument("--fs", type=parse_fs, default=16000)
    ap.add_argument("--ngpu", type=int, default=0)
    ap.add_argument("--debug", type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--progress", type=lambda x: str(x).lower()=="true", default=True)
    args=ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    wavscp = read_wavscp(args.wav_scp)

    # 1) Build segments per recording from either source
    if args.segments:
        seg_by_rec = read_segments(args.segments)
    elif args.rttm:
        seg_by_rec = read_rttm(args.rttm)
    else:
        raise SystemExit("ERROR: Provide --segments or --rttm")

    # tmp set
    args.backend = "speechbrain_ecapa"
    args.embed_model = None

    for rec in wavscp.keys():
        # 2) Loop over recordings & extract embeddings
        segs = seg_by_rec.get(rec, [])
        # segs is list of (uid, st, ed)
        if not segs:
            continue
        
        embs = extract_embeds(
            args.backend, args.embed_model, wavscp, segs,
            fs=args.fs, ngpu=args.ngpu, work_dir=args.work_dir, rec_id=rec, show_progress=args.progress
        )

        # 3) Cluster embeddings → assign global speaker labels
        if args.method == "AHC":
            utts, labels = cluster_ahc(embs, args.ahc_stop_thr)
        else:
            raise NotImplementedError("VBx backend not wired yet.")

        uniq = sorted(set(labels.tolist()))
        gid = {lab: f"spk{chr(ord('A')+i)}" for i, lab in enumerate(uniq)}
        lab_by_utt = {u: gid[l] for u,l in zip(utts, labels)}

        # 4) Write per-record RTTM
        uid2seg = {u:(st,ed) for (u,st,ed) in segs}
        turns=[]
        for u in utts:
            st,ed = uid2seg[u]
            turns.append((st, ed-st, lab_by_utt[u]))
        turns.sort(key=lambda x:x[0])

        out_rttm = os.path.join(args.out_dir, f"{rec}.rttm")
        write_rttm(out_rttm, rec, turns)

        if args.debug:
            with open(os.path.join(args.work_dir, f"{rec}.embed_cluster.debug.json"), "w") as w:
                json.dump({
                    "rec": rec,
                    "n_segments": len(segs),
                    "ahc_stop_thr": args.ahc_stop_thr,
                    "clusters": {u: lab_by_utt[u] for u in utts}
                }, w, indent=2)

if __name__ == "__main__":
    main()
