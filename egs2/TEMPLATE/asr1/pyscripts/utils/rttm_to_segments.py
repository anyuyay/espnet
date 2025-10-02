#!/usr/bin/env python3
import argparse, sys
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm", required=True)
    ap.add_argument("--out_segments", required=True)
    ap.add_argument("--out_utt2spk_local", required=True)
    args = ap.parse_args()

    by_rec = defaultdict(list)
    with open(args.rttm, "r") as f:
        for L in f:
            if not L or L[0]=="#": continue
            a=L.strip().split()
            if len(a)<9 or a[0]!="SPEAKER": continue
            rec=a[1]; st=float(a[3]); du=float(a[4]); spk=a[7]
            by_rec[rec].append((st,du,spk))
    for rec in by_rec:
        by_rec[rec].sort(key=lambda x:x[0])

    cnt = defaultdict(int)
    with open(args.out_segments,"w") as seg, open(args.out_utt2spk_local,"w") as u2s:
        for rec, lst in by_rec.items():
            for st,du,spk in lst:
                cnt[(rec,spk)] += 1
                uid=f"{rec}_{spk}_{cnt[(rec,spk)]:06d}"
                ed=st+du
                seg.write(f"{uid} {rec} {st:.3f} {ed:.3f}\n")
                u2s.write(f"{uid} {spk}\n")

if __name__ == "__main__":
    main()
