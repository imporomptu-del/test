import json, sys
# Usage: python filter_by_class_thresholds.py preds.txt class_thresholds.json > preds.filtered.txt
pred=sys.argv[1]; cfg=sys.argv[2]
thr=json.load(open(cfg))["class_thresholds"]
out=[]
for ln in open(pred):
    p=ln.strip().split()
    if len(p)<6: continue
    c=int(p[0]); conf=float(p[-1])
    if conf >= thr.get(c, 0.25): out.append(ln)
sys.stdout.write("".join(out))
