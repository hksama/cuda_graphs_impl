import json;
c=json.load(open('calibration/rtx5060.json'))['shapes']
for k,v in sorted(c.items()):
    s=v['eager_latency_p50_s']-v['replay_latency_p50_s']
    print(k, f'save {s*1000:.3f} ms/hit')