#!/usr/bin/env python3
"""Export small protocol goldens from the read-only pinned upstream runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main(root, output):
    import numpy as np
    sys.path.insert(0, str(root.resolve()))
    import d3_compose as dc
    import d3_runtime as runtime
    config = json.loads((root / "decision_config.json").read_text())
    codes = config["codes"]
    cases = []
    inputs = [
        (None, {"type":"noul"}, {}, 0, 0),
        ({"x":[1e-5, True, "中文"]}, {"type":"choice", "instructions":"Compare @ab with @a.",
            "criteria":{"A":["@ab", "x @a"], "B":None}}, {"@a":"a.png", "@ab":"b.png"}, 1, 0),
        ("[image 1]", {"type":"box", "instructions":"the red glass"}, {}, 1, 0),
        ("state", {"type":"noul", "instructions":"\"@a\"", "criteria":{"false":False,"true":"@a"}}, {"@a":"a.png"}, 0, 0),
        ("state", {"type":"score", "instructions":"quality", "criteria":["bad", {"x":1e-5},"good"]}, {}, 0, 0),
        ("clip", {"type":"box", "instructions":"the red glass", "at":1.25}, {}, 1, 1),
        ("clip", {"type":"track", "instructions":"the red glass", "at":1.25}, {}, 0, 1),
    ]
    for state, question, assets, front, videos in inputs:
        if not assets and question["type"] not in ("box", "track"):
            q = runtime.normalize_question(question)
            parts = [(dc.IMAGE,i) for i in range(front)] + [(dc.TEXT,runtime.user_prompt(state,q.rendered,codes))]
        else:
            parts = dc.user_parts(state,question,codes,front,videos,dc.check_references(list(assets)))
        content = [{"type":"text","text":item} if kind==dc.TEXT else {"type":kind} for kind,item in parts]
        own = [assets[item] for kind,item in parts if kind==dc.IMAGE and isinstance(item,str)]
        cases.append(dict(state=state, question=question, assets=assets, front=front, videos=videos, content=content,
            own_images=own, suffix=dc.suffix(question["type"]=="track") if question["type"] in ("box","track") else ""))
    boxes=[]
    for mode in range(3):
        logits = np.zeros((4,128),np.float32)
        if mode==1:
            logits.fill(-10)
            for edge,index in enumerate([20,100,10,110]): logits[edge,index]=10
        if mode==2:
            logits=np.sin(np.arange(512,dtype=np.float32).reshape(4,128)/7)*3
        probs=np.exp(logits-logits.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
        box=dc.decode(probs.tolist())
        calibration=config["systemone"]["box_calibration"]
        boxes.append(dict(logits=logits.tolist(),box=box,calibration=calibration,
            confidence=dc.calibrate(dc.confidence(probs.tolist(),box),calibration)))
    output.write_text(json.dumps(dict(codes=codes,cases=cases,boxes=boxes,
        sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ["d3_compose.py","d3_runtime.py","d3_format.py","decision_config.json"]}),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    main(a.model,a.output)
