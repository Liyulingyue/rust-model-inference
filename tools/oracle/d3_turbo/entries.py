#!/usr/bin/env python3
"""Sequential real CLI/HTTP probes; keeps only one loaded model at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


def post(port, route, request):
    message = urllib.request.Request(f"http://127.0.0.1:{port}{route}",
        data=json.dumps(request).encode(), headers={"Content-Type":"application/json"})
    start=time.monotonic()
    try:
        with urllib.request.urlopen(message, timeout=1200) as response:
            return dict(status=response.status, seconds=time.monotonic()-start, response=json.load(response))
    except urllib.error.HTTPError as error:
        return dict(status=error.code, seconds=time.monotonic()-start, response=json.load(error))


def execution(log, compute):
    submits=re.findall(r"device_submissions=(\d+)",log)
    failed="Vulkan disabled" in log or "Vulkan device failed" in log
    return dict(requested=compute,device_submissions=[int(n) for n in submits],
        device_error=failed,actual="failed" if failed and compute=="vulkan" else "cpu" if compute=="cpu" or failed else "mixed" if submits and int(submits[-1]) else "unverified")


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    request=json.loads(args.request.read_text())
    env=os.environ.copy();env["RMI_COMPUTE_TRACE"]="1"
    rows=[]
    for kind in args.formats:
        model=args.model if kind=="raw" else args.ggufs/f"d3-turbo-{kind}.gguf"
        for compute in args.compute:
            row=dict(format=kind,compute=compute,request=str(args.request),request_sha256=hashlib.sha256(args.request.read_bytes()).hexdigest())
            stem=args.out/f"{kind}-{compute}"
            if not model.exists():
                row["error"]="missing model asset";rows.append(row)
                (args.out/"entries.json").write_text(json.dumps(rows,indent=2))
                continue
            command=[str(args.native),"--model",str(model),"--threads",str(args.threads),"--compute",compute]
            route="/v1/systemone"
            if args.jev:
                assert len(request["questions"])==1
                question=request["questions"][0]
                command += ["--jev","--jev-context",request.get("context",""),"--jev-question",question["text"],"--jev-output","json"]
                for option in question["options"]:command += ["--jev-option",option]
                if request.get("positive"):command += ["--jev-positive",request["positive"]]
                if request.get("image_url"):command += ["--image",request["image_url"]]
                route="/v1/jev/image" if request.get("image_url") else "/v1/jev/score"
            else:
                command += ["--decision-request",str(args.request)]
            start=time.monotonic()
            cli_env=env.copy()
            if args.trace:
                trace_path=stem.with_suffix(".jsonl").resolve()
                trace_path.unlink(missing_ok=True)
                cli_env["RMI_PARITY_TRACE"]=str(trace_path)
            with stem.with_suffix(".cli.json").open("w") as output,stem.with_suffix(".cli.log").open("w") as log:
                try:
                    result=subprocess.run(command,stdout=output,stderr=log,env=cli_env,timeout=1200)
                    code=result.returncode
                except subprocess.TimeoutExpired:
                    code=124
            cli=dict(exit=code,seconds=time.monotonic()-start,command=command,
                execution=execution(stem.with_suffix(".cli.log").read_text(),compute))
            if code==0:
                cli["response"]=json.loads(stem.with_suffix(".cli.json").read_text())
            row["cli"]=cli
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
            command=[str(args.server),"--model",str(model),"--host","127.0.0.1","--port",str(port),"--threads",str(args.threads),"--compute",compute]
            with stem.with_suffix(".server.log").open("w") as log:
                server=subprocess.Popen(command,stdout=log,stderr=log,env=env)
                try:
                    deadline=time.monotonic()+120
                    while True:
                        if server.poll() is not None: raise RuntimeError("server exited before listening")
                        try:
                            with socket.create_connection(("127.0.0.1",port),timeout=0.2):break
                        except OSError:
                            if time.monotonic()>deadline:raise RuntimeError("server startup timeout")
                            time.sleep(0.2)
                    http=post(port,route,request)
                    http["route"]=route
                    http["command"]=command
                    row["http"]=http
                    if http["status"]==200 and cli.get("response"):
                        row["cli_http_equal"]=cli["response"]==(http["response"]["results"][0] if args.jev else http["response"])
                    # These trust-boundary failures must be rejected before file or video decoding.
                    bad=dict(state="x",questions={"q":dict(type="noul")},assets={"@x":"/etc/passwd"})
                    row["local_asset_rejected"]=post(port,"/v1/systemone",bad)["status"]==400
                    bad.pop("assets");bad["videos"]=["/etc/passwd"]
                    row["local_video_rejected"]=post(port,"/v1/systemone",bad)["status"]==400
                except Exception as error:
                    row["http_error"]=str(error)
                finally:
                    server.terminate()
                    try:server.wait(timeout=30)
                    except subprocess.TimeoutExpired:server.kill();server.wait()
            if "http" in row:
                row["http"]["execution"]=execution(stem.with_suffix(".server.log").read_text(),compute)
            rows.append(row)
            (args.out/"entries.json").write_text(json.dumps(rows,indent=2))
            print(json.dumps({"format":kind,"compute":compute,"cli":cli["exit"],"http":row.get("http",{}).get("status"),"equal":row.get("cli_http_equal")}),flush=True)
    (args.out/"entries.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",type=Path,default=Path("target/release-fast/rust-model-inference"))
    p.add_argument("--server",type=Path,default=Path("target/release-fast/rust-model-server"))
    p.add_argument("--model",type=Path,default=Path("models/d3-turbo"))
    p.add_argument("--ggufs",type=Path,default=Path("target/d3-turbo/gguf"))
    p.add_argument("--formats",nargs="+",default=["raw","f32","f16","bf16","q8_0","q4_0","q4_1","q4_k","q5_k","q6_k"])
    p.add_argument("--compute",nargs="+",default=["cpu","auto","vulkan"])
    p.add_argument("--threads",type=int,default=12)
    p.add_argument("--jev",action="store_true",help="probe a single-question JEV request")
    p.add_argument("--trace",action="store_true",help="keep CLI parity checkpoints on the model host")
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
