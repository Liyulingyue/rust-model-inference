"""Measure real CLI launches and repeated forecasts with one resident model.

Build both binaries with --release --features vulkan, without parity-trace.
All measurements run sequentially. Fresh processes retain the OS file cache.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time

import numpy as np


# wait4 measures the child alone, including peak RSS, without GNU time on Linux.
RESOURCE_WRAPPER = '''
import json, os, subprocess, sys, time
started = time.perf_counter()
child = subprocess.Popen(sys.argv[1:])
_, status, usage = os.wait4(child.pid, 0)
elapsed = (time.perf_counter() - started) * 1000
child.returncode = os.waitstatus_to_exitcode(status)
print('benchmark_resources=' + json.dumps(dict(wall_ms=elapsed,
    max_rss_bytes=usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024))), file=sys.stderr)
sys.exit(child.returncode if child.returncode >= 0 else 128 - child.returncode)
'''


def cases():
    def wave(n, shift=0):
        return [math.sin(i / 17) + 0.3 * math.cos(i / 71) + i / 4096 + shift for i in range(n)]

    standard = dict(context=wave(512), prediction_length=96)
    return {
        'short': dict(context=wave(96), prediction_length=24),
        'standard': standard,
        'without_symmetry': dict(**standard, symmetric_averaging=False),
        'long_context': dict(context=wave(4096), prediction_length=96),
        'multivariate8': dict(context=[wave(512, i / 3) for i in range(8)], prediction_length=96),
        'covariates': dict(**standard, past_covariates=[wave(512, 2)],
                           known_covariates=[wave(608, 3), [i % 2 for i in range(608)]]),
        'long_horizon': dict(context=wave(512), prediction_length=1025),
        'max_context': dict(context=wave(32768), prediction_length=96),
    }


def active_compilers():
    rows = subprocess.check_output(['ps', '-axo', 'pid,comm'], text=True)
    return [row.strip() for row in rows.splitlines()[1:]
            if row.strip().split()[-1].rsplit('/', 1)[-1] in ('rustc', 'ld')]


def execute(command, env, log, timeout, wait_idle=False):
    if wait_idle:
        deadline, quiet = time.monotonic() + 600, 0
        while quiet < 3:
            quiet = 0 if active_compilers() else quiet + 1
            if time.monotonic() > deadline:
                raise RuntimeError('no compiler-free sampling window within 600 seconds')
            time.sleep(1)
    process = subprocess.Popen([sys.executable, '-c', RESOURCE_WRAPPER, *map(str, command)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    compilers = []
    deadline = time.monotonic() + timeout
    while True:
        try:
            stdout, stderr = process.communicate(timeout=min(2, max(0.01, deadline - time.monotonic())))
            break
        except subprocess.TimeoutExpired:
            compilers.extend(active_compilers())
            if time.monotonic() >= deadline:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                log.write_bytes(stderr)
                raise RuntimeError(f'timeout after {timeout}s; see {log}')
    compilers.extend(active_compilers())
    log.write_bytes(stderr)
    if process.returncode:
        raise RuntimeError(f'exit {process.returncode}; see {log}')
    resources = re.findall(r'^benchmark_resources=(.+)$', stderr.decode(), re.MULTILINE)
    assert len(resources) == 1, 'missing child resource measurement'
    return stdout, dict(**json.loads(resources[0]), log=str(log),
                        observed_compilers=sorted(set(compilers)))


def backend_evidence(log, mode):
    text = log.read_text()
    backends = re.findall(r'scope=timestral\.transformer backend=(\w+)', text)
    expected = 'Cpu' if mode == 'cpu' else 'Vulkan'
    assert backends and all(value == expected for value in backends), (mode, backends)
    counters = []
    for line in text.splitlines():
        if 'counters=device_cumulative' in line:
            counters.append({key: int(value) for key, value in re.findall(r'(\w+): (\d+)', line)})
    runs, current = [], []
    for line in text.splitlines():
        if re.fullmatch(r'benchmark run=\d+ begin', line):
            current = []
        elif 'counters=device_cumulative' in line:
            current.append({key: int(value) for key, value in re.findall(r'(\w+): (\d+)', line)})
        elif re.fullmatch(r'benchmark run=\d+ end', line):
            runs.append(current[-1] if current else {})
    return dict(backends=sorted(set(backends)), transformer_blocks=len(backends),
                final_counters=counters[-1] if counters else {}, resident_run_counters=runs)


def compare(reference, actual):
    assert reference['quantile_levels'] == actual['quantile_levels']
    expected, got = (np.asarray(value['quantiles'], dtype=np.float32) for value in (reference, actual))
    assert got.shape == expected.shape and np.isfinite(got).all() and np.isfinite(expected).all()
    delta = got.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    amplitude = float(np.max(np.abs(expected)))
    rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    passed = maximum <= 3e-5 + 3e-4 * amplitude and rmse <= 3e-5 + 5e-5 * rms
    return dict(passed=passed, max_abs=maximum, rmse=rmse, nrmse=rmse / max(rms, 1e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/release/rust-model-inference'))
    parser.add_argument('--resident-binary', type=Path, default=Path('target/release/examples/timestral_bench'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--case', action='append', choices=list(cases()))
    parser.add_argument('--mode', action='append', choices=['cpu', 'vulkan', 'auto'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warm-repeats', type=int, default=5)
    parser.add_argument('--cli-repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--wait-idle', action='store_true', help='wait for a compiler-free start; observed interference is still reported')
    args = parser.parse_args()
    assert min(args.threads, args.warm_repeats, args.cli_repeats, args.timeout) > 0
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('RMI_', 'RUST_GPU_', 'RAYON_'))}
    env['RMI_COMPUTE_TRACE'] = '1'
    with args.model.open('rb') as stream:
        model_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(platform=platform.platform(), started_at_utc=datetime.now(timezone.utc).isoformat(),
        load_average_before=os.getloadavg(), git_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), threads=args.threads,
        warm_repeats=args.warm_repeats, cli_repeats=args.cli_repeats,
        model=str(args.model.resolve()), model_sha256=model_sha256, model_bytes=args.model.stat().st_size,
        binaries={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.binary, args.resident_binary)}, cases={})
    selected = args.case or [name for name in cases() if name != 'max_context']
    for name, request in cases().items():
        if name not in selected:
            continue
        request_path = root / f'{name}.json'
        request_path.write_text(json.dumps(request))
        result = dict(request=str(request_path), modes={})
        report['cases'][name] = result
        for mode in args.mode or ['cpu', 'vulkan']:
            prefix = root / f'{name}-{mode}'
            print(f'begin {name} {mode} threads={args.threads}', flush=True)
            command = [args.resident_binary.resolve(), args.model.resolve(), request_path,
                       mode, args.threads, args.warm_repeats, prefix.with_suffix('.forecast.json')]
            output, resource = execute(command, env, prefix.with_suffix('.resident.log'), args.timeout, args.wait_idle)
            resident = json.loads(output)
            resident.update(resource)
            resident['backend_evidence'] = backend_evidence(Path(resource['log']), mode)
            first = json.loads(Path(resident['output_path']).read_text())
            cli = []
            for repeat in range(args.cli_repeats):
                command = [args.binary.resolve(), '--model', args.model.resolve(), '--timestral-request',
                           request_path, '--compute', mode, '--threads', args.threads]
                output, measurement = execute(command, env, Path(f'{prefix}.cli-{repeat}.log'), args.timeout, args.wait_idle)
                assert json.loads(output) == first, 'CLI differs from resident-model forecast'
                measurement['backend_evidence'] = backend_evidence(Path(measurement['log']), mode)
                cli.append(measurement)
            warm = statistics.median(resident['warm_forecast_ms'])
            result['modes'][mode] = dict(resident=resident, cli=cli, warm_median_ms=warm,
                warm_min_ms=min(resident['warm_forecast_ms']), warm_max_ms=max(resident['warm_forecast_ms']),
                cli_median_ms=statistics.median(row['wall_ms'] for row in cli),
                horizon_points_per_second=len(first['quantiles']) * request['prediction_length'] * 1000 / warm)
            (root / 'report.json').write_text(json.dumps(report, indent=2))
            print(f'end {name} {mode} warm={warm:.3f}ms cli={result["modes"][mode]["cli_median_ms"]:.3f}ms', flush=True)
        if 'cpu' in result['modes']:
            reference = json.loads(Path(result['modes']['cpu']['resident']['output_path']).read_text())
            for mode, measured in result['modes'].items():
                actual = json.loads(Path(measured['resident']['output_path']).read_text())
                measured['numeric_vs_cpu'] = compare(reference, actual)
            for mode in ('vulkan', 'auto'):
                if mode in result['modes']:
                    result['modes'][mode]['warm_speedup_vs_cpu'] = result['modes']['cpu']['warm_median_ms'] / result['modes'][mode]['warm_median_ms']
        (root / 'report.json').write_text(json.dumps(report, indent=2))
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['load_average_after'] = os.getloadavg()
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    assert all(mode.get('numeric_vs_cpu', {}).get('passed', True)
               for case in report['cases'].values() for mode in case['modes'].values())
    print(root / 'report.json', flush=True)


if __name__ == '__main__':
    main()
