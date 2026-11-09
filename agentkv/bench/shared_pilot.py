"""Owned Qwen3 shared-GPU pilot and two-card dmon monitoring; outputs stay outside Git."""
import argparse
import contextlib
import csv
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from agentkv.provenance import git_commit
from agentkv.bench.workload import ROOT
from agentkv.bench.trace_loader import sha256_file

GPU_TOKENS=182880
CPU_TOKENS=274320
KV_BYTES_PER_TOKEN=147456


def server_arguments(vllm_dir,port,gpu_index=1,*,cache_policy="lru",cache_policy_module=None):
    node=0 if gpu_index==0 else 2
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',kv_connector_extra_config=dict(
        cpu_bytes_to_use=CPU_TOKENS*KV_BYTES_PER_TOKEN,offload_prompt_only=True,eviction_policy=cache_policy))
    if cache_policy_module is not None:transfer['kv_connector_extra_config']['cache_policy_module_path']=cache_policy_module
    return ['numactl',f'--cpunodebind={node}',f'--preferred={node}',str(vllm_dir/'.venv/bin/vllm'),'serve','/data/models/Qwen3-8B',
        '--served-model-name','Qwen/Qwen3-8B','--host','127.0.0.1','--port',str(port),
        '--shutdown-timeout','120','--disable-uvicorn-access-log','--gpu-memory-utilization','0.38',
        '--kv-cache-memory-bytes',str(GPU_TOKENS*KV_BYTES_PER_TOKEN),'--num-gpu-blocks-override',str(GPU_TOKENS//16),
        '--block-size','16','--max-model-len','16384','--max-num-batched-tokens','8192','--max-num-seqs','8',
        '--enable-prefix-caching','--kv-transfer-config',json.dumps(transfer,separators=(',',':'))]


def stop_owned(process,*,grace_s=45,signal_group=True):
    if process is None:return
    try:
        if signal_group:os.killpg(process.pid,signal.SIGTERM)
        else:process.send_signal(signal.SIGTERM)
    except ProcessLookupError:pass
    deadline=time.monotonic()+grace_s
    try:
        while process.poll() is None:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise subprocess.TimeoutExpired(process.args,grace_s)
            try:process.wait(timeout=min(10,remaining))
            except subprocess.TimeoutExpired:continue
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=15)


def run(output,gpu_index=1,port=8012,comparison_dir=None, *, duration=1200,warmup=300,rate=1.,population_seed=0,run_id=None,analyze_output=True,cache_policy="lru",cache_policy_module=None,hint_tier="none",extra_environment=None):
    commit=git_commit()
    vllm_dir=ROOT.parent/'vllm';backend_commit=git_commit(repository=vllm_dir)
    if output.resolve().is_relative_to(ROOT):
        raise ValueError('Stage pilot outside the Git worktree, then publish after completion')
    if output.exists():raise FileExistsError('New staging directory required')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise ValueError('Pilot port is already in use')
    output.mkdir(parents=True)
    clock=dict(wall_time_s=time.time(),monotonic_ns=time.monotonic_ns())
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu_index),VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_KV_CACHE_LAYOUT='LBHNC',VLLM_PLUGINS='agentkv_replay_observer',AGENTKV_TRACE='0',
        AGENTKV_OBSERVATION_DIR=str(output/'server-observations'),OMP_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        AGENTKV_EXPERIMENT_CLOCK=str(output/'experiment-clock.json'),
        PYTHONPATH=os.pathsep.join([str(ROOT/'agentkv/bench/plugins'),str(ROOT),os.environ.get('PYTHONPATH','')]))
    if extra_environment:environment.update(extra_environment)
    command=server_arguments(vllm_dir,port,gpu_index,cache_policy=cache_policy,cache_policy_module=cache_policy_module)
    node=0 if gpu_index==0 else 2
    run_id=run_id or f'm1-shared-lru-seed{population_seed}'
    manifest=dict(vllm_commit=backend_commit,agentkv_commit=commit,model='Qwen/Qwen3-8B',
        gpu_index=gpu_index,gpu_cache_tokens=GPU_TOKENS,cpu_cache_tokens=CPU_TOKENS,
        gpu_cache_bytes=GPU_TOKENS*KV_BYTES_PER_TOKEN,cpu_bytes=CPU_TOKENS*KV_BYTES_PER_TOKEN,
        block_size=16,gpu_blocks=GPU_TOKENS//16,policy=cache_policy,hint_tier=hint_tier,
        max_model_len=16384,max_num_seqs=8,max_num_batched_tokens=8192,
        shared_gpu=True,command=command,clock_pair=clock,
        duration_s=duration,warmup_s=warmup,rate_per_minute=rate,population_seed=population_seed,
        gpu_before=subprocess.check_output(['nvidia-smi'],text=True))
    (output/'server-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    status=dict(git_commit=commit,status='starting',started_at=datetime.now(timezone.utc).isoformat())
    def save(**updates):
        status.update(updates);(output/'pilot-status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    server=dmon=client=None
    handles=[]
    try:
        monitor_environment=os.environ.copy();monitor_environment['TZ']='UTC'
        dlog=(output/'dmon.log').open('w');handles.append(dlog)
        dmon=subprocess.Popen(['nvidia-smi','dmon','-i','0,1','-d','1','-s','pucm','-o','DT'],
                              stdout=dlog,stderr=subprocess.STDOUT,env=monitor_environment,start_new_session=True)
        slog=(output/'server.log').open('w');handles.append(slog)
        server=subprocess.Popen(command,cwd=vllm_dir,env=environment,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
        save(server_pid=server.pid,dmon_pid=dmon.pid)
        deadline=time.monotonic()+420
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError(f'Server exited: {server.returncode}')
            if dmon.poll() is not None:raise RuntimeError('dmon exited before trial')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                    if response.status==200:break
            except OSError:pass
            time.sleep(2)
        else:raise TimeoutError('Server health timeout')
        # All server logs/manifests live in /tmp, so the client remains clean.
        save(status='replaying',replay_started_at=datetime.now(timezone.utc).isoformat())
        args=['numactl',f'--cpunodebind={node}',f'--preferred={node}',sys.executable,'-m','agentkv.bench.replay',
            '--workload',str(ROOT/'results/m1-workload-seed0-v3-16k/workload.parquet'),
            '--schedule',str(ROOT/'results/m1-capacity-seed0-v3/schedule-16384.parquet'),
            '--output-dir',str(output),'--base-url',f'http://127.0.0.1:{port}','--run-id',run_id,
            '--policy',cache_policy,'--hint-tier',hint_tier,'--arrival-rate',str(rate),'--population-seed',str(population_seed),'--steady-state',
            '--duration',str(duration),'--warmup',str(warmup),'--cooldown','0','--max-inflight','128',
            '--request-timeout','180','--drain-timeout','120','--observation-dir',str(output/'server-observations'),
            '--server-manifest',str(output/'server-manifest.json')]
        (output/'client-command.json').write_text(json.dumps(args,indent=2)+'\n')
        clog=(output/'client.log').open('w');handles.append(clog)
        client=subprocess.Popen(args,cwd=ROOT,env=environment,stdout=clog,stderr=subprocess.STDOUT,start_new_session=True)
        save(client_pid=client.pid)
        while client.poll() is None:
            if server.poll() is not None:raise RuntimeError('Server exited during replay')
            if dmon.poll() is not None:raise RuntimeError('dmon exited during replay')
            time.sleep(2)
        if client.returncode:raise RuntimeError(f'Client exited: {client.returncode}')
        save(status='analyzing')
    except BaseException as error:
        save(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for process in (client,server,dmon):
            stop_owned(process,grace_s=135 if process is server else 45,signal_group=process is not server)
        for handle in handles:handle.close()
    if analyze_output:analyze_pilot(output,comparison_dir)
    save(status='complete',completed_at=datetime.now(timezone.utc).isoformat())


def parse_dmon(text):
    rows=[]
    for line in text.splitlines():
        parts=line.split()
        if len(parts)<17 or parts[0].startswith('#'):continue
        try:
            stamp=datetime.strptime(parts[0]+' '+parts[1],'%Y%m%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
            rows.append(dict(wall_time_s=stamp,gpu=int(parts[2]),sm=float(parts[6]),memory_util=float(parts[7]),fb_mib=float(parts[14])))
        except ValueError:continue
    return rows


def analyze_pilot(directory,comparison_dir=None):
    from agentkv.bench.analyze_replay import analyze
    run=json.loads((directory/'run.json').read_text())
    with (directory/'analysis.log').open('w') as sink,contextlib.redirect_stdout(sink):
        analysis=analyze(directory)
    manifest=json.loads((directory/'server-manifest.json').read_text())
    clock=manifest['clock_pair']
    origin_epoch=clock['wall_time_s']+(run['origin_ns']-clock['monotonic_ns'])/1e9
    dmon=parse_dmon((directory/'dmon.log').read_text())
    for row in dmon:row['elapsed_s']=row['wall_time_s']-origin_epoch
    if {r['gpu'] for r in dmon}!={0,1}:raise ValueError('Missing dmon samples for one GPU')
    metrics=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
    parameters=run['parameters'];duration=parameters['duration'];warmup=parameters['warmup'];cooldown=parameters['cooldown']
    window=[s for s in metrics if warmup<=s['elapsed_s']<=duration-cooldown and 'vllm:num_requests_running' in s.get('values',{})]
    running=[s['values']['vllm:num_requests_running'] for s in window]
    actual=run['summary']['nominal_token_weighted_fraction']
    baseline=json.loads((ROOT/'results/m1-two-tier-seed0/run.json').read_text())
    target=next(r for r in baseline['results'] if r['directory']=='config-11')
    expected=target['summary']['source_fractions']
    comparison=dict(git_commit=run['git_commit'],actual_token_weighted_fraction=actual,simulation_fraction=expected,
        difference_pp={k:100*(actual[k]-expected[k]) for k in expected},
        running=dict(p50=float(np.median(running)),p90=float(np.quantile(running,.9)),maximum=max(running),
                     mean=float(np.mean(running)),fraction_at_8=sum(v>=8 for v in running)/len(running)),
        waiting=analysis['server_waiting'],stability=analysis['stability'],
        dmon=dict(samples=len(dmon),sampled_gpus=sorted({r['gpu'] for r in dmon})),
        dmon_sha256=sha256_file(directory/'dmon.log'),comparison_baseline_sha256=sha256_file(ROOT/'results/m1-two-tier-seed0/run.json'))
    if comparison_dir and (comparison_dir/'run.json').exists():
        oracle=json.loads((comparison_dir/'run.json').read_text())
        comparison['oracle_comparison_status']=oracle['status']
        if oracle['status']=='complete':
            comparison['noticeably_larger_oracle_configurations']=oracle['noticeably_larger_than_target']
            comparison['oracle_ranking']=[dict(id=r['id'],reduction_pp=r['recompute_reduction_pp']) for r in oracle['results']]
            if 'target_matched_window' in oracle:comparison['matched_window_simulation']=oracle['target_matched_window']['summary']['source_fractions']
    (directory/'pilot-analysis.json').write_text(json.dumps(comparison,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(7.1,4.4),sharex=True)
    for gpu,color in ((0,'#D55E00'),(1,'#0072B2')):
        data=[r for r in dmon if r['gpu']==gpu and 0<=r['elapsed_s']<=duration]
        axes[0].plot([r['elapsed_s']/60 for r in data],[r['sm'] for r in data],label=f'GPU {gpu}',lw=.7,color=color)
        axes[1].plot([r['elapsed_s']/60 for r in data],[r['fb_mib']/1024 for r in data],label=f'GPU {gpu}',lw=.7,color=color)
    axes[0].set(ylabel='SM utilization (%)',ylim=(0,100));axes[0].legend(frameon=False,ncol=2)
    axes[1].set(ylabel='Framebuffer (GiB)',xlabel='Elapsed minutes',xlim=(0,duration/60))
    for ax in axes:ax.axvspan(0,warmup/60,color='gray',alpha=.15);ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(directory/f'gpu_utilization.{ext}',dpi=300)
    plt.close(fig)
    lines=['# 共享 GPU：Qwen3-8B、LRU、none、20 分钟','','λ=1 根/分钟；16K；8 槽位；GPU池182880 token、CPU层274320 token；稳态初始化；预热5分钟。保留现有 GPU 进程，独立端口；nvidia-smi dmon 同时记录 GPU0、GPU1。',
           '',f"AgentKV 提交 `{run['git_commit']}`，vLLM 提交 `{manifest['vllm_commit']}`；两仓库启动前均检查干净。原始输入/请求输出 SHA256 在 run.json，未复制实现源码。",'',
           '| 指标 | 实测 |','|---|---:|',f"| 窗口请求 / 成功 / 失败 | {run['summary']['window_requests']} / {run['summary']['window_success']} / {run['summary']['window_failed']} |",
           f"| 运行请求 P50 / P90 / 最大值 | {comparison['running']['p50']:.1f} / {comparison['running']['p90']:.1f} / {comparison['running']['maximum']:.0f} |",
           f"| 稳定性（新判据） | {comparison['stability']['status']} |",'',
           '| token 来源 | 模拟（40分/预热10分） | 实测（20分/预热5分） | 实测−模拟百分点 |','|---|---:|---:|---:|']
    for k in expected:lines.append(f"| {k} | {expected[k]:.3%} | {actual[k]:.3%} | {comparison['difference_pp'][k]:+.3f} |")
    lines+=['','名义口径按 token 加权，三类相加为100%。有限窗口、固定服务时间模型、真实 vLLM 块生命周期/异步搬运、共享资源均可能造成差异，本次对照不能独立归因。',
            '', '![等待队列和运行数](server_queue.png)','','![两卡利用率与显存](gpu_utilization.png)','',
            '[run.json](run.json) · [分析](pilot-analysis.json) · [原始 dmon](dmon.log) · [逐请求名义来源](per_request_nominal.csv)',
            '', '指定配置是否有更大 Oracle 空间的替代项见 pilot-analysis.json 的 oracle_ranking 和 noticeably_larger_oracle_configurations；完整对照见相邻 Oracle 结果目录。']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-index',type=int,default=1)
    parser.add_argument('--port',type=int,default=8012)
    parser.add_argument('--comparison-dir',type=Path)
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    if args.analyze_only:analyze_pilot(args.output_dir,args.comparison_dir)
    else:run(args.output_dir,args.gpu_index,args.port,args.comparison_dir)


if __name__=='__main__':main()
