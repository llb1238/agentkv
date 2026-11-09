"""Frozen, preregistered 24-run E1: CPU predictions, registration, durable resume."""
import argparse
import functools
from collections import defaultdict
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
import multiprocessing
import socket
from pathlib import Path
import random
import signal
import shutil
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit,current_commit
from agentkv.bench.e0 import ROOT,WORKLOAD,SCHEDULE,load_inputs,population_hash,write_json
from agentkv.bench.steady_state import sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.e1_simulation import _schedule_fingerprint
from agentkv.policy.prefix_adapter import frozen_options
from agentkv.sim.hint_adapter import HintPolicyAdapter
from agentkv.sim.two_tier import simulate

RATES=(1.,1.4)
SEEDS=(3,4,5)
POLICIES=('lru','lru-final','online','perfect')
ORDER_SEED=20261104
BACKEND='4749dd5961841243596559b10ba0a22c48980c45'
FREEZE=ROOT/'docs/m5_scoring_frozen.json'
PROTOCOL=ROOT/'docs/e1_gpu_preregistration.json'
DOC=ROOT/'docs/e1_gpu_preregistration.md'
RESULTS=ROOT/'results/e1-gpu-config11-20261105'
PINNED=('agentkv/bench/e1_gpu.py','agentkv/analysis/e1_gpu.py','agentkv/bench/effective_observer.py',
        'agentkv/bench/observability_plugin.py','agentkv/bench/shared_pilot.py','agentkv/bench/replay.py',
        'agentkv/policy/light_metrics.py','agentkv/policy/vllm_plugin.py','agentkv/policy/measured_policy.py',
        'agentkv/policy/prefix_adapter.py','agentkv/policy/optimized_adapter.py','agentkv/policy/optimized_policy.py',
        'agentkv/policy/hint_aware.py','agentkv/policy/remaining_life.py','agentkv/analysis/stability.py',
        'agentkv/analysis/recomputation.py','agentkv/bench/steady_state.py','agentkv/hints/protocol.py')


def read(path):return json.loads(path.read_text())


def plan():
    rng=random.Random(ORDER_SEED);groups=[]
    for rate in RATES:
        for seed in SEEDS:
            order=list(POLICIES);rng.shuffle(order)
            groups.append(dict(name=f'lambda-{rate:g}-seed-{seed}',rate=rate,seed=seed,order=order))
    return groups


def registered_plan(protocol):
    if protocol.get('format')=='agentkv.stable_limit.v1':
        rng=random.Random(protocol['order_seed']);groups=[]
        assert protocol['policies']==['lru','online','perfect']
        for seed in SEEDS:
            order=list(protocol['policies']);rng.shuffle(order)
            groups.append(dict(name=f'lambda-1.2-seed-{seed}',rate=1.2,seed=seed,order=order))
        return groups
    groups=plan();labels=protocol.get('policies',POLICIES)
    for group in groups:group['order']=[label for label in group['order'] if label in labels]
    return groups


def model_policy(label):
    if label=='lru':return 'lru'
    if label=='lru-final':return HintPolicyAdapter(274320//16,hint_tier='final',scoring='lru_final',tail_first=False,final_first=True,timeout_decay=False)
    return HintPolicyAdapter(274320//16,hint_tier=label,**frozen_options(label))


_MODEL_ROOTS=_MODEL_BLOCKS=None


def _predict_group(arguments):
    group,output,commit=arguments
    roots,blocks=_MODEL_ROOTS,_MODEL_BLOCKS
    cases=[]
    instances=sample_population({k:r.duration_s for k,r in roots.items()},group['rate'],group['seed'],2400,True)
    folder=output/group['name'];folder.mkdir()
    write_json(folder/'population.json',dict(instances=[asdict(i) for i in instances],population_sha256=population_hash(instances)))
    reference=None
    for label in POLICIES:
        summary,rows=simulate(roots,instances,blocks,182880,274320,8,duration=2400,warmup=600,service_time=.5,cpu_policy=model_policy(label))
        fingerprint=_schedule_fingerprint(rows)
        if reference is None:reference=fingerprint
        assert fingerprint==reference,'CPU cache policy changed fixed-service GPU/schedule'
        window=[r for r in rows if r['in_window']]
        prompt=sum(r['prompt_tokens'] for r in window)
        # Qwen causal generation must forward the terminal prompt token.
        work=sum(max(1,r['recomputed']) for r in window)
        destination=folder/label;destination.mkdir()
        pq.write_table(pa.Table.from_pylist(rows),destination/'requests.parquet',compression='zstd')
        record=dict(git_commit=commit,status='complete',group=group['name'],rate=group['rate'],seed=group['seed'],policy=label,
            population_sha256=population_hash(instances),schedule_gpu_sha256=fingerprint,
            predicted_effective_recompute=work/prompt,terminal_token_correction=work-summary['source_tokens']['recomputed'],
            summary=summary,requests_sha256=sha256_file(destination/'requests.parquet'))
        write_json(destination/'run.json',record);cases.append(record)
        print('SIM',group['name'],label,record['predicted_effective_recompute'],flush=True)
    return cases


def predictions(output,workers=4):
    global _MODEL_ROOTS,_MODEL_BLOCKS
    commit=git_commit()
    freeze=read(FREEZE)
    assert sha256_file(ROOT/'agentkv/policy/hint_aware.py')==freeze['source_hashes']['agentkv/policy/hint_aware.py']
    assert sha256_file(ROOT/'agentkv/policy/data/remaining_life_v1.json')==freeze['remaining_sha256']
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage predictions outside Git')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);_MODEL_ROOTS,_MODEL_BLOCKS=load_inputs()
    with multiprocessing.get_context('fork').Pool(workers) as pool:
        batches=pool.map(_predict_group,[(g,output,commit) for g in plan()])
    cases=[c for batch in batches for c in batch]
    write_json(output/'run.json',dict(git_commit=commit,status='complete',groups=plan(),cases=cases,
        source=dict(workload_sha256=sha256_file(WORKLOAD),schedule_sha256=sha256_file(SCHEDULE),freeze_sha256=sha256_file(FREEZE))))


def register(models):
    if PROTOCOL.exists() or DOC.exists():raise FileExistsError('Never overwrite preregistration')
    simulations=read(models/'run.json')
    assert simulations['status']=='complete' and len(simulations['cases'])==24
    for case in simulations['cases']:
        folder=models/case['group']/case['policy']
        assert read(folder/'run.json')==case and sha256_file(folder/'requests.parquet')==case['requests_sha256']
    protocol=dict(format='agentkv.e1_gpu.v1',order_seed=ORDER_SEED,groups=plan(),backend_commit=BACKEND,
        semantic_freeze_commit=read(FREEZE)['freeze_code_commit'],optimized_core_freeze_commit='e2501f0d63fec130ada2574803b044c9cd195310',
        parameters=dict(gpu_tokens=182880,cpu_tokens=274320,max_num_seqs=8,context_budget=16384,duration_s=2400,warmup_s=600,
            cooldown_s=0,max_inflight=128,request_timeout_s=180,drain_timeout_s=120,service_model_s=.5),
        stability_preregistration='a81dd86',source=simulations['source'],simulation_commit=simulations['git_commit'],
        simulation_run_sha256=sha256_file(models/'run.json'),simulation_directory=str(models.resolve().relative_to(ROOT)),predictions=simulations['cases'],
        runtime_source_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'agentkv').rglob('*.py'))})
    write_json(PROTOCOL,protocol)
    lines=['# E1 GPU 正式实验预注册','',
        '本节在任何GPU运行之前提交。提交后不改预注册正文；结果只在末尾追加或放结果目录。所有24项CPU预测已运行，输入与原始请求指纹随预注册提交。','',
        '配置：config-11，16K、8槽位、GPU182880/CPU274320 token；Qwen3-8B；vLLM fork `'+BACKEND+'`。评分语义冻结提交 `'+protocol['semantic_freeze_commit']+'`，优化核心冻结于 `'+protocol['optimized_core_freeze_commit']+'`；具体运行实现按配套JSON的文件SHA256锁定，不改变评分/校准。',
        'λ∈{1,1.4}根会话/分钟，人口种子3/4/5，T=2400秒、预热600秒、收尾0。稳态初始化、人口有放回抽样与实例置换沿用M1。','',
        '## 策略与共同输入','',
        'LRU使用vLLM注册的内置lru类，仅加轻量计时；LRU+final只用身份、轮次和final，非final间隔在适配层归零；online使用冻结剩余寿命评分及online校准表；perfect使用冻结的deadline规则。四种策略的运行参数不变。',
        'LRU与LRU+final使用online信封传输；内置LRU忽略hint，final基线不读间隔预测。online/perfect使用各自hint。所有实例、根到达和每个逻辑请求的token序列固定，置换与策略无关。闭环中服务完成时刻会改变后续请求实际时刻、总发送集合和跨链顺序，不强称40分钟内的实际发送列表相同；共同请求逐项比较token哈希/置换种子/长度。','',
        '## 固定运行顺序','',f'按(λ,种子)分组，组内连续运行四策略，Python random.Random({ORDER_SEED})依次shuffle同一初始策略列表。重跑紧随被中断的策略，不改后续顺序。','',
        '| λ | 人口种子 | 组内顺序 |','|---:|---:|---|']
    for g in plan():lines.append(f"| {g['rate']:g} | {g['seed']} | {' → '.join(g['order'])} |")
    lines+=['','## 测量口径与分析','',
        '主要指标仅纳入稳定运行：有效重算比例=成功、非is_entry、发送时刻在[600,2400]秒的请求，其实际完成前向中的prompt token工作量之和/这些请求prompt token之和；不把现有名义重算冒充有效值。CPU观察器在_update_after_schedule前保存prompt区间，在update_from_output收到模型输出后累计，decode位置排除。分块、末token replay与重复prefill均计工作量；未执行/未返回的batch不计。',
        '同时保留名义三分区和有效去重三分区；设备来源可确定时有效去重GPU+CPU+重算=prompt长度。若抢占后重新命中使设备来源无法从首轮名义计数确定，GPU/CPU有效来源置空并报告未知来源命中数；实际前向工作量仍直接观测，不从名义计数推断。重复prefill工作量另列，主指标包含重复工作量，因此有重复时可大于去重比例。有效前向工作量未观测则不能纳入主指标，不以名义字段补齐。',
        'TTFT P50/P99使用相同发送窗口内成功非is_entry请求。稳定性严格沿用a81dd86：零等待采样≥80%、等待P99≤8；等待对实际时间OLS斜率×1800<1；后半TTFT中位数/前半≤1.5；有未开始请求因KV不足阻塞且等待>0时运行数≥7。轻量模式精确累计统计窗口内阻塞事件的running直方图，不用每秒最大运行数替代事件值。缺观察记无法判定，不默认稳定。',
        '每个λ按人口做配对，只有策略及其同人口LRU都稳定时进入主配对均值/最小–最大；明确报告有效配对数，不把缺失补零。各策略绝对指标按自身稳定运行等权汇总，逐人口原值全部列出。仅3个人口，不做显著性检验。',
        '所有运行报告稳定/不稳定/无法判定/失败以及窗口内完成的请求数。完成数按end_ns落在[600,2400]且成功计，分别列全部与非is_entry数量；它与按发送时刻的主指标分母不同。',
        '次要指标：三类重算沿既有16-token整块名义来源定义（不可避免、间隙期间含排队驱逐、其他驱逐），与有效token工作量的差额、尾块、末token replay和重复工作单列；hint传输与策略消费计数对账；所有回调轻量直方图、scheduler.schedule耗时；两卡dmon。',
        'TTFT CDF按稳定人口等权混合，仅展示分布；主表仍对每人口分位数等权汇总。主图按策略显示稳定实测及同人口模型预测，所有不稳定实测用不同标记列出但不进入稳定均值。λ=1.4另外列稳定次数、完成数和等待曲线；全部策略逐人口列实测与预测差值（不稳定明确标记）。','',
        '## CPU预测','',
        '服务时间固定0.5秒、搬运时间为0，沿用原两层模型及原冻结策略；预测有效重算为每请求max(1,名义重算)，反映因果生成至少计算末prompt token，不改变缓存决策或服务时长。模型不预测TTFT、不宣称真实驱逐/排队时序相同。','',
        '| λ | 种子 | LRU | LRU+final | online | perfect |','|---:|---:|---:|---:|---:|---:|']
    for g in plan():
        rows={c['policy']:c for c in simulations['cases'] if c['group']==g['name']}
        lines.append(f"| {g['rate']:g} | {g['seed']} | "+' | '.join(f"{rows[k]['predicted_effective_recompute']:.6%}" for k in POLICIES)+' |')
    lines+=['','## 中断、断点与保留','',
        '每策略独立服务和新目录，启动前验证干净工作区、固定后端与协议SHA。以flock防并发controller，原子checkpoint保存当前组/策略/attempt，完成项按产物SHA验证后跳过，未完成不覆盖。支持--max-groups分夜运行；同一命令续跑。',
        '非策略原因中断（不含策略栈异常/观测正确性错误）立即记录原因并原配置重跑一次；第二次失败如实报告，无第三次自动重跑。策略异常或无法保证观测正确性则停止，保留证据并报告，不能用重跑掩盖。资源未满足启动条件时等待，不算已开始的attempt；不终止其他GPU任务。人为中断可在续跑时使用一次中断重跑额度，已完成数据不变。',
        '默认24次连续执行。shared GPU与外部进程状态原样记录，不凭瞬时空闲声称算力独占。所有成功、失败、不稳定结果保留；原始run.json不改，退出后写单独e1-analysis.json关联来源。大prompt文件保留本机并忽略Git，不复制实现代码。','',
        f"机器协议SHA256：`{sha256_file(PROTOCOL)}`；模拟代码提交：`{simulations['git_commit']}`；模拟总清单SHA256：`{protocol['simulation_run_sha256']}`。"]
    DOC.write_text('\n'.join(lines)+'\n')
    print(PROTOCOL,sha256_file(PROTOCOL),flush=True)


def verify_protocol(protocol_path=None):
    protocol_path=PROTOCOL if protocol_path is None else protocol_path.resolve()
    commit=git_commit();backend=git_commit(repository=ROOT.parent/'vllm')
    if backend!=BACKEND:raise ValueError('Pinned vLLM commit differs')
    p=read(protocol_path)
    registration=subprocess.check_output(['git','log','-1','--format=%H','--',str(protocol_path.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    if not registration:raise ValueError('Preregistration must be committed before GPU execution')
    raw=subprocess.check_output(['git','show',f'{registration}:{protocol_path.relative_to(ROOT)}'],cwd=ROOT)
    if raw!=protocol_path.read_bytes():raise ValueError('Protocol changed')
    expected=registered_plan(p)
    assert p['groups']==expected and p['backend_commit']==BACKEND
    registered=ROOT/p['simulation_directory']
    assert sha256_file(registered/'run.json')==p['simulation_run_sha256']
    for name,digest in p['runtime_source_sha256'].items():
        if sha256_file(ROOT/name)!=digest:raise ValueError(f'Frozen implementation changed: {name}')
    for key,path in [('workload_sha256',WORKLOAD),('schedule_sha256',SCHEDULE),('freeze_sha256',FREEZE)]:
        if sha256_file(path)!=p['source'][key]:raise ValueError(f'Input changed: {key}')
    return commit,registration,p


def failure_kind(directory,error):
    text=(directory/'server.log').read_text(errors='replace') if (directory/'server.log').exists() else ''
    policy_frames=('agentkv/policy/hint_aware.py','agentkv/policy/optimized_policy.py','agentkv/policy/optimized_adapter.py',
                   'agentkv/policy/chain_policy.py','agentkv/policy/chain_adapter.py',
                   'agentkv/policy/vllm_plugin.py','agentkv/policy/measured_policy.py')
    if any(s in text for s in policy_frames) or isinstance(error,(AssertionError,ValueError,KeyError,TypeError)):
        return 'strategy_or_correctness'
    return 'non_strategy_interruption'


def restore_sigterm(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        previous=signal.getsignal(signal.SIGTERM)
        try:return function(*args,**kwargs)
        finally:signal.signal(signal.SIGTERM,previous)
    return wrapped


@restore_sigterm
def run(output,*,gpu=1,port=8016,max_groups=None,protocol_path=None):
    verify=(lambda:verify_protocol()) if protocol_path is None else (lambda:verify_protocol(protocol_path))
    commit,registration,protocol=verify()
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage GPU output outside Git')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'checkpoint.json';digest=sha256_file(PROTOCOL if protocol_path is None else protocol_path)
        state=read(path) if path.exists() else dict(git_commit=commit,preregistration_commit=registration,
            protocol_sha256=digest,status='running',runs=[])
        if state['protocol_sha256']!=digest:raise ValueError('Cannot resume a different protocol')
        for group in protocol['groups']:
            for label in group['order']:
                identity=group['name']+'/'+label
                if not any(r['id']==identity for r in state['runs']):
                    state['runs'].append(dict(id=identity,group=group['name'],rate=group['rate'],seed=group['seed'],
                        policy=label,status='pending',attempts=[]))
        state['status']='running';write_json(path,state)
        roots,blocks=load_inputs();finished_groups=0
        def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,interrupt)
        from agentkv.bench.shared_pilot import run as serve
        from agentkv.analysis.e1_gpu import analyze,render,verify_result
        for group in protocol['groups']:
            group_had_work=False
            for label in group['order']:
                identity=group['name']+'/'+label
                item=next((r for r in state['runs'] if r['id']==identity),None)
                if item is None:
                    item=dict(id=identity,group=group['name'],rate=group['rate'],seed=group['seed'],policy=label,status='pending',attempts=[])
                    state['runs'].append(item)
                if item['status']=='complete':
                    record=output/item['selected_attempt']/'e1-analysis.json'
                    if sha256_file(record)!=item['analysis_sha256']:raise ValueError('Completed artifact changed')
                    verify_result(record.parent,read(record))
                    continue
                if item['status']=='failed':continue
                if item['status']=='blocked':raise ValueError(f"Blocked correctness failure: {identity}")
                group_had_work=True
                if item['attempts'] and item['attempts'][-1]['status']=='running':
                    recovered=output/item['attempts'][-1]['directory']/'e1-analysis.json'
                    if recovered.exists():
                        result=read(recovered);verify_result(recovered.parent,result)
                        item['attempts'][-1]['status']='complete'
                        item.update(status='complete',selected_attempt=item['attempts'][-1]['directory'],
                            result=result,analysis_sha256=sha256_file(recovered));write_json(path,state)
                        continue
                # An unclean previous controller must never overlap its old server.
                if item['attempts'] and item['attempts'][-1]['status']=='running':
                    old=output/item['attempts'][-1]['directory'];status=read(old/'pilot-status.json') if (old/'pilot-status.json').exists() else {}
                    for key in ('server_pid','client_pid','dmon_pid'):
                        pid=status.get(key)
                        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError(f'Owned prior process still alive; investigate {old}: {pid}')
                    item['attempts'][-1].update(status='interrupted',reason='Previous controller stopped before recording completion')
                while len(item['attempts'])<2:
                    verify()
                    free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    available=dict((int(a),float(b)) for a,b in (line.split(',') for line in free.splitlines()))
                    if available[gpu]<58000:
                        state.update(status='waiting_for_gpu',waiting_for=identity);write_json(path,state)
                        print('WAIT_GPU',identity,available,flush=True);time.sleep(30);continue
                    with socket.socket() as sock:
                        busy=sock.connect_ex(('127.0.0.1',port))==0
                    if busy:
                        state.update(status='waiting_for_port',waiting_for=identity);write_json(path,state)
                        print('WAIT_PORT',port,flush=True);time.sleep(30);continue
                    state['status']='running';state.pop('waiting_for',None)
                    number=len(item['attempts'])+1;directory=output/group['name']/label/f'attempt-{number}'
                    directory.parent.mkdir(parents=True,exist_ok=True)
                    config=dict(format='agentkv.vllm_hint_policy.v1',hint_tier='perfect' if label=='perfect' else 'online',
                        freeze_sha256=sha256_file(FREEZE),residual_table_path=str(ROOT/'agentkv/policy/data/remaining_life_v1.json'))
                    if label=='lru-final':config['policy_variant']='lru-final'
                    if label!='lru' and protocol.get('policy_implementation'):
                        config['implementation']=protocol['policy_implementation']
                    config_path=directory.with_name(f'attempt-{number}-policy-config.json');write_json(config_path,config)
                    attempt=dict(directory=str(directory.relative_to(output)),number=number,status='running',git_commit=commit)
                    item['attempts'].append(attempt);item['status']='running';write_json(path,state)
                    print('BEGIN',identity,'attempt',number,flush=True)
                    try:
                        serve(directory,gpu,port,duration=2400,warmup=600,rate=group['rate'],population_seed=group['seed'],
                            run_id='e1-'+group['name'],analyze_output=False,cache_policy='lru' if label=='lru' else 'HintAwareCachePolicy',
                            cache_policy_module=None if label=='lru' else 'agentkv.policy.vllm_plugin',hint_tier=config['hint_tier'],
                            extra_environment=dict(AGENTKV_INSTRUMENTATION='light',AGENTKV_EFFECTIVE_PREFILL='1',
                                AGENTKV_BUILTIN_LRU_TIMING='1' if label=='lru' else '0',AGENTKV_HINT_POLICY_CONFIG=str(config_path),PYTHONHASHSEED='0'))
                        shutil.copyfile(config_path,directory/'policy-config.json')
                        result=analyze(directory,roots,blocks,protocol,group,label)
                        attempt['status']='complete';item.update(status='complete',selected_attempt=attempt['directory'],
                            analysis_sha256=sha256_file(directory/'e1-analysis.json'),result=result)
                        write_json(path,state);render(output,state,protocol)
                        print('DONE',identity,result['stability']['status'],result['effective_recompute'],flush=True)
                        break
                    except BaseException as error:
                        kind=failure_kind(directory,error)
                        attempt.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',kind=kind,
                            reason=f'{type(error).__name__}: {error}')
                        directory.mkdir(parents=True,exist_ok=True)
                        try:
                            from agentkv.bench.e0 import gpu_snapshot
                            attempt['gpu_after_failure']=gpu_snapshot()
                        except (OSError,subprocess.SubprocessError) as snapshot_error:
                            attempt['gpu_snapshot_error']=str(snapshot_error)
                        write_json(directory/'failure.json',attempt)
                        item['status']='blocked' if kind=='strategy_or_correctness' else 'pending' if number<2 else 'failed'
                        state['status']='blocked' if kind=='strategy_or_correctness' else 'interrupted' if isinstance(error,KeyboardInterrupt) else 'running'
                        write_json(path,state);render(output,state,protocol)
                        print('FAILED',identity,number,kind,repr(error),flush=True)
                        if isinstance(error,KeyboardInterrupt) or kind=='strategy_or_correctness':raise
                if item['status']=='pending':item['status']='failed';write_json(path,state)
            if group_had_work:finished_groups+=1
            if max_groups is not None and finished_groups>=max_groups:break
        complete=sum(r['status']=='complete' for r in state['runs'])
        terminal=sum(r['status'] in ('complete','failed') for r in state['runs'])
        total=sum(len(g['order']) for g in protocol['groups'])
        state['status']='complete' if complete==total else 'complete_with_failures' if terminal==total else 'partial'
        write_json(path,state);render(output,state,protocol)
        return state


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('predict');a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--workers',type=int,default=4)
    a=sub.add_parser('register');a.add_argument('--models',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--gpu-index',type=int,default=1)
    a.add_argument('--port',type=int,default=8016);a.add_argument('--max-groups',type=int)
    a.add_argument('--protocol',type=Path)
    a=sub.add_parser('status');a.add_argument('--output-dir',type=Path,required=True)
    a=sub.add_parser('report');a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--protocol',type=Path,default=PROTOCOL)
    args=p.parse_args()
    if args.command=='predict':predictions(args.output_dir,args.workers)
    elif args.command=='register':register(args.models)
    elif args.command=='run':run(args.output_dir,gpu=args.gpu_index,port=args.port,max_groups=args.max_groups,protocol_path=args.protocol)
    elif args.command=='status':
        s=read(args.output_dir/'checkpoint.json');print(json.dumps(dict(status=s['status'],completed=sum(r['status']=='complete' for r in s['runs']),runs=[{k:r[k] for k in ('id','status')} for r in s['runs']]),indent=2))
    else:
        from agentkv.analysis.e1_gpu import render
        render(args.output_dir,read(args.output_dir/'checkpoint.json'),read(args.protocol))


if __name__=='__main__':main()
