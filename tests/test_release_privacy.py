import hashlib
import json

import pytest

from agentkv.analysis.publication_audit import audit_release, scan_text
from agentkv.analysis.release_privacy import (
    HOST_REDACTION, PROCESS_REDACTION, redact_tree, transform_json,
)


HOME = '/' + 'home' + '/'


def kinds(data):
    return {v['kind'] for v in scan_text(json.dumps(data).encode(), 'sample.json')}


def test_audit_rejects_other_users_home_and_redacts_finding_values(tmp_path):
    secret = HOME + 'another_user/tool/bin/python'
    path = tmp_path/'sample.json'
    path.write_text(json.dumps({'command':secret}))
    assert 'foreign_home' in kinds({'command':secret})
    with pytest.raises(ValueError, match='Publication privacy audit failed') as failure:
        audit_release(tmp_path)
    assert secret not in str(failure.value)
    assert 'foreign_home' not in kinds({'command':HOME+'llb/my_soft/vllm/.venv/bin/python'})


def test_audit_rejects_other_users_environment_paths():
    folder = 'mini' + 'conda3'
    assert 'foreign_environment' in kinds({'command':'/opt/'+folder+'/envs/other/bin/python'})
    assert 'foreign_environment' in kinds({'command':'/shared/'+'another_user/'+'venv/bin/python'})


def test_audit_rejects_hostnames():
    assert 'hostname' in kinds({'hostname':'private'+'-compute-123'})
    changed, _ = transform_json({'hostname':'private'+'-compute-123'})
    assert changed['hostname']==HOST_REDACTION
    assert 'hostname' not in kinds(changed)


def test_audit_rejects_unattributed_gpu_process_even_without_a_home_path():
    line = '|    0 N/A N/A  123456 C   unrelated-python  612MiB |'
    assert 'foreign_gpu_process' in kinds({'gpu_before':line})
    assert 'foreign_gpu_process' in kinds({'processes':'GPU-example, 123456, unrelated-python, 612'})


def test_redaction_preserves_gpu_memory_and_explicit_project_processes(tmp_path):
    original = {
        'processes':'GPU-example, 123456, '+HOME+'another_user/tools/bin/python, 612\n',
        'nested':{'gpu_before':'|    1 N/A N/A 987654 C '+('sg'+'lang')+'::scheduler 13327... |\n'},
        'own':{'processes':'GPU-example, 111, '+HOME+'llb/my_soft/vllm/.venv/bin/python, 2048\n'},
        'metrics':{'gpu':1,'free_MiB':10240,'recompute':.25},
        'gpus':'GPU-example, 182880, 155000, 27880',
    }
    raw=(json.dumps(original,indent=2)+'\n').encode()
    path=tmp_path/'a/nested/run.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    result=redact_tree(tmp_path)
    published=json.loads(path.read_text())
    assert published['own']==original['own'] and published['metrics']==original['metrics']
    assert published['gpus']==original['gpus']
    assert published['processes']=='GPU-example, '+PROCESS_REDACTION+', '+PROCESS_REDACTION+', 612\n'
    assert '1 N/A N/A '+PROCESS_REDACTION+' C 13327...' in published['nested']['gpu_before']
    assert not kinds(published)
    assert result['changed_files'][0]['original_sha256']==hashlib.sha256(raw).hexdigest()
    assert result['changed_files'][0]['published_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert not redact_tree(tmp_path)['changed_files']


def test_structured_process_records_and_whole_tree_audit(tmp_path):
    raw={'processes':[{'gpu':0,'pid':123,'process_name':'foreign-python','used_gpu_memory':99}]}
    changed, _=transform_json(raw)
    assert changed['processes'][0]==dict(gpu=0,pid=PROCESS_REDACTION,
                                       process_name=PROCESS_REDACTION,used_gpu_memory=99)
    assert not kinds(changed)
    target=tmp_path/'unexpected-experiment/deeper/analysis.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        audit_release(tmp_path)
