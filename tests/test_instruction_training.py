"""Target masking, disjoint datasets and real tiny-model instruction updates."""

import ast
import json

import pytest
import torch

from data.build_instructions import CODE, build, examples, split_rows
from evals.evidence import file_sha256
from finetune import batch, encode_examples, tune
from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def test_seed_dataset_is_reproducible_and_family_disjoint(tmp_path):
    first = build(tmp_path)
    second = build(tmp_path)
    assert first == second
    training, validation = split_rows(examples())
    assert len(training) == 300 and len(validation) == 68
    assert not {r['family'] for r in training} & {r['family'] for r in validation}
    assert len({r['id'] for r in training + validation}) == 368
    for arguments, body, _ in CODE.values():
        ast.parse(f'def checked({arguments}):\n    {body}\n')


def test_training_does_not_copy_benchmark_prompts():
    from evals.tasks import default_task_files, load_tasks
    prompts = {t['prompt'].strip() for t in load_tasks(default_task_files())}
    assert not prompts & {r['prompt'].strip() for r in examples()}


def test_authored_templates_match_reference_outputs():
    # These are reviewed project-authored constants, never generated model output.
    cases = {
        'square': [((-3,), 9)], 'absolute': [((-4,), 4)],
        'clip': [((20, 1, 9), 9), ((-1, 0, 5), 0)],
        'nonempty': [(([0, '', None, 'a', 2],), ['a', 2])],
        'trim': [((' hi\n',), 'hi')], 'normalize': [((' Hi ',), 'hi')],
        'positive': [(([-2, 0, 4, 3],), 7)],
        'prefix': [(('abc', 'ab'), True)], 'suffix': [(('abc', 'bc'), True)],
        'default': [((None, 4), 4), ((0, 4), 0)],
        'lengths': [((['a', 'abc'],), [1, 3])],
        'lookup': [(({'a': 1}, 'missing', 2), 2)],
        'count_lines': [(('',), 0), (('one\ntwo\n',), 2)],
        'log_line': [(('one\r\ntwo',), 'one  two')],
        'valid_port': [((True,), False), ((65536,), False), ((443,), True)],
        'mask': [(('abcdef',), '**cdef'), (('ab',), '**'), (('',), '')],
    }
    assert cases.keys() == CODE.keys()
    for family, tests in cases.items():
        arguments, body, _ = CODE[family]
        namespace = {}
        exec(f'def f({arguments}):\n    {body}\n', namespace)
        for args, expected in tests:
            assert namespace['f'](*args) == expected, family


def test_response_targets_exclude_prompt_and_padding(tmp_path):
    tokenizer = CharTokenizer('abcdef\n ')
    path = tmp_path/'rows.jsonl'
    path.write_text(json.dumps({'id':'a','family':'one','prompt':'abc','response':'de'})+'\n')
    _, encoded = encode_examples(path, tokenizer, 8)
    x, y = encoded[0]
    assert x.tolist() == tokenizer.encode('abcd')
    assert y.tolist() == [-100, -100, *tokenizer.encode('de')]
    short = (x[:2], y[-2:])
    tokens, targets = batch([encoded[0], short], 16, torch.Generator().manual_seed(0), 'cpu')
    assert targets.shape == tokens.shape
    for row in range(len(tokens)):
        if torch.equal(tokens[row, :2], short[0]) and (tokens[row, 2:] == 0).all():
            assert (targets[row, 2:] == -100).all()
    with pytest.raises(ValueError, match='exceeds context'):
        encode_examples(path, tokenizer, 3)


def test_real_training_saves_new_weights_and_keeps_source(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    training = {'id':'a','family':'train','prompt':'Q: hi\nA: ','response':'hello\n'}
    validation = {'id':'b','family':'validation','prompt':'Q: bye\nA: ','response':'bye\n'}
    (data/'train.jsonl').write_text(json.dumps(training)+'\n')
    (data/'validation.jsonl').write_text(json.dumps(validation)+'\n')
    tok = CharTokenizer(training['prompt']+training['response']+validation['prompt']+validation['response'])
    cfg = GPTConfig(tok.vocab_size, 32, n_layer=1, n_head=1, n_embd=8, dropout=0)
    model = GPT(cfg)
    source = tmp_path/'base.pt'
    torch.save({'model_state':model.state_dict(),'model_cfg':cfg.__dict__,'tokenizer':tok.to_dict()},source)
    original = file_sha256(source)
    destination = tmp_path/'candidate'/'ckpt.pt'
    report = tune(source,data,destination,steps=2,batch_size=1,learning_rate=.001,context=32,threads=1)
    assert file_sha256(source) == original
    last = torch.load(destination.with_name('last.pt'),weights_only=True)
    assert last['instruction_tuning']['step'] == 2
    assert last['optimizer_state']['state']
    assert any(not torch.equal(v,last['model_state'][key]) for key,v in model.state_dict().items())
    assert report['best_validation_loss'] <= report['base_validation_loss']
    with pytest.raises(ValueError,match='base checkpoint'):
        tune(source,data,source,steps=1)
    validation['family']='train'
    (data/'validation.jsonl').write_text(json.dumps(validation)+'\n')
    with pytest.raises(ValueError,match='families overlap'):
        tune(source,data,tmp_path/'other.pt',steps=1,context=32,threads=1)
