#!/usr/bin/env python3
"""Stage a frozen self-play corpus and paired continuation runs."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--games', type=int, default=2032)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    binary, model, out = (path.resolve() for path in (a.binary, a.model, a.out))
    if not binary.is_file() or not model.is_file():
        p.error('binary and seed model must exist')
    if a.games < 32:
        p.error('--games must be at least 32')
    if out.exists():
        p.error('choose a fresh output directory')
    trainer = Path(__file__).with_name('train_resnet.py').resolve()
    commands = []
    for name, games, seed in [('train', a.games, 2026112801), ('holdout', 400, 2026112802)]:
        commands.append([str(binary), '--mode', 'selfplay', '--model', str(model),
                         '--out', str(out / f'{name}.bin'), '--games', str(games),
                         '--batch-games', '32', '--groups', '3', '--chunk', '1024',
                         '--ruleset', 'nopass', '--komi', '0.5', '--visits', '1024',
                         '--opening', '2', '--temp', '1.0', '--temp-moves', '12',
                         '--root-k', '32', '--top-k', '16', '--noise', '0.25',
                         '--dirichlet', '0.3', '--seed', str(seed)])
    for seed in (11, 29):
        for epochs in (2, 27):
            name = f'e{epochs}-seed{seed}'
            commands.append([sys.executable, str(trainer), '--init-model', str(model),
                             '--data', str(out / 'train.bin'), '--out', str(out / f'{name}.bin'),
                             '--log-csv', str(out / f'{name}.csv'), '--epochs', str(epochs),
                             '--channels', '96', '--blocks', '6', '--input-planes', '37',
                             '--a-cond', '--a-anchor', '0.5', '--value-weight', '1.0',
                             '--score-weight', '0.5', '--ruleset', 'nopass', '--device', 'cuda',
                             '--lr', '0.0002', '--batch', '512', '--val', '0.06', '--seed', str(seed)])
    if a.execute:
        out.mkdir(parents=True)
        (out / 'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
    for index, command in enumerate(commands, 1):
        print(shlex.join(command), flush=True)
        if a.execute:
            with (out / f'step-{index:02}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
