"""Write mbirtorch/data/gpu_speeds.json from the newest nightly performance records.

Run before each release:

    python dev_scripts/update_gpu_speeds.py

The script downloads the newest nightly records for the prerelease branch from the
cabouman/mbirtorch_metrics repository, fits the time lines that mbirtorch.estimate_resources
uses, prints the old and new values, and writes the speed file.  It is the only code that
knows the format of the nightly records.  If the records are missing or have changed
format, it stops and writes nothing.

The "gpus" section of the speed file (card memory, driver reserve, and time factor of each
GPU model) is entered by hand and kept as it is.
"""
import json
import pathlib
import re
import sys
import urllib.request

import numpy as np
import yaml

import mbirtorch
from mbirtorch.resources import projection_work

REPOSITORY = 'cabouman/mbirtorch_metrics'
BRANCH = 'prerelease'
RAW = f'https://raw.githubusercontent.com/{REPOSITORY}/main'
RECORDS_URL = f'{RAW}/results/gpu/{BRANCH}/records_gpu.yaml'
TABLES_URL = f'https://api.github.com/repos/{REPOSITORY}/contents/results/gpu/{BRANCH}'
HARNESS_URL = f'{RAW}/tooling/scaling_tests/performance_tracking.py'
SPEED_FILE = pathlib.Path(__file__).resolve().parents[1] / 'mbirtorch' / 'data' / 'gpu_speeds.json'

#: The GPU models the speed file starts with when it does not exist yet.
INITIAL_GPUS = {
    'H100': {'card_memory_gb': 80, 'driver_reserve_gib': 2.0, 'time_factor': 1.0},
    # Memory speed of an H100 is 1.7 times that of an 80 GB A100 and 2.2 times that of a 40 GB A100.
    'A100': {'card_memory_gb': None, 'driver_reserve_gib': 2.0, 'time_factor': 2.0},
}


def fail(message):
    """Stop without writing anything."""
    sys.exit(f'update_gpu_speeds: {message}  The speed file was not changed.')


def fetch(url):
    """Return the text at ``url``."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read().decode()
    except OSError as error:
        fail(f'could not download {url} ({error}).')


def nightly_model(geometry, size):
    """Return the model the nightly runs build for ``geometry`` and sinogram ``size``."""
    num_views, num_rows, num_channels = size
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    if geometry == 'parallel':
        model = mbirtorch.ParallelBeamModel(size, angles)
    else:
        source_detector_dist = 4.0 * num_channels
        model = mbirtorch.ConeBeamModel(size, angles, source_detector_dist=source_detector_dist,
                                        source_iso_dist=source_detector_dist / 2.0)
        model.set_params(recon_shape=(num_channels, num_channels, num_rows), no_warning=True)
    return model


def iso_date(date):
    """Return a nightly run date such as 20261010 as 2026-10-10."""
    text = str(date)
    return f'{text[:4]}-{text[4:6]}-{text[6:8]}' if re.fullmatch(r'\d{8}', text) else text


def fit_line(work, seconds):
    """Return a and b of seconds = a + b * work, by least squares with a >= 0."""
    work, seconds = np.asarray(work, float), np.asarray(seconds, float)
    b, a = np.polyfit(work, seconds, 1) if len(work) > 1 else (seconds[0] / work[0], 0.0)
    if a < 0:
        a, b = 0.0, float(np.dot(work, seconds) / np.dot(work, work))
    return float(a), float(b)


def main():
    records = yaml.safe_load(fetch(RECORDS_URL))
    if not isinstance(records, dict) or not records:
        fail('the nightly records are empty or not a table.')
    match = re.search(r'vcd_iterations:\s*int\s*=\s*(\d+)', fetch(HARNESS_URL))
    if match is None:
        fail('the nightly harness no longer states vcd_iterations.')
    vcd_iterations = int(match.group(1))
    tables = sorted(item['name'] for item in json.loads(fetch(TABLES_URL))
                    if re.fullmatch(r'regression_gpu_\d{8}T\d{6}Z_[0-9a-f]+_table\.yaml', item['name']))
    if not tables:
        fail('no nightly run table was found.')
    run = yaml.safe_load(fetch(f'{RAW}/results/gpu/{BRANCH}/{tables[-1]}'))['run']
    device = run['device']
    if 'H100' not in device:
        fail(f'the nightly runs are on "{device}", but the speed file measures H100 times.')

    # Each record key is "geometry|operation|views x rows x channels|GPU count".
    times = {}
    for geometry in ('cone', 'parallel'):
        cells = {}
        for key, record in records.items():
            parts = key.split('|')
            if len(parts) != 4 or parts[0] != geometry:
                continue
            _, op, size_label, count = parts
            try:
                cells[(op, size_label, int(count))] = record['min_ms']['value'] / 1000.0
            except (KeyError, TypeError, ValueError):
                fail(f'record {key} has no min_ms value.')
        lines = {'recon': {}, 'direct': {}}
        for count in sorted({c for (_, _, c) in cells}):
            sizes = sorted({s for (op, s, c) in cells if c == count and op == 'vcd_nonconst'})
            work, iteration, direct = [], [], []
            for size_label in sizes:
                needed = [('vcd_nonconst', size_label, count), ('direct_filter', size_label, count),
                          ('back', size_label, count)]
                if not all(k in cells for k in needed):
                    continue
                size = tuple(int(n) for n in size_label.split('x'))
                work.append(projection_work(nightly_model(geometry, size)))
                iteration.append(cells[needed[0]] / vcd_iterations)
                direct.append(cells[needed[1]] + cells[needed[2]])
            if len(work) < 2:
                continue
            for name, seconds in (('recon', iteration), ('direct', direct)):
                a, b = fit_line(work, seconds)
                lines[name][str(count)] = {'a': a, 'b': b, 'largest_work': max(work)}
        if not lines['recon']:
            fail(f'the nightly records have no usable {geometry} reconstruction times.')
        times[geometry] = lines

    old = json.loads(SPEED_FILE.read_text()) if SPEED_FILE.exists() else None
    new = {
        'format_version': 1,
        'source': {'records': REPOSITORY, 'branch': BRANCH, 'date': iso_date(run['date']),
                   'mbirtorch_commit': str(run['commit']), 'device': device},
        'reference_gpu': 'H100',
        'gpus': old['gpus'] if old else INITIAL_GPUS,
        'times': times,
    }

    print(f'Nightly run of {run["date"]} on {device}, commit {run["commit"]}.')
    print('Seconds for a 1024-cube problem (old -> new):')
    for geometry in times:
        for workload, lines in times[geometry].items():
            for count, line in lines.items():
                work = 1024 * 992 * 992 * 1008
                new_s = line['a'] + line['b'] * work
                try:
                    o = old['times'][geometry][workload][count]
                    old_s = f'{o["a"] + o["b"] * work:8.2f}'
                except (TypeError, KeyError):
                    old_s = '     new'
                print(f'  {geometry:9s} {workload:7s} {count} GPUs: {old_s} -> {new_s:8.2f}')
    SPEED_FILE.parent.mkdir(exist_ok=True)
    SPEED_FILE.write_text(json.dumps(new, indent=2) + '\n')
    print(f'Wrote {SPEED_FILE}')


if __name__ == '__main__':
    main()
