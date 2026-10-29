#!/bin/bash
# exp120 descriptive counts (aicr 1582199), submitted as:
# sbatch -J oeix-exp120-describe -A p2026_0089_neu -p cpu -c 1 --mem=4G -t 00:10:00 --wrap "cd .../checks/exp120 && venv/bin/python -I exp/exp120_describe.py --units out/exp120_units.json --tile out/exp120_tile.json --out out/exp120_describe.json"
