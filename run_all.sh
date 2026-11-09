#!/usr/bin/env bash
set -e
python generate_data.py
python part_a.py
python validate.py
python part_b.py
python part_c.py
python part_d.py
