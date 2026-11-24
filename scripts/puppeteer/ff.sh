#!/bin/bash
# render de primeros cuadros (8 vistas) con el renderer de Puppeteer. Uso: ff.sh <nombre>
N=$1; A=/root/puppeteer/animation
source /root/venv-pup/bin/activate; export PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$A
mkdir -p $A/inputs/$N/objs; cp /root/work/$N/objs/* $A/inputs/$N/objs/
cd $A; /root/gpurun.sh ${N}_firstframe python utils/render_first_frame.py --input_path inputs --seq_name $N --size 512
