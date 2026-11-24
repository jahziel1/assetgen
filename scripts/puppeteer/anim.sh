#!/bin/bash
# Animacion guiada por video con Puppeteer + export a GLB animado.
# Uso: anim.sh <nombre> <video.mp4> <glb_original> [opciones extra de optimization.py]
# Prerrequisito: rig.sh <nombre> ya corrido (rig.txt y mesh.obj en animation/inputs/<nombre>/objs).
set -e
N=$1; VID=$2; GLB=$3; shift 3
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
P=/root/puppeteer; A=$P/animation; W=/root/work/$N
source /root/venv-pup/bin/activate
mkdir -p $A/inputs/$N/objs; cp $W/objs/* $A/inputs/$N/objs/
cd $A/inputs/$N; mkdir -p imgs; rm -f imgs/*.png
ffmpeg -y -loglevel error -i $VID -vf "fps=10,scale=960:960" imgs/frame_%04d.png
cp $VID input.mp4
cd $A; export PYTHONPATH=$A
/root/gpurun.sh ${N}_flow python utils/save_flow.py --input_path inputs --seq_name $N
/root/gpurun.sh ${N}_optim python optimization.py --save_path ../results/animation --iter 200 --input_path inputs --img_size 960 \
   --seq_name $N --save_name ${N}_demo --gauss_filter "$@"
R=$P/results/animation/$N/${N}_demo
RAW=$R/smoothed; [ -d $RAW ] || RAW=$R/raw
python $P/pt2npz.py $RAW $W/objs/mesh.obj $W/anim.npz
/root/venv-bpy/bin/python $P/export_glb.py --src $GLB --rig $W/objs/rig.txt --anim $W/anim.npz --fps 10 --out $W/${N}_animated.glb
