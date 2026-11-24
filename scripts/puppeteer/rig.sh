#!/bin/bash
# Rigging con Puppeteer (esqueleto + skinning) y export a GLB riggeado.  Uso: rig.sh <nombre> <glb>
# Requiere venv-pup (puppeteer) y venv-bpy (export). Tiempos/VRAM en /root/stats/<nombre>_{skeleton,skinning,export_rig}.json
set -e
N=$1; GLB=$2
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
P=/root/puppeteer; W=/root/work/$N
mkdir -p $W/rig_in $W/skeletons $W/final_rigging
source /root/venv-pup/bin/activate
# 1) glb -> obj (+mtl +png), orden de vertices conservado
python -I $P/glb2obj.py $GLB $W/objs
sed -i 's/^Ks .*/Ks 0.0 0.0 0.0/; s/^Ns .*/Ns 10.0/; s/^K\([ad]\) .*/K\1 1.0 1.0 1.0/' $W/objs/material.mtl
cp $W/objs/mesh.obj $W/rig_in/$N.obj
# 2) esqueleto
cd $P/skeleton
/root/gpurun.sh ${N}_skeleton python demo.py --input_dir $W/rig_in \
  --pretrained_weights skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth \
  --output_dir $W/ --save_name skel_results --input_pc_num 8192 \
  --save_render --apply_marching_cubes --joint_token --seq_shuffle
cp $W/skel_results/${N}_pred.txt $W/skeletons/$N.txt
# 3) skinning
cd $P/skinning
/root/gpurun.sh ${N}_skinning torchrun --nproc_per_node=1 --master_port=10009 main.py \
  --num_workers 1 --batch_size 1 --generate --save_skin_npy \
  --pretrained_weights skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth \
  --input_skel_folder $W/skeletons --mesh_folder $W/rig_in \
  --post_filter --depth 1 --save_folder $W/skin_results
cp $W/skin_results/generate/${N}_skin.txt $W/final_rigging/$N.txt
cp $W/final_rigging/$N.txt $W/objs/rig.txt
# 4) GLB riggeado
time /root/venv-bpy/bin/python $P/export_glb.py --src $GLB --rig $W/objs/rig.txt --out $W/${N}_rigged.glb
