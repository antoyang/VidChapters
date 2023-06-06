dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip 
results_root=results
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=qvhighlights_features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root})
  (( v_feat_dim += 768 ))
fi

#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python moment_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dim ${v_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
${@:1}
