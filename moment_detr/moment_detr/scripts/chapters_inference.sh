dset_name=chapters
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip
results_root=TOFILL
exp_id=chapters_inference

######## data paths
train_path=TOFILL/chapters_vmr_train.jsonl
eval_path=TOFILL/chapters_vmr_test.jsonl
eval_split_name=test

######## setup video+text features
feat_root=TOFILL
feat_root_eval=TOFILL
subtitles_path=TOFILL

# video features
v_feat_dim=0
v_feat_dirs=()
v_feat_dirs_eval=()
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root})
  v_feat_dirs_eval+=(${feat_root_eval})
  (( v_feat_dim += 768 ))
fi

#### training
bsz=256
num_workers=3
n_epoch=50
max_es_cnt=50

PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dirs_eval ${v_feat_dirs_eval[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dim ${v_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--num_workers ${num_workers} \
--exp_id ${exp_id} \
--n_epoch ${n_epoch} \
--max_es_cnt ${max_es_cnt} \
--eval_path ${eval_path} \
--subtitles_path ${subtitles_path} \
${@:1}
