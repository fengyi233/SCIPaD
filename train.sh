python train.py --config configs/tridepth_baseline.yaml

for i in $(seq 15 29); do
  python evaluate_depth.py --config configs/tridepth_baseline.yaml \
    load_weights_folder runs/tridepth_baseline/models/weights_$i
done
#eval teacher
for i in $(seq 15 29); do
  python evaluate_depth.py --config configs/tridepth_baseline.yaml \
    load_weights_folder runs/tridepth_baseline/models/weights_$i \
    eval.eval_teacher True
done

python train.py --config configs/vkitti2.yaml
python evaluate_depth.py --config configs/vkitti2.yaml \
  load_weights_folder runs/vkitti2/models/weights_10

python train.py --config configs/posenet.yaml \
  exp_name posenetv2 \
  posenet.version PoseNetV2

python train.py --config configs/posenet.yaml \
  exp_name posenetv3 \
  posenet.version PoseNetV3

python train.py --config configs/posenet.yaml \
  exp_name posenetv4 \
  posenet.version PoseNetV4

python train.py --config configs/posenet.yaml \
  exp_name posenetv5 \
  posenet.version PoseNetV5
for i in $(seq 13 15); do
  python evaluate_depth.py --config configs/posenet.yaml \
    load_weights_folder runs/posenetv5/models/weights_$i \
    posenet.version PoseNetV5 \
    eval.batch_size 1
done

python evaluate_depth.py --config configs/posenet.yaml \
  load_weights_folder runs/posenetv5/models/weights_15 \
  posenet.version PoseNetV5 \
  eval.split eigen_benchmark

# server
python evaluate_depth.py --config configs/posenet.yaml \
  load_weights_folder runs/server/posenetv5_70k/weights_15 \
  posenet.version PoseNetV5 \
  save_pred_disps True

# ============================ kitti odom ============================
python train.py --config configs/odom.yaml
# eval
for i in $(seq 16 17); do
  python evaluate_pose.py --config configs/odom.yaml \
    load_weights_folder runs/kitti_odom/models/weights_$i \
    eval.split odom_09
  python evaluate_pose.py --config configs/odom.yaml \
    load_weights_folder runs/kitti_odom/models/weights_$i \
    eval.split odom_10
  python ./utils/kitti_odom_eval/eval_odom.py --result=runs/kitti_odom/models/weights_$i --align='7dof'
done


python train.py --config configs/odom_posenetv5.yaml
for i in $(seq 5 18); do
  python evaluate_pose.py --config configs/odom_posenetv5.yaml \
    load_weights_folder runs/kitti_odom_posenetv5/models/weights_$i \
    eval.split odom_09
  python evaluate_pose.py --config configs/odom_posenetv5.yaml \
    load_weights_folder runs/kitti_odom_posenetv5/models/weights_$i \
    eval.split odom_10
  python ./utils/kitti_odom_eval/eval_odom.py --result=runs/kitti_odom_posenetv5/models/weights_$i --align='7dof'
done



python train.py --config configs/posenet.yaml \
  exp_name posenet
