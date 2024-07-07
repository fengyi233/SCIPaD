# ====================== metrics in paper ======================
# KITTI Raw
python evaluate_depth.py --config configs/posenet.yaml \
  load_weights_folder checkpoints/KITTI \
  eval.batch_size 1
#0.0902	0.6504	4.0560	0.1656	0.9184	0.9695	0.9845


python evaluate_depth.py --config configs/posenet.yaml \
  load_weights_folder checkpoints/KITTI \
  eval.batch_size 1 \
  eval.split eigen_benchmark
#0.0585	0.2868	2.8712	0.0946	0.9643	0.9926	0.9976

# KITTI Odom
python evaluate_pose.py --config configs/odom.yaml \
  load_weights_folder checkpoints/KITTI_Odom/ \
  eval.split odom_09
python evaluate_pose.py --config configs/odom.yaml \
  load_weights_folder checkpoints/KITTI_Odom/ \
  eval.split odom_10
python ./utils/kitti_odom_eval/eval_odom.py --result=checkpoints/KITTI_Odom/ --align='7dof'
#Sequence: 	 9
#Trans. err. (%): 	 7.438
#Rot. err. (deg/100m): 	 2.463
#ATE (m): 	 26.198
#RPE (m): 	 0.061
#RPE (deg): 	 0.083
#
#Sequence: 	 10
#Trans. err. (%): 	 9.824
#Rot. err. (deg/100m): 	 3.868
#ATE (m): 	 15.509
#RPE (m): 	 0.065
#RPE (deg): 	 0.100




