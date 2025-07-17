device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir= trained_on_mvtecad
        save_dir=/home/zhujiawen/AnomalyCLIP/checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset SDD \
        --data_path /home/zhujiawen/PDA/data/SDD_anomaly_detection --save_path ./results/${base_dir}/zero_shot \
        --checkpoint_path /home/zhujiawen/AnomalyCLIP/checkpoints/visa_full3_bias/epoch_7.pth \
         --features_list 6 12 18 24 --image_size 518 --seed 111 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --metrics image-pixel-level
    wait
    done
done


# BrainMRI_anomaly_detection brainmri2
# HeadCT headct2
# SDD SDD
# AITEX AITEX
# elpv elpv
# Brain_MRI Brain_MRI  LAG LAG
# mvtecad_anomaly_detection/mvtecad mvtecad
# visa_anomaly_detection/visa visa
# DAGM_anomaly_detection DAGM
# br35_anomaly_detection br35
# ClinicDB_anomaly_detection, ColonDB, EndoTect, Kvasir colon
# covid19_anomaly_detection chest
# ISBI_anomaly_detection ISBI
# tn3k thyroid
