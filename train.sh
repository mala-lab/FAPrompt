device=2

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=trained_on_mvtecad
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtecad --train_data_path /home/zhujiawen/PDA/data/mvtecad_anomaly_detection/mvtecad \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done

#LOG=${save_dir}"res.log"
#echo ${LOG}
#depth=(9)
#n_ctx=(12)
#t_n_ctx=(4)
#for i in "${!depth[@]}";do
#    for j in "${!n_ctx[@]}";do
#    ## train on the VisA dataset
#        base_dir=trained_on_visa
#        save_dir=./checkpoints/${base_dir}/
#        CUDA_VISIBLE_DEVICES=${device} python train_cls.py --dataset visa --train_data_path /home/zhujiawen/PDA/data/visa_anomaly_detection/visa \
#        --save_path ${save_dir} \
#        --features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
#        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
#    wait
#    done
#done