for i in $(seq 2 11);
do
    python3 real2sim.py --task_name hang_mug --data_index $i
done

#python3 real2sim.py --task_name hang_mug --data_index 1
#python3 video.py --task_name hang_mug --data_index 1
# easy: 
# lift_ball # huge error on EE
# stack blocks



# hard 
# close pen
# hand_over block
# ziploc # huge error on EE
# 
