python ./tools/vos_inference.py \
    --sam2_cfg ./configs/sam2.1/sam2.1_hiera_t.yaml \
    --sam2_checkpoint ../../../checkpoints/checkpoint.pt \
    --base_video_dir ../../../inputs/bsc_imgs \
    --input_mask_dir ../../../inputs/masks \
    --input_mv_dir ../../../inputs/mv_maps \
    --input_pm_dir ../../../inputs/frame_types \
    --video_list_file ../../../inputs/test_list.txt \
    --output_mask_dir ../../../inputs/results \
    --track_object_appearing_later_in_video

# python outputs/seg_eval.py