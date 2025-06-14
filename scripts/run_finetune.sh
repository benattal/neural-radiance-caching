bash scripts/train_dgp.sh --scene $1 --stage material_light_resample_finetune \
    --take_stage material_light_from_scratch_resample --batch_size 1024 --grad_accum_steps 1 \
    --sample_factor 4 --sample_render_factor 4 --early_exit_steps 100000 \
    --sl_relight --eval_train

bash scripts/eval_dgp.sh --scene $1 --stage material_light_resample_finetune --sample_render_factor 1 --sl_relight --eval_path