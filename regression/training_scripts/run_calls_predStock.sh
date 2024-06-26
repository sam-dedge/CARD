export EXP_DIR=./results
export N_STEPS=100
export RUN_NAME=run_3
export LOSS=card_conditional
export TASK=options_pred
export N_SPLITS=2
export N_THREADS=4
export DEVICE_ID=1
export CAT_F_PHI=_cat_f_phi
export CONFIG_FILE_NAME=calls_predStock
export MODEL_VERSION_DIR=card_conditional_options_preds/${N_STEPS}steps/nn/${RUN_NAME}/f_phi_prior${CAT_F_PHI}
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${CONFIG_FILE_NAME}.yml #--train_guidance_only
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test