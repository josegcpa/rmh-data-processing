set -e

source .env

if [[ $SERIES == "" ]]
then
    echo "SERIES must be defined as an environment variable"
    exit 1
fi

echo Arranging DICOM files
uv run scripts/arrange_dicom.py \
    --data_dir $DICOM_IN_DIR \
    --metadata_path $METADATA_PATH \
    --output_dir $DICOM_OUT_DIR

echo Converting DICOM to NIFTI
bash dicom-to-nifti.sh $DICOM_OUT_DIR $NIFTI_DIR
rm -f $NIFTI_DIR/*.json
rm -f $NIFTI_DIR/*.bval

echo Resampling to spacing of first image
uv run scripts/resample_to_first.py --input_dir $NIFTI_DIR

echo Predicting with nnUNetv2 and model in $MODEL_DIR
uv run nnUNetv2_predict_from_modelfolder \
    -i $NIFTI_DIR \
    -o $PREDICTIONS_DIR \
    -m $MODEL_DIR \
    --c \
    --save_probabilities

echo Converting probability predictions to Nifti
uv run python scripts/probability_to_nifti.py \
    --input_path $PREDICTIONS_DIR