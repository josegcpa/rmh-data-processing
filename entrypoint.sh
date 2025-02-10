set -e

source .env

OUT_JSON=$1
STOP_AFTER_SEG=$2

if [[ $OUT_JSON == "" ]]
then
    OUT_JSON=out.json
fi

if [[ $SERIES == "" ]]
then
    echo "SERIES must be defined as an environment variable"
    exit 1
fi

if [[ $TMP_DIR == "" ]]
then
    TMP_DIR=.tmp
fi

if [[ $SEGMENTATION_SERIES == "" ]]
then
    SEGMENTATION_SERIES=$SERIES
fi

echo Arranging DICOM files
uv run scripts/arrange_dicom.py \
    --data_dir $DICOM_IN_DIR \
    --metadata_path $METADATA_PATH \
    --output_dir $DICOM_OUT_DIR

echo Converting DICOM to NIFTI
bash scripts/dicom-to-nifti.sh $DICOM_OUT_DIR $NIFTI_DIR
rm -f $NIFTI_DIR/*.json
rm -f $NIFTI_DIR/*.bval

echo Resampling to spacing of first image
uv run scripts/resample_to_first.py --input_dir $NIFTI_DIR --series $SEGMENTATION_SERIES

echo Linking relevant niftis for segmentation
bash scripts/link-for-segmentation.sh $NIFTI_DIR $TMP_DIR

echo Predicting with model in $SEGMENTATION_MODEL_DIR
export nnUNet_raw=$NIFTI_DIR
export nnUNet_preprocessed=$TMP_DIR
export nnUNet_results=$SEGMENTATION_MODEL_DIR
uv run nnUNetv2_predict_from_modelfolder \
    -i $TMP_DIR \
    -o $SEGMENTATION_PREDICTIONS_DIR \
    -m $SEGMENTATION_MODEL_DIR \
    --c \
    --save_probabilities

echo Converting probability predictions to Nifti
echo uv run python scripts/probability_to_nifti.py \
    --input_path $SEGMENTATION_PREDICTIONS_DIR

if [[ $STOP_AFTER_SEG == 1 ]]
then
    exit 0
fi

echo Predicting
uv run python scripts/get_prediction.py \
    --image_folder $NIFTI_DIR \
    --mask_folder $SEGMENTATION_PREDICTIONS_DIR \
    --mask_pattern '[0-9]\.nii\.gz' \
    --t2_pattern _0000 \
    --dwi_pattern _0001 \
    --adc_pattern _0002 \
    --id_pattern '[0-9\\.]+[0-9]' \
    --model_folder $RADIOMICS_MODEL_DIR \
    -o $OUT_JSON