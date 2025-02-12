set -e

source .env

OUT_PATH=$1
STOP_AFTER_SEG=$2
SKIP_PROBA_CONVERSION=$3

if [[ $OUT_PATH == "" ]]
then
    OUT_PATH=data/radiomic_preds
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
echo uv run scripts/arrange_dicom.py \
    --data_dir $DICOM_IN_DIR \
    --metadata_path $METADATA_PATH \
    --output_dir $DICOM_OUT_DIR

echo Converting DICOM to NIFTI
echo bash scripts/dicom-to-nifti.sh $DICOM_OUT_DIR $NIFTI_DIR
rm -f $NIFTI_DIR/*.json
rm -f $NIFTI_DIR/*.bval
rm -f $NIFTI_DIR/*bvec

echo Resampling to spacing of first image
echo uv run scripts/resample_to_first.py --input_dir $NIFTI_DIR --series $SEGMENTATION_SERIES

echo Linking relevant niftis for segmentation
bash scripts/link-for-segmentation.sh $NIFTI_DIR $TMP_DIR

echo Predicting with model in $SEGMENTATION_MODEL_DIR
if [[ $SKIP_PROBA == 1 ]]
then
	PROBA_FLAG=""
else
	PROBA_FLAG="--save_probabilities"
fi
export nnUNet_raw=$NIFTI_DIR
export nnUNet_preprocessed=$TMP_DIR
export nnUNet_results=$SEGMENTATION_MODEL_DIR
uv run nnUNetv2_predict_from_modelfolder \
    -i $TMP_DIR \
    -o $SEGMENTATION_PREDICTIONS_DIR \
    -m $SEGMENTATION_MODEL_DIR \
    --c $PROBA_FLAG

if [[ $SKIP_PROBA_CONVERSION == 1 ]]
then
    echo Skipping probability conversion
else
    echo Converting probability predictions to Nifti
    uv run python scripts/probability_to_nifti.py \
        --input_path $SEGMENTATION_PREDICTIONS_DIR
fi

if [[ $STOP_AFTER_SEG == 1 ]]
then
    exit 0
fi

echo Predicting
uv run python scripts/get_prediction.py \
    --image_folder $NIFTI_DIR \
    --mask_folder $SEGMENTATION_PREDICTIONS_DIR \
    --mask_pattern '[0-9]\.nii\.gz' \
    --t2_pattern _0000.nii.gz \
    --dwi_pattern _0001.nii.gz \
    --adc_pattern _0002.nii.gz \
    --id_pattern '[0-9\\.]+[0-9]' \
    --model_folder $RADIOMICS_MODEL_DIR \
    -o $OUT_PATH
