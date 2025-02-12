IN_DIR=$1
OUT_DIR=$2

mkdir -p $OUT_DIR

declare -A SERIES_TO_NUM
SERIES_TO_NUM[T2]="0000"
SERIES_TO_NUM[HBV]="0001"
SERIES_TO_NUM[ADC]="0002"

for series_type in $SEGMENTATION_SERIES
do
    for file in $IN_DIR/*${SERIES_TO_NUM[$series_type]}.nii.gz
    do
        cp $file $OUT_DIR/$(basename $file)
    done
done
