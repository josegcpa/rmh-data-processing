source .env

IN_DIR=$1
OUT_DIR=$2

mkdir -p $OUT_DIR

declare -A SERIES_TO_NUM
SERIES_TO_NUM[T2]="0000"
SERIES_TO_NUM[HBV]="0001"
SERIES_TO_NUM[ADC]="0002"

for series_type in $SERIES
do
    for file in $IN_DIR/*/$series_type
    do
        uv run dcm2niix -o $OUT_DIR -z o -9 -w 1 -f %k_${SERIES_TO_NUM[$series_type]} $file
    done
done