datetime=$(date "+%Y%m%d-%H%M%S")

python -u evaluation.py > ./evaluation_log/${datetime}.log 2>&1 &