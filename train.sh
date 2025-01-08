datetime=$(date "+%Y%m%d-%H%M%S")

python -u main.py > ./train_log/${datetime}.log 2>&1 &