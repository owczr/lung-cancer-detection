python -m scripts.azure.machine_learning.run_job \
  --model mobilenet \
  --optimizer adam \
  --loss binary_crossentropy \
  --epochs 2 \
  --batch_size 128 \
  --distributed
