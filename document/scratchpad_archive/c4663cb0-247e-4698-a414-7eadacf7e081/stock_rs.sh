# Stock ZipVoice resynthesis on CPU (GPU is shared), 8 kHz and 24 kHz prompts, both corpora.
export PYTHONPATH="$(pwd -W)" CUDA_VISIBLE_DEVICES="" PYTHONUTF8=1
for c in libritts:bwe_eval vctk:vctk_bwe_eval; do n=${c%%:*}; d=${c##*:}
  for rate in 8000 24000; do
    list=C:/zipvoice_data/data/$d/test_sr$rate.tsv
    out=C:/zipvoice_data/exp/results_rs/$n/stock_sr$rate
    [ -f $list ] || { echo "missing $list"; continue; }
    echo "== $n $rate $(date +%T)"
    python -u -m zipvoice.bin.infer_zipvoice --model-name zipvoice --tokenizer emilia \
      --test-list $list --res-dir $out --num-step 16 --seed 666 || echo "FAILED $n $rate"
    echo "done $n $rate $(date +%T) $(ls $out/*.wav 2>/dev/null | wc -l) wavs"
  done
done
