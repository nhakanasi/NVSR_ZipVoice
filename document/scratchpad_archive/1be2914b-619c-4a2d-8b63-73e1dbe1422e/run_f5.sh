#!/bin/bash
set -e
SP='C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/1be2914b-619c-4a2d-8b63-73e1dbe1422e/scratchpad'
cd 'G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice'
export PYTHONPATH="$PWD:$PWD/third_party/F5-TTS/src"
for corpus in libritts vctk; do
  if [ $corpus = libritts ]; then eval_dir=/c/zipvoice_data/data/bwe_eval; else eval_dir=/c/zipvoice_data/data/vctk_bwe_eval; fi
  for rate in 24000 8000; do
    python "$SP/f5_stock_infer.py" $eval_dir/test_sr$rate.tsv C:/zipvoice_data/exp/results_rs/$corpus/f5_sr$rate 666
  done
done
echo ALLDONE
