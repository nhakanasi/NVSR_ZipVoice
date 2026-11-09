param(
    [int]$NumIters = 1000,
    [string]$ExpDir = 'C:\zipvoice_data\exp\adapter_multistep_k3_20260918',
    [string]$ResumeFrom = ''
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
$trainArgs = @(
    '-m', 'zipvoice.bin.train_zipvoice_adapter',
    '--world-size', '1', '--use-fp16', '1',
    '--max-duration', '24', '--max-len', '12', '--num-workers', '0', '--scan-oom', '0',
    '--model-config', 'C:\zipvoice_data\exp\adapter\model.json',
    '--checkpoint', 'C:\zipvoice_data\exp\adapter\iter-30000-avg-2.pt',
    '--tokenizer', 'emilia', '--token-file', 'C:\zipvoice_data\exp\adapter\tokens.txt',
    '--dataset', 'libritts', '--manifest-dir', 'C:\zipvoice_data\data\fbank',
    '--finetune', '1', '--base-lr', '0.0001', '--num-iters', "$NumIters",
    '--save-every-n', '100', '--valid-interval', '500',
    '--tensorboard', '0', '--wandb', '0', '--adapter-trainable', 'adapter',
    '--recon-loss-weight', '1', '--curriculum-steps', '0',
    '--multistep-loss-weight', '100', '--multistep-steps', '3', '--multistep-max-utts', '1',
    '--exp-dir', $ExpDir
)
if ($ResumeFrom) { $trainArgs += @('--resume-from', $ResumeFrom) }
& python @trainArgs
exit $LASTEXITCODE
