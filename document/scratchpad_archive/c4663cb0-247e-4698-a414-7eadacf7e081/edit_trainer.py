p = "zipvoice/bin/train_zipvoice_dual.py"
s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)

rep('''        than noise, and a
        discriminator trained to call noise fake learns the noise level rather
        than the band.""",
    )
''', '''        than noise, and a
        discriminator trained to call noise fake learns the noise level rather
        than the band. Ignored with --gan-source solve.""",
    )

    parser.add_argument(
        "--gan-source",
        type=str,
        default="endpoint",
        choices=["endpoint", "solve"],
        help="""What the discriminators judge as fake. "endpoint" is the
        branch's one-shot endpoint estimate at the training state, above
        --gan-t-min. "solve" is the evaluation-matched solve of the first
        --solve-utts utterances, i.e. the mel inference produces, with the
        target's low band put under it so the real and the fake differ only in
        the band the branch generates. The solve reuses the one --solve-loss-
        weight already pays for; keep that weight above zero so the band level
        stays anchored while the discriminators shape its texture.""",
    )
''')
