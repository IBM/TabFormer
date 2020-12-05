import argparse


def define_main_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--jid", type=int,
                        default=1,
                        help="job id: 1[default] used for job queue")
    parser.add_argument("--seed", type=int,
                        default=9,
                        help="seed to use: 9[default]")

    parser.add_argument("--lm_type", default='bert', choices=['bert', 'gpt2'],
                        help="gpt or bert choice.")
    parser.add_argument("--flatten", action='store_true',
                        help="enable flattened input, no hierarchical")
    parser.add_argument("--field_ce", action='store_true',
                        help="enable field wise CE")
    parser.add_argument("--mlm", action='store_true',
                        help="masked lm loss; pass it for BERT")
    parser.add_argument("--mlm_prob", type=float,
                        default=0.15,
                        help="mask mlm_probability")

    parser.add_argument("--data_type", type=str,
                        default="card", choices=['card', 'prsa'],
                        help='root directory for files')
    parser.add_argument("--data_root", type=str,
                        default="./data/credit_card/",
                        help='root directory for files')
    parser.add_argument("--data_fname", type=str,
                        default="card_transaction.v1",
                        help='file name of transaction')
    parser.add_argument("--data_extension", type=str,
                        default="",
                        help="file name extension to add to cache")
    parser.add_argument("--vocab_file", type=str,
                        default='vocab.nb',
                        help="cached vocab file")
    parser.add_argument('--user_ids', nargs='+',
                        default=None,
                        help='pass list of user ids to filter data by')
    parser.add_argument("--cached", action='store_true',
                        help='use cached data files')
    parser.add_argument("--nrows", type=int,
                        default=None,
                        help="no of transactions to use")

    parser.add_argument("--output_dir", type=str,
                        default='checkpoints',
                        help="path to model dump")
    parser.add_argument("--checkpoint", type=int,
                        default=0,
                        help='set to continue training from checkpoint')
    parser.add_argument("--do_train", action='store_true',
                        help="enable training flag")
    parser.add_argument("--do_eval", action='store_true',
                        help="enable evaluation flag")
    parser.add_argument("--save_steps", type=int,
                        default=500,
                        help="set checkpointing")
    parser.add_argument("--num_train_epochs", type=int,
                        default=3,
                        help="number of training epochs")
    parser.add_argument("--stride", type=int,
                        default=5,
                        help="stride for transaction sliding window")

    parser.add_argument("--field_hs", type=int,
                        default=768,
                        help="hidden size for transaction transformer")
    parser.add_argument("--skip_user", action='store_true',
                        help="if user field to be skipped or added (default add)")

    return parser
