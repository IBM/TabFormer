from os import makedirs
from os.path import join
import logging
import numpy as np
import torch
import random
from args import define_main_parser

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from dataset.prsa import PRSADataset
from dataset.card import TransactionDataset
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import TransDataCollatorForLanguageModeling


logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def main(args):
    # random seeds
    seed = args.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    if args.data_type == 'card':
        dataset = TransactionDataset(root=args.data_root,
                                     fname=args.data_fname,
                                     fextension=args.data_extension,
                                     vocab_dir=args.output_dir,
                                     nrows=args.nrows,
                                     user_ids=args.user_ids,
                                     mlm=args.mlm,
                                     cached=args.cached,
                                     stride=args.stride,
                                     flatten=args.flatten,
                                     return_labels=False,
                                     skip_user=args.skip_user)
    elif args.data_type == 'prsa':
        dataset = PRSADataset(stride=args.stride,
                              mlm=args.mlm,
                              return_labels=False,
                              use_station=False,
                              flatten=args.flatten,
                              vocab_dir=args.output_dir)

    else:
        raise Exception(f"data type '{args.data_type}' not defined")

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

    if args.lm_type == "bert":
        tab_net = TabFormerBertLM(custom_special_tokens,
                               vocab=vocab,
                               field_ce=args.field_ce,
                               flatten=args.flatten,
                               ncols=dataset.ncols,
                               field_hidden_size=args.field_hs
                               )
    else:
        tab_net = TabFormerGPT2(custom_special_tokens,
                             vocab=vocab,
                             field_ce=args.field_ce,
                             flatten=args.flatten,
                             )

    log.info(f"model initiated: {tab_net.model.__class__}")

    if args.flatten:
        collactor_cls = "DataCollatorForLanguageModeling"
    else:
        collactor_cls = "TransDataCollatorForLanguageModeling"

    log.info(f"collactor class: {collactor_cls}")
    data_collator = eval(collactor_cls)(
        tokenizer=tab_net.tokenizer, mlm=args.mlm, mlm_probability=args.mlm_prob
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        logging_dir=args.log_dir,  # directory for storing logs
        save_steps=args.save_steps,
        do_train=args.do_train,
        # do_eval=args.do_eval,
        # evaluation_strategy="epoch",
        prediction_loss_only=True,
        overwrite_output_dir=True,
        # eval_steps=10000
    )

    trainer = Trainer(
        model=tab_net.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if args.checkpoint:
        model_path = join(args.output_dir, f'checkpoint-{args.checkpoint}')
    else:
        model_path = args.output_dir

    trainer.train(model_path=model_path)


if __name__ == "__main__":

    parser = define_main_parser()
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    if opts.mlm and opts.lm_type == "gpt2":
        raise Exception("Error: GPT2 doesn't need '--mlm' option. Please re-run with this flag removed.")

    if not opts.mlm and opts.lm_type == "bert":
        raise Exception("Error: Bert needs '--mlm' option. Please re-run with this flag included.")

    main(opts)
