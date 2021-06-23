import argparse
import logging
import os
import pickle

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.datacollator import TransDataCollatorForLanguageModeling

from dataset.card import TransactionDataset
from models.modules import TabFormerGPT2

logger = logging.getLogger(__name__)


def gpt_eval(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    user_model = f'gpt2-userid-{args.user_ids[0]}'

    if args.num_bins:
        user_model += f'_nbins-{args.num_bins}'
    if args.hidden_size:
        user_model += f'_hsz-{args.hidden_size}'

    user_model_checkpoint = os.path.join(f'checkpoint-{args.checkpoint}')

    logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

    logger.info(f'***** Running GPT2 Evaluation for User ID {args.user_ids} from Checkpoint {args.checkpoint}*****')
    logger.info(f'  Saved weights loaded from: {os.path.join(args.output_dir, user_model_checkpoint)}')
    logger.info(f'  Batch size = {args.batch_size}')
    logger.info(f'  Stride = {args.stride}')
    logger.info(f'  Number of transactions used to seed decoder = {args.num_seed_trans}')

    if args.decoding == 'greedy':
        logger.info(f'  Decoding methodology = {args.decoding}')
    else:
        logger.info(f'  Decoding methodology = {args.decoding} with temperature of {args.temperature}')

    # Initialize and load components
    vocab_dir = os.path.join(args.output_dir, user_model)

    dataset = TransactionDataset(root=args.data_dir,
                                 fname=args.data_fname,
                                 fextension=args.data_extension,
                                 vocab_dir=args.output_dir,
                                 nrows=args.nrows,
                                 user_ids=args.user_ids,
                                 mlm=False,
                                 cached=True,
                                 stride=args.stride,
                                 flatten=True,
                                 return_labels=False)

    custom_special_tokens = dataset.vocab.get_special_tokens()
    tab_net = TabFormerGPT2(custom_special_tokens,
                         vocab=dataset.vocab,
                         field_ce=True,
                         flatten=True
                         )
    checkpoint = os.path.join(args.output_dir, user_model_checkpoint)

    generator = tab_net.model.from_pretrained(pretrained_model_name_or_path=checkpoint, vocab=dataset.vocab).to(device)
    data_collator = TransDataCollatorForLanguageModeling(tokenizer=tab_net.tokenizer, mlm=False)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator.collate_batch)

    # Create path to save directory
    path_to_save_dir = os.path.join(args.output_dir,
                                    user_model,
                                    f'checkpoint-{args.checkpoint}-eval')
    os.makedirs(path_to_save_dir, exist_ok=True)
    # File prefix of where to save output
    if args.decoding == 'greedy':
        fname = f'bsz-{args.batch_size}_stride-{args.stride}_seedtxn-{args.num_seed_trans}_decode-{args.decoding}-eval'
    else:
        fname = f'bsz-{args.batch_size}_stride-{args.stride}_seedtxn-{args.num_seed_trans}_decode-{args.decoding}_' + \
                f'temp-{args.temperature}-eval'

    # Generate transactions
    field_names = dataset.vocab.get_field_keys(input_only=True, ignore_special=True)
    if args.store_csv:
        with open(os.path.join(path_to_save_dir, fname+'.csv'), 'w') as csv_file:
            headers = ','.join(field_names) + '\n'
            csv_file.write(headers)
    num_fields = len(field_names)
    predicted_data = {f: [0] * len(dataset.vocab.get_field_ids(f)) for f in field_names}
    real_data = {f: [0] * len(dataset.vocab.get_field_ids(f)) for f in field_names}
    dl_iter = iter(data_loader)
    for _ in tqdm(range(len(data_loader)), desc="Batch"):
        inputs = next(dl_iter)
        seq_len = inputs['labels'].shape[1] - 2  # subtract two for [BOS] and [EOS] tokens
        # We will generate the next 10 minus num_seed_trans txns
        max_output = (seq_len//num_fields - args.num_seed_trans) * num_fields
        generator_input = inputs['labels'][:, 1:args.num_seed_trans*num_fields+1].to(device)
        for i in range(max_output):
            field_name = field_names[i % num_fields]
            true_local_id = dataset.vocab.get_from_global_ids(inputs['labels'][:, args.num_seed_trans*num_fields+1+i])
            global_ids_field = dataset.vocab.get_field_ids(field_name)
            generator_output = generator(generator_input)[0]
            lm_logits_field = generator_output[:, -1, global_ids_field]
            if args.decoding == 'greedy':
                next_field_local_id = torch.max(lm_logits_field, dim=1)[1]  # greedy max decoding of field level local id
            else:
                softmax_distribution = Categorical(logits=lm_logits_field / args.temperature)
                next_field_local_id = softmax_distribution.sample()  # softmax decoding of field level local id
            for b in range(true_local_id.shape[0]):
                real_data[field_name][true_local_id[b].item()] += 1
                predicted_data[field_name][next_field_local_id[b].item()] += 1
            token_id_to_add = dataset.vocab.get_from_local_ids(field_name, next_field_local_id)
            # Extend the context that GPT sees with the transaction we just predicted
            generator_input = torch.cat((generator_input, token_id_to_add.unsqueeze(1)), dim=1)
        if args.store_csv:
            generator_input_tokens = dataset.vocab.get_from_global_ids(generator_input, 'tokens')
            with open(os.path.join(path_to_save_dir, fname+'.csv'), 'a') as csv_file:
                batch_to_list = generator_input_tokens.tolist()
                for b in range(generator_input_tokens.shape[0]):
                    for t in range(seq_len//num_fields):  # write each txn as new line in csv
                        txn = (t*num_fields, (t+1)*num_fields)
                        txn_line = ','.join(batch_to_list[b][txn[0]:txn[1]]) + '\n'
                        csv_file.write(txn_line)

    # Save pkl file
    if args.store_pkl:
        with open(os.path.join(path_to_save_dir, fname+'.pkl'), 'wb') as results_file:
            pickle.dump({'predictions': predicted_data, 'ground_truth': real_data, 'field_names': field_names},
                        results_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Evaluation Arguments')
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoints',
                        help='Parent directory for checkpoint files.')
    parser.add_argument('--data_dir', type=str,
                        default='./data/card_transaction/data',
                        help='Directory where card transaction data is stored.')
    parser.add_argument("--data_fname", type=str,
                        default="card_transaction.v1",
                        help='file name of transaction')
    parser.add_argument('--user_ids', nargs='+',
                        default=None,
                        help='pass list of user ids to filter data by')
    parser.add_argument('--checkpoint', type=int, required=True,
                        help='Checkpoint from which to load saved model.')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins used in creating the vocabulary.')
    parser.add_argument('--hidden_size', type=int,
                        help='GPT2 embedding sized used during training.')
    parser.add_argument('--batch_size', type=int,
                        default=8,
                        help='Batch size for evaluation.')
    parser.add_argument('--stride', type=int,
                        default=3,
                        help='Stride to use for sliding window in dataloader.')
    parser.add_argument('--num_seed_trans', type=int, choices=range(1, 10),
                        default=1,
                        help='Number of transactions with which to seed the GPT2 decoder.')
    parser.add_argument('--decoding', type=str, choices=['greedy', 'softmax'],
                        default='greedy',
                        help='Method for decoding GPT2 generated tokens: greedy vs. softmax.')
    parser.add_argument('--temperature', type=float,
                        default=0.5,
                        help='Temperature to use in softmax decoding.')
    parser.add_argument('--store_pkl', action='store_true',
                        help='Set this flag to create pkl file of generated data for histogram comparisons.')
    parser.add_argument('--store_csv', action='store_true',
                        help='Set this flag to create csv file of generated data for visualizations.')
    parser.add_argument("--nrows", type=int,
                        default=None,
                        help="no of transactions to use")
    parser.add_argument("--data_extension", type=str,
                        default="",
                        help="file name extension to add to cache")

    opts = parser.parse_args()

    gpt_eval(opts)