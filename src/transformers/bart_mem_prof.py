from transformers import *
import torch
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def runner(source_path, out_file, batch_size=8, device=DEFAULT_DEVICE, prof_generate=False,
           max_length=1024):

    tokenizer = BartTokenizer.from_pretrained('bart-large')
    lns = [" " + x.rstrip() for x in open(source_path).readlines()][:batch_size]

    dct = tokenizer.batch_encode_plus(lns, max_length=max_length, return_tensors="pt", pad_to_max_length=True)
    ids = dct['input_ids'].to(DEFAULT_DEVICE)
    msk = dct['attention_mask'].to(DEFAULT_DEVICE)
    model = BartForConditionalGeneration.from_pretrained('bart-large-cnn', output_past=prof_generate).to(DEFAULT_DEVICE)

    if prof_generate:
        summaries = model.generate(
            input_ids=ids,
            attention_mask=msk,
            num_beams=4,
            length_penalty=2.0,
            max_length=9,  # +2 from original because we start at step=1 and stop before max_length
            min_length=6,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,
            decoder_start_token_id=model.config.eos_token_ids[0],
        )
        model.log_mem('done')
        dec = [tokenizer.decode(s) for s in summaries]
        print(f'summary 0: {dec[0]}')
    else:
        #model.decoder.generation_mode = Fals
        with torch.no_grad():
            model(
                input_ids=ids,
                attention_mask=msk,
            )

    log_df = model.combine_logs()
    log_df.to_csv(out_file)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_path", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "--source_path", type=str, default="/home/shleifer/transformers_fork/notebooks/test.source",
        help="like cnn_dm/test.source", required=False
    )
    parser.add_argument(
        "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, required=False,
    )
    parser.add_argument(
        "--do-generate", action='store_true', required=False, help="batch size: how many to summarize at a time",
    )

    args = parser.parse_args()
    runner(args.source_path, args.output_path, batch_size=args.bs, device=args.device, prof_generate=args.do_generate)



