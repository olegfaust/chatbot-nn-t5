import nlp

import tqdm

from argparse import ArgumentParser


def export(out_data_dir: str, split_type: str, src_data_dir=None):
    """Export dataset converted to seq2seq format
    """

    context_filename = out_data_dir + "/squad2/" + split_type + ".source"
    target_filename = out_data_dir + "/squad2/" + split_type + ".target"

    # Load squad v2 dataset
    squad_dataset = nlp.load_dataset("squad_v2", data_dir=src_data_dir, split=split_type)
    progress = tqdm.tqdm(unit="docs", total=squad_dataset.num_rows)

    with open(context_filename, "w", encoding="utf-8") as context_file, \
            open(target_filename, "w", encoding="utf-8") as target_file:
        for item in squad_dataset:
            # *** write concatenated source text (= question + context) ***

            # remove newline symbols as they are present in "context" text (why?)
            context_text = item["question"] + " \\n " + item["context"].replace('\r', '').replace('\n', '')
            context_text = context_text.lower() + ' [EOS]\n'

            # convert to lowercase targets and context texts as some uppercase words are not present...
            # ... in vocabulary (for instance "France" is absent, but, surprisingly, there is "france" instead)
            # Remark: seems like T5Tokenizer from transformers library is broken now ...
            # ... (it also incorrectly works with eos symbol and there are many other weird things)
            context_file.write(context_text)

            # *** write target text (= correct answer) ***
            answers = item["answers"]["text"]
            # Important!!! addition of "end of sequence" symbol is needed to avoid
            # an exception due to empty sequence (which is thrown otherwise inside t5 model code)
            # custom '[EOS]' symbol is used instead of '</s>' as '</s>' is incorrectly mapped by T5Tokenizer
            # see https://github.com/huggingface/transformers/issues/5142 for more info on the problem and workaround
            if answers:
                target_text = answers[0].lower() + '. [EOS]\n'
            else:
                target_text = '. [EOS]\n'
            target_file.write(target_text)

            progress.update()


def main(params):
    export(params.output_dir, "train", params.src_data_dir)
    export(params.output_dir, "validation", params.src_data_dir)


def add_general_args(arg_parser):
    arg_parser.add_argument(
        "--src_data_dir",
        default=None,
        type=str,
        required=False,
        help="source data directory (original dataset will be downloaded into this directory)",
    )
    arg_parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="output directory (where converted dataset will be saved)",
    )


if __name__ == "__main__":
    # init argument parser
    parser = ArgumentParser(add_help=False)
    # add general arguments
    add_general_args(parser)
    # get parsed params
    params = parser.parse_args()
    # run
    main(params)
