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

            # remove newline symbols as they are present in "context" (why?)
            context_text = item["question"] + " \\n " + item["context"].replace('\r', '').replace('\n', '') + '\n'
            # convert to lowercase targets and context texts as uppercase words (at least some of them) ...
            # ... are not present in vocabulary (for instance "France" is absent, but there is "france" instead)
            context_file.write(context_text.lower())

            # *** write target text (= correct answer) ***
            answers = item["answers"]["text"]
            if answers:
                # add "<answer>" at the beginning of answer to guarantee it contains at least one known word
                target_text = "<answer> " + answers[0] + '\n'
            else:
                # replace empty strings with "<no answer>" to avoid exception in t5 model code
                target_text = "<no answer>" + '\n'
            target_file.write(target_text.lower())
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
