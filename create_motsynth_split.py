""" Generates motsynth split files """
import argparse
import random 
from pathlib import Path

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to MOTSynth sequences')
parser.add_argument('--test_file_path', type=str, required=True,
                            help='Path to the file with test sequences')
parser.add_argument('--output_path', type=str, required=True,
                    help='Output path for the split files')


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(42)

    all_seqs = [p.parts[-1] for p in Path(args.data_path).glob('*')]

    with open(str(Path(args.test_file_path)), 'r') as file:
        test_seqs = sorted(["{:03d}".format(int(line.strip())) for line in file.readlines()])

    train_val_seqs = [seq for seq in all_seqs if seq not in test_seqs]

    assert not bool(set(train_val_seqs) & set(test_seqs))
    
    train_num = int(len(train_val_seqs) * 0.8)
    print("Number of training seqs: {}, number of validation seqs: {}".format(train_num, len(train_val_seqs) - train_num))
    train_seqs = random.sample(train_val_seqs, train_num)
    val_seqs = [seq for seq in train_val_seqs if seq not in train_seqs]

    assert not bool(set(train_seqs) & set(val_seqs))

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_ids = [i for i in range(1800)]  # every sequence has 1800 frames 

    with open(str(output_path / 'motsynth_train.txt'), 'w') as file:
        for seq in train_seqs:
            ids = random.sample(frame_ids, 360)
            for idx in ids:
                print("{}/{}.jpg {}/{}.png".format(seq, idx, seq, idx), file=file)
    
    with open(str(output_path / 'motsynth_test.txt'), 'w') as file:
        for seq in test_seqs:
            ids = random.sample(frame_ids, 360)
            for idx in ids:
                print("{}/{}.jpg {}/{}.png".format(seq, idx, seq, idx), file=file)
