import os
import csv
import argparse
import logging
import hr.document as doc
from hr.recommendation import QuestionBank

parser = argparse.ArgumentParser(description='Update question data.')
parser.add_argument('orig_file', help='original question file')
parser.add_argument('files', nargs='*', help='files to be appended')
parser.add_argument('--model_file', '-m', help='clustering model path')
args = parser.parse_args()


if __name__ == '__main__':
    ques_list = doc.load(args.orig_file)

    for file in args.files:
        ext = os.path.splitext(file)[1]
        if ext == '.json':
            ques_list.extend(doc.load(file))
        elif ext == '.csv':
            # ques_list.append(doc.Document(main=main, answer=answer))
            pass
        else:
            logging.warn('Unknown data type: %s', file)

    doc.dump(ques_list, args.orig_file)

    if args.model_file:
        bank = QuestionBank(ques_list, [], keep_n=1000)
        bank.save_model(args.model_file)
