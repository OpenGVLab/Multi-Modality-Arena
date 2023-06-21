
import json
import argparse

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from vqa_tools.vqa_eval import VQAEval


def load_tokenizer():
    # Sets up the BERT tokenizer using tf-text.

    VOCAB_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt'  #@param {type:"string"}

    vocab_table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=VOCAB_PATH,
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER
            ), 
            num_oov_buckets=1)
    cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
    tokenizer = text.BertTokenizer(vocab_lookup_table=vocab_table, 
                                token_out_type=tf.int64, 
                                preserve_unused_token=True, 
                                lower_case=True)
    return tokenizer, cls_id, sep_id

def preprocess_text_vqav2(prompt: str) -> str:
    '''Preprocess text/prompt in the way the same as VQAv2Eval beofore evaluation.
    '''
    prompt = prompt.replace('\n', ' ')
    prompt = prompt.replace('\t', ' ')
    prompt = prompt.strip()
    prompt = VQAEval.processPunctuation(prompt)
    prompt = VQAEval.processDigitArticle(prompt)
    return prompt

#@title Helper functions for converting examples to BERT inputs.

def bertify_example(example, tokenizer, cls_id, sep_id):
    question = tokenizer.tokenize(example['question']).merge_dims(1, 2)
    reference = tokenizer.tokenize(example['reference']).merge_dims(1, 2)
    candidate = tokenizer.tokenize(example['candidate']).merge_dims(1, 2)

    input_ids, segment_ids = text.combine_segments(
        (candidate, reference, question), cls_id, sep_id)

    return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}

def pad(a, length=512):
    return np.append(a, np.zeros(length - a.shape[-1], np.int32))

def bertify_examples(examples, tokenizer, cls_id, sep_id):
    input_ids = []
    segment_ids = []
    for example in examples:
        example_inputs = bertify_example(example, tokenizer, cls_id, sep_id)
        input_ids.append(pad(example_inputs['input_ids']))
        segment_ids.append(pad(example_inputs['segment_ids']))

    return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}

def get_args_parser():
    parser = argparse.ArgumentParser('whoops vqa evaluation', add_help=False)
    # parser.add_argument('--infer_dir', type=str, help='vqa inference directory')
    parser.add_argument('--infer_json', type=str, help='json file of inference results')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size of bem inference')
    parser.add_argument(
        '--question_path', type=str,
        default='datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        help='question path')
    parser.add_argument(
        '--annotation_path', type=str,
        default='datasets/vqav2/v2_mscoco_val2014_annotations.json',
        help='annotation path')

    return parser

def main(resFile, quesFile, annFile):

    preds = json.load(open(resFile, 'r'))
    annos = json.load(open(quesFile, 'r'))
    # annos_ids = [x['question_id'] for x in annos]
    # annos_dict = dict(zip(annos_ids, annos))
    counter = 0
    num_samples = len(preds)
    for x in preds:
    #    question_id = x['question_id']
    #    x['question'] = annos_dict[question_id]['question']
       x['reference'] = x['gt_answers']
       x['candidate'] = x['answer']
       if preprocess_text_vqav2(x['reference']) == preprocess_text_vqav2(x['candidate']):
           counter += 1
    print(f'>>> exact match score: {counter / num_samples * 100:.2f}%')

    # load tokenizer
    tokenizer, cls_id, sep_id = load_tokenizer()
    # Load BEM model.
    # bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
    bem = hub.load('checkpoints/bem_1')

    scores = []
    bsz = args.batch_size
    for i in tqdm(range(0, num_samples, bsz)):
        examples = preds[i:i+bsz]
        inputs = bertify_examples(examples, tokenizer, cls_id, sep_id)
        # The outputs are raw logits.
        raw_outputs = bem(inputs)
        # They can be transformed into a classification 'probability' like so:
        bem_score = list(softmax(raw_outputs, axis=1)[:, 1])
        scores.extend(bem_score)

    print(f'>>> mean BEM score: {np.mean(scores)*100:.2f}%')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    question_json_path = args.question_path
    annotation_json_path = args.annotation_path
    infer_json = args.infer_json
    out_file = infer_json
    main(out_file, question_json_path, annotation_json_path)
