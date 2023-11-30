import json
from datasketch import MinHash, MinHashLSH
import pandas as pd
import multiprocessing
import tqdm
import defopt
import re
import datasets
import utils,os
from datasets import concatenate_datasets, load_dataset
import random

def load_mmlu(mode='test'):
    if mode == 'train':
        mode = 'auxiliary_train'
    data = datasets.load_dataset("cais/mmlu", 'all')[mode]
    data = data.map(
        lambda content: {
            'instruction': content["question"],
            'output': content['choices'][content['answer']]
        }, 
        remove_columns=data.features,
        num_proc=os.cpu_count(),
    )        
    return data

def load_truthfulqa(mode='test'):
    if mode == 'train':
        raise Exception("no train!")

    def process_choice(text):
        # mc1_targets 比 mc2_targets 少
        dict_data = text['mc1_targets']
        right_idx = dict_data['labels'].index(1)
        right_choice = dict_data['choices'][right_idx]
        return right_choice

    data = datasets.load_dataset("truthful_qa", 'multiple_choice')['validation']
    data = data.map(
        lambda content: {
            'instruction': content["question"] ,
            'output': process_choice(content)
        }, 
        remove_columns=data.features,
        num_proc=os.cpu_count(),
    )        
    return data

def load_bbq(mode='test'):
    raw_datasets = load_dataset('lighteval/bbq_helm', 'all', split=mode)
    raw_datasets = raw_datasets.map(lambda x:{
        'instruction': 'Passage:' + x['context'].replace('\n','') + '\nQuestion:' + x['question'],
        'output': x['choices'][x['gold_index']]
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    return raw_datasets

def load_cnn_dm(mode='test'):
    raw_datasets = load_dataset('cnn_dailymail', name='3.0.0')[mode]
    # 'article' 'highlights'
    raw_datasets = raw_datasets.map(
        lambda x: {
            'instruction': " ".join(x['article'].replace("\n", " ").split()[:512]) + '\nSummarize the above article in 3 sentences.',
            'output': x['highlights'].replace("\n", " "),
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    return raw_datasets

def load_gsm8k(mode='test'):
    raw_datasets = load_dataset('gsm8k', name='main')[mode]
    # 'article' 'highlights'
    def process(text):
        text = text.replace('\n', ' ')
        text = text.split('####')
        text, answer = text[0], text[1]
        text = utils.period(text)
        text += 'The answer is' + answer
        # text = re.sub(r'<<[^<>]+>>', '', text)
        return text

    raw_datasets = raw_datasets.map(
        lambda x: {
            'instruction': f"Q: {x['question']}",
            'output': process(x['answer']),
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    # raw_datasets.to_json('../gsm8k_formatted.json')
    return raw_datasets

def load_flan(mode='train'):
    # 有fewshot
    # dialog_data = utils.from_jsonl('/home/lzy/nips/data/platypus/flan/FLAN/dialog_submix_data.jsonl')
    cot_data = utils.from_json('flan/FLAN/cot_zs_submix_data.json')
    flan_data = utils.from_json('flan/FLAN/flan2021_zsnoopt_submix_data.json')
    niv2_data = utils.from_json('flan/FLAN/niv2_zs_submix_data.json')
    t0_data = utils.from_json('flan/FLAN/t0_zsnoopt_submix_data.json')
    import multiprocess, os
    
    def map_cot(data):
        inputs = data['inputs'].split('\nOptions\n')[0]
        targets = data['targets']
        return {
            'instruct': inputs,
            'output': targets,
        }

    def map_flan(data):
        return {
            'instruct': data['inputs'],
            'output': data['targets'],
        }

    def map_t0(data):
        return {
            'instruct': data['inputs'],
            'output': data['targets'].replace('\nAnswer:', '').replace('\nA:','').replace('\nOutput:', '').strip(),
        }

    result = []

    process_num = os.cpu_count()
    for data_to_map, map_fn in  zip(
        [cot_data,flan_data,niv2_data,t0_data],
        [map_cot,map_flan,map_t0,map_t0],
    ):
        with multiprocess.Pool() as pool:
            result += list(tqdm.tqdm(
                pool.imap(map_fn, data_to_map, chunksize=100), 
                total=len(data_to_map), 
                desc=f'MAP({process_num})',
            ))
    
    return {
        'instruction': [ans['instruct'] for ans in result],
        'output': [ans['output'] for ans in result]
    }

def load_bigbench(mode='train'):
    dataset_list = []
    for task in [
        'abstract_narrative_understanding_zero_shot', 'anachronisms_zero_shot', 'analogical_similarity_zero_shot', 'analytic_entailment_zero_shot', 'arithmetic_zero_shot', 'ascii_word_recognition_zero_shot', 'authorship_verification_zero_shot', 'auto_categorization_zero_shot', 'auto_debugging_zero_shot', 'bbq_lite_json_zero_shot', 'bridging_anaphora_resolution_barqa_zero_shot', 'causal_judgment_zero_shot', 'cause_and_effect_zero_shot', 'checkmate_in_one_zero_shot', 'chess_state_tracking_zero_shot', 'chinese_remainder_theorem_zero_shot', 'cifar10_classification_zero_shot', 'code_line_description_zero_shot', 'codenames_zero_shot', 'color_zero_shot', 'common_morpheme_zero_shot', 'conceptual_combinations_zero_shot', 'conlang_translation_zero_shot', 'contextual_parametric_knowledge_conflicts_zero_shot', 'crash_blossom_zero_shot', 'crass_ai_zero_shot', 'cryobiology_spanish_zero_shot', 'cryptonite_zero_shot', 'cs_algorithms_zero_shot', 'dark_humor_detection_zero_shot', 'date_understanding_zero_shot', 'disambiguation_qa_zero_shot', 'discourse_marker_prediction_zero_shot', 'disfl_qa_zero_shot', 'dyck_languages_zero_shot', 'elementary_math_qa_zero_shot', 'emoji_movie_zero_shot', 'emojis_emotion_prediction_zero_shot', 'empirical_judgments_zero_shot', 'english_proverbs_zero_shot', 'english_russian_proverbs_zero_shot', 'entailed_polarity_hindi_zero_shot', 'entailed_polarity_zero_shot', 'epistemic_reasoning_zero_shot', 'evaluating_information_essentiality_zero_shot', 'fact_checker_zero_shot', 'fantasy_reasoning_zero_shot', 'few_shot_nlg_zero_shot', 'figure_of_speech_detection_zero_shot', 'formal_fallacies_syllogisms_negation_zero_shot', 'gem_zero_shot', 'gender_inclusive_sentences_german_zero_shot', 'general_knowledge_zero_shot', 'geometric_shapes_zero_shot', 'goal_step_wikihow_zero_shot', 'gre_reading_comprehension_zero_shot', 'hhh_alignment_zero_shot', 'hindi_question_answering_zero_shot', 'hindu_knowledge_zero_shot', 'hinglish_toxicity_zero_shot', 'human_organs_senses_zero_shot', 'hyperbaton_zero_shot', 'identify_math_theorems_zero_shot', 'identify_odd_metaphor_zero_shot', 'implicatures_zero_shot', 'implicit_relations_zero_shot', 'intent_recognition_zero_shot', 'international_phonetic_alphabet_nli_zero_shot', 'international_phonetic_alphabet_transliterate_zero_shot', 'intersect_geometry_zero_shot', 'irony_identification_zero_shot', 'kanji_ascii_zero_shot', 'kannada_zero_shot', 'key_value_maps_zero_shot', 'known_unknowns_zero_shot', 'language_games_zero_shot', 'language_identification_zero_shot', 'linguistic_mappings_zero_shot', 'linguistics_puzzles_zero_shot', 'list_functions_zero_shot', 'logic_grid_puzzle_zero_shot', 'logical_args_zero_shot', 'logical_deduction_zero_shot', 'logical_fallacy_detection_zero_shot', 'logical_sequence_zero_shot', 'mathematical_induction_zero_shot', 'matrixshapes_zero_shot', 'metaphor_boolean_zero_shot', 'metaphor_understanding_zero_shot', 'minute_mysteries_qa_zero_shot', 'misconceptions_russian_zero_shot', 'misconceptions_zero_shot', 'mnist_ascii_zero_shot', 'modified_arithmetic_zero_shot', 'moral_permissibility_zero_shot', 'movie_dialog_same_or_different_zero_shot', 'movie_recommendation_zero_shot', 'mult_data_wrangling_zero_shot', 'multiemo_zero_shot', 'natural_instructions_zero_shot'
    ]:
        raw_datasets = load_dataset('hails/bigbench', task, split=mode)
        # ['idx', 'inputs', 'targets', 'multiple_choice_targets', 'multiple_choice_scores']
        dataset_list.append( 
            raw_datasets.map(lambda x:{
                'instruction': x['inputs'],
                'output': random.choice(x['targets']),
            },
            remove_columns=raw_datasets.features,
            num_proc=os.cpu_count(),
            )
        )
    datasets = concatenate_datasets(dataset_list)
    print(datasets)
    return datasets
def load_dolly(mode='train'):
    dolly = load_dataset('databricks/databricks-dolly-15k',split='train')
    df = pd.DataFrame(dolly)
    df['output'] = df['response']
    data = df[['instruction','context', 'output']].to_dict('records')
    data_reordered = []
    for record in data:
        # TODO: filter context
        if len(record['context']):
            continue
        data_reordered.append({
            'instruction': record['instruction'], 
            'output': record['output']
        })
    dolly = datasets.Dataset.from_list(data_reordered)
    return dolly

def load_guanaco(mode='train'):
    guanaco = load_dataset('timdettmers/openassistant-guanaco',split='train')
    guanaco_test = load_dataset('timdettmers/openassistant-guanaco',split='test')
    df = pd.DataFrame(guanaco)
    df_test = pd.DataFrame(guanaco_test)
    df = pd.concat([df, df_test])
    df.head()

    def split_text(text):
        split_marker = "### Assistant:"
        instruction, output = text.split(split_marker, 1)
        instruction = instruction.replace("### Human:", "").strip()
        output = output.replace("### Human:", "### Instruction:\n").replace("### Assistant:", "### Response:\n").strip()
        return pd.Series([instruction, output])

    # Apply the function to the filtered dataframe
    df[['instruction', 'output']] = df['text'].apply(split_text)

    guanaco = datasets.Dataset.from_pandas(df[['instruction', 'output']]).remove_columns(['__index_level_0__'])
    return guanaco
def load_stackexchange(mode='train'):
    # 450w
    datasets = utils.from_jsonl('/home/lzy/nips/data/stackexchange.jsonl')
    # import multiprocess, os
    # def map_fn(x):
    #     return {
    #         'instruction': x['title_body'],
    #         'output': x['upvoted_answer'],
    #     }

    # process_num = os.cpu_count()
    # with multiprocess.Pool() as pool:
    #     ans = list(tqdm.tqdm(pool.imap(map_fn, datasets, chunksize=100), total=len(datasets), desc=f'MAP({process_num})'))

    # import pdb; pdb.set_trace()
    # utils.to_jsonl(ans, '/home/lzy/nips/data/stackexchange.jsonl')
    return {
        'instruction': [ans['instruction'] for ans in datasets],
        'output': [ans['output'] for ans in datasets]
    }


def load_wikihow(mode='train'):
    # 20w
    result = utils.from_jsonl('/home/lzy/nips/data/wikihow.json')
    return {
        'instruction': [ans['instruction'] for ans in result],
        'output': [ans['output'] for ans in result]
    }
def load_bestmmlu(mode='train'):
    #mmlu: 70.88
    result = utils.from_jsonl('/home/lzy/nips/data/platypus/match/filter_out/load_truthfulqa/load_bbq/filter-load_bbq-load_truthfulqa-threshold_1.0_7.504231720157954_30163.json')
    return {
        'instruction': [ans['instruction'] for ans in result],
        'output': [ans['output'] for ans in result]
    }
def load_besttruthfulqa(mode='train'):
    #truthfulqa: 72
    result = utils.from_jsonl('/home/lzy/nips/data/platypus/match/filter_out/load_bbq/load_bigbench/filter-load_bigbench-load_bbq-threshold_3.0_61.914473413460584_6589.json')
    return {
        'instruction': [ans['instruction'] for ans in result],
        'output': [ans['output'] for ans in result]
    }
def load_bestbbq(mode='train'):
    #bbq: 88
    result = utils.from_jsonl('/home/lzy/nips/data/platypus/match/filter_out/load_mmlu/load_bigbench/filter-load_bigbench-load_mmlu-threshold_4.0_98.87415648960648_25607.json')
    return {
        'instruction': [ans['instruction'] for ans in result],
        'output': [ans['output'] for ans in result]
    }
def load_bestcnn(mode='train'):
    #cnn: 21.8 
    result = utils.from_jsonl('/home/lzy/nips/data/platypus/match/filter_out/load_cnn_dm/load_cnn_dm/filter-load_cnn_dm-load_cnn_dm-threshold_8.0_494.33520505538934_1255.json')
    return {
        'instruction': [ans['instruction'] for ans in result],
        'output': [ans['output'] for ans in result]
    }
def load_xsum(mode='test'):
    # 20w
    raw_datasets = load_dataset('EdinburghNLP/xsum', name='3.0.0')[mode]
 
    def _xsum_filter( x):
        article = x['document']
        summary = x['summary']
        art_len = len(article.split())
        summ_len = len(summary.split())

        if art_len==0 or summ_len==0:
            return False

        if "Media playback is unsupported on your device" in article:
            return False

        if "Last updated at" in article:
            return False

        if summ_len <= 10:
            return False

        if summ_len / art_len > 0.2:
            return False

        return True

    # 'article' 'highlights'
    raw_datasets = raw_datasets.filter(
        lambda x: _xsum_filter(x)
    )
    raw_datasets = raw_datasets.map(
        lambda x: {
            'instruction': ' '.join(x['document'].replace("\n", " ").split()[:512]) + "\nSummarize the above article in 1 sentence.",
            'output': x['summary'].replace("\n", " "),
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    # raw_datasets.select([
    #     random.randint(0, len(raw_datasets)-1) for _ in range(50000)
    # ]).to_json('../xsum1.json')
    # import pdb; pdb.set_trace()
    return raw_datasets 
def load_narrativeqa(mode='train'):
    raw_datasets = load_dataset('lighteval/narrative_qa_helm',split=mode)
    raw_datasets = raw_datasets.map(lambda x:{
        'instruction': 'Passage: '+x['passage'].replace('\n','') + '\nQuestion: '+x['question'],
        'output': random.choice(x['references'])
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    #raw_datasets.to_json('../narrativeqa_formatted.json')
    return raw_datasets

def load_synthetic_reasoning(mode='train'):
    raw_datasets = concatenate_datasets([
        load_dataset('lighteval/synthetic_reasoning_natural','easy',split=mode),
        load_dataset('lighteval/synthetic_reasoning_natural','hard',split=mode)
    ])

    raw_datasets = raw_datasets.map(lambda x:{
        'instruction': x['question'],
        'output': x['target']
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    #raw_datasets.to_json('../synthetic_reasoning_formatted.json')
    return raw_datasets

def load_babi(mode='train'):
    raw_datasets = load_dataset('Muennighoff/babi',split=mode)

    raw_datasets = raw_datasets.map(lambda x:{
        'instruction': 'Passage: ' + x['passage'] + '\nQuestion: ' + x['question'] ,
        'output': x['answer']
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    #raw_datasets.to_json('../babi_formatted.json')
    return raw_datasets
def load_MATH(mode='train'):
    datasets = utils.from_jsonl(f'/home/lzy/nips/data/platypus/MATH_formatted.{mode}.jsonl')
    return {
        'instruction': [ans['instruction'] for ans in datasets],
        'output': [ans['output'] for ans in datasets]
    }
def load_coarse_truthfulqa(mode='train'):
    data = []
    with open("/home/lzy/nips/data/platypus/coarse_data_truthfulqa.txt", "r") as f:
        text_now = ""
        for i in f:
            text = i.strip()
            if text == "\n":
                data.append(text_now)
                text_now = ""
            else:
                text_now += text
        if text_now != "":
            data.append(text_now)
    return {
        'instruction': [i for i in data],
        'output': ["" for ans in datasets]
    }
    
def main(
    *,
    candidate: str = 'load_flan',
    targets: list[str] = ['load_mmlu','load_truthfulqa', 'load_bbq', 'load_gsm8k', 'load_cnn_dm'],
    out_file: str = 'out.json',
):
    #init    
    candidate_datasets = eval(candidate)('train')
    candidate_dir = os.path.join("/home/lzy/nips/data/platypus/match/loads", candidate)
    if not os.path.exists(candidate_dir):
        os.makedirs(candidate_dir)
    out_jsonl_path = f"{candidate_dir}/passage_output.jsonl"
    f_out =  open(out_jsonl_path, 'w')
    for idx, (i0, i1) in enumerate(zip(candidate_datasets["instruction"], candidate_datasets["output"])):
        data_item = {
            "id": idx,
            "contents": i0 + " Answer: " + i1,
            "instruction": i0,
            "output": i1
        }
        f_out.write(json.dumps(data_item) + "\n")


if __name__ == '__main__':

    try:
        import defopt
        defopt.run(main)
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
    
