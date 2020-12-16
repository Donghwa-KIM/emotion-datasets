import os
import argparse
import yaml
import json 
import logging

from src.rawdataset import CmuMoseiDataset, Iemocap, Meld


def get_args(parser):
    if 'IPython' in __doc__ :
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
    
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../configs/cmu_mosei.yaml", help='configuration')
    parser.add_argument("--output_folder", type=str, default='../data', help='output folder')
    return parser

def parse_args(parser):
    args = get_args(parser)
    if args.config_file:
        with open(args.config_file) as f:
            y_dict = yaml.load(f, Loader=yaml.FullLoader)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in y_dict.items():
            arg_dict[key] = value
    return args



if __name__=='__main__':
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    args = parse_args(create_parser())
    
    if args.DATA_FILE =='CMU_MOSEI':
        
        data_path = os.path.join(args.DATA_ROOT, args.DATA_FILE)
        label_path = os.path.join(data_path, args.LABEL_FILE)

        audio_root = os.path.join(data_path, args.AUDIO_FOLDER)
        meta_root = os.path.join(data_path, args.META_FOLDER)
        comb_root = os.path.join(data_path, args.COMB_FOLDER)

        cmuMosei = CmuMoseiDataset(meta_root, comb_root, audio_root, label_path)

        logger.info("[1/7] Get data index ...")
        data_ix = cmuMosei.get_data_ix()

        logger.info("[2/7] Get labed dataframe ...")
        label_df = cmuMosei.get_label_df()

        logger.info("[3/7] Get split index ...")
        label_df['split'] = label_df['wav_id'].map(data_ix)

        logger.info("[4/7] Get Text ...")
        label_df = cmuMosei.add_sentence(label_df)

        logger.info("[5/7] Get Audio ...")
        dataset_ = cmuMosei.add_audio_wav(label_df, data_ix)

        logger.info("[6/7] Drop NA (no audio) ...")
        dataset = dataset_.dropna()

        logger.info("[7/7] Save ...")
        os.makedirs(args.output_folder, exist_ok=True)
        dataset.to_pickle(os.path.join(args.output_folder,'CMU_mosei_data.pkl'))
    
    elif args.DATA_FILE =='IEMOCAP_full_release':
        data_path = os.path.join(args.DATA_ROOT, args.DATA_FILE)
        label_root = [os.path.join(data_path, f, args.LABEL_FOLDER) for f in os.listdir(data_path) if 'Session' in f]
        audio_root = [os.path.join(data_path,f, args.AUDIO_FOLDER) for f in os.listdir(data_path) if 'Session' in f]
        text_root = [os.path.join(data_path,f, args.TEXT_FOLDER) for f in os.listdir(data_path) if 'Session' in f]
        
        iemocap = Iemocap(label_root, audio_root, text_root)

        logger.info("[1/7] Get labed dataframe ...")
        label_dict = iemocap.add_label_dataset()

        logger.info("[2/7] Get labelname to count ...")
        label_df = iemocap.name2df(label_dict)

        logger.info("[3/7] Get text ...")
        text_dataset = iemocap.add_text_dataset()
        label_df['text'] = label_df.index.map(text_dataset)

        logger.info("[4/7] remove [garbage] token ...")
        label_df['text']= label_df.text.map(iemocap.remove_garbage)

        logger.info("[5/7] Get Audio ...")
        audio_dataset = iemocap.add_audio_dataset()
        label_df['audio'] = label_df.index.map(audio_dataset)

        logger.info("[6/7] Drop NA ...")
        dataset = label_df.dropna()

        logger.info("[7/7] Save ...")
        os.makedirs(args.output_folder, exist_ok=True)
        dataset.to_pickle(os.path.join(args.output_folder,'iemocap_data.pkl'))
        
    elif args.DATA_FILE =='MELD':
        data_path = os.path.join(args.DATA_ROOT, args.DATA_FILE)
        data_folder = [os.path.join(data_path, f) 
                      for f in os.listdir(data_path) if f in ['train', 'dev', 'test']]
        wav_folders= [os.path.join(subdir, 'wav') for subdir in data_folder]
        label_root = os.path.join(data_path, args.LABEL_FOLDER)
        audio_folders = [os.path.join(data_path,'train', 'train_splits'),
                       os.path.join(data_path,'dev', 'dev_splits_complete'),
                       os.path.join(data_path,'test', 'output_repeated_splits_test')]
        
        meld = Meld(label_root, audio_folders, wav_folders)
        
        logger.info("[1/5] Get labed dataframe ...")
        data_parts = [os.path.join(label_root, f) for f in os.listdir(label_root)]
        df = meld.concat_parts(data_parts).reset_index(drop=True)

        logger.info("[2/5] Extract Audio.wav from mp4 ...")
        meld.make_wavfiles()

        logger.info("[3/5] Add audio array ...")
        audio_dataset = meld.add_audio_dataset(df)
        df['audio'] = df.index.map(audio_dataset)

        logger.info("[4/5] Drop non-match ...")
        dataset = meld.drop_na(df)

        logger.info("[5/5] Save ...")
        os.makedirs(args.output_folder, exist_ok=True)
        dataset.to_pickle(os.path.join(args.output_folder,'meld_data.pkl'))
    
    else:
        raise Exception('Unexpected dataset!') 