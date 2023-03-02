import os
import argparse
from six import iteritems
from itertools import product
from time import gmtime, strftime

def read_command_line(argv=None):
    parser = argparse.ArgumentParser(description='Large Scale Pretraining for Visual Dialog')
    base_path = 'data/'

    #-------------------------------------------------------------------------
    # Data input settings (VisDial)
    parser.add_argument('-visdial_processed_train', default=base_path + 'visdial/visdial_1.0_train_processed.json', 
                                 help='json file containing train split of visdial data')
    parser.add_argument('-visdial_processed_val', default=base_path + 'visdial/visdial_1.0_val_processed.json',
                            help='json file containing val split of visdial data')
    parser.add_argument('-visdial_processed_test', default=base_path + 'visdial/visdial_1.0_test_processed.json',
                            help='json file containing test split of visdial data')
    parser.add_argument('-visdial_processed_train_0.9', default=base_path + 'visdial/visdial_0.9_train_processed.json', 
                                 help='json file containing train split of visdial data')
    parser.add_argument('-visdial_processed_val_0.9', default=base_path + 'visdial/visdial_0.9_val_processed.json',
                            help='json file containing val split of visdial data')
    parser.add_argument('-visdial_image_feats', default=base_path + 'visdial/visdial_img_feat.lmdb',
                            help='json file containing image feats for train,val and splits of visdial data')
    parser.add_argument('-visdial_processed_train_dense', default=base_path + 'visdial/visdial_1.0_train_dense_processed.json',
                            help='samples on the train split for which dense annotations are available')
    parser.add_argument('-train_dense', action='store_true', help='use additional pseudo-labels in training')
    parser.add_argument('-visdial_processed_val_dense_annotations', default=base_path + 'visdial/visdial_1.0_val_dense_annotations_processed.json',
                            help='JSON file with dense annotations')
    parser.add_argument('-visdial_processed_val_coref_dependency', default=base_path + 'visdial/visdial_1.0_val_coref_dependency.json',
                            help='semantic dependencies between dialog rounds')
    parser.add_argument('-cos_sim_counter_fitting', default=base_path + 'visdial/cos_sim_counter_fitting.npy',
                            help='cosine similarity matrix for synonym substitution in textual adversarial attack')
    parser.add_argument('-cos_sim_idx2word', default=base_path + 'visdial/cos_sim_idx2word.pickle')
    parser.add_argument('-cos_sim_word2idx', default=base_path + 'visdial/cos_sim_word2idx.pickle')
    parser.add_argument('-start_path', default='', help='path of starting model checkpt')
    parser.add_argument('-start_path_q', default='', help='path of starting questioner model checkpt')
    parser.add_argument('-start_path_a', default='', help='path of starting answerer model checkpt')
    parser.add_argument('-model_enc_config', default='config/bert_base_6layer_6conect_enc.json', help='model definition of the bert model')
    parser.add_argument('-model_dec_config', default='config/bert_base_6layer_6conect_dec.json', help='model definition of the bert model')
    #-------------------------------------------------------------------------
    # Data input settings (CC12M)
    parser.add_argument('-cc12m_processed_train', default=base_path + 'cc12m/dialogs/', 
                                 help='json file containing train split of visdial data')
    parser.add_argument('-cc12m_image_feats', default=base_path + 'cc12m/features/',
                            help='json file containing image feats for train,val and splits of visdial data')
    parser.add_argument('-cc12m_caption', default='', 
                                 help='json file containing train split of visdial data')
    parser.add_argument('-chunk', default='', help='the number of chunks to use')
    parser.add_argument('-threshold', default=50, type=int, help='perplexity-based data selection threshold')
    #-------------------------------------------------------------------------
    # Optimization / training params
    # Other training environmnet settings
    parser.add_argument('-vd_version', default='1.0', type=str, choices=['1.0', '0.9'])
    parser.add_argument('-mode', default='vd_train', type=str, choices=['vd_train', 'vd_eval_val', 'vd_eval_test', 'vd_gen_val', 'cc12m_gen', 'cc12m_train'])
    parser.add_argument('-model', default='enc_dec_a', type=str, choices=['enc_only_a', 'enc_dec_a', 'enc_dec_q'])
    parser.add_argument('-iter', default=1, type=int, help='current iteration in self-training')
    parser.add_argument('-num_workers', default=8, type=int,
                            help='Number of worker threads in dataloader')  
    parser.add_argument('-batch_size', default=72, type=int,
                            help='size of mini batch')
    parser.add_argument('-num_epochs', default=100, type=int,
                            help='total number of epochs')
    parser.add_argument('-batch_multiply', default=1, type=int,
                            help='amplifies batch size in mini-batch training')
    parser.add_argument('-select_data', action='store_true', help='using perplexity-based data selection')
    parser.add_argument('-lr',default=2e-5,type=float,help='learning rate')
    parser.add_argument('-image_lr',default=2e-5,type=float,help='learning rate for vision params')
    parser.add_argument('-overfit', action='store_true', help='overfit for debugging')
    parser.add_argument('-continue', action='store_true', help='continue training')
    parser.add_argument('-num_train_samples',default=0,type=int, help='number of train samples, set 0 to include all')
    parser.add_argument('-num_val_samples',default=0, type=int, help='number of val samples, set 0 to include all')
    parser.add_argument('-num_options',default=100, type=int, help='number of options to use. Max: 100 Min: 2')
    parser.add_argument('-gpu_ids', nargs="+", type=int, default=[0, 1, 2, 3], help="List of ids of GPUs to use.")
    parser.add_argument('-sequences_per_image',default=1, type=int, help='number of sequences sampled from an image during training')
    parser.add_argument('-visdial_tot_rounds',default=11, type=int,  \
               help='number of rounds to use in visdial,caption is counted as a separate round, therefore a maximum of 11 rounds possible')
    parser.add_argument('-max_seq_len',default=256, type=int, help='maximum sequence length for the dialog sequence')
    parser.add_argument('-max_utt_len',default=25, type=int, help='maximum sequence length for each utterance')
    parser.add_argument('-num_negative_samples',default=1, type=int, help='number of negative samples for every positive sample for the nsp loss')
    parser.add_argument('-lm_loss_coeff',default=1,type=float,help='Coeff for lm loss')
    parser.add_argument('-nsp_loss_coeff',default=1,type=float,help='Coeff for nsp loss')
    parser.add_argument('-img_loss_coeff',default=1,type=float,help='Coeff for img masked loss')
    parser.add_argument('-mask_prob',default=0.15,type=float,help='prob used to sample masked tokens')
    parser.add_argument('-attack',default='fgsm', type=str, choices=['fgsm', 'random_token', 'coreference'])
    parser.add_argument('-save_path', default='checkpoints/',
                            help='Path to save checkpoints')
    parser.add_argument('-save_name', default='', help='file name to save')

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    try:
        parsed = vars(parser.parse_args(args=argv))
        if parsed['save_path'] == 'checkpoints/':
            # Standard save path with time stamp
            import random
            timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
            parsed['save_path'] = os.path.join(parsed['save_path'], timeStamp)
   
        assert parsed['sequences_per_image'] <= 8 
        assert parsed['visdial_tot_rounds'] <= 11

    except IOError as msg:
        parser.error(str(msg))

    return parsed
