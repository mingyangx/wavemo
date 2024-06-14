from wavemo import *
import argparse

parser = argparse.ArgumentParser(description='Train WaveMo')
parser.add_argument('--save_folder', type=str, default='Train_WaveMo', help='Save folder')
parser.add_argument('--training_data_dir', type=str, default='/fs/vulcan-datasets/mit_places/data_large', help='Training data directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--use_wandb', action='store_true')
args = parser.parse_args()
print('use_wandb', args.use_wandb)

wavemo(
      sim=True, 
      use_modulation=True, 
      learn_slm=True, 
      hidden_dim=16,
      nframe=16, 
      batch_size=args._get_args,
      save_folder=args.save_folder, 
      training_data_dir=args.training_data_dir,
      use_wandb=args.use_wandb,
)  

'''
Example Usage:

python main.py --save_folder OUTPUT_DIRECTORY --training_data_dir DATASET_DIRECTORY

'''


