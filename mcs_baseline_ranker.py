
import os
os.chdir("C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/")




import numpy as np
import open_clip
import torch
import yaml
from loguru import logger
from sklearn.preprocessing import normalize
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from cv_project.modeling_code_and_architecture.data_utils.augmentations import get_val_aug_query, get_val_aug_gallery
from cv_project.modeling_code_and_architecture.data_utils.dataset import SubmissionDataset
from cv_project.modeling_code_and_architecture.utils import convert_dict_to_tuple

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



        
class Head(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Head, self).__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x)


class ModelToUse(torch.nn.Module):
    def __init__(self, vit_backbone):
        super(ModelToUse, self).__init__()
        self.model = vit_backbone
        self.head = Head(1024)

    def forward(self, images):
        x = self.model(images)
        return self.head(x)


class MCS_BaseLine_Ranker:
    def __init__(self, dataset_path, gallery_csv_path, queries_csv_path):
        
        self.dataset_path = dataset_path
        self.gallery_csv_path = gallery_csv_path
        self.queries_csv_path = queries_csv_path
        self.max_predictions = 1000

        checkpoint_path1 = '/experiments/vit/convnext_final_bigdatast.pt'
        self.batch_size = 32
        self.input_size = 272

        # self.exp_cfg = 'config/baseline_mcs.yml'
        self.inference_cfg = 'config/inference_config.yml'

        self.device = torch.device('cpu')

        # with open(self.exp_cfg) as f:
        #     data = yaml.safe_load(f)
        # self.exp_cfg = convert_dict_to_tuple(data)

        with open(self.inference_cfg) as f:
            data = yaml.safe_load(f)
        self.inference_cfg = convert_dict_to_tuple(data)

        logger.info('Creating model and loading checkpoint')
        self.model_scripted = self.get_model_raw(checkpoint_path1, self.device)
        logger.info('Weights are loaded!')

    def get_model_raw(self, model_path: str, device_type: torch.device):
        vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms("convnext_xxlarge")
        # Create an instance of the ModelToUse class (assuming ModelToUse is a PyTorch module)
        model = ModelToUse(vit_backbone.visual)
        
        # Load the model's architecture using torch.jit.load without loading weights
        model_scripted = torch.jit.load(model_path, map_location=device_type)
        
        # Trace the loaded model to capture its structure and initialize its weights
        traced_model = torch.jit.trace(model, torch.rand(1, 3, 224, 224).to(device_type))
        
        # Load the state_dict from the model_scripted into the traced_model
        traced_model.load_state_dict(model_scripted.state_dict())
        
        # Set the traced_model to evaluation mode and move it to the specified device
        traced_model = traced_model.eval().to(device=device_type).to(memory_format=torch.channels_last)
        
        return traced_model


    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)

    def predict_product_ranks(self):                                              # This function return a numpy array of shape `(num_queries, 1000)`.For each query image the model will need to predict
                                                                                  # a set of 1000 unique gallery indexes, in order of best match first. 
        gallery_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.gallery_csv_path,
            transforms=get_val_aug_gallery(self.input_size)
        )

        query_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.queries_csv_path,
            transforms=get_val_aug_query(self.input_size), with_bbox=True
        )

        datasets = ConcatDataset([query_dataset, gallery_dataset])
        combine_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=self.inference_cfg.num_workers
        )

        logger.info('Calculating embeddings')
        embeddings = []
        # with torch.no_grad(), torch.cuda.amp.autocast():
        with torch.no_grad():
            logger.info('Start')
            for i, images in tqdm(enumerate(combine_loader), total=len(combine_loader)):
                images = images.to(self.device).to(memory_format=torch.channels_last).half()
                outputs = self.model_scripted(images).cpu().float().numpy()
                embeddings.append(outputs)

        embeddings = np.concatenate(embeddings)
        embeddings = np.nan_to_num(embeddings, posinf=0, neginf=0)
        query_embeddings = embeddings[:len(query_dataset)]
        gallery_embeddings = embeddings[len(query_dataset):]

        logger.info('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)

        logger.info(f"whitening, query:{query_embeddings.shape[0]} gallery:{gallery_embeddings.shape[0]}")

        similarities = -pairwise_distances(query_embeddings, gallery_embeddings)
        topk_final = min(similarities.shape[1], self.max_predictions)
        class_ranks = torch.topk(torch.from_numpy(similarities), topk_final, dim=1)[1].numpy()
        logger.info("Finished")
        return class_ranks



dataset_path = "C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/Academic Project/MCS2023_development_test_data/development_test_data/"
gallery_csv_path = "C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/Academic Project/MCS2023_development_test_data/development_test_data/gallery.csv"
queries_csv_path = "C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/Academic Project/MCS2023_development_test_data/development_test_data/queries.csv"

baseline_model = MCS_BaseLine_Ranker(dataset_path,gallery_csv_path, queries_csv_path)

mat = baseline_model.predict_product_ranks()

import pandas as pd
df = pd.DataFrame(mat)
df.to_csv('class_rank.csv')
