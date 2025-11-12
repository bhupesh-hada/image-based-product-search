# image-based-product-search

### 🚀 Project Overview

#### 🎯 Objective
The primary objective of this project is to accurately identify matching products from a collection of seller photos when presented with a user’s search image. Product images are categorized into two groups:
- User Photos – typically captured in real-world, cluttered environments using smartphone cameras.
- Seller Photos – professionally captured product images designed for online marketplaces.
User photos also include object bounding boxes highlighting the target product within the image, which are used as search queries for retrieval.

#### 🧠 Approach & Methodology
There are several state-of-the-art solutions in the image-based product search domain. Some of the most widely used include:
- CNN-based feature extraction using architectures like VGG or EfficientNet, pretrained on large-scale datasets such as ImageNet.
- SIFT-based methods that identify descriptors and keypoints across images to compute overlap-based similarity scores.

##### 🔹 Baseline Model
The project began with a baseline solution using a ResNet18 CNN architecture to find associations between seller and user images. Model performance was evaluated using the Mean Average Precision (mAP) metric.

##### 🔹 Iterative Improvements
After establishing the baseline, I iteratively refined the model by integrating advanced methods such as ArcFace and XBM (Cross-Batch Memory). Each training iteration leveraged the weights from the previous run, progressively improving feature representations. The iterations varied in terms of image resolution and training dataset size to optimize retrieval accuracy.

#### Commands to run the training scripts for different models
##### running the resnet18 classifier
python modeling_code_and_architecture/main.py --cfg config/baseline_mcs.yml

##### running the arcface_1st model
python -m visual_search.main_arcface --cfg config/arcface_1st.yml --name arcface_1st

##### running the arcface_2st model
python -m visual_search.main_arcface --cfg config/arcface_2nd.yml --name arcface_2nd

##### running the xbm_1st model
python -m visual_search.main_arcface --cfg config/xbm_1st.yml --name xbm_1st

#### 📊 Key Highlights
- Developed a scalable, deep learning–based product image retrieval system.
- Integrated advanced architectures (ArcFace, XBM) for enhanced feature learning.
- Achieved improved mAP through sequential fine-tuning and data scaling.
