# Ensemble-Pre-trained-Multimodal-Models-for-Image-text-Retrieval-in-the-NewsImages-MediaEval-2023
For each of the test datasets (RT, GDELT-P1, GDELT-P2), we submitted the results of five runs
separately, the implementation details are presented as follows:

Run #1: Using the BLIP-2 model as the feature extractor, we encode article text and images
separately to obtain their respective features. We rank the images by calculating the cosine
similarity between text and all test images. From these ranking results, we select the top 100
most relevant images as our predicted results.

Run #2: Using the ViT-H/14 model of CLIP as the feature extractor, we encode article text
and images separately. We calculate the similarity between text features and all test image
features. We add the dual softmax method to calculate the similarity ranking between text and
images. The top 100 most relevant images are selected as our predicted results.

Run #3: By designing a multi-task contrastive learning model, we process the test set
similarly to the training set. For each text, we calculate cosine similarity with all test images
and keep only the top 100 text-image pairs based on similarity as our predicted results.

Run #4: As Run #2, we use the ViT-H/14@336px model of CLIP to encode article text (or
article titles) and images separately.

Run #5: Based on Runs #2, #3, and #4, we retrained three models, the results of each model
include all texts, with each text corresponding to 100 images and the cosine similarity between
each text and image. We then select a specific text URL and sum the cosine similarities for all
identical images. The results are sorted in descending order, and the top 100 most relevant
images are selected as our predicted results.
