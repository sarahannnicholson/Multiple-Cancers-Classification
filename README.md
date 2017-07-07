# Classifying-Multiple-Cancers-Using-Machine-Learning
This project is part of the CENG 420 (AI) course at the University of Victoria.

## Motivation
Finding the right treatment plan for a cancer patient is a difficult process. Various treatment options are available depending on the stage of the cancer, type of cancer, and the patient's overall health. This means that in order to provide the best outcome for the patient, a proper diagnosis must be made. The way we diagnosis, treatment or prevention across different areas of Cancer is very primitive as compared to what we can accomplish with the help of computer science. Currently the Doctors start the process by mapping patient information into structured data by hand and then run basic statistical analyses to identify correlations [1]. The purpose of this project is to accurately classify a patient's type of cancer based on gene expressions.

Our DNA, is constantly being replicated and made into proteins. Protein expression2 is a reliable way to distinguish between two cells within the body, this can be extended to help recognize cancer cells. However, this method is not widely applied because many cancer molecular markers have yet to be identified [2]. Rather than concentrating on specific markers, a study that analyzed multiclass cancer diagnosis using tumor gene expressions showed that clustering the gene expressions of tumor cells can be used to classify common malignancies [3].

## Baseline
There is little research regarding the classification of multiple cancers using a single database; this is partially due to the lack of comprehensive gene expression datasets. We intend to use “Multiclass cancer diagnosis using tumor gene expression signatures” study as a baseline for our project [3]. The study had created an extensive database with cancer tumor gene expression profiles relating their corresponding cancer class; which we intend to use for our project. The 14 common malignancies we aim to predict are as follows: Breast, Prostate, Lung, Colorectal, Lymphoma, Bladder, Melanoma, Uterus, Leukemia, Renal, Pancreas, Ovary, Mesothelioma, and Central Nervous System cancer.

## Model
The study compared two different machine learning models, supervised and unsupervised clustering with an overall accuracy of 78%. Their model was unable to classify poorly differentiated tumors. We will use hierarchical clustering to differentiate between the major groups and investigate further to find partitions in the poorly differentiated groups.

## Installation and run
1. clone the project `git clone https://github.com/sarahannnicholson/Classifying-Multiple-Cancers-Using-Machine-Learning.git`
2. Change directory `cd Classifying-Multiple-Cancers-Using-Machine-Learning`
3. Install stuff `sudo apt-get install libfreetype6-dev libxft-dev`
4. Install project dependencies `pip install -r requirements.txt`
5. Run the code `python featureData.py`

## References
1. Murphy,Meg (2017, February 17) Empowering Cancer Treatment with Machine Learning.
2. Connolly, J. L., Schnitt, S. J., Wang, H. H., Dvorak, A. M. & Dvorak, H. F. (1997) in Cancer Medicine, eds. Holland, J. F., Frei, E., Bast, R. C., Kufe, D. W., Morton, D. L. & Weichselbaum, R. R. (Williams & Wilkins, Baltimore), pp. 533–555.
3. [S. Ramaswamy, et al Multiclass cancer diagnosis using tumor gene expression signatures.](http://portals.broadinstitute.org/cgi-bin/cancer/publications/view/61)
