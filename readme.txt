This repository contains the code sample for my Tubi Builders Program application. The code can be used to replicate a subset of the experiments from our AAAI-26 paper [1]. That paper shows that Average Treatment Effect (ATE) estimates from linear and non‑linear methods (including neural networks and random forests) can be manipulated by adversarial missingness attacks.

The code first runs the BLAMM algorithm, which iteratively learns which entries to make missing so that the resulting partially observed dataset yields a biased ATE estimate. Next, the efficacy of the learned missingness is measured and compared with a baseline attack. The experiments use the real‑world TWINS benchmark dataset, which must be downloaded first as explained below.

All code provided, except the folder External/ is written by me.

Steps:

0. Install the required packages below.

1. Change the directory to this folder.

2. Download Twin_Data.csv.gz from the below link as provided in the CATENets package by running the following command:
curl -o Twin_Data.csv https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz

3. Apply the BLAMM to learn an adversarial missingness mechanism.
python main_blamm.py --log 0

4. Test the attacks effectiveness against ATE estimation using mean imputation + linear model
python main_modeler.py 5 2


[1] Koyuncu, D.; Gittens, A.; Yener, B.; Yung, M. Exploiting Missing Data Remediation Strategies Using Adversarial Missingness Attacks. To appear at the Proceedings of the Fortieth AAAI Conference on Artificial Intelligence; 2026. Accessible at https://arxiv.org/abs/2409.04407.

#Required packages

matplotlib                3.3.3
numpy                     1.19.5 
scipy                     1.5.3 
scikit-learn              1.0.2 
pandas (with excel read)  1.1.4 
tensorflow                2.4.0 
causallearn               0.1.3.1 
networkx                  2.7.1
joblib                    0.17.0
seaborn                   0.12.0
statsmodels               0.12.2 
ucimlrepo                 0.0.3 

