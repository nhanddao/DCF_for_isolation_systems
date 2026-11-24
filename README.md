# DCF_for_isolation_systems
This repository contains data and core Python codes for the paper entitled "Damping correction factor for an equivalent linear force procedure to predict the peak displacement of isolation systems" (DOI: https://xxxx)

The repository has two folders. Their contents are explained below.

## Folder Data
This folder contains the main data for the investigation of the research. Below is the explanation for the files of the Mix group. Other group (i.e. PulseLike and NoPulse) have similar meaning.

| File name | Description |
|---|---|
| GMSets_Mix.txt | Contain 100 Mix ground motion sets. Each set has both pulse-like and non-pulse motions. Column 1 is the set ID number. Columns 2-31 are the NGA number of the ground motion in the set. Ignore any 0-NGA number. |
| GMSets_Mix_AnalysisSF.txt | Contain the scale factors of the ground motions in file GMSets_Mix.txt. This file has 100 rows and 30 columns, corresponding to 100 sets and 30 ground motions/set in file GMSets_Mix.txt. |
| Disp_Train_Mix.out | Contain the peak displacement obtained from OpenSees analysis. This file has 20,000 rows corresponding to 20,000 samples for training models.<br>Column 1 = f_d; Column 2 = T_d; Column 3 = k_1d; Column 4 = Ground motion set ID (corresponding to the first column in file GMSets_Mix.txt); Column 5 = D_M = average peak displacement (= average of non-zero values in Columns 6-35); Colunns 6-35: peak displacement subjected to individual ground motion in the set. |
| Disp_Test_Mix.out | Similar to Disp_Train_Mix.out, but for validating models. |
| BM_Train_Mix_HighDM.txt | Contain data for training ANN models and fitting simple equation. This file was derived from Disp_Train_Mix.out. Only data rows in Disp_Train_Mix.out with D_M > D_0 were used to generate the data in this file. This file has 9 columns: Column 1 = f_d; Column 2 = T_d; Column 3 = k_1d; Column 4 = Ground motion set ID; Column 5 = T_M/T_0; Column 6 = y_0 = D_M/D_0; Column 7 = T_d; Column 8 = beta_M; Column 9 = DCF. |
| BM_Train_Mix_LowDM.txt | Similar to file BM_Train_Mix_HighDM.txt, but for D_M <= D_0. This file was not used for developing the DCF models. |
| BM_Test_Mix_HighDM.txt | Similar to file BM_Train_Mix_HighDM.txt, but for validating the proposed models. |
