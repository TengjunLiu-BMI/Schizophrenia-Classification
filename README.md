### Project Environment:

- Python==3.6.13
- pytorch==1.8.1
- matplotlib==3.2.2
- numpy==1.19.5
- h5py==2.10.0
- scikit-learn==0.24.2
- scipy==1.5.3


### Workflow:

1. **Data Segmentation & Feature Extraction**: In the filefolder "1. Data Segmentation & Feature Extraction", excute the file "dataSegmentationAndFeatureExtraction.m" to segment the data and extract the features, and obtain the samples to be classified. The ready-to-classify samples at different levels (5s-sample and individual) were already stored in the file "Segmented Data".
2. **Baseline Model Training**: In the filefolder "2. Baseline Model Training & Direct Transferring Prediction & Fine Tuning\\*** Level", where "\*\*\*" indicates different levels, execute the file "cnnClassification_\*\*\*-PolandTrained.py" to train the baseline models.
3. **Direct Transferring Prediction**: In the filefolder "2. Baseline Model Training & Direct Transferring Prediction & Fine Tuning\\*** Level", where "\*\*\*" indicates different levels, execute the file "cnnClassification_\*\*\*-RussiaTested.py" to obtain the baseline transferring accuracy.
4. **Fine Tuning the Trained Models**: In the filefolder "2. Baseline Model Training & Direct Transferring Prediction & Fine Tuning\\*** Level", where "\*\*\*" indicates different levels, execute the file "cnnClassification_individual-RussiaTested-FineTuning.py" to obtain the fine-tuning transferring accuracy.
5. **Applying Transfer Component Analysis (TCA)**: TCA is implemented in the file "3. TCA & TCA_FT\\cnnClassification_Individual-TCA-TCA_FT.py".
6. **Train the Models on the Transformed Samples and Fine Tuning the Trained Models**: Execute the file "3. TCA & TCA_FT\\cnnClassification_Individual-TCA-TCA_FT.py" to obtain the direct transferring predictions of model trained on the TCA-transformed samples, and the corresponding fine-tuning transferring accuracy.
7. **Figure Plotting**: Execute the file "4. Figure Plotting\\resultAnalysis.m". The plotted results were already in the filefolder "4. Figure Plotting".

More details please refer to the file "EEG精神分裂症分类任务的迁移学习.pdf" (In mandarin btw).

### References:
1. Singh, K., S. Singh, and J. Malhotra, Spectral features based convolutional neural network for accurate and prompt identification of schizophrenic patients. Proc Inst Mech Eng H, 2021. 235(2): p. 167-184.
2. Pan, S.J., et al., Domain Adaptation via Transfer Component Analysis. IEEE Transactions on Neural Networks, 2011. 22(2): p. 199-210.
