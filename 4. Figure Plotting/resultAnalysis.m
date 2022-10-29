%% Baseline Acc & Loss
close all;
clear, clc;

fileIndividual = FindAllFiles('.\baseline', 'Poland Individual', 0, 0);
fileSample = FindAllFiles('.\baseline', 'Poland Sample', 0, 0);

numEpoch = 200;

acc_val_best_all_S = zeros(size(fileSample,1),1);
acc_val_all_S = zeros(size(fileSample,1),numEpoch);
acc_tr_all_S = zeros(size(fileSample,1),numEpoch);
loss_all_S = zeros(size(fileSample,1),numEpoch);

for idx_file = 1:size(fileSample,1)
    load([fileSample{idx_file}]);
    acc_val = acc_val(1:numEpoch);
    acc_tr = acc(1:numEpoch);
    loss = loss(1:numEpoch);
    acc_val_all_S(idx_file,:) = acc_val;
    acc_tr_all_S(idx_file,:) = acc_tr;
    acc_val_best_all_S(idx_file) = max(acc_val);
    loss_all_S(idx_file,:) = loss;
end

figure,
shadedErrorBar(1:numEpoch,mean(acc_val_all_S,1),std(acc_val_all_S),'lineProps',{'LineWidth',3});
hold on;
shadedErrorBar(1:numEpoch,mean(acc_tr_all_S,1),std(acc_tr_all_S),'lineProps',{'-.','Color','#D95319','LineWidth',3});
xlabel('Epoch');
ylabel('Accuracy/%');
legend({'','','','Validation','','','','Training'},'Location','Southeast'); 
ylim([50 120]);
set(gca, 'FontSize', 40);
box on;

figure,
shadedErrorBar(1:numEpoch,mean(loss_all_S,1),std(loss_all_S),'lineProps',{'LineWidth',3});
xlabel('Epoch');
ylabel('Loss');
% ylim([50 120]);
set(gca, 'FontSize', 40);
box on;


acc_val_best_all_I = zeros(size(fileIndividual,1),1);
acc_val_all_I = zeros(size(fileIndividual,1),numEpoch);
acc_tr_all_I = zeros(size(fileIndividual,1),numEpoch);
loss_all_I = zeros(size(fileIndividual,1),numEpoch);

for idx_file = 1:size(fileIndividual,1)
    load([fileIndividual{idx_file}]);
    acc_val = acc_val(1:numEpoch);
    acc_tr = acc(1:numEpoch);
    loss = loss(1:numEpoch);
    acc_val_all_I(idx_file,:) = acc_val;
    acc_tr_all_I(idx_file,:) = acc_tr;
    acc_val_best_all_I(idx_file) = max(acc_val);
    loss_all_I(idx_file,:) = loss;
end

figure,
shadedErrorBar(1:numEpoch,mean(acc_val_all_I,1),std(acc_val_all_I),'lineProps',{'LineWidth',3});
hold on;
shadedErrorBar(1:numEpoch,mean(acc_tr_all_I,1),std(acc_tr_all_I),'lineProps',{'-.','Color','#D95319','LineWidth',3});
xlabel('Epoch');
ylabel('Accuracy/%');
legend({'','','','Validation','','','','Training'},'Location','Southeast'); %,'LSTM more #para'
ylim([50 120]);
set(gca, 'FontSize', 40);
box on;

figure,
shadedErrorBar(1:numEpoch,mean(loss_all_I,1),std(loss_all_I),'lineProps',{'LineWidth',3});
xlabel('Epoch');
ylabel('Loss');
set(gca, 'FontSize', 40);
box on;

%% Baseline, Direct Tranferring Prediction

close all;
clear, clc;

fileIndividual_Baseline = FindAllFiles('.\baseline', 'Baseline*Individual', 0, 0);
fileSample_Baseline = FindAllFiles('.\baseline', 'Baseline*Sample', 0, 0);

numK = 10;

acc_transfer_all_I = zeros(size(fileIndividual_Baseline,1),numK);

for idx_file = 1:size(fileIndividual_Baseline,1)
    load([fileIndividual_Baseline{idx_file}]);
    acc_transfer_all_I(idx_file,:) = acc_ts_baseline_all';
end

disp([mean(acc_transfer_all_I(:)),std(acc_transfer_all_I(:))])


acc_transfer_all_S = zeros(size(fileSample_Baseline,1),numK);

for idx_file = 1:size(fileSample_Baseline,1)
    load([fileSample_Baseline{idx_file}]);
    acc_transfer_all_S(idx_file,:) = acc_ts_baseline_all';
end

disp([mean(acc_transfer_all_S(:)),std(acc_transfer_all_S(:))])

%% Comparison Bwteen Fine Tuning & Baseline

close all;
clear, clc;

fileIndividual_FT = FindAllFiles('.\fineTuning', 'Adaptation*individual', 0, 0);
fileSample_FT = FindAllFiles('.\fineTuning', 'Adaptation*sample', 0, 0);

fileIndividual_Baseline = FindAllFiles('.\baseline', 'Baseline*Individual', 0, 0);
fileSample_Baseline = FindAllFiles('.\baseline', 'Baseline*Sample', 0, 0);

numEpoch = 200;
numK = 10;

acc_val_best_all_I_apt = zeros(size(fileIndividual_FT,1),1);

for idx_file = 1:size(fileIndividual_FT,1)
    load([fileIndividual_FT{idx_file}]);
    acc_val_best_all_I_apt(idx_file) = max(acc_ts_apt);
end


acc_val_best_all_S_apt = zeros(size(fileSample_FT,1),1);

for idx_file = 1:size(fileIndividual_FT,1)
    load([fileSample_FT{idx_file}]);
    acc_val_best_all_S_apt(idx_file) = max(acc_ts_apt);
end

acc_transfer_all_I_BL = zeros(size(fileIndividual_Baseline,1),numK);

for idx_file = 1:size(fileIndividual_Baseline,1)
    load([fileIndividual_Baseline{idx_file}]);
    acc_transfer_all_I_BL(idx_file,:) = acc_ts_baseline_all';
end

acc_transfer_all_S_BL = zeros(size(fileSample_Baseline,1),numK);

for idx_file = 1:size(fileSample_Baseline,1)
    load([fileSample_Baseline{idx_file}]);
    acc_transfer_all_S_BL(idx_file,:) = acc_ts_baseline_all';
end


results_I = [mean(acc_val_best_all_I_apt(:)),mean(acc_transfer_all_I_BL(:))]';
stds_I = [std(acc_val_best_all_I_apt(:)),std(acc_transfer_all_I_BL(:))]';
        
results_S = [mean(acc_val_best_all_S_apt(:)),mean(acc_transfer_all_S_BL(:))]';
stds_S = [std(acc_val_best_all_S_apt(:)),std(acc_transfer_all_S_BL(:))]';    
results = [results_I,results_S]';
stds = [stds_I,stds_S]';
        
[h_I, p_I] = ttest2(acc_val_best_all_I_apt(:), acc_transfer_all_I_BL(:));
[h_S, p_S] = ttest2(acc_val_best_all_S_apt(:), acc_transfer_all_S_BL(:));

figure,
bar(results, 'grouped');
hold on;
% Find the number of groups and the number of bars in each group
[ngroups, nbars] = size(results);
% Calculate the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, results(:,i), stds(:,i), 'k', 'linestyle', 'none','LineWidth',3,'CapSize',42);
end
hold off

set(gca, 'XTickLabel', {'Individual','Sample'});

legend({'Fine Tuning','Baseline'});

ylabel('Accuracy/%');
ylim([0,100])
set(gca, 'FontSize', 50);


%% Comparison Bwteen Domain Adaptation & Baseline

close all;
clear, clc;

fileIndividual_TCA = FindAllFiles('.\TCA', 'individual', 0, 0);

fileIndividual_Baseline = FindAllFiles('.\baseline', 'Baseline*Individual', 0, 0);

numEpoch = 200;
numK_R = 10;
numK_P = 2;

acc_transfer_all_TCA = zeros(size(fileIndividual_TCA,1),numK_R*numK_P);

for idx_file = 1:size(fileIndividual_TCA,1)
    load([fileIndividual_TCA{idx_file}]);
    acc_transfer_all_TCA(idx_file,:) = acc_ts_TCA_all(:)';
end


acc_transfer_all_I_BL = zeros(size(fileIndividual_Baseline,1),numK_R);

for idx_file = 1:size(fileIndividual_Baseline,1)
    load([fileIndividual_Baseline{idx_file}]);
    acc_transfer_all_I_BL(idx_file,:) = acc_ts_baseline_all';
end


results_I = [mean(acc_transfer_all_TCA(:)),mean(acc_transfer_all_I_BL(:))]';
stds_I = [std(acc_transfer_all_TCA(:)),std(acc_transfer_all_I_BL(:))]';

results = results_I';
stds = stds_I';
        
[h_I, p_I] = ttest2(acc_transfer_all_TCA(:), acc_transfer_all_I_BL(:));

figure,
bar(results, 'grouped');
hold on;

x = [1,2];
errorbar(x, results, stds, 'k', 'linestyle', 'none','LineWidth',3,'CapSize',42);
hold off

set(gca, 'XTickLabel', {'DA','Baseline'});

ylabel('Accuracy/%');
ylim([0,100])
set(gca, 'FontSize', 50);


%% Comparison Bwteen (Domain Adaptation + Fine Tuning), Domain Adaptation & Baseline


close all;
clear, clc;

fileIndividual_TCA = FindAllFiles('.\TCA', 'individual', 0, 0);
fileIndividual_TCA_FT = FindAllFiles('.\TCAandFineTuning', 'individual', 0, 0);
fileIndividual_Baseline = FindAllFiles('.\baseline', 'Baseline*Individual', 0, 0);

numEpoch = 200;
numK_R = 10;
numK_P = 2;

acc_transfer_all_TCA = zeros(size(fileIndividual_TCA,1),numK_R*numK_P);

for idx_file = 1:size(fileIndividual_TCA,1)
    load([fileIndividual_TCA{idx_file}]);
    acc_transfer_all_TCA(idx_file,:) = acc_ts_TCA_all(:)';
end

acc_transfer_all_TCA_FT = zeros(size(fileIndividual_TCA_FT,1),numK_R*numK_P);

for idx_file = 1:size(fileIndividual_TCA_FT,1)
    load([fileIndividual_TCA_FT{idx_file}]);
    acc_transfer_all_TCA_FT(idx_file,:) = acc_ts_TCA_fineTuning_all(:)';
end


acc_transfer_all_I_BL = zeros(size(fileIndividual_Baseline,1),numK_R);

for idx_file = 1:size(fileIndividual_Baseline,1)
    load([fileIndividual_Baseline{idx_file}]);
    acc_transfer_all_I_BL(idx_file,:) = acc_ts_baseline_all';
end


results_I = [mean(acc_transfer_all_TCA_FT(:)),mean(acc_transfer_all_TCA(:)),mean(acc_transfer_all_I_BL(:))]';
stds_I = [std(acc_transfer_all_TCA_FT(:)),std(acc_transfer_all_TCA(:)),std(acc_transfer_all_I_BL(:))]';

results = results_I';
stds = stds_I';
        
[h_TCA_BL, p_TCA_BL] = ttest2(acc_transfer_all_TCA(:), acc_transfer_all_I_BL(:));
[h_TCAFT_BL, p_TCAFT_BL] = ttest2(acc_transfer_all_TCA_FT(:), acc_transfer_all_I_BL(:));
[h_TCAFT_TCA, p_TCAFT_TCA] = ttest2(acc_transfer_all_TCA_FT(:), acc_transfer_all_TCA(:));

figure,
bar(results, 'grouped');
hold on;

x = [1,2,3];
errorbar(x, results, stds, 'k', 'linestyle', 'none','LineWidth',3,'CapSize',42);
hold off

set(gca, 'XTickLabel', {'DA+FT','DA','Baseline'});

ylabel('Accuracy/%');
ylim([0,100])
set(gca, 'FontSize', 50);



