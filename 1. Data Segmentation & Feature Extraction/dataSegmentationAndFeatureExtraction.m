close all;
clear,clc;

dataSets = {'Poland', 'Russia'};
class = {'Scz', 'Norm'};
dataSetType = {'Individual', 'Sample'};
fileSczPoland = FindAllFiles(['.\',dataSets{1},'\',dataSets{1},'_',class{1},'_clean\clean'], '.mat', 0, 0);
fileNormPoland = FindAllFiles(['.\',dataSets{1},'\',dataSets{1},'_',class{2},'_clean\clean'], '.mat', 0, 0);
fileSczRussia = FindAllFiles(['.\',dataSets{2},'\',dataSets{2},'_',class{1},'_clean\clean'], '.mat', 0, 0);
fileNormRussia = FindAllFiles(['.\',dataSets{2},'\',dataSets{2},'_',class{2},'_clean\clean'], '.mat', 0, 0);

timeWin_Individual = 1*55; % 1 min
timeWin_Sample = 5; %  5 s
timeStep_Individual = 20; % 20 s
timeStep_Sample = 2; % 2 s
channels_Russia = [1:6, 8:11, 13:16]; % F7 F3 F4 F8 T3 C3 C4 T4 T5 P3 P4 T6 O1 O2
channels_Poland = [11 3 4 12 13 5 6 14 15 7 8 16 9 10]; % F7 F3 F4 F8 T3 C3 C4 T4 T5 P3 P4 T6 O1 O2
filterBands = [0.1, 4; 4, 6; 6, 8; 8, 12; 12, 30; 30, 50];

for idx_dataset_type = 1:size(dataSetType, 2)
    for idx_dataset = 1:size(dataSets, 2)
        specFeas = [];
        labels = [];
        for idx_class = 1:size(class, 2)  
            eval(['fileIterated = file',class{idx_class},dataSets{idx_dataset},';']);
            for idx_file = 1:size(fileIterated,1)
                disp([idx_dataset_type, idx_dataset, idx_class, idx_file]);
                load(fileIterated{idx_file});
                eval(['lengthSeg = EEG_Clean.srate*timeWin_',dataSetType{idx_dataset_type},';']);
                idx_length = 1;
                idx_sample = 1;
                while(idx_length+lengthSeg-1 <= size(EEG_Clean.data, 2))
                    eval(['fftData = fft(EEG_Clean.data(channels_', dataSets{idx_dataset},', idx_length:idx_length+lengthSeg-1), [], 2);']);
                    Fs = EEG_Clean.srate;
                    T = 1/Fs;
                    L = lengthSeg;
                    P2 = abs(fftData/L);
                    P1 = P2(:,1:L/2+1);
                    P1(2:end-1) = 2*P1(2:end-1);
                    f = Fs*(0:(L/2))/L;
    %                 plot(f,P1);
                    eval(['idx_length = idx_length+timeStep_',dataSetType{idx_dataset_type},'*EEG_Clean.srate;']);
                    specFea = zeros(size(channels_Poland, 2), 5*size(filterBands, 1));
                    for idx_band = 1:size(filterBands, 1) % Feature Extraction
                        filteredSpect = P1(:,f>filterBands(idx_band, 1) & f<filterBands(idx_band, 2));
                        MSA = mean(filteredSpect, 2);
                        MSP = mean(filteredSpect.^2, 2);
                        Act = var(filteredSpect, 1, 2);
                        Mob = mobility(filteredSpect);
                        Com = mobility(diff(filteredSpect, 1, 2))./mobility(filteredSpect);
                        specFea(:, 5*(idx_band-1)+1 : 5*idx_band) = [MSA, MSP, Act, Mob, Com];
                    end
                    idx_sample = idx_sample+1;
                    specFeas = cat(3,specFeas, specFea);
                    labels = [labels; idx_class-1];
                end
            end
        end      
        rng(1);
        randIndex = randperm(size(labels,1));
        specFeas = specFeas(:,:,randIndex);
        labels = labels(randIndex);
        eval(['save .\dataset_', dataSetType{idx_dataset_type},'_', dataSets{idx_dataset}, '.mat specFeas labels'] );
    end
end    


