function [ VecFiles ] = FindAllFiles( InputDir,ext,IsReturnNameOnly,FlagSubFile )
%查找当前路径下所有文件夹
%InputDir: 输入路径
%ext：查找文件名字符段
%IsReturnNameOnly：是否只返回文件名

%check当前路径是否合法
if ~isdir(InputDir)
    msgbox('The input isnot a valid directory','Warning','warn');
    return 
else
    if nargin == 1
        ext = '*';
        IsReturnNameOnly = 1;
    elseif nargin == 2
        IsReturnNameOnly = 1;
    elseif nargin>4||nargin<1
        msgbox('1 or 2 inputs are required','Warning','warn');
        return
    end
    if nargout>1
        msgbox('Too many output arguments','Warning','warn');
        return
    end
end
    %初始化文件列表
    filesname = {};
    %文件夹临时存放列表
    folder{1} = InputDir;
    flag = 1; %当无文件夹时为0
    while flag
        currfolders = folder;
        folder = {};
        
        for m = 1:1:length(currfolders)
            cfolder = char(currfolders(m));
            %查找当前路径下所有含有ext字段文件
            strtmp = strcat(cfolder,'\*',ext,'*');
            files = dir(strtmp);
            L = length(files);
            num = length(filesname);
            for i =1:1:L
                %排除.和..
                if ~(strcmp(files(i).name,'.')||strcmp(files(i).name,'..'))
                    tmp = files(i).name;  
                    if ~files(i).isdir        
                        num = num + 1;
                        if IsReturnNameOnly    %返回文件夹列表
                            filesname{num} = tmp;
                        else               %返回文件夹全路径列表
                                tmp = fullfile(cfolder,tmp);
                                filesname{num} = tmp;
                        end
                    end
                end
            end %end for i =1:1:L
            
            %保存当前路径下文件夹
            if FlagSubFile == 1
                allfiles = dir(cfolder);
            else
                allfiles = files;
            end
            AL = length(allfiles);
            AF = length(folder);
            for i =1:1:AL
                if ~(strcmp(allfiles(i).name,'.')||strcmp(allfiles(i).name,'..'))
                    tmp = allfiles(i).name;  
                    if allfiles(i).isdir        
                        AF = AF + 1;
                        tmp = fullfile(cfolder,tmp);
                        folder{AF} = tmp;
                    end
                end
            end
        end   %end for m = 1:1:length(currfolders)
    
        if isempty(folder)
            flag = 0;
        end
    end  % end of while
        
    %赋值到返回变量VecFolders
    if nargout==1
        VecFiles = filesname';
    end
end
