
%% Get list of filenames
fnames = dir("*.png");


for i = 1:length(fnames)
%% Parse start epoch from filename (first token before first dot)
% Example name: 1729173600.Thu.Oct.17_14_00_00.GMT.2024.CACO03.c1.ras_transect70.png
    tokens = regexp(fnames(i).name, '^(\d+)\.', 'tokens', 'once')
    if isempty(tokens)
        error('Could not parse epoch from filename: %s', fname);
    end
    start_epoch = str2double(tokens{1});  % 1729173600 (seconds since Unix epoch, GMT)
    
    %% Create directory
    
    t = datetime(start_epoch, 'ConvertFrom', 'epochtime', 'Epoch', '1970-01-01');
    
    doy = day(t, 'dayofyear');
    monDay = datestr(t, 'mmm.dd'); % e.g., Aug.21

    yearDir = "data/" + t.Year + "/cx/"
    dirName = sprintf('%03d_%s', doy, monDay);
    fullName = yearDir+dirName

        if exist(fullName)
            movefile(fnames(i).name,fullName)
        else
            mkdir(fullName);
            movefile(fnames(i).name,fullName)
        end
    end