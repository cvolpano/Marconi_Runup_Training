% Move all folders from subfolders to the main directory
% Author: Senior MATLAB Developer
% Date: 2026-10-01

% === CONFIGURATION ===
mainDir = "C:\Users\cvolpano\Marconi_Runup_Training\runupTool-Toolbox-master\runup\"; % Main directory (current folder)
% You can also set manually: mainDir = 'C:\path\to\main';

% === FIND SUBFOLDERS ===
subDirs = dir(mainDir);
subDirs = subDirs([subDirs.isdir]); % Keep only directories
subDirs = subDirs(~ismember({subDirs.name}, {'.', '..'})); % Remove . and ..

for i = 1:length(subDirs)
    subDirPath = fullfile(mainDir, subDirs(i).name);
    % Get files inside this subfolder
    files = dir(fullfile(subDirPath, '*'));
    files = files(~[files.isdir]); % Keep only files
    destFolder = fullfile(mainDir);
    movefile(subDirPath, mainDir);
    
end

disp('Folder move operation completed.');
