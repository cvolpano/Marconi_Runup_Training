%% runup_label_qc.m
%
% Batch QC tool for runupTool-digitized waterlines.
%
% For each digitized runup coord file (*.runup.mat) found under ruCoordDir:
%   1. Derive the matching raw image file name (strip '.runup90' segment... 
%      see deriveImageFilename() below -- EDIT to match your exact pattern)
%   2. Load RAW image + runup.Ri / T
%   3. Plot RAW with the digitized waterline overlaid
%   4. Wait for keypress:
%        y -> ACCEPT: generate 2D grayscale label jpg {0=other, 1=water}
%        n -> REJECT: move both .mat files into a 'rejected' subfolder
%
% Label logic:
%   - T is a per-row timestamp vector, length(T) == size(RAW,1)
%   - runup.Ri(k) is the sub-pixel column (X) position of the waterline
%     at row k
%   - For row k: columns > runup.Ri(k) are water (1), columns <= are other (0)
%   - If length(runup.Ri) < size(RAW,1) (partial digitization), the script
%     ALERTS you and skips generating a label for that file (goes to a
%     separate "incomplete" list) rather than silently guessing.
%
% Outputs:
%   - Label pngs saved to a 'labels' folder mirroring the input folder
%     structure under ruCoordDir
%   - Rejected .mat pairs moved to 'rejected' subfolder (mirroring structure)
%   - Incomplete files (partial digitization) reported in a summary at the end

function runup_label_qc(ruImageDir, ruCoordDir)

    if nargin < 1 || isempty(ruImageDir)
        ruImageDir = '.\data\';
    end
    if nargin < 2 || isempty(ruCoordDir)
        ruCoordDir = '.\runup\';
    end

    labelsRootDir  = fullfile(fileparts(ruCoordDir), 'labels');
    rejectRootDir  = fullfile(fileparts(ruCoordDir), 'rejected');

    % --- Find all runup coord files recursively ---
    coordFiles = dir(fullfile(ruCoordDir, '**', '*.runup.mat'));
    coordFiles = coordFiles(~[coordFiles.isdir]);

    if isempty(coordFiles)
        fprintf('No *.runup.mat files found under %s\n', ruCoordDir);
        return;
    end

    fprintf('Found %d runup coord files. Press y = accept, n = reject.\n\n', ...
            numel(coordFiles));

    fig = figure('Name', 'Runup QC', 'NumberTitle', 'off');

    incompleteList = {};   % track partially-digitized files
    skippedList    = {};   % track files where image wasn't found

    for idx = 1:numel(coordFiles)
        coordPath = fullfile(coordFiles(idx).folder, coordFiles(idx).name);

        % --- Derive matching image path ---
        imagePath = deriveImageFilename(coordPath, ruCoordDir, ruImageDir);

        if ~isfile(imagePath)
            fprintf('[%d/%d] SKIP (image not found): %s\n', ...
                    idx, numel(coordFiles), coordFiles(idx).name);
            skippedList{end+1} = coordPath; %#ok<AGROW>
            continue;
        end

        % --- Load data ---
        imgData   = load(imagePath);   % expects variable RAW
        coordData = load(coordPath);   % expects variable runup, with .Ri and .T

        if ~isfield(imgData, 'RAW')
            fprintf('[%d/%d] SKIP (no RAW variable in image file): %s\n', ...
                    idx, numel(coordFiles), imagePath);
            skippedList{end+1} = coordPath; %#ok<AGROW>
            continue;
        end
        if ~isfield(coordData, 'runup') || ~isfield(coordData.runup, 'Ri') ...
                || ~isfield(coordData.runup, 'T')
            fprintf('[%d/%d] SKIP (missing runup.Ri/T): %s\n', ...
                    idx, numel(coordFiles), coordPath);
            skippedList{end+1} = coordPath; %#ok<AGROW>
            continue;
        end

        RAW   = imgData.RAW;
        runup = coordData.runup;
        nRows = size(RAW, 1);
        nRi   = length(runup.Ri);

        isPartial = (length(runup.T) < nRows) || (nRi < nRows);
        if isPartial
            fprintf(['[%d/%d] *** ALERT: partial digitization *** %s\n' ...
                     '        (RAW has %d rows, runup.Ri/T has %d entries)\n'], ...
                     idx, numel(coordFiles), coordFiles(idx).name, nRows, nRi);
        end

        % --- Plot for review ---
        clf(fig);
        ax = axes(fig);
        imagesc(RAW, 'Parent', ax); axis(ax, 'image');  % RAW is RGB
        hold(ax, 'on');
        rows = 1:nRi;
        scatter(ax, runup.Ri, rows, 10, 'r', 'filled');
        titleStr = sprintf('%s   [%d/%d]   (y = accept, n = reject)', ...
                            coordFiles(idx).name, idx, numel(coordFiles));
        if isPartial
            titleStr = ['[PARTIAL] ' titleStr];
        end
        title(ax, titleStr, 'Interpreter', 'none');
        drawnow;

        % --- Wait for valid keypress ---
        key = '';
        fig.CurrentCharacter = char(0);
        while ~ismember(key, {'y', 'n'})
            waitfor(fig, 'CurrentCharacter');
            if ~ishandle(fig)
                fprintf('Figure closed. Stopping.\n');
                return;
            end
            key = lower(fig.CurrentCharacter);
            fig.CurrentCharacter = char(0);
        end

        if strcmp(key, 'n')
            % --- REJECT: move both files, preserving relative structure ---
            relPath = erase(coordFiles(idx).folder, ruCoordDir);
            destDir = fullfile(rejectRootDir, relPath);
            if ~exist(destDir, 'dir'); mkdir(destDir); end
            movefile(coordPath, fullfile(destDir, coordFiles(idx).name));
            [~, imgName, imgExt] = fileparts(imagePath);
            movefile(imagePath, fullfile(destDir, [imgName imgExt]));
            fprintf('[%d/%d] %s -> REJECTED (files moved)\n', ...
                    idx, numel(coordFiles), coordFiles(idx).name);

        else
            % --- ACCEPT ---
            if isPartial
                fprintf(['[%d/%d] %s -> accepted, but SKIPPING label generation ' ...
                         '(partial digitization)\n'], idx, numel(coordFiles), ...
                         coordFiles(idx).name);
                incompleteList{end+1} = coordPath; %#ok<AGROW>
                continue;
            end

            label = buildLabel(RAW, runup.Ri);

            relPath = erase(coordFiles(idx).folder, ruCoordDir);
            destDir = fullfile(labelsRootDir, relPath);
            if ~exist(destDir, 'dir'); mkdir(destDir); end

            [~, imgName, ~] = fileparts(imagePath);
            labelPath = fullfile(destDir, [imgName '_label.png']);

            % Save as exact {0,1} grayscale png (lossless, class map, not scaled for viewing)
            imwrite(uint8(label), labelPath);

            fprintf('[%d/%d] %s -> accepted, label saved: %s\n', ...
                    idx, numel(coordFiles), coordFiles(idx).name, labelPath);
        end
    end

    if ishandle(fig)
        close(fig);
    end

    fprintf('\n=== Done ===\n');
    fprintf('Labels saved under: %s\n', labelsRootDir);
    fprintf('Rejected pairs moved under: %s\n', rejectRootDir);

    if ~isempty(incompleteList)
        fprintf('\n%d file(s) accepted but SKIPPED for label generation (partial digitization):\n', ...
                numel(incompleteList));
        for k = 1:numel(incompleteList)
            fprintf('  %s\n', incompleteList{k});
        end
    end
    if ~isempty(skippedList)
        fprintf('\n%d file(s) skipped entirely (missing image/fields):\n', numel(skippedList));
        for k = 1:numel(skippedList)
            fprintf('  %s\n', skippedList{k});
        end
    end
end


function label = buildLabel(RAW, Ri)
    % Build a 2D binary label the same size as RAW (first two dims).
    % Row k: columns > Ri(k) are water (1); columns <= Ri(k) are other (0).
    [nRows, nCols, ~] = size(RAW);
    label = false(nRows, nCols);
    colIdx = 1:nCols;
    for k = 1:length(Ri)
        label(k, :) = colIdx > Ri(k);
    end
    % Any rows beyond length(Ri) (shouldn't happen if isPartial check passed)
    % are left as 0 (other) by default.
end


function imagePath = deriveImageFilename(coordPath, ruCoordDir, ruImageDir)
    % EDIT THIS FUNCTION if your naming pattern differs.
    %
    % Example transform based on your sample filenames:
    %   coord: 1573920000.Sat.Nov.16_16_00_00.GMT.2019.madbeach.cx.runup90.runup.mat
    %   image: 1573920000.Sat.Nov.16_16_00_00.GMT.2019.madbeach.cx.runup90.mat
    %
    % i.e. remove the trailing '.runup' just before '.mat'

    [coordFolder, coordName, coordExt] = fileparts(coordPath); %#ok<ASGLU>
    imageName = coordName; % coordName has no extension here since coordExt='.mat'
    % coordName currently: '...runup90.runup'
    imageName = regexprep(imageName, '\.runup$', '');  % -> '...runup90'
    imageFileName = [imageName '.mat'];

    % Mirror the relative folder structure from ruCoordDir -> ruImageDir
    relFolder = erase(coordFolder, ruCoordDir);
    imagePath = fullfile(ruImageDir, relFolder, imageFileName);
end