%% runup_label_qc.m
%
% Batch QC tool for runupTool-digitized waterlines.
%
% For each digitized runup coord file (*.runup.mat) found under ruCoordDir:
%   1. Derive the matching raw image file name (strip trailing '.runup'
%      before '.mat' -- see deriveImageFilename() below)
%   2. Load RAW image + runup.Ri / T
%   3. Plot RAW (RGB) with the digitized waterline overlaid
%   4. Wait for keypress:
%        y -> ACCEPT: generate 2D grayscale label png {0=other, 1=water}
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
%   - Preview (human-viewable, {0,255}) pngs saved to a 'labels_preview'
%     folder mirroring the same structure -- NOT used by Segmentation Gym,
%     for visual sanity-checking only
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
    previewRootDir = fullfile(fileparts(ruCoordDir), 'labels_preview');
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

    % --- Set up figure with key capture via appdata flag ---
    fig = figure('Name', 'Runup QC', 'NumberTitle', 'off');
    setappdata(fig, 'lastKey', '');
    set(fig, 'KeyPressFcn', @(src, event) setappdata(src, 'lastKey', event.Key));

    incompleteList = {};   % track partially-digitized files
    skippedList    = {};   % track files where image wasn't found

    for idx = 1:numel(coordFiles)
        coordPath = fullfile(coordFiles(idx).folder, coordFiles(idx).name);

        % --- Derive matching image path ---
        imagePath = deriveImageFilename(coordPath, ruCoordDir, ruImageDir);

        if ~isfile(imagePath)
            fprintf('[%d/%d] SKIP (image not found): %s\n', ...
                    idx, numel(coordFiles), coordFiles(idx).name);
            fprintf('        looked for: %s\n', imagePath);
            skippedList{end+1} = coordPath; %#ok<AGROW>
            continue;
        end

        % --- Load data ---
        imgData   = load(imagePath);   % expects variables RAW, T
        coordData = load(coordPath);   % expects variable runup, with .Ri

        if ~isfield(imgData, 'RAW') || ~isfield(imgData, 'T')
            fprintf('[%d/%d] SKIP (missing RAW or T in image file): %s\n', ...
                    idx, numel(coordFiles), imagePath);
            skippedList{end+1} = coordPath; %#ok<AGROW>
            continue;
        end
        if ~isfield(coordData, 'runup') || ~isfield(coordData.runup, 'Ri')
            fprintf('[%d/%d] SKIP (missing runup.Ri): %s\n', ...
                    idx, numel(coordFiles), coordPath);
            skippedList{end+1} = coordPath; %#ok<AGROW>
            continue;
        end

        RAW   = imgData.RAW;
        T     = imgData.T;
        runup = coordData.runup;
        nRows = size(RAW, 1);
        nRi   = length(runup.Ri);

        isPartial = (length(T) < nRows) || (nRi < nRows);
        if isPartial
            fprintf(['[%d/%d] *** ALERT: partial digitization *** %s\n' ...
                     '        (RAW has %d rows, runup.Ri/T has %d entries)\n'], ...
                     idx, numel(coordFiles), coordFiles(idx).name, nRows, nRi);
        end

        % --- Plot for review ---
        clf(fig);
        ax = axes('Parent', fig);
        imagesc(RAW, 'Parent', ax); axis(ax, 'image');  % RAW is RGB, no colormap needed
        hold(ax, 'on');
        rows = 1:nRi;
        scatter(ax, runup.Ri, rows, 10, 'r', 'filled');
        titleStr = sprintf('%s   [%d/%d]   (y = accept, n = reject)', ...
                            coordFiles(idx).name, idx, numel(coordFiles));
        if isPartial
            titleStr = ['[PARTIAL] ' titleStr];
        end
        title(ax, titleStr, 'Interpreter', 'none');

        figure(fig);      % bring to front
        drawnow;

        % --- Wait for valid keypress ---
        setappdata(fig, 'lastKey', '');
        key = '';
        while ~ismember(key, {'y', 'n'})
            if ~ishandle(fig)
                fprintf('Figure closed. Stopping.\n');
                return;
            end
            uiwait(fig, 1);           % briefly yield so KeyPressFcn can fire
            key = lower(getappdata(fig, 'lastKey'));
            if isempty(key)
                key = '';
            elseif ~ismember(key, {'y', 'n'})
                setappdata(fig, 'lastKey', '');  % clear invalid key, keep waiting
                key = '';
            end
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

            try
                % Save as exact {0,1} grayscale png (lossless, class map, not scaled for viewing)
                imwrite(uint8(label), labelPath);

                % Also save a human-viewable preview copy, scaled to {0,255}.
                % This is for visual sanity-checking only -- NOT used by
                % Segmentation Gym. Saved in a separate 'labels_preview' folder
                % so it never gets confused with the real class-map labels.
                previewDir = fullfile(previewRootDir, relPath);
                if ~exist(previewDir, 'dir'); mkdir(previewDir); end
                previewPath = fullfile(previewDir, [imgName '_label_preview.png']);
                imwrite(uint8(label) * 255, previewPath);

                fprintf('[%d/%d] %s -> accepted, label saved: %s\n', ...
                        idx, numel(coordFiles), coordFiles(idx).name, labelPath);
                fprintf('        preview saved: %s\n', previewPath);
            catch ME
                fprintf('[%d/%d] %s -> accepted, but FAILED to save label/preview: %s\n', ...
                        idx, numel(coordFiles), coordFiles(idx).name, ME.message);
                skippedList{end+1} = coordPath; %#ok<AGROW>
            end
        end
    end

    if ishandle(fig)
        close(fig);
    end

    fprintf('\n=== Done ===\n');
    fprintf('Labels saved under: %s\n', labelsRootDir);
    fprintf('Preview images saved under: %s\n', previewRootDir);
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
    % Example transform based on your file naming scheme:
    %   coord: 1573920000.Sat.Nov.16_16_00_00.GMT.2019.madbeach.cx.runup90.runup.mat
    %   image: 1573920000.Sat.Nov.16_16_00_00.GMT.2019.madbeach.cx.runup90.mat
    %
    % i.e. remove the trailing '.runup' just before '.mat'

    [coordFolder, coordName, ~] = fileparts(coordPath);
    % coordName currently: '...runup90.runup' (extension '.mat' already stripped)
    imageName = regexprep(coordName, '\.runup$', '');  % -> '...runup90'
    imageFileName = [imageName '.mat'];

    % Mirror the relative folder structure from ruCoordDir -> ruImageDir
    relFolder = erase(coordFolder, ruCoordDir);
    imagePath = fullfile(ruImageDir, relFolder, imageFileName);
end