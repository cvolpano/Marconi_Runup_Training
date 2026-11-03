%% Converts CoastCam timestack image to .mat file
% Required inputs: transect (u,v) coordinates of pix instruments (yaml)
% Split timestack transect image
%% CONFIG
fs = 2;                           % sampling frequency (Hz)
burst_minutes = 10;
time = burst_minutes*60*fs;       % 1200
coords_path = 'CACO03_c1_timestack_20240920_transect70.yaml'; % CSV with columns: u,v (integers or floats)
img_dir = 'C:\Users\cvolpano\OneDrive - DOI\Desktop\Problem_Timestacks\'
meters_per_pixel = [];             % optional: set if you have calibration (e.g., 0.015)

% Start filename (first frame)
fname = '1728993600.Tue.Oct.15_12_00_00.GMT.2024.CACO03.c1.ras_transect70.png';        
% folder holding burst PNGs
[filepath, basename, ext] = fileparts(fname);

out_mat = basename+".mat";         % output .mat
%% Parse start epoch from filename (first token before first dot)
% Example name: 1729173600.Thu.Oct.17_14_00_00.GMT.2024.CACO03.c1.ras_transect70.png
tokens = regexp(fname, '^(\d+)\.', 'tokens', 'once');
if isempty(tokens)
    error('Could not parse epoch from filename: %s', fname);
end
start_epoch = str2double(tokens{1});  % 1729173600 (seconds since Unix epoch, GMT)

%% Time vector T (1 x time) in epoch seconds
T = start_epoch + (0:(time-1))/fs;    % row vector

%% Load (u,v) transect coordinates

yaml_text = fileread(coords_path);

% Extract the entire bracket-free numeric list after "U:"
u_token = regexp(yaml_text, 'U:\s*([^#V]+)', 'tokens', 'once');
v_token = regexp(yaml_text, 'V:\s*([^#]+)', 'tokens', 'once');

% Clean whitespace/newlines from multi-line lists
u_str = regexprep(u_token{1}, '\s+', ' ');
v_str = regexprep(v_token{1}, '\s+', ' ');

% Parse comma-separated numbers into vectors
U = sscanf(u_str, '%f,');
V = sscanf(v_str, '%f,');

coords_uv = [U(:)  V(:)];
% Ensure column vector ordering and double
coords_uv = double(coords_uv);
timestacklength = size(coords_uv, 1);

XYZ = zeros(timestacklength,3);

XYZ(:,1)= coords_uv(:,1);
XYZ(:,2)= coords_uv(:,2);

% Leave XYZ(:,3) (Z value) zeros

%% Preallocate RAW
% We'll read the first frame to discover colorbands
img0 = imread(fullfile(fname));
img0 = rgb2gray(img0);
RAW = round(min(255, max(0, img0)));
RAW = double(RAW);


%% Save .mat
save(out_mat, 'RAW', 'T', 'XYZ');

fprintf('Saved %s with RAW [%d x %d x %d], T [1 x %d], XYZ [%d x 3]\n', ...
    out_mat, size(RAW,1), size(RAW,2), size(RAW,3), numel(T), size(XYZ,1));
