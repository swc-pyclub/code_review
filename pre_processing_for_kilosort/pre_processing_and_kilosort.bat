SET MOUSE=VTA_drive2
SET EXPERIMENT=day4__2019-03-30_18-03-45

cd C:\Users\francescag\Documents\SourceTree_repos\Python_git\pre_processing_for_kilosort
python pre_process_start.py %MOUSE% %EXPERIMENT%

w: 
cd W:\Tetrode_kilosort\templates
set join="call_kilosort2('%MOUSE%','%EXPERIMENT%');exit;"
matlab -r %join%
pause
