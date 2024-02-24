function RUN_TESTS 

% RUN_TESTS

% Runs all unit tests contained in this directory
% (NOTE: must be called from the unittests directory!)

% Get list of files in current directory
DIRLIST=dir;

% Set to 1 to pause between unit tests (eg, to examine plots)
PAUSEON = 1;

% Run all scripts except this one
for TEST_ITER=3:length(DIRLIST)
    UNITTESTNAME = DIRLIST(TEST_ITER).name;
    if ~strcmp(UNITTESTNAME(1),'.') && ~strcmp(UNITTESTNAME(1:3),'RUN') && ~strcmp(UNITTESTNAME(end-2:end),'asv')
        fprintf('\n\n-----------------------------------\n'); 
        fprintf('RUNNING UNIT TEST: %s\n', UNITTESTNAME);
        runTest(UNITTESTNAME(1:end-2))
        if PAUSEON
            fprintf('press any key to run next test...\n');
            pause;
        end
    end
end

end

function runTest(funname)
    % creates new local environment for evaluating each unit test
    eval(funname);
end