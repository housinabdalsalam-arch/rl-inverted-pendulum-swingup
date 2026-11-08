%% train_InvertedPendulum_SwingUpBalance_RL.m
% Train a TD3 agent to swing up and balance the Quanser IP02 cart-pole.
%
% theta = 0 is down, theta = pi is up. Time step Ts = 0.02 s (50 Hz).
% Observation: [x, xdot, sin(theta), cos(theta), thetadot].
% Action: motor voltage, -6 to 6 V.
%
% Stages:
%   S1  balance at the top, small kicks
%   S2  start 45-140 deg from upright, learn to pump energy
%   S3  swing up from the bottom
%   S4  mixed tasks, random plant values
%   S5  smoother balance at the top (optional)
clear; clc; close all;
here = fileparts(mfilename('fullpath'));
addpath(here);
assert(exist('scaleToVoltage','file')==2, 'scaleToVoltage.m not found on path');

START_STAGE = 1;                % 1 = new run; N = start at stage N
RUN_STAGE5 = true;              % run S5
rngSeed = 0; rng(rngSeed);
mdl = "InvertedPendulum_SwingUpBalance_RL";

%% Use GPU if there is one
device = "cpu";
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
    if gpuDeviceCount("available") > 0, gpuDevice(1); device = "gpu"; end
catch ME, warning('train:noGPU','%s',ME.message); end
fprintf("Device: %s\n", device);

%% Settings and plant values
Ts = 0.02; uMin = -6; uMax = 6; obsDim = 5; actDim = 1; actScale = (uMax-uMin)/2;
% The rig measured speed 1.42x too high (now fixed). Train with the real speed.
VEL_GAIN = 1.0;
p = struct();
p.mCart=1.025; p.mPend=0.110; p.L=0.3365; p.lp=0.1778; p.g=9.81;
p.gear_ratio=3.71; p.pinion_radius=0.00635; p.Kd=0.02; p.Bp=0.00017;
p.Fc=1.8; p.rho_coulomb=100; p.xmax=0.407; p.x_safe=0.35;
p.voltage_scale=0.80; p.voltage_bias=-0.181; p.voltage_rate_limit=1e4; p.voltage_deadzone=0.0;
p.encoder_res=4096; p.noise_x=0.00044; p.noise_theta=0.0018; p.noise_v=0.031; p.noise_thetadot=0.213;

%% Load model, write reward code
assert(exist(mdl+".slx",'file')~=0, 'Model not found: %s.slx', mdl);
load_system(fullfile(here, mdl+".slx"));
set_param(mdl,'SimulationMode','Accelerator');
configureModelSampleTime(mdl, Ts);
set_param(char(mdl+"/RL Agent"),'Commented','off');
writeReward(mdl, buildReward(10,1.5,0.08,0.05));   % penalties near the top: cart position and fast changes

wks = get_param(mdl,"ModelWorkspace");
assignin(wks,"mCart",p.mCart); assignin(wks,"mPend",p.mPend); assignin(wks,"L",p.L); assignin(wks,"lp",p.lp);
assignin(wks,"g",p.g); assignin(wks,"Kd",p.Kd); assignin(wks,"Bp",p.Bp); assignin(wks,"xmax",p.xmax);
assignin(wks,"x_safe",p.x_safe); assignin(wks,"pinion_radius",p.pinion_radius);
assignin(wks,"Fc_cart",p.Fc); assignin(wks,"rho_coulomb",p.rho_coulomb); assignin(wks,"enable_coulomb_cart_friction",1);
assignin(wks,"voltage_scale",p.voltage_scale); assignin(wks,"voltage_bias",p.voltage_bias);
assignin(wks,"voltage_rate_limit",p.voltage_rate_limit); assignin(wks,"voltage_deadzone",p.voltage_deadzone);
assignin(wks,"use_sensor_effects",0); assignin(wks,"action_delay_steps",0); assignin(wks,"obs_delay_steps",0);
assignin(wks,"encoder_res",p.encoder_res); assignin(wks,"vel_obs_gain",VEL_GAIN);
assignin(wks,"theta0",0); assignin(wks,"angle_limit_for_done",Inf); assignin(wks,"Vp",100);
if evalin(wks,"exist('W_ENERGY','var')"), assignin(wks,"W_ENERGY",0.0); end

%% Observation and action limits
obsInfo = rlNumericSpec([obsDim 1],"LowerLimit",[-p.xmax;-inf;-1;-1;-inf],"UpperLimit",[p.xmax;inf;1;1;inf]); obsInfo.Name="obs";
actInfo = rlNumericSpec([actDim 1],"LowerLimit",uMin,"UpperLimit",uMax); actInfo.Name="voltage_V";

%% Output folders
outDir = fullfile(here,'run'); ckptDir = fullfile(outDir,'checkpoints');
if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(ckptDir,'dir'), mkdir(ckptDir); end
fprintf('Output folder: %s\n', outDir);

%% New agent, or load the last stage
if isempty(START_STAGE) || START_STAGE<=1
    agent = buildAgent(obsInfo, actInfo, obsDim, actDim, Ts, actScale, device);
else
    prev = START_STAGE-1; f = fullfile(outDir, sprintf('stage%d_agent.mat', prev));
    assert(isfile(f), 'Agent file not found: %s', f);
    S = load(f); agent = S.agent;
    fprintf('Loaded %s, starting at stage %d\n', f, START_STAGE);
end
fprintf('Actor parameters: %d\n', sum(cellfun(@numel,agent.getActor().Learnables)));
env = rlSimulinkEnv(mdl,char(mdl+"/RL Agent"),obsInfo,actInfo); env.UseFastRestart='off';

% QUICK_TEST=1: quick test with 1% of the episodes
QUICK = 0; v_=getenv('QUICK_TEST'); if ~isempty(v_), QUICK=str2double(v_); end
ep = @(n) max(1, round(n * (1 - 0.99*(QUICK>0))));
runStage = @(N) isempty(START_STAGE) || START_STAGE<=N;

%% S1: balance, small kicks
if runStage(1)
    fprintf('\n--- Stage 1: balance (%d episodes, 6 s each) ---\n', ep(1500));
    set_param(mdl,'StopTime','6'); assignin(wks,'W_ENERGY',0.0);
    env.ResetFcn = @(in) resetBalance(in, mdl, VEL_GAIN);
    setStageOptions(agent, 3e-4, 1e-3, 0.4, 0.05, 5e-5, 500, 0.99);
    train(agent, env, makeTrainOptions(ep(1500), ceil(6/Ts), ckptDir, 100, device, QUICK));
    save(fullfile(outDir,'stage1_agent.mat'),'agent'); quickTest(env,agent,wks,mdl,'stage 1');
end

%% S2: start 45-140 deg from upright, pump energy
if runStage(2)
    fprintf('\n--- Stage 2: pump energy (%d episodes, 12 s each) ---\n', ep(1500));
    set_param(mdl,'StopTime','12'); assignin(wks,'W_ENERGY',1.0);
    env.ResetFcn = @(in) resetEnergy(in, mdl, VEL_GAIN, p.Fc);
    setStageOptions(agent, 1e-4, 5e-4, 0.7, 0.10, 3e-5, 1000, 0.99);
    train(agent, env, makeTrainOptions(ep(1500), ceil(12/Ts), ckptDir, 100, device, QUICK));
    save(fullfile(outDir,'stage2_agent.mat'),'agent'); quickTest(env,agent,wks,mdl,'stage 2');
end

%% S3: swing up from the bottom
if runStage(3)
    fprintf('\n--- Stage 3: swing-up (%d episodes, 12 s each) ---\n', ep(2000));
    set_param(mdl,'StopTime','12'); assignin(wks,'W_ENERGY',1.0);
    env.ResetFcn = @(in) resetSwingUp(in, mdl, VEL_GAIN, p.Fc);
    setStageOptions(agent, 5e-5, 3e-4, 0.9, 0.10, 3e-5, 256, 0.995);
    train(agent, env, makeTrainOptions(ep(2000), ceil(12/Ts), ckptDir, 100, device, QUICK));
    save(fullfile(outDir,'stage3_agent.mat'),'agent'); quickTest(env,agent,wks,mdl,'stage 3');
end

%% S4: mixed tasks, random plant
if runStage(4)
    fprintf('\n--- Stage 4: mixed tasks, random plant (%d episodes, 12 s each) ---\n', ep(1500));
    set_param(mdl,'StopTime','12'); assignin(wks,'W_ENERGY',1.0);
    env.ResetFcn = @(in) resetMixed(in, mdl, VEL_GAIN);
    setStageOptions(agent, 5e-5, 5e-4, 0.4, 0.05, 3e-5, 256, 0.995);
    train(agent, env, makeTrainOptions(ep(1500), ceil(12/Ts), ckptDir, 100, device, QUICK));
    save(fullfile(outDir,'stage4_agent.mat'),'agent'); quickTest(env,agent,wks,mdl,'stage 4');
end

%% S5: smoother balance (optional)
if runStage(5) && RUN_STAGE5
    fprintf('\n--- Stage 5: smoother balance (%d episodes, 8 s each) ---\n', ep(300));
    writeReward(mdl, buildReward(10,1.5,0.28,0.15));   % bigger penalty for fast changes at the top
    set_param(mdl,'StopTime','8'); assignin(wks,'W_ENERGY',1.0);
    env.ResetFcn = @(in) resetSmooth(in, mdl, VEL_GAIN);
    setStageOptions(agent, 3e-5, 3e-4, 0.1, 0.05, 0, 256, 0.995);
    train(agent, env, makeTrainOptions(ep(300), ceil(8/Ts), ckptDir, 50, device, QUICK));
    save(fullfile(outDir,'stage5_agent.mat'),'agent'); quickTest(env,agent,wks,mdl,'stage 5');
end

%% Save policy for the rig
exportPolicy(agent, here, outDir);
fprintf('\nTraining done.\n');

%% Local functions
function agent = buildAgent(obsInfo, actInfo, obsDim, actDim, Ts, actScale, device)
    actorLG = layerGraph([
        featureInputLayer(obsDim,"Normalization","none","Name","observation")
        fullyConnectedLayer(400,"Name","fc1"); reluLayer("Name","relu1")
        fullyConnectedLayer(300,"Name","fc2"); reluLayer("Name","relu2")
        fullyConnectedLayer(actDim,"Name","fc3"); tanhLayer("Name","tanh")
        functionLayer(@scaleToVoltage,"Name","scale_to_voltage","Formattable",true)]);
    actor = rlContinuousDeterministicActor(dlnetwork(actorLG), obsInfo, actInfo, "ObservationInputNames","observation");
    try actor.UseDevice = device; catch; end
    criticLG = layerGraph([
        concatenationLayer(1,2,"Name","concat_obs_act")
        fullyConnectedLayer(400,"Name","cfc1"); reluLayer("Name","crelu1")
        fullyConnectedLayer(300,"Name","cfc2"); reluLayer("Name","crelu2")
        fullyConnectedLayer(1,"Name","q_value")]);
    criticLG = addLayers(criticLG, featureInputLayer(obsDim,"Name","observation"));
    criticLG = addLayers(criticLG, featureInputLayer(actDim,"Name","action"));
    criticLG = connectLayers(criticLG,"observation","concat_obs_act/in1");
    criticLG = connectLayers(criticLG,"action","concat_obs_act/in2");
    c1 = rlQValueFunction(dlnetwork(criticLG),obsInfo,actInfo,"ObservationInputNames","observation","ActionInputNames","action");
    c2 = rlQValueFunction(dlnetwork(criticLG),obsInfo,actInfo,"ObservationInputNames","observation","ActionInputNames","action");
    try c1.UseDevice=device; c2.UseDevice=device; catch; end
    ao = rlTD3AgentOptions("SampleTime",Ts,"DiscountFactor",0.99,"MiniBatchSize",256, ...
        "ExperienceBufferLength",2e6,"PolicyUpdateFrequency",2,"TargetUpdateFrequency",2,"TargetSmoothFactor",0.005);
    ao.ActorOptimizerOptions = rlOptimizerOptions("LearnRate",3e-4);
    for i=1:2, ao.CriticOptimizerOptions(i)=rlOptimizerOptions("LearnRate",1e-3); end
    ao.ExplorationModel.StandardDeviation=0.4; ao.ExplorationModel.StandardDeviationMin=0.05; ao.ExplorationModel.StandardDeviationDecayRate=5e-5;
    try ao.ExplorationModel.LowerLimit=-0.3*actScale; ao.ExplorationModel.UpperLimit=0.3*actScale; catch; end
    ao.TargetPolicySmoothModel.StandardDeviation=0.1*actScale; ao.TargetPolicySmoothModel.StandardDeviationMin=0.1*actScale; ao.TargetPolicySmoothModel.StandardDeviationDecayRate=0;
    try ao.TargetPolicySmoothModel.LowerLimit=-0.3*actScale; ao.TargetPolicySmoothModel.UpperLimit=0.3*actScale; catch; end
    ao.NumWarmStartSteps=500;
    agent = rlTD3Agent(actor,[c1 c2],ao);
end

function configureModelSampleTime(mdl, Ts)
    % set Ts on all sampled blocks
    s=num2str(Ts); set_param(mdl,'FixedStep',s);
    for zz=find_system(char(mdl),'LookUnderMasks','on','FollowLinks','on','BlockType','ZeroOrderHold')',set_param(zz{1},'SampleTime',s);end
    b=find_system(char(mdl),'LookUnderMasks','on','FollowLinks','on','BlockType','DiscreteTransferFcn'); for i=1:numel(b),set_param(b{i},'SampleTime',s);end
    for dd=find_system(char(mdl),'LookUnderMasks','on','FollowLinks','on','Name','Observation Delay')',try,set_param(dd{1},'SampleTime',s);catch,end,end
    for dd=find_system(char(mdl),'LookUnderMasks','on','FollowLinks','on','Name','Action Delay')',try,set_param(dd{1},'SampleTime',s);catch,end,end
end

function setStageOptions(agent, aLR, cLR, eV, eMin, eDec, warm, gam)
    % learn rates, exploration noise (V), warm-up steps, discount
    o=agent.AgentOptions; o.ActorOptimizerOptions.LearnRate=aLR;
    for ii=1:2, o.CriticOptimizerOptions(ii).LearnRate=cLR; end
    o.ExplorationModel.StandardDeviation=eV; o.ExplorationModel.StandardDeviationMin=eMin; o.ExplorationModel.StandardDeviationDecayRate=eDec;
    o.NumWarmStartSteps=warm; if nargin>=8&&~isempty(gam), o.DiscountFactor=gam; end
    agent.AgentOptions=o;
end

function t = makeTrainOptions(maxEps, maxSteps, ckptDir, ckptFreq, device, quick)
    % no plot in quick test or without desktop
    plt='training-progress'; if (nargin>=6 && quick>0) || ~usejava('desktop'), plt='none'; end
    t=rlTrainingOptions("MaxEpisodes",maxEps,"MaxStepsPerEpisode",maxSteps,"ScoreAveragingWindowLength",30, ...
        "StopTrainingCriteria","AverageReward","StopTrainingValue",1e6,"Plots",plt,"Verbose",true);
    t.SaveAgentCriteria="EpisodeFrequency"; t.SaveAgentValue=ckptFreq; t.SaveAgentDirectory=ckptDir;
    try t.UseDevice=device; catch; end
end

function rc = findRewardBlock(mdl)
    % find the reward block
    rt=sfroot; a=rt.find('-isa','Stateflow.EMChart'); rc=[];
    for i=1:numel(a), if contains(a(i).Path,char(mdl))&&contains(a(i).Path,'Environment')&&strcmp(a(i).Name,'Reward'), rc=a(i); break; end, end
end
function writeReward(mdl, code)
    % replace reward code and check it
    rc=findRewardBlock(mdl); assert(~isempty(rc),'Reward block not found'); rc.Script=code;
    assert(contains(code,'W_X = upright_blend *')&&~contains(code,'W_X = 5.0 + upright_blend'),'Reward check failed');
end

function code = buildReward(w_x_up, w_xdot_up, wdu_up, wthd_up)
    % reward code; most penalties only count near the top (upright_blend)
    bonus_delta = min((w_x_up-5.0)/5, 3.0); sigx = min(0.03, 0.03*(w_x_up-5.0)/10);
    L={ 'function r = reward_fcn(theta, thetadot, x, xdot, u, u_prev, W_ENERGY)', ...
    'OMEGA_MAX = 10.0; X_MAX = 0.407; V_MAX = 1.5; U_MAX = 6.0;','W_THETA = 3.0; W_U = 0.01; W_RAIL = 15.0;', ...
    'SIGMA_THETA = 0.15; SIGMA_XDOT = 0.30;','cos_th = cos(theta);','upright_blend = max(0, min(1, (-cos_th - 0.5) / 0.5));', ...
    sprintf('W_X = upright_blend * %.4f;', w_x_up), sprintf('W_XDOT = upright_blend * %.4f;', w_xdot_up), ...
    sprintf('W_THETADOT = upright_blend * %.4f;', wthd_up), sprintf('W_DU = 0.02 + upright_blend * %.4f;', wdu_up), ...
    sprintf('BONUS = 1.0 + upright_blend * %.4f;', bonus_delta), sprintf('SIGMA_X = 0.05 - upright_blend * %.4f;', sigx), ...
    'c_theta = (1 + cos_th) / 2;','c_thetadot = (thetadot / OMEGA_MAX)^2;','c_x = (x / X_MAX)^2;','c_xdot = (xdot / V_MAX)^2;', ...
    'c_u = (u / U_MAX)^2;','c_du = ((u - u_prev) / U_MAX)^2;','RAIL_SOFT = 0.30;','x_abs = abs(x);', ...
    'if x_abs > RAIL_SOFT','    c_rail = ((x_abs - RAIL_SOFT) / (X_MAX - RAIL_SOFT))^2;','else','    c_rail = 0;','end', ...
    'theta_err = atan2(sin(theta - pi), cos(theta - pi));','goal_err_sq = (theta_err / SIGMA_THETA)^2 + (x / SIGMA_X)^2 + (xdot / SIGMA_XDOT)^2;', ...
    'b = BONUS * exp(-0.5 * goal_err_sq);','if W_ENERGY > 0','    E_pe = (1 - cos(theta)) / 2;','    E_ke = 0.5 * (thetadot / OMEGA_MAX)^2;', ...
    '    E_norm = E_pe + E_ke;','    r_energy = -W_ENERGY * (1.0 - E_norm)^2;','else','    r_energy = 0;','end', ...
    'r = -W_THETA*c_theta - W_THETADOT*c_thetadot - W_X*c_x - W_XDOT*c_xdot - W_U*c_u - W_DU*c_du - W_RAIL*c_rail + b + r_energy;','end'};
    code = strjoin(L, newline);
end

% Reset functions: start state and plant for each episode
function in = setPlant(in, mdl, Fc, Kd, dz, vs, vb, sensor, vg)
    in=setVariable(in,'Fc_cart',Fc,'Workspace',mdl); in=setVariable(in,'Kd',Kd,'Workspace',mdl);
    in=setVariable(in,'voltage_deadzone',dz,'Workspace',mdl); in=setVariable(in,'voltage_scale',vs,'Workspace',mdl);
    in=setVariable(in,'voltage_bias',vb,'Workspace',mdl); in=setVariable(in,'use_sensor_effects',sensor,'Workspace',mdl);
    in=setVariable(in,'vel_obs_gain',vg,'Workspace',mdl);
end
function in = setStart(in, mdl, th, thd, x0, xd0)
    in=setVariable(in,'theta0',th,'Workspace',mdl); in=setVariable(in,'thetadot0',thd,'Workspace',mdl);
    in=setVariable(in,'x0',x0,'Workspace',mdl); in=setVariable(in,'xdot0',xd0,'Workspace',mdl);
end
function in = resetBalance(in, mdl, vg)   % near the top, small kick
    in=setPlant(in,mdl, 1.4+0.8*rand, 0.3*rand, 0.4*rand, 0.76+0.08*rand, -0.22+0.08*rand, 1, vg);
    th=pi+deg2rad(25)*(2*rand-1); thd=3*(2*rand-1); in=setStart(in,mdl,th,thd,0.06*randn,0.15*randn);
end
function in = resetEnergy(in, mdl, vg, Fc)   % 45-140 deg from upright
    in=setPlant(in,mdl, Fc, 0.02, 0.0, 0.80, -0.181, 0, vg);
    sgn=2*(rand>0.5)-1; sgn_d=2*(rand>0.5)-1; delta=deg2rad(45+95*rand);
    th=sgn*pi + sgn_d*delta; thd=0.5*randn; in=setStart(in,mdl,th,thd,0.05*randn,0.1*randn);
end
function in = resetSwingUp(in, mdl, vg, Fc)   % near the bottom (+-15 deg)
    in=setPlant(in,mdl, Fc, 0.02, 0.0, 0.80, -0.181, 0, vg);
    th=deg2rad(15)*(2*rand-1); thd=0.3*randn; in=setStart(in,mdl,th,thd,0.05*randn,0.1*randn);
end
function in = resetMixed(in, mdl, ~)   % mixed tasks, random plant
    vgr = 0.85 + 0.30*rand;   % keep this range small; a wide range stops the swing-up
    in=setPlant(in,mdl, 1.0+2.0*rand, 1.5*rand, 1.0*rand, 0.72+0.16*rand, -0.27+0.17*rand, 1, vgr);
    r=rand;
    if r<0.5,      th=deg2rad(15)*(2*rand-1); thd=0.3*randn;             % swing-up
    elseif r<0.8,  th=pi+deg2rad(25)*(2*rand-1); thd=0.5*randn;         % balance
    else,          th=pi+deg2rad(15)*(2*rand-1); thd=(2+rand)*(2*(rand>0.5)-1);  % push at top
    end
    in=setStart(in,mdl,th,thd,0.08*randn,0.2*randn);
end
function in = resetSmooth(in, mdl, vg)   % top only, small pushes
    in=setPlant(in,mdl, 1.4+0.8*rand, 0.3*rand, 0.3*rand, 0.76+0.08*rand, -0.22+0.08*rand, 1, vg);
    r=rand;
    if r<0.7,  th=pi+deg2rad(20)*(2*rand-1); thd=0.5*randn;
    else,      th=pi+deg2rad(15)*(2*rand-1); thd=(2+rand)*(2*(rand>0.5)-1);
    end
    in=setStart(in,mdl,th,thd,0.06*randn,0.15*randn);
end

function quickTest(env, agent, wks, mdl, tag)
    % quick test: swing up, then hold at top
    rf=env.ResetFcn; A=copy(agent); try A.UseExplorationPolicy=false; catch, end
    assignin(wks,'Fc_cart',1.8); assignin(wks,'Kd',0.05); assignin(wks,'voltage_deadzone',0.2);
    assignin(wks,'voltage_scale',0.80); assignin(wks,'voltage_bias',-0.181); assignin(wks,'use_sensor_effects',0); assignin(wks,'vel_obs_gain',1.0);
    so=rlSimulationOptions('MaxSteps',1000);
    env.ResetFcn=@(in)setStart(in,mdl,0.06,0,0,0); set_param(mdl,'StopTime','20');
    pk=NaN; try, e=sim(env,A,so); o=squeeze(e.Observation.obs.Data); pk=max(abs(atan2(o(3,:),o(4,:)))); catch, end
    env.ResetFcn=@(in)setStart(in,mdl,pi,0,0,0); set_param(mdl,'StopTime','12');
    bal=NaN; try, e=sim(env,A,rlSimulationOptions('MaxSteps',600)); o=squeeze(e.Observation.obs.Data); ue=atan2(-o(3,:),-o(4,:)); h=round(numel(ue)/2):numel(ue); bal=mean(abs(ue(h))<0.3); catch, end
    fprintf('  Quick test after %s: swing-up peak %.2f rad (upright = 3.14), balance held %.0f%%\n', tag, pk, 100*bal);
    env.ResetFcn=rf;
end

function exportPolicy(agent, ~, outDir)
    % save the policy file for the rig, no template needed
    td=tempname; mkdir(td); oldpwd=pwd; cu=onCleanup(@()cd(oldpwd)); %#ok<NASGU>
    pre=find_system('type','block_diagram'); cd(td);
    generatePolicyBlock(agent);                         % makes blockAgentData.mat
    post=find_system('type','block_diagram');
    hw=load(fullfile(td,'blockAgentData.mat'));
    cd(oldpwd);
    for m=reshape(setdiff(post,pre),1,[]), try, bdclose(m{1}); catch, end, end  % close generated models
    model=hw.model; modelData=hw.modelData; metaData=hw.metaData; %#ok<NASGU>
    policyData=hw.policyData; policyData.SampleTime=agent.SampleTime; %#ok<NASGU>
    of=fullfile(outDir,'swingup_policy.mat');
    save(of,'model','modelData','policyData','metaData');
    % check the saved network matches the actor
    net=getModel(getActor(agent)); me=0; rs=rng; rng(42);
    for i=1:50, o=single(randn(5,1)); o(1)=o(1)*0.3; o(3)=tanh(o(3)); o(4)=tanh(o(4));
        me=max(me, abs(double(extractdata(predict(net,dlarray(o,'CB'))))-double(extractdata(predict(model,dlarray(o,'CB')))))); end
    rng(rs);
    fprintf('Policy saved: %s (check: max error %.2e V, %s)\n', of, me, ternary(me<1e-6,'PASS','FAIL'));
end
function s = ternary(c,a,b), if c, s=a; else, s=b; end, end
