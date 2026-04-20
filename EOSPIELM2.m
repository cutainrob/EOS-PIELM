function [TrainingTime, Acc_stream, Rec_stream, Spe_stream, Gmean_stream, D_stream, AUC_stream] = ...
         EAFWOSELM(dataFile, classFile, N0, Block, TimePoints)
% EAFWOSELM 流式预测-训练框架下的在线分类算法，实现自适应遗忘因子加权OSELM集成
%（已改进：混淆矩阵与类不平衡处理已泛化到多类情况）

    %% 0. 默认参数处理
    if nargin < 1 || isempty(dataFile)
        dataFile = 'NEweather_data.csv';
    end
    if nargin < 2 || isempty(classFile)
        classFile = 'NEweather_class.csv';
    end
    if nargin < 3 || isempty(N0)
        N0 = 200;
    end
    if nargin < 4 || isempty(Block)
        Block = 200;
    end
    if nargin < 5 || isempty(TimePoints)
        TimePoints = [];
    end

    %% 1. 数据加载与预处理
    X_raw = readmatrix(dataFile);      % 特征矩阵 N×d
    Y_raw = readmatrix(classFile);     % 原始标签向量
    [Y, labels] = check_labels(Y_raw);
    nClass = numel(labels);
    if nClass < 2, error('需至少两类标签'); end

    mu    = mean(X_raw(1:min(N0,end),:), 1);
    sigma = std(X_raw(1:min(N0,end),:), [], 1);
    X_norm = (X_raw - mu) ./ (sigma + (sigma==0));
    X = 1./(1 + exp(-X_norm));   % 干涉层 sigmoid 映射（可按需修改）
   %X=X_norm; 
   N = size(X,1);

    %% 2. 算法参数与基分类器初始化
    M = 12;
    input_dim = size(X,2);
    hidden_neurons = [input_dim, 2*input_dim, 3*input_dim, 4*input_dim];
    action_funcs   = {'sig','softplus','tanh'};
    % 计算初始化样本的类分布并进行少类加权（泛化到多类）
    init_n = min(N0,N);
    init_labels = Y(1:init_n);
    counts_init = histcounts(init_labels, 1:(nClass+1));
    minCount = min(counts_init);
    % 若所有类均有样本，则 minClassIdx 为第一个最少类索引
    minClassIdx = find(counts_init==minCount,1);
    IR_init = max(counts_init)/max(minCount,1); % 不会除以0

    for m = 1:M
        hn = hidden_neurons(mod(m-1,4)+1);
        af = action_funcs{ceil(m/4)};
        % 随机初始化
        C_m(m).IW     = randn(hn, input_dim)*sqrt(2/input_dim);
        C_m(m).Bias   = zeros(1, hn);
        C_m(m).lambda = 0.999;
        C_m(m).weight = 1/M;
        C_m(m).CM     = zeros(nClass, nClass); % 每个基分类器的混淆矩阵（nClass x nClass）
        C_m(m).CF     = 1.0;

        % 初始化样本权重 W0：对最少类样本放大权重
        W0 = ones(init_n,1);
        W0(init_labels==minClassIdx) = IR_init;

        H0 = activate(X(1:init_n,:), C_m(m).IW, C_m(m).Bias, af);
        C = 0.1;
        C_m(m).P = inv(H0'*diag(W0)*H0 + (1/C)*eye(hn));
        Y0_onehot = full(ind2vec(Y(1:init_n)', nClass))';
        C_m(m).beta = C_m(m).P * (H0'*diag(W0)*Y0_onehot);
    end

    EC_CM      = zeros(nClass, nClass); % 全局混淆矩阵
    Gmax       = 0;
    lambda_glb = 0.999;
    mu_thr     = 0.30;

    %% 3. 流式在线学习
    TrainingTimer = tic;
    corrects = 0; total = 0;
    nStream = N - min(N0,N);
    Acc_stream   = zeros(max(nStream,0),1);
    Rec_stream   = zeros(max(nStream,0),1);
    Spe_stream   = zeros(max(nStream,0),1);
    Gmean_stream = zeros(max(nStream,0),1);
    D_stream     = zeros(max(nStream,0),1);
    scores       = zeros(max(nStream,0),1); % 仅二分类时会被赋值
    idx_stream   = 0;

    % 将“正类索引”定义为 nClass（兼容原来二分类把 1/0 映射到 1/2 的情形）
    posIdx = nClass;

    for i = min(N0,N)+1 : Block : N
        for j = i : min(i+Block-1, N)
            idx_stream = idx_stream + 1;
            x_j = X(j,:); y_j = Y(j);

            % 预测（每个基分类器投票并更新其局部混淆）
            preds = zeros(M,1);
            for m = 1:M
                H_j = activate(x_j, C_m(m).IW, C_m(m).Bias, action_funcs{ceil(m/4)});
                out = H_j * C_m(m).beta;      % 1 x nClass
                [~, preds(m)] = max(out);     % 预测类索引（1..nClass）

                % 指数衰减更新基分类器局部混淆矩阵
                C_m(m).CM = C_m(m).lambda * C_m(m).CM;
                C_m(m).CM(y_j, preds(m)) = C_m(m).CM(y_j, preds(m)) + 1;
            end

            % 加权投票（基于当前各基分类器权重）
            weights_vec = [C_m.weight];
            votes = accumarray(preds, weights_vec', [nClass,1])';  % 1 x nClass
            [~, y_hat] = max(votes);

            % 更新全局混淆矩阵（指数衰减）
            EC_CM = lambda_glb * EC_CM;
            EC_CM(y_j, y_hat) = EC_CM(y_j, y_hat) + 1;

            % 统计累计准确率
            total = total + 1;
            if y_hat==y_j, corrects = corrects + 1; end
            Acc_stream(idx_stream) = corrects/total;

            % 计算 TP,FN,FP,TN（按类）
            TP = diag(EC_CM);                       % nClass x 1
            FN = sum(EC_CM,2) - TP;                 % nClass x 1
            FP = sum(EC_CM,1)' - TP;               % nClass x 1
            total_samples = sum(EC_CM(:));
            TN = total_samples - TP - FN - FP;     % nClass x 1

            if nClass==2
                Rec_cur = TP(posIdx)/(TP(posIdx)+FN(posIdx)+eps);
                Spe_cur = TN(posIdx)/(TN(posIdx)+FP(posIdx)+eps);
            else
                Rec_cur = mean(TP./(TP+FN+eps));
                Spe_cur = mean(TN./(TN+FP+eps));
            end
            Rec_stream(idx_stream)   = Rec_cur;
            Spe_stream(idx_stream)   = Spe_cur;
            Gnow = sqrt(Rec_cur * Spe_cur);
            Gmean_stream(idx_stream)= Gnow;
            D_stream(idx_stream)     = abs(Rec_cur - Spe_cur);
            if nClass==2
                scores(idx_stream)   = votes(posIdx)/sum(weights_vec + eps);
            end

            % 漂移检测（保持原逻辑）
            if Gnow > Gmax, Gmax = Gnow; end
            CDI = Gnow/(Gmax+eps);
            lambda_glb = (CDI<=0.9) * (0.9+0.1*CDI) + (CDI>0.9)*0.999;

            % 在线更新：计算当前各类计数，找出最少类用于加权（泛化）
            counts = histcounts(Y(1:j), 1:(nClass+1));
            minCount = min(counts);
            minClassIdx = find(counts==minCount,1);
            IR_cur = max(counts)/max(minCount,1);

            % 更新每个基分类器参数
            for m = 1:M
                TP_m = C_m(m).CM(y_j,preds(m));
                FN_m = sum(C_m(m).CM(y_j,:)) - TP_m;
                FP_m = sum(C_m(m).CM(:,preds(m))) - TP_m;
                % 基分类器的 TP_m, FN_m, FP_m 已计算
                Rec_m = TP_m/(TP_m+FN_m+eps);
                % 修正：TN 计算正确 -> specificity 计算分母应为 (TN + FP) = total - TP - FN
                total_m = sum(C_m(m).CM(:));
                TN_m = total_m - TP_m - FN_m - FP_m;
                Spe_m = TN_m/(TN_m+FP_m+eps);
                Dval = abs(Rec_m - Spe_m);

                % 自适应 CF （保持原逻辑）
                if Dval >= mu_thr
                    C_m(m).CF = max(0.5, C_m(m).CF * 0.95);
                else
                    C_m(m).CF = min(2.0, C_m(m).CF * 1.05);
                end

                % 类别加权：如果样本属于当前最少类则放大权重 wk
                if y_j == minClassIdx
                    wk = IR_cur * C_m(m).CF;
                else
                    wk = 1;
                end

                H_j = activate(x_j, C_m(m).IW, C_m(m).Bias, action_funcs{ceil(m/4)});
                lam = C_m(m).lambda;
                Pp  = C_m(m).P / lam;
                denom = (lam/wk + H_j*Pp*H_j');
                Ki  = (Pp*H_j')/(denom + eps);  % hn x 1
                err = full(ind2vec(y_j,nClass))' - H_j*C_m(m).beta;

                C_m(m).P      = Pp - Ki*H_j*Pp;
                C_m(m).beta   = C_m(m).beta + Ki*wk*err;
                % 更新基分类器权重（保持原逻辑，但归一化可能需要后续处理）
                C_m(m).weight = 0.8*C_m(m).weight + 0.2*(preds(m)==y_j) + 0.1*(1-Dval);
                % 避免权重为负
                if C_m(m).weight < 0, C_m(m).weight = 0; end
            end
        end
    end

    TrainingTime = toc(TrainingTimer);

    %% 4. 流式 AUC（二分类）
    if nClass==2
        [~,~,~,AUC_stream] = perfcurve(Y(min(N0,N)+1:end), scores, posIdx);
    else
        AUC_stream = NaN;
    end

    %% 5. 绘图：Gmean 流式折线图
    if nStream>0
        figure;
        t = 1:nStream;
        plot(t, Gmean_stream, 'LineWidth', 1.2);
        hold on;
        if ~isempty(TimePoints)
            tp = TimePoints(TimePoints>=1 & TimePoints<=nStream);
            gm_tp = Gmean_stream(tp);
            plot(tp, gm_tp, 'ro', 'MarkerSize',6, 'LineWidth',1.5);
            for k = 1:length(tp)
                text(tp(k), gm_tp(k), sprintf('%.3f', gm_tp(k)), ...
                     'VerticalAlignment','bottom', 'HorizontalAlignment','right');
            end
        end
        hold off;
        xlabel('样本序号');
        ylabel('Gmean');
        title('分时 Gmean 指标折线图');
        grid on;
        legend({'Gmean\_stream', '标注点'}, 'Location', 'best');
    end

    %% 6. 控制台输出
    if nStream>0
        FinalAcc   = Acc_stream(end);
        FinalRec   = Rec_stream(end);
        FinalSpe   = Spe_stream(end);
        FinalGmean = Gmean_stream(end);
        FinalD     = D_stream(end);

        fprintf('\n====== 实时流式学习结果 ======\n');
        fprintf('训练时间: %.2f 秒\n', TrainingTime);
        fprintf('最终累积准确率: %.2f%%\n', FinalAcc*100);
        fprintf('最终累积召回率: %.2f%%\n', FinalRec*100);
        fprintf('最终累积特异度: %.2f%%\n', FinalSpe*100);
        fprintf('最终累积Gmean: %.2f%%\n', FinalGmean*100);
        fprintf('最终分类距离 D: %.2f%%\n', FinalD*100);
        if ~isnan(AUC_stream)
            fprintf('流式AUC: %.3f\n', AUC_stream);
        end
        fprintf('==========================\n');
    else
        fprintf('没有流式样本（N <= N0），仅完成初始化。\n');
    end
end

%% 辅助函数
function [Y, labels] = check_labels(Y_raw)
    labels = unique(Y_raw);
    if any(labels<1) || any(mod(labels,1)~=0)
        Y = Y_raw - min(Y_raw) + 1;
        labels = unique(Y);
    else
        Y = Y_raw;
    end
end

function H = activate(X, IW, Bias, func)
    Z = X*IW' + Bias;
    switch func
        case 'sig'
            H = 1./(1+exp(-max(min(Z,20),-20)));
        case 'softplus'
            H = log(1+exp(-abs(Z))) + max(Z,0);
        case 'tanh'
            H = tanh(Z);
        otherwise
            error('Unsupported activation');
    end
end
