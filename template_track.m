clc;
close all;
clear;

%%
videoFile = 'IMG_2104_1mM_3s_crop.mp4'; 
v = VideoReader(videoFile);

% 讀取第一幀
firstFrame = readFrame(v);

% 顯示第一幀以框選 ROI
figure;
imshow(firstFrame);
title('框選要追蹤的ROI (綠色螢光點)');
roi = drawrectangle();
roiPosition = roi.Position;  % [x y width height]

% 從 ROI 取得 template
template = firstFrame(round(roiPosition(2)):round(roiPosition(2)+roiPosition(4)), ...
                     round(roiPosition(1)):round(roiPosition(1)+roiPosition(3)), :);

% 計算模板的特徵
templateGreen = template(:,:,2);
templateFeatures.intensity = mean(templateGreen(:));
templateFeatures.std = std(double(templateGreen(:)));
templateFeatures.size = sum(templateGreen(:) > mean(templateGreen(:)));
templateFeatures.maxIntensity = max(templateGreen(:));

% 初始化追蹤結果
numFrames = v.NumFrames;
trackingResults = zeros(numFrames, 2);
trackingResults(1,:) = [roiPosition(1)+roiPosition(3)/2, roiPosition(2)+roiPosition(4)/2];

% 基礎權重設定
weights.correlation = 0.4;
weights.intensity = 0.3;
weights.size = 0.15;
weights.motion = 0.15;

% 自適應參數設定
adaptiveWeights = struct();
adaptiveWeights.base = weights;
adaptiveWeights.current = weights;

% 運動預測參數
alpha = 0.3;  % 速度平滑因子(降低以減少歷史速度的影響)
velocity = [0, 0];
lastDirection = [0, 0];
directionChangeThreshold = 0.7;  % 方向變化閾值
randomFactor = 0.2;  % 隨機擾動因子

% 搜索範圍參數
extensionFactor = 1.2;
searchWidth = round(roiPosition(3) * extensionFactor);
searchHeight = round(roiPosition(4) * extensionFactor);

% 顯示設定
v.CurrentTime = 0;
h = figure;
set(h, 'Position', [100 100 800 400]);

frameCount = 1;
while hasFrame(v)
    % 讀取當前幀
    currentFrame = readFrame(v);
    frameCount = frameCount + 1;
    
    % 獲取前一幀的位置
    prevPos = trackingResults(frameCount-1, :);
    
    % 預測位置（使用自適應速度估計）
    if frameCount > 2
        currentVelocity = trackingResults(frameCount-1,:) - trackingResults(frameCount-2,:);
        
        % 計算方向變化
        if norm(lastDirection) > 0
            directionChange = dot(currentVelocity, lastDirection) / ...
                             (norm(currentVelocity) * norm(lastDirection));
        else
            directionChange = 1;
        end
        
        % 根據方向變化調整速度估計和權重
        if directionChange < directionChangeThreshold
            % 方向變化大時，降低速度預測的影響
            velocity = 0.1 * currentVelocity;
            % 調整權重，增加相關性和強度的重要性
            adaptiveWeights.current.motion = 0.05;
            adaptiveWeights.current.correlation = 0.5;
            adaptiveWeights.current.intensity = 0.35;
            adaptiveWeights.current.size = 0.1;
        else
            % 方向穩定時，使用正常速度估計
            velocity = alpha * velocity + (1-alpha) * currentVelocity;
            % 恢復基礎權重
            adaptiveWeights.current = adaptiveWeights.base;
        end
        
        % 加入隨機擾動
        randomOffset = randomFactor * (rand(1,2) - 0.5) .* [searchWidth, searchHeight];
        predictedPos = trackingResults(frameCount-1,:) + velocity + randomOffset;
        
        % 更新上一幀的方向
        lastDirection = currentVelocity;
    else
        predictedPos = prevPos;
    end
    
    % 計算搜索區域（根據預測位置）
    searchX = max(1, round(predictedPos(1) - searchWidth/2));
    searchY = max(1, round(predictedPos(2) - searchHeight/2));
    actualSearchWidth = min(size(currentFrame,2) - searchX, searchWidth);
    actualSearchHeight = min(size(currentFrame,1) - searchY, searchHeight);
    
    % 截取搜索區域
    searchRegion = currentFrame(searchY:searchY+actualSearchHeight, ...
                               searchX:searchX+actualSearchWidth, :);
    
    searchRegionGreen = searchRegion(:,:,2);
    
    % 計算相關係數矩陣
    c = normxcorr2(templateGreen, searchRegionGreen);
    
    % 找到所有可能的匹配位置
    threshold = 0.5;  % 相關係數閾值
    peakMask = c > threshold*max(c(:));
    [peakRows, peakCols] = find(peakMask);
    
    bestScore = -inf;
    bestPos = predictedPos;
    
    % 評估每個候選位置
    for i = 1:length(peakRows)
        y = peakRows(i);
        x = peakCols(i);
        
        % 計算ROI在原圖中的位置
        roiY = searchY + y - size(template,1);
        roiX = searchX + x - size(template,2);
        
        % 確保位置在影像範圍內
        if roiY > 0 && roiX > 0 && ...
           roiY + size(template,1) <= size(currentFrame,1) && ...
           roiX + size(template,2) <= size(currentFrame,2)
            
            % 提取候選區域
            candidateRoi = currentFrame(roiY:roiY+size(template,1)-1, ...
                                      roiX:roiX+size(template,2)-1, 2);
            
            % 計算特徵
            roiFeatures.intensity = mean(candidateRoi(:));
            roiFeatures.std = std(double(candidateRoi(:)));
            roiFeatures.size = sum(candidateRoi(:) > mean(candidateRoi(:)));
            roiFeatures.maxIntensity = max(candidateRoi(:));
            
            % 計算各項得分
            corrScore = c(y,x);
            intensityScore = 1 - abs(roiFeatures.intensity - templateFeatures.intensity) / ...
                               max(templateFeatures.intensity, 1);
            sizeScore = 1 - abs(roiFeatures.size - templateFeatures.size) / ...
                           max(templateFeatures.size, 1);
            
            % 計算位置得分（基於預測位置）
            pos = [roiX + size(template,2)/2, roiY + size(template,1)/2];
            distToPredicted = norm(pos - predictedPos);
            motionScore = 1 / (1 + distToPredicted/searchWidth);
            
            % 使用自適應權重計算綜合得分
            totalScore = adaptiveWeights.current.correlation * corrScore + ...
                        adaptiveWeights.current.intensity * intensityScore + ...
                        adaptiveWeights.current.size * sizeScore + ...
                        adaptiveWeights.current.motion * motionScore;
            
            % 更新最佳位置
            if totalScore > bestScore
                bestScore = totalScore;
                bestPos = pos;
                globalXoffSet = roiX;
                globalYoffSet = roiY;
            end
        end
    end
    
    % 檢查追蹤是否可靠並進行恢復
    distance = norm(bestPos - prevPos);
    if distance > max(searchWidth, searchHeight)/2 || bestScore < 0.5
        % 增加隨機搜索
        randomSearch = rand(1,2) .* [searchWidth, searchHeight];
        predictedPos = prevPos + 0.3 * randomSearch;
        bestPos = predictedPos;
        globalXoffSet = predictedPos(1) - size(template,2)/2;
        globalYoffSet = predictedPos(2) - size(template,1)/2;
    end
    
    % 更新追蹤結果
    trackingResults(frameCount,:) = bestPos;
    
    % 顯示結果
    subplot(1,2,1);
    imshow(currentFrame);
    hold on;
    % 顯示搜索區域
    rectangle('Position', [searchX, searchY, actualSearchWidth, actualSearchHeight], ...
             'EdgeColor', 'y', 'LineWidth', 1, 'LineStyle', '--');
    % 顯示追蹤框
    rectangle('Position', [globalXoffSet, globalYoffSet, size(template,2), size(template,1)], ...
             'EdgeColor', 'r', 'LineWidth', 2);
    % 顯示預測位置
    plot(predictedPos(1), predictedPos(2), 'b+', 'MarkerSize', 10);
    hold off;
    title(sprintf('Frame %d (Score: %.2f)', frameCount, bestScore));
    
    subplot(1,2,2);
    plot(trackingResults(1:frameCount,1), trackingResults(1:frameCount,2), 'g-', 'LineWidth', 2);
    axis ij;
    grid on;
    title('追蹤軌跡');
    xlabel('X');
    ylabel('Y');
    
    drawnow;
end

% 顯示完整軌跡
figure;
plot(trackingResults(:,1), trackingResults(:,2), 'g-', 'LineWidth', 2);
axis ij;
grid on;
title('完整追蹤軌跡');
xlabel('X');
ylabel('Y');
