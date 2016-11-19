function img = viewColorPatches(inpPatches,patchSize)
    % constrast stretching for each patch
    
    numPatches = size(inpPatches,2);

    temp = patchSize^2;



    % Sort patches according to variance
    var_patches = var(inpPatches(1:temp,:)) + ...
                    var(inpPatches(temp+1:temp*2,:)) + ...
                    var(inpPatches(2*temp+1:3*temp));
    [~, idx] = sort(var_patches, 'descend');

    inpPatches = inpPatches(:, idx);
    
    imSize = patchSize*floor(sqrt(numPatches));
    
    for i=1:numPatches
        minIntensity = min(min(inpPatches(:,i)));
        maxIntensity = max(max(inpPatches(:,i)));
        
        inpPatches(:,i) = (inpPatches(:,i)-minIntensity)/(maxIntensity-minIntensity);
    end
    
    imgRed = col2im(inpPatches(1:temp,:), ...
                [patchSize patchSize],[imSize imSize],'distinct');
    imgGreen = col2im(inpPatches(temp+1:temp*2,:),...
                [patchSize patchSize],[imSize imSize],'distinct');
    imgBlue = col2im(inpPatches(2*temp+1:temp*3,:),...
                [patchSize patchSize],[imSize imSize],'distinct');
    
    img = zeros(size(imgRed,1),size(imgRed,2),3);
    img(:,:,1) = imgRed;
    img(:,:,2) = imgGreen;
    img(:,:,3) = imgBlue;
