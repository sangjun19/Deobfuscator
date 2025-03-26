// Repository: maxdiefenbach/trabecular_bone_QSM
// File: code/common/functions/FiniteDifferences/compute_backward_difference_gradient.m

function grad = compute_backward_difference_gradient(array, voxelSize_mm)
    
    if nargin < 2
        voxelSize_mm = [1, 1, 1];
    end
    
    switch ndims(array)
      case 3
        grad = compute_backward_difference_gradient_3d(array, voxelSize_mm);
      case 2
        grad = compute_backward_difference_gradient_2d(array, voxelSize_mm);
      case 1
        grad = compute_backward_difference_gradient_1d(array, voxelSize_mm);
    end    
    
end


function grad  = compute_backward_difference_gradient_3d(array, voxelSize_mm)
    grad = repmat(zeros('like', array), [1, 1, 1, 3]);
    
    grad(2:end, :, :, 1) = (array(2:end, :, :) - array(1:(end-1), :, :)) / voxelSize_mm(1);
    grad(1, :, :, 1) = (array(1, :, :) - array(end, :, :)) / voxelSize_mm(1);
    
    grad(:, 2:end, :, 2) = (array(:, 2:end, :) - array(:, 1:(end-1), :)) / voxelSize_mm(2);
    grad(:, 1, :, 2) = (array(:, 1, :) - array(:, end, :)) / voxelSize_mm(2);
    
    grad(:, :, 2:end, 3) = (array(:, :, 2:end) - array(:, :, 1:(end-1))) / voxelSize_mm(3);
    grad(:, :, 1, 3) = (array(:, :, 1) - array(:, :, end)) / voxelSize_mm(3);
end


function grad  = compute_backward_difference_gradient_2d(array, voxelSize_mm)
    grad = repmat(zeros('like', array), [1, 1, 2]);
    
    grad(2:end, :, 1) = (array(2:end, :) - array(1:(end-1), :)) / voxelSize_mm(1);
    grad(1, :, 1) = (array(1, :) - array(end, :)) / voxelSize_mm(1);
    
    grad(:, 2:end, 2) = (array(:, 2:end) - array(:, 1:(end-1))) / voxelSize_mm(2);
    grad(:, 1, 2) = (array(:, 1) - array(:, end)) / voxelSize_mm(2);
end


function grad  = compute_backward_difference_gradient_1d(array, voxelSize_mm)
    grad = zeros('like', array);
    grad(2:end) = (array(2:end) - array(1:(end-1))) / voxelSize_mm(1);
    grad(1) = (array(1) - array(end)) / voxelSize_mm(1);
end

