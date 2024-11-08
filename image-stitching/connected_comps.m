function cc = connected_comps(adj, images)

N = size(adj, 1);
remaining = 1:N;
cluster = [];
cc = {};

while ~isempty(remaining)
    stack = remaining(1);
    
    while ~isempty(stack)
        i = stack(1);
        stack(1) = [];
        cluster(end+1) = i;
        for j = (i+1):N
            if adj(i,j) == 1
                stack(end+1) = j;
            end
        end
    end
    
    % Remove each index in cluster from remaining
    for z = cluster
        remaining(remaining == z) = [];
    end

    % Register the cluster as a connected component
    m = length(cc)+1;
    if length(cluster) > 1
        for i = 1:length(cluster)
            cc{m}{i} = images{cluster(i)};
        end
    end
    
    cluster = [];
end

end

