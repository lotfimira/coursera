function idx = closestPoint(x, centroids)
  
k = size(centroids, 1);
  
min_distance = realmax;
min_idx = -1;
  
for i = 1 : k
  
  distance = norm(x - centroids(i,:));
  
  if distance < min_distance
    min_distance = distance;
    min_idx = i;
  end  

end
  
idx = min_idx;
  
end