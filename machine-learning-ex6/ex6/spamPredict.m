function pred = spamPredict(model, filename)

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
pred = svmPredict(model, x);

if(pred == 1)
  printf("%s == SPAM\n", filename);
else
  printf("%s == LEGIT\n", filename);
end

end