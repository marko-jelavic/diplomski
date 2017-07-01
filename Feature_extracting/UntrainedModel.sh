for i in {1..10}
  do 
     echo "Iteration number: $i" >> ResultsUntrained
     python3 UntrainedModel.py >> ResultsUntrained
 done
