for i in {1..10}
  do 
     echo "Iteration number: $i" >> ResultsTrained
     python3 DCGAN.py
     python3 LoadedModel.py >> ResultsTrained
 done
