for i in {1..10}
do
  echo "Iteration $i"
  python3 Discriminator10.py >> Results10
  python3 Discriminator100.py >> Results100
  python3 DiscriminatorEntire.py >> ResultsEntire
done
