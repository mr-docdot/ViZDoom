mkdir -p out

# Create exactly 100 valid maps for training
for i in {1..204}
do
    python map_create.py -s $i -d regular -m none -z none
done
