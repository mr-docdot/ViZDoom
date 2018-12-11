mkdir -p out_val

# Create exactly 20 validation maps
for i in {205..247}
do
    python map_create.py -s $i -d regular -m none -z none -o './out_val/'
done
