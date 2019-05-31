# Understand traffic as images
### Use Convolutional Neural Network to predict travel time on I-80E from Davis to West Sacramento

## Data 
see ./Data/Data Fact.pdf

## Generate images from traffic data
Use *ImageGenerate.py* (already generated into ./Pics)

## Pare images with traveltime to csvs
Use *TravelTimeGenerate.py* (already generated into ./TravelTime, 4 days' test data in /test_data)

## Generate Pytorch training data sets
Use *GetOneChannel.py* for One Channel CNN (speed/flow/occupancy)  

Use *GetThreeChannel.py* for Three Channels CNN (speed+flow+occupancy)  

Use *GetChannel.py* for Four Channels CNN (speed+flow+occupancy+observation rate)

## Train CNNs
Use *CNN_1channel.py* for One Channel CNN (speed/flow/occupancy)  

Use *CNN_3channel.py* for Three Channels CNN (speed+flow+occupancy)  

Use *CNN_4channel.py* for Four Channels CNN (speed+flow+occupancy+observation rate)

## Outputs
Output results are stored in ./outputs

## TSNE
to be overrided

## Visualization of CNN
Use *./NetVisualize.py*
