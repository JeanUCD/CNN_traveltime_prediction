# import packages
library(ggplot2)

## Channel 1
### Select batch size and learning rate
#### Compare how batch size and learning rate influence convergent speed
trainloss_C1_b10_lr0.005_spd = read.csv('../outputs/Channel1/speed/training_loss/500epochs_10batchsize_0.005lr_teston1030.csv')
names(trainloss_C1_b10_lr0.005_spd) = c('epochs','training.loss')
trainloss_C1_b10_lr0.005_spd$label = 'Batch.size: 10, Learning.rate: 0.005'

trainloss_C1_b10_lr0.01_spd = read.csv('../outputs/Channel1/speed/training_loss/500epochs_10batchsize_0.01lr_teston1030.csv')
names(trainloss_C1_b10_lr0.01_spd) = c('epochs','training.loss')
trainloss_C1_b10_lr0.01_spd$label = 'Batch.size: 10, Learning.rate: 0.01'
trainloss_C1_b10_lr0.01_spd$variables = 'speed'

trainloss_C1_b50_lr0.005_spd = read.csv('../outputs/Channel1/speed/training_loss/500epochs_50batchsize_0.005lr_teston1030.csv')
names(trainloss_C1_b50_lr0.005_spd) = c('epochs','training.loss')
trainloss_C1_b50_lr0.005_spd$epochs = trainloss_C1_b50_lr0.005_spd$epochs*5
trainloss_C1_b50_lr0.005_spd$label = 'Batch.size: 50, Learning.rate: 0.005'

trainloss_C1_spd = rbind(trainloss_C1_b10_lr0.005_spd,trainloss_C1_b10_lr0.01_spd,trainloss_C1_b50_lr0.005_spd)
# show the plot of training loss
p.trainloss_C1_spd <- ggplot(trainloss_C1_spd, aes(x=epochs, y=training.loss,color=label)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Training Loss ",
    title = paste("Training loss over epochs of 1-channel CNN")
  )
p.trainloss_C1_spd

### Select dependent variable(speed/flow/occupancy)
#### Compare the performance of trainloss
trainloss_C1_b10_lr0.01_occ = read.csv('../outputs/Channel1/occupancy/training_loss/500epochs_10batchsize_0.01lr_teston1030.csv')
names(trainloss_C1_b10_lr0.01_occ) = c('epochs','training.loss')
trainloss_C1_b10_lr0.01_occ$label = 'Batch.size: 10, Learning.rate: 0.01'
trainloss_C1_b10_lr0.01_occ$variables = 'occupancy'

trainloss_C1_b10_lr0.01_flw = read.csv('../outputs/Channel1/flow/training_loss/500epochs_10batchsize_0.01lr_teston1030.csv')
names(trainloss_C1_b10_lr0.01_flw) = c('epochs','training.loss')
trainloss_C1_b10_lr0.01_flw$label = 'Batch.size: 10, Learning.rate: 0.01'
trainloss_C1_b10_lr0.01_flw$variables = 'flow'

trainloss_C1 = rbind(trainloss_C1_b10_lr0.01_occ,trainloss_C1_b10_lr0.01_flw,trainloss_C1_b10_lr0.01_spd)
# show the plot of training loss
p.trainloss_C1 <- ggplot(trainloss_C1, aes(x=epochs, y=training.loss,color=variables)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Training Loss ",
    title = paste("Training loss over epochs of 1-channel CNN")
  )
p.trainloss_C1

#### Compare the performance of test loss
testloss_C1_b10_lr0.01_spd = read.csv('../outputs/Channel1/speed/test_loss/500epochs_10batchsize_0.01lr_teston1030.csv')
names(testloss_C1_b10_lr0.01_spd) = c('epochs','test.loss')
testloss_C1_b10_lr0.01_spd$label = 'Batch.size: 10, Learning.rate: 0.01'
testloss_C1_b10_lr0.01_spd$variables = 'speed'

testloss_C1_b10_lr0.01_occ = read.csv('../outputs/Channel1/occupancy/test_loss/500epochs_10batchsize_0.01lr_teston1030.csv')
names(testloss_C1_b10_lr0.01_occ) = c('epochs','test.loss')
testloss_C1_b10_lr0.01_occ$label = 'Batch.size: 10, Learning.rate: 0.01'
testloss_C1_b10_lr0.01_occ$variables = 'occupancy'

testloss_C1_b10_lr0.01_flw = read.csv('../outputs/Channel1/flow/test_loss/500epochs_10batchsize_0.01lr_teston1030.csv')
names(testloss_C1_b10_lr0.01_flw) = c('epochs','test.loss')
testloss_C1_b10_lr0.01_flw$label = 'Batch.size: 10, Learning.rate: 0.01'
testloss_C1_b10_lr0.01_flw$variables = 'flow'

testloss_C1 = rbind(testloss_C1_b10_lr0.01_occ,testloss_C1_b10_lr0.01_flw,testloss_C1_b10_lr0.01_spd)
# show the plot of training loss
p.testloss_C1 <- ggplot(testloss_C1, aes(x=epochs, y=test.loss,color=variables)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Test Loss ",
    title = paste("Test loss over epochs of 1-channel CNN")
  )
p.testloss_C1

#### Compare the performance of prediction accuracy
PreAcc_C1_b10_lr0.01_spd = read.csv('../outputs/Channel1/speed/acc_test/500epochs_10batchsize_0.01lr_teston1030.csv')
names(PreAcc_C1_b10_lr0.01_spd) = c('epochs','prediction.accuracy')
PreAcc_C1_b10_lr0.01_spd$label = 'Batch.size: 10, Learning.rate: 0.01'
PreAcc_C1_b10_lr0.01_spd$variables = 'speed'

PreAcc_C1_b10_lr0.01_occ = read.csv('../outputs/Channel1/occupancy/acc_test/500epochs_10batchsize_0.01lr_teston1030.csv')
names(PreAcc_C1_b10_lr0.01_occ) = c('epochs','prediction.accuracy')
PreAcc_C1_b10_lr0.01_occ$label = 'Batch.size: 10, Learning.rate: 0.01'
PreAcc_C1_b10_lr0.01_occ$variables = 'occupancy'

PreAcc_C1_b10_lr0.01_flw = read.csv('../outputs/Channel1/flow/acc_test/500epochs_10batchsize_0.01lr_teston1030.csv')
names(PreAcc_C1_b10_lr0.01_flw) = c('epochs','prediction.accuracy')
PreAcc_C1_b10_lr0.01_flw$label = 'Batch.size: 10, Learning.rate: 0.01'
PreAcc_C1_b10_lr0.01_flw$variables = 'flow'

PreAcc_C1 = rbind(PreAcc_C1_b10_lr0.01_occ,PreAcc_C1_b10_lr0.01_flw,PreAcc_C1_b10_lr0.01_spd)
# show the plot of training loss
p.PreAcc_C1 <- ggplot(PreAcc_C1, aes(x=epochs, y=prediction.accuracy,color=variables)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Predition Accuracy ",
    title = paste("Prediction accuracy on test set over epochs of 1-channel CNN")
  )
p.PreAcc_C1

max(PreAcc_C1$prediction.accuracy)#46.2222

### Compare performance on different test sets
PreAcc_C1_b10_lr0.01_spd_1106 = read.csv('../outputs/Channel1/speed/acc_test/500epochs_10batchsize_0.01lr_teston1106.csv')
names(PreAcc_C1_b10_lr0.01_spd_1106) = c('epochs','prediction.accuracy')
PreAcc_C1_b10_lr0.01_spd_1106$label = 'Batch.size: 10, Learning.rate: 0.01'
PreAcc_C1_b10_lr0.01_spd_1106$variables = 'speed'
PreAcc_C1_b10_lr0.01_spd_1106$testdate = '1106'

PreAcc_C1_b10_lr0.01_spd_1031 = read.csv('../outputs/Channel1/speed/acc_test/500epochs_10batchsize_0.01lr_teston1031.csv')
names(PreAcc_C1_b10_lr0.01_spd_1031) = c('epochs','prediction.accuracy')
PreAcc_C1_b10_lr0.01_spd_1031$label = 'Batch.size: 10, Learning.rate: 0.01'
PreAcc_C1_b10_lr0.01_spd_1031$variables = 'speed'
PreAcc_C1_b10_lr0.01_spd_1031$testdate = '1031'

PreAcc_C1_2test = rbind(PreAcc_C1_b10_lr0.01_spd_1105,PreAcc_C1_b10_lr0.01_spd_1031)
p.PreAcc_C1_2test <- ggplot(PreAcc_C1_2test, aes(x=epochs, y=prediction.accuracy,color=testdate)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Predition Accuracy ",
    title = paste("Prediction accuracy on 2 test sets over epochs of 1-channel CNN")
  )
p.PreAcc_C1_2test
max(PreAcc_C1_2test$prediction.accuracy)#55.15464

## Channel 3
### Compare training loss(convergent speed) with Channel 1
trainloss_C3_b10_lr0.01 = read.csv('../outputs/Channel3/training_loss/500epochs_10batchsize_0.01lr_teston1031.csv')
names(trainloss_C3_b10_lr0.01) = c('epochs','training.loss')
trainloss_C3_b10_lr0.01$label = 'Batch.size: 10, Learning.rate: 0.01'
trainloss_C3_b10_lr0.01$Channel.Number = '3'

trainloss_C1_b10_lr0.01_spd = read.csv('../outputs/Channel1/speed/training_loss/500epochs_10batchsize_0.01lr_teston1031.csv')
names(trainloss_C1_b10_lr0.01_spd) = c('epochs','training.loss')
trainloss_C1_b10_lr0.01_spd$label = 'Batch.size: 10, Learning.rate: 0.01'
trainloss_C1_b10_lr0.01_spd$Channel.Number = '1(speed)'

trainloss_C1C3 = rbind(trainloss_C3_b10_lr0.01,trainloss_C1_b10_lr0.01_spd)
p.trainloss_C1C3 <- ggplot(trainloss_C1C3, aes(x=epochs, y=training.loss,color=Channel.Number)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Training Loss ",
    title = paste("Training loss over epochs of 2 CNNs")
  )
p.trainloss_C1C3

### Compare prediction accuracy with Channel 1
PreAcc_C1_b10_lr0.01_spd = read.csv('../outputs/Channel1/speed/acc_test/500epochs_10batchsize_0.01lr_teston1031.csv')
names(PreAcc_C1_b10_lr0.01_spd) = c('epochs','prediction.accuracy')
PreAcc_C1_b10_lr0.01_spd$label = 'Batch.size: 10, Learning.rate: 0.01'
PreAcc_C1_b10_lr0.01_spd$variables = 'speed'

PreAcc_C3_b10_lr0.01 = read.csv('../outputs/Channel3/acc_test/500epochs_10batchsize_0.01lr_teston1031.csv')
names(PreAcc_C3_b10_lr0.01) = c('epochs','prediction.accuracy')
PreAcc_C3_b10_lr0.01$label = 'Batch.size: 10, Learning.rate: 0.01'
PreAcc_C3_b10_lr0.01$variables = '3 Channels'
PreAcc_C1C3 = rbind(PreAcc_C1_b10_lr0.01_spd,PreAcc_C3_b10_lr0.01)
# show the plot of training loss
p.PreAcc_C1C3 <- ggplot(PreAcc_C1C3, aes(x=epochs, y=prediction.accuracy,color=variables)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Predition Accuracy ",
    title = paste("Prediction accuracy on 2 test sets over epochs of CNNs")
  )
p.PreAcc_C1C3

max(PreAcc_C1C3$prediction.accuracy)#64.44444

### Compare prediction accuracy BTW 1031 and 1106
PreAcc_C3_b10_lr0.01_1106 = read.csv('../outputs/Channel3/acc_test/500epochs_10batchsize_0.01lr_teston1106.csv')
names(PreAcc_C3_b10_lr0.01_1106) = c('epochs','prediction.accuracy')
PreAcc_C3_b10_lr0.01_1106$testdate = '1106'

PreAcc_C3_b10_lr0.01_1031 = read.csv('../outputs/Channel3/acc_test/500epochs_10batchsize_0.01lr_teston1031.csv')
names(PreAcc_C3_b10_lr0.01_1031) = c('epochs','prediction.accuracy')
PreAcc_C3_b10_lr0.01_1031$testdate = '1031'

PreAcc_C3_2test = rbind(PreAcc_C3_b10_lr0.01_1106,PreAcc_C3_b10_lr0.01_1031)
p.PreAcc_C3_2test <- ggplot(PreAcc_C3_2test, aes(x=epochs, y=prediction.accuracy,color=testdate)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Predition Accuracy ",
    title = paste("Prediction accuracy on 2 test sets over epochs of 3-Channel CNN ")
  )
p.PreAcc_C3_2test
max(PreAcc_C3_b10_lr0.01_1106$prediction.accuracy)#46.90722
max(PreAcc_C3_b10_lr0.01_1031$prediction.accuracy)#64.44444

## Channel 4
### Compare prediction accuracy on 1031 BTW 3 Channel and 4 channel
PreAcc_C4_b10_lr0.01_1031 = read.csv('../outputs/Channel4/acc_test/500epochs_10batchsize_0.01lr_teston1031.csv')
names(PreAcc_C4_b10_lr0.01_1031) = c('epochs','prediction.accuracy')
PreAcc_C4_b10_lr0.01_1031$testdate = '1031'
PreAcc_C4_b10_lr0.01_1031$Channel.Number = '4'

PreAcc_C3_b10_lr0.01_1031$Channel.Number = '3'

PreAcc_C3C4_1031 = rbind(PreAcc_C3_b10_lr0.01_1031,PreAcc_C4_b10_lr0.01_1031)
p.PreAcc_C3C4_1031 <- ggplot(PreAcc_C3C4_1031, aes(x=epochs, y=prediction.accuracy,color=Channel.Number)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Predition Accuracy ",
    title = paste("Prediction accuracy on 1031 test set over epochs of CNNs ")
  )
p.PreAcc_C3C4_1031
max(PreAcc_C4_b10_lr0.01_1031$prediction.accuracy)# 55.55556

### Compare prediction accuracy on 1106 BTW 3 Channel and 4 channel
PreAcc_C4_b10_lr0.01_1106 = read.csv('../outputs/Channel4/acc_test/500epochs_10batchsize_0.01lr_teston1106.csv')
names(PreAcc_C4_b10_lr0.01_1106) = c('epochs','prediction.accuracy')
PreAcc_C4_b10_lr0.01_1106$testdate = '1106'
PreAcc_C4_b10_lr0.01_1106$Channel.Number = '4'

PreAcc_C3_b10_lr0.01_1106$Channel.Number = '3'

PreAcc_C3C4_1106 = rbind(PreAcc_C3_b10_lr0.01_1106,PreAcc_C4_b10_lr0.01_1106)
p.PreAcc_C3C4_1106 <- ggplot(PreAcc_C3C4_1106, aes(x=epochs, y=prediction.accuracy,color=Channel.Number)) + geom_line()+
  theme_bw()+
  labs(
    x = "Epochs",
    y= "Predition Accuracy ",
    title = paste("Prediction accuracy on 1106 test set over epochs of CNNs ")
  )
p.PreAcc_C3C4_1106
max(PreAcc_C4_b10_lr0.01_1106$prediction.accuracy)# 58.24742

## Plot prediction accuracies on 2 test sets with 1/3/4 channel CNNs
### 1031
#### 46.2222  64.44444  55.55556

### 1106
#### 45.36082  46.90722  58.24742
PreACC_C1C3C4_1031 = as.data.frame(cbind(c('1 Channel','3 Channels','4 Channels'),c(46.2222,64.4444,55.5556)))
names(PreACC_C1C3C4_1031) = c('Channels','Prediction.Accuracy')
PreACC_C1C3C4_1031$TestSet = '1031'
PreACC_C1C3C4_1106 = as.data.frame(cbind(c('1 Channel','3 Channels','4 Channels'),c(45.3608,46.9072,58.2474)))
names(PreACC_C1C3C4_1106) = c('Channels','Prediction.Accuracy')
PreACC_C1C3C4_1106$TestSet = '1106'
PreACC_C1C3C4 = rbind(PreACC_C1C3C4_1031,PreACC_C1C3C4_1106)
PreACC_C1C3C4$Prediction.Accuracy = as.numeric(as.character(PreACC_C1C3C4$Prediction.Accuracy))
p.PreAcc_C1C3C4 <- ggplot(PreACC_C1C3C4, aes(x=Channels, y=Prediction.Accuracy,color=TestSet)) + 
  geom_line(aes(Channels,Prediction.Accuracy,group = TestSet))+
  theme_bw()+
  labs(
    x = "Channels",
    y= "Predition Accuracy (%)",
    title = paste("Prediction accuracy on test sets of CNNs ")
  )
p.PreAcc_C1C3C4