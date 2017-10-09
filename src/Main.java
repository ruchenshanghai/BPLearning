import bmp.BMPResolver;
import network.DataNode;
import network.NeuronNode;
import network.NeuronSystem;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class Main {
    private static final String classificationPath = "TRAIN";
    private static final int classificationType = 14;
    private static final int imageWidth = 28;
    private static final int imageHeight = 28;

    public static void main(String[] args) throws IOException {
//        testSin(100, 50000, NeuronNode.SIGMOID_FUNCTION_TYPE.logFunction);
        testClassification(50, 0.8, NeuronNode.SIGMOID_FUNCTION_TYPE.logFunction);


    }

    private static void testSin(int dataSize, int trainingTimes, NeuronNode.SIGMOID_FUNCTION_TYPE sigmoidFunctionType) {
        if (dataSize <= 0) {
            return;
        }

        // tanFunction
//        int[] hiddenArray = {7, 7};
//        double rating = 0.134;
//        NeuronSystem sinSystem = new NeuronSystem(1, hiddenArray, 1, rating, NeuronNode.SIGMOID_FUNCTION_TYPE.tanFunction);
//        sinSystem.paramsReset();
//        // generate data
//        ArrayList<DataNode> dataList = new ArrayList<>();
//        double min = -Math.PI;
//        double max = Math.PI;
//        for (int i = 0; i < dataSize; i++) {
//            double x = min + new Random().nextDouble() * (max - min);
//            double y = Math.sin(x);
//            double[] tempInput = {x};
//            double[] tempOutput = {y};
//            DataNode tempNode = new DataNode(tempInput, tempOutput);
//            dataList.add(tempNode);
//        }
//        // training
//        for (int i = 0; i < trainingTimes; i++) {
//            for (int j = 0; j < dataSize; j++) {
//                sinSystem.train(dataList.get(j).getInput(), dataList.get(j).getOutput());
//            }
//            Collections.shuffle(dataList);
//        }
//        // verify
//        double errorSum = 0;
//        for (int i = 0; i < dataSize; i++) {
//            double x = min + new Random().nextDouble() * (max - min);
//            double y = Math.sin(x);
//            double[] tempInput = {x};
//            double[] tempOutput = {y};
//            double[] tempPredict = sinSystem.predict(tempInput);
//            for (int j = 0; j < tempOutput.length; j++) {
//                errorSum += Math.abs(tempPredict[j] - tempOutput[j]);
//            }
//        }
//        System.out.printf("rating = %5f average predict error %10.7f \n", rating, (errorSum / dataSize));


        int[] hiddenArray = {};
        double rating = 0;
        if (sigmoidFunctionType == NeuronNode.SIGMOID_FUNCTION_TYPE.tanFunction) {
            hiddenArray = new int[2];
            hiddenArray[0] = 7;
            hiddenArray[1] = 7;
            rating = 0.134;
        } else {
            hiddenArray = new int[1];
            hiddenArray[0] = 9;
            rating = 3.5;
        }
        NeuronSystem sinSystem = new NeuronSystem(1, hiddenArray, 1, rating, sigmoidFunctionType);
        sinSystem.paramsReset();
        // generate data
        ArrayList<DataNode> dataList = new ArrayList<>();
        double min = -Math.PI;
        double max = Math.PI;
        for (int i = 0; i < dataSize; i++) {
            double x = min + new Random().nextDouble() * (max - min);
            double y = Math.sin(x);
            double[] tempInput = {x};
            double[] tempOutput = new double[1];
            if (sigmoidFunctionType == NeuronNode.SIGMOID_FUNCTION_TYPE.logFunction) {
                tempOutput[0] = (y + 1) / 2;
            } else {
                tempOutput[0] = y;
            }
            DataNode tempNode = new DataNode(tempInput, tempOutput);
            dataList.add(tempNode);
        }
        // training
        for (int i = 0; i < trainingTimes; i++) {
            Collections.shuffle(dataList);
            for (int j = 0; j < dataSize; j++) {
                sinSystem.trainSin(dataList.get(j).getInput()[0], dataList.get(j).getOutput()[0]);
            }
        }
        // verify
        double errorSum = 0;
        for (int i = 0; i < dataSize * 100; i++) {
            double x = min + new Random().nextDouble() * (max - min);
            double y = Math.sin(x);
            double tempInput = min + new Random().nextDouble() * (max - min);
            double tempOutput = 0;
            if (sigmoidFunctionType == NeuronNode.SIGMOID_FUNCTION_TYPE.logFunction) {
                tempOutput = (y + 1) / 2;
            } else {
                tempOutput = y;
            }
            double tempPredict = sinSystem.predictSin(x);
            errorSum += Math.abs(tempPredict - tempOutput);
        }
        System.out.printf("rating = %5f average predict error %10.7f \n", rating, (errorSum / dataSize) / 100);

    }

    private static void testClassification(int trainingTime, double trainingPercent, NeuronNode.SIGMOID_FUNCTION_TYPE sigmoidFunctionType) throws IOException {

        int[] hiddenArray;
        double rating = 1.05;
        int inputCount = imageWidth * imageHeight;
        int outputCount = classificationType;
        int[][] hiddenArraySample = new int[][]{{90, 74}, {94, 72}, {94, 73}, {91, 74}, {93, 73}};


        for (int hiddenInex = 0; hiddenInex < hiddenArraySample.length; hiddenInex++) {
            hiddenArray = hiddenArraySample[hiddenInex];
            NeuronSystem classificationSystem = new NeuronSystem(inputCount, hiddenArray, outputCount, rating, sigmoidFunctionType);
            double verifyAccurateSum = 0;
            double minAccurate = 1;

            for (int verifyTimes = 0; verifyTimes < 20; verifyTimes++) {
                classificationSystem.paramsReset();
                // get data
                ArrayList<DataNode> trainingList = new ArrayList<>();
                ArrayList<DataNode> verifyList = new ArrayList<>();
                for (int typeIndex = 0; typeIndex < classificationType; typeIndex++) {
                    File tempFile = new File(classificationPath + "/" + (typeIndex + 1));
                    int tempFileNum = tempFile.list().length;
                    ArrayList<DataNode> tempList = new ArrayList<>();
                    for (int fileIndex = 0; fileIndex < tempFileNum; fileIndex++) {
                        BMPResolver resolver = new BMPResolver(imageWidth, imageHeight, classificationPath + "/" + (typeIndex + 1) + "/" + fileIndex + ".bmp");
                        int[] tempInput = resolver.getInputVector();
                        int[] tempOutput = new int[classificationType];
                        for (int i = 0; i < classificationType; i++) {
                            if (i == typeIndex) {
                                tempOutput[i] = 1;
                            } else {
                                tempOutput[i] = 0;
                            }
                        }
                        DataNode tempNode = new DataNode(tempInput, tempOutput);
                        tempList.add(tempNode);
                    }

                    // divide data
                    Collections.shuffle(tempList);
                    int tempLength = tempList.size();
                    int tempTrainingLength = (int) (tempLength * trainingPercent);
                    for (int tempIndex = 0; tempIndex < tempLength; tempIndex++) {
                        if (tempIndex < tempTrainingLength) {
                            trainingList.add(tempList.get(tempIndex));
                        } else {
                            verifyList.add(tempList.get(tempIndex));
                        }
                    }

                }

                // training
                int dataSize = trainingList.size();
                double accurateTimes = 0;
                for (int i = 0; i < trainingTime; i++) {
                    accurateTimes = 0;
                    for (int j = 0; j < dataSize; j++) {
                        accurateTimes += classificationSystem.trainClassification(trainingList.get(j).getInput(), trainingList.get(j).getOutput());
                    }
                    Collections.shuffle(trainingList);
//                    System.out.println("index: " + i + ", rate: " + (accurateTimes / dataSize));
                }

                // verify
                int verifySize = verifyList.size();
                accurateTimes = 0;
                Collections.shuffle(verifyList);
                for (int i = 0; i < verifySize; i++) {
                    accurateTimes += classificationSystem.predictClassification(verifyList.get(i).getInput(), verifyList.get(i).getOutput());
                }
                System.out.printf("\t%5.4f", (accurateTimes / verifySize));

                if ((accurateTimes / verifySize) < minAccurate) {
                    minAccurate = (accurateTimes / verifySize);
                }
                verifyAccurateSum += (accurateTimes / verifySize);
            }
            System.out.println();
            System.out.println("hidden layer structure: " + Arrays.toString(hiddenArray) + ", rating: " + rating);
            System.out.println("min average predict accurate rate: " + minAccurate);
            System.out.println("total average predict accurate rate: " + (verifyAccurateSum / 20));
        }

    }

}
