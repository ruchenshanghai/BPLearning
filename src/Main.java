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
    private static final String classificationPath = "../TRAIN";
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

        int[] hiddenArray = {};
        double rating = 0;
        int inputCount = imageWidth * imageHeight;
        int outputCount = classificationType;

        for (int ratingLength = 0; ratingLength < 10; ratingLength++) {
            rating = 1.05 + (ratingLength * 0.2);
            hiddenArray = new int[2];

            for (int layerTimes = 0; layerTimes < 10; layerTimes++) {

                hiddenArray[0] = 90 + (int) (Math.random() * 5);
                hiddenArray[1] = 70 + (int) (Math.random() * 5);

                NeuronSystem classificationSystem = new NeuronSystem(inputCount, hiddenArray, outputCount, rating, sigmoidFunctionType);
                classificationSystem.paramsReset();

                // get data
                ArrayList<DataNode> trainingList = new ArrayList<>();
                ArrayList<DataNode> verifyList = new ArrayList<>();
                for (int typeIndex = 0; typeIndex < classificationType; typeIndex++) {
                    File tempFile = new File(classificationPath + "/" + (typeIndex + 1));

                    int tempFileNum = tempFile.list().length;
                    int tempTrainingNum = (int) (tempFileNum * trainingPercent);
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
                        if (fileIndex < tempTrainingNum) {
                            trainingList.add(tempNode);
                        } else {
                            verifyList.add(tempNode);
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
                    System.out.println("index: " + i + ", rate: " + (accurateTimes / dataSize));
                }
                System.out.println("hidden layer structure: " + Arrays.toString(hiddenArray) + ", rating: " + rating);

                // verify
                int verifySize = verifyList.size();
                accurateTimes = 0;
                Collections.shuffle(verifyList);
                for (int i = 0; i < verifySize; i++) {
                    accurateTimes += classificationSystem.predictClassification(verifyList.get(i).getInput(), verifyList.get(i).getOutput());
                }
                System.out.println("average predict accurate rate: " + (accurateTimes / verifySize));
                System.out.println();
            }

//            for (int hiddenLayer = 2; hiddenLayer < 3; hiddenLayer++) {
//                hiddenArray = new int[hiddenLayer];
//                for (int layerIndex = 0; layerIndex < hiddenLayer; layerIndex++) {
//                    hiddenArray[layerIndex] = (int) (Math.random() * 100) + 50;
//                }
//            }

        }

    }

}
