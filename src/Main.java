import bmp.BMPResolver;
import network.DataNode;
import network.NeuronNode;
import network.NeuronSystem;

import java.io.*;
import java.util.*;

public class Main {
    private static final String classificationPath = "TRAIN";
    private static final int classificationType = 14;
    private static final int imageWidth = 28;
    private static final int imageHeight = 28;
    private static final String testPath = "./test";

    public static void main(String[] args) throws IOException {
//        simulateSin(100, 50000, NeuronNode.SIGMOID_FUNCTION_TYPE.logFunction);

//        simulateClassification(50, 0.8, NeuronNode.SIGMOID_FUNCTION_TYPE.logFunction);


        predictClassification(200, NeuronNode.SIGMOID_FUNCTION_TYPE.logFunction, "./test");
    }

    private static void simulateSin(int dataSize, int trainingTimes, NeuronNode.SIGMOID_FUNCTION_TYPE sigmoidFunctionType) {
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

    private static void simulateClassification(int trainingTimes, double trainingPercent, NeuronNode.SIGMOID_FUNCTION_TYPE sigmoidFunctionType) throws IOException {

        int[] hiddenArray = {90, 74};
        double rating = 1.05;
        int inputCount = imageWidth * imageHeight;
        int outputCount = classificationType;

        NeuronSystem classificationSystem = new NeuronSystem(inputCount, hiddenArray, outputCount, rating, sigmoidFunctionType);
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
                double[] tempInput = resolver.getInputVector();
                double[] tempOutput = new double[classificationType];
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
        for (int i = 0; i < trainingTimes; i++) {
            accurateTimes = 0;
            for (int j = 0; j < dataSize; j++) {
                accurateTimes += classificationSystem.trainClassification(trainingList.get(j).getInput(), trainingList.get(j).getOutput());
            }
            Collections.shuffle(trainingList);
            System.out.println("index: " + i + ", accurate rate: " + (accurateTimes / dataSize));
        }

        // verify
        int verifySize = verifyList.size();
        accurateTimes = 0;
        Collections.shuffle(verifyList);
        for (int i = 0; i < verifySize; i++) {
            accurateTimes += classificationSystem.predictClassification(verifyList.get(i).getInput(), verifyList.get(i).getOutput());
        }

        System.out.println("hidden layer structure: " + Arrays.toString(hiddenArray) + ", rating: " + rating);
        System.out.println("average predict accurate rate: " + (accurateTimes / verifySize));

    }

    private static void predictClassification(int trainingTimes, NeuronNode.SIGMOID_FUNCTION_TYPE sigmoidFunctionType, String dirPath) throws IOException {

        int[] hiddenArray = {90, 74};
        double rating = 1.05;
        int inputCount = imageWidth * imageHeight;
        int outputCount = classificationType;

        NeuronSystem classificationSystem = new NeuronSystem(inputCount, hiddenArray, outputCount, rating, sigmoidFunctionType);
        classificationSystem.paramsReset();
        // get data
        ArrayList<DataNode> trainingList = new ArrayList<>();
        ArrayList<DataNode> verifyList = new ArrayList<>();
        for (int typeIndex = 0; typeIndex < classificationType; typeIndex++) {
            File tempFile = new File(classificationPath + "/" + (typeIndex + 1));
            int tempFileNum = tempFile.list().length;
            for (int fileIndex = 0; fileIndex < tempFileNum; fileIndex++) {
                BMPResolver resolver = new BMPResolver(imageWidth, imageHeight, classificationPath + "/" + (typeIndex + 1) + "/" + fileIndex + ".bmp");
                double[] tempInput = resolver.getInputVector();
                double[] tempOutput = new double[classificationType];
                for (int i = 0; i < classificationType; i++) {
                    if (i == typeIndex) {
                        tempOutput[i] = 1;
                    } else {
                        tempOutput[i] = 0;
                    }
                }
                DataNode tempNode = new DataNode(tempInput, tempOutput);
                trainingList.add(tempNode);
            }
            // divide data
            Collections.shuffle(trainingList);
        }

        // training
        int dataSize = trainingList.size();
        double accurateTimes = 0;
        for (int i = 0; i < trainingTimes; i++) {
            accurateTimes = 0;
            for (int j = 0; j < dataSize; j++) {
                accurateTimes += classificationSystem.trainClassification(trainingList.get(j).getInput(), trainingList.get(j).getOutput());
            }
            Collections.shuffle(trainingList);
            System.out.println("index: " + i + ", accurate rate: " + (accurateTimes / dataSize));
        }



        ArrayList<double[]> testDataList = importTestData(testPath);
        int[] predictArray = new int[testDataList.size()];
        // predict
        int testSize = testDataList.size();
        for (int i = 0; i < testSize; i++) {
            predictArray[i] = classificationSystem.predictClassification(testDataList.get(i));
            System.out.println(predictArray[i]);
        }
        File predictFile = new File("pred.txt");
        if (!predictFile.exists()) {
            predictFile.createNewFile();
        }
        FileWriter predictWriter = new FileWriter(predictFile, false);
        for (int i = 0; i < predictArray.length; i++) {
            predictWriter.write("" + predictArray[i] + "\n");
        }
        predictWriter.flush();
        predictWriter.close();
    }

    private static ArrayList<double[]> importTestData(String dirPath) throws IOException {
        ArrayList<double[]> dataArrayList = new ArrayList<>();
        // get filename array
        File dirFile = new File(dirPath);
        if (!dirFile.exists()) {
            System.out.println(dirPath + " not exist!");
            return null;
        }
        File[] fileArray = dirFile.listFiles();
        String[] fileNameArray = new String[fileArray.length];
        for (int i = 0; i < fileArray.length; i++) {
            if (fileArray[i].isDirectory()) {
                System.out.println(dirPath + "/" + fileArray[i].getName() + " is not file!");
                return null;
            }
//            System.out.println(dirPath + "/" + fileArray[i].getName());
            fileNameArray[i] = fileArray[i].getName();
        }
        List nameList = Arrays.asList(fileNameArray);
        Collections.sort(nameList, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return Integer.parseInt(o1.replaceAll("[^0-9]*", "")) - Integer.parseInt(o2.replaceAll("[^0-9]*", ""));
            }
        });
//        System.out.println(nameList);
        for (int i = 0; i < nameList.size(); i++) {
            BMPResolver resolver = new BMPResolver(imageWidth, imageHeight, dirPath + "/" + nameList.get(i));
            double[] tempInput = resolver.getInputVector();
            dataArrayList.add(tempInput);
        }
        return dataArrayList;
    }
}
