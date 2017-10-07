import network.DataNode;
import network.NeuronNode;
import network.NeuronSystem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        testSin(100, 20000);

    }

    public static void testSin(int dataSize, int testTimes) {
        if (dataSize <= 0) {
            return;
        }
        int[] hiddenArray = {10, 10};
        NeuronSystem sinSystem = new NeuronSystem(1, hiddenArray, 1, 0.1);
        // generate data
        ArrayList<DataNode> dataList = new ArrayList<>();
        double min = -Math.PI;
        double max = Math.PI;
        for (int i = 0; i < dataSize; i++) {
            double x = min + new Random().nextDouble() * (max - min);
            double y = Math.sin(x);
            double[] tempInput = {x};
            double[] tempOutput = {y};
            DataNode tempNode = new DataNode(tempInput, tempOutput);
            dataList.add(tempNode);
        }
        // training
        for (int i = 0; i < testTimes; i++) {
            for (int j = 0; j < dataSize; j++) {
                sinSystem.train(dataList.get(j).getInput(), dataList.get(j).getOutput());
            }
            Collections.shuffle(dataList);
        }
        // verify
        for (int k = 0; k < dataSize; k++) {
            double error = 0;
            for (int i = 0; i < dataSize; i++) {
                double x = min + new Random().nextDouble() * (max - min);
                double y = Math.sin(x);
                double[] tempInput = {x};
                double[] tempOutput = {y};
                double[] tempPredict = sinSystem.predict(tempInput);
                for (int j = 0; j < tempOutput.length; j++) {
                    error += Math.abs(tempPredict[j] - tempOutput[j]);
                }
            }
            System.out.printf("average predict error %10.7f \n", (error / dataSize));
        }
//        double error = 0;
//        for (int i = 0; i < dataSize; i++) {
//            double x = min + new Random().nextDouble() * (max - min);
//            double y = Math.sin(x);
//            double[] tempInput = {x};
//            double[] tempOutput = {y};
//            double[] tempPredict = sinSystem.predict(tempInput);
//            for (int j = 0; j < tempOutput.length; j++) {
//                error += Math.abs(tempPredict[j] - tempOutput[j]);
//            }
//        }
//        System.out.printf("average predict error %10.7f \n", (error / dataSize));
    }
}
