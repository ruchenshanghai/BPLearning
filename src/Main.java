import network.NeuronNode;
import network.NeuronSystem;

import java.util.Random;

public class Main {
    public static void main(String[] args) {

        testSin(100);

    }

    public static void testSin(int dataSize) {
        if (dataSize <= 0) {
            return;
        }
        int[] hiddenArray = {10, 10};
        NeuronSystem sinSystem = new NeuronSystem(1, hiddenArray, 1, 0.01);
        double[][] targetArray = new double[dataSize][2];
        double min = -Math.PI;
        double max = Math.PI;
        for (int i = 0; i < dataSize; i++) {
            double x = min + new Random().nextDouble() * (max - min);
            double y = Math.sin(x);
            double[] tempInput = {x};
            double[] tempOutput = {y};
            sinSystem.train(tempInput, tempOutput);
        }
    }
}
