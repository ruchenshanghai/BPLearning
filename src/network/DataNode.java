package network;

public class DataNode {
    private double[] input;
    private double[] output;
    public DataNode(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }

    public DataNode(int[] tempInput, int[] tempOutput) {
        input = new double[tempInput.length];
        for (int i = 0; i < tempInput.length; i++) {
            input[i] = tempInput[i];
        }
        output = new double[tempOutput.length];
        for (int i = 0; i < tempOutput.length; i++) {
            output[i] = tempOutput[i];
        }
    }

    public double[] getInput() {
        return input;
    }

    public double[] getOutput() {
        return output;
    }
}
