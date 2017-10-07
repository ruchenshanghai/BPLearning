package network;

public class DataNode {
    private double[] input;
    private double[] output;
    public DataNode(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }

    public double[] getInput() {
        return input;
    }

    public double[] getOutput() {
        return output;
    }
}
