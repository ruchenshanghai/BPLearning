package network;

public class NeuronSystem {
    private int inputCount;
    private int hiddenLayerCount;
    private int[] hiddenCount;
    private int outputCount;

    private NeuronNode[] inputNodes;
    private NeuronNode[][] hiddenNodes;
    private double[][] biasArray;
    private NeuronNode[] outputNodes;
    private double[][][] weightArray;
    private double rating;
    private NeuronNode.SIGMOID_FUNCTION_TYPE sigmoidFunctionType = NeuronNode.SIGMOID_FUNCTION_TYPE.tanFunction;

    public NeuronSystem(int inputCount, int[] hiddenCount, int outputCount, double rating) {
        this.inputCount = inputCount;
        this.hiddenLayerCount = hiddenCount.length;
        this.hiddenCount = hiddenCount;
        this.outputCount = outputCount;
        this.rating = rating;
        reset();
    }

    public NeuronSystem(int inputCount, int[] hiddenCount, int outputCount, double rating, NeuronNode.SIGMOID_FUNCTION_TYPE sigmoidFunctionType) {
        this.inputCount = inputCount;
        this.hiddenLayerCount = hiddenCount.length;
        this.hiddenCount = hiddenCount;
        this.outputCount = outputCount;
        this.rating = rating;
        this.sigmoidFunctionType = sigmoidFunctionType;
        reset();
    }

    public int trainClassification(double[] input, double[] output) {
        forward(input);
        int maxOutputIndex = 0;
        double maxOutputValue = 0;
        int maxExpectOutputIndex = 0;
        double maxExpectOutputValue = 0;
        for (int i = 0; i < outputCount; i++) {
//            System.out.println("i: " + i + " value: " + outputNodes[i].getForwardOutputValue());
//            System.out.println("i: " + i + " expect: " + output[i]);
            if (outputNodes[i].getForwardOutputValue() > maxOutputValue) {
                maxOutputIndex = i;
                maxOutputValue = outputNodes[i].getForwardOutputValue();
            }
            if (output[i] > maxExpectOutputValue) {
                maxExpectOutputIndex = i;
                maxExpectOutputValue = output[i];
            }
        }
        backward(output);
        updateParams();
        if (maxExpectOutputIndex != maxOutputIndex) {
            return 0;
        } else {
            return 1;
        }
    }

    public void trainSin(double input, double output) {
        forward(new double[]{input});
        backward(new double[]{output});
        updateParams();
    }

    public void train(double[] input, double[] output) {
        forward(input);
        backward(output);
        updateParams();
    }

    private void forward(double[] input) {
        if (input.length != inputCount) {
            return;
        }
        // set input
        for (int i = 0; i < inputCount; i++) {
            inputNodes[i].setForwardInputValue(input[i]);
        }
        // input to first hidden
        for (int i = 0; i < hiddenCount[0]; i++) {
            double temp = 0;
            for (int j = 0; j < inputCount; j++) {
                temp += (weightArray[0][j][i] * inputNodes[j].getForwardOutputValue());
            }
            temp += biasArray[0][i];
            hiddenNodes[0][i].setForwardInputValue(temp);
        }
        // hidden to hidden
        for (int i = 1; i < hiddenLayerCount; i++) {
            for (int j = 0; j < hiddenCount[i]; j++) {
                double temp = 0;
                int lastLayerIndex = i - 1;
                int lastLayerCount = hiddenCount[lastLayerIndex];
                for (int k = 0; k < lastLayerCount; k++) {
                    temp += (weightArray[i][k][j] * hiddenNodes[lastLayerIndex][k].getForwardOutputValue());
                }
                temp += biasArray[i][j];
                hiddenNodes[i][j].setForwardInputValue(temp);
            }
        }
        // hidden to output
        for (int i = 0; i < outputCount; i++) {
            double temp = 0;
            for (int j = 0; j < hiddenCount[hiddenLayerCount - 1]; j++) {
                temp += (weightArray[hiddenLayerCount][j][i] * hiddenNodes[hiddenLayerCount - 1][j].getForwardOutputValue());
            }
            temp += biasArray[hiddenLayerCount][i];
            outputNodes[i].setForwardInputValue(temp);
        }
    }

    private void backward(double[] expect) {
        // output
        for (int i = 0; i < outputCount; i++) {
            outputNodes[i].setBackwardInputValue(outputNodes[i].getForwardOutputValue() - expect[i]);
        }
        // output to hidden
        for (int i = 0; i < hiddenCount[hiddenLayerCount - 1]; i++) {
            double temp = 0;
            for (int j = 0; j < outputCount; j++) {
                temp += (weightArray[hiddenLayerCount][i][j] * outputNodes[j].getBackwardOutputValue());
            }
            hiddenNodes[hiddenLayerCount - 1][i].setBackwardInputValue(temp);
        }
        // hidden to hidden
        for (int i = hiddenLayerCount - 1; i > 0; i--) {
            for (int j = 0; j < hiddenCount[i - 1]; j++) {
                double temp = 0;
                for (int k = 0; k < hiddenCount[i]; k++) {
                    temp += (weightArray[i][j][k] * hiddenNodes[i][k].getBackwardOutputValue());
                }
                hiddenNodes[i - 1][j].setBackwardInputValue(temp);
            }
        }
    }

    private void updateParams() {
        // input to hidden
        for (int i = 0; i < inputCount; i++) {
            for (int j = 0; j < hiddenCount[0]; j++) {
                weightArray[0][i][j] -= (rating * inputNodes[i].getForwardOutputValue() * hiddenNodes[0][j].getBackwardOutputValue());
            }
        }
        for (int i = 0; i < hiddenCount[0]; i++) {
            biasArray[0][i] -= (rating * hiddenNodes[0][i].getBackwardOutputValue());
        }
        // hidden to hidden
        for (int i = 1; i < hiddenLayerCount; i++) {
            for (int j = 0; j < hiddenCount[i - 1]; j++) {
                for (int k = 0; k < hiddenCount[i]; k++) {
                    weightArray[i][j][k] -= (rating * hiddenNodes[i - 1][j].getForwardOutputValue() * hiddenNodes[i][k].getBackwardOutputValue());
                }
            }
            for (int j = 0; j < hiddenCount[i]; j++) {
                biasArray[i][j] -= (rating * hiddenNodes[i][j].getBackwardOutputValue());
            }
        }
        // hidden to output
        for (int i = 0; i < hiddenCount[hiddenLayerCount - 1]; i++) {
            for (int j = 0; j < outputCount; j++) {
                weightArray[hiddenLayerCount][i][j] -= (rating * hiddenNodes[hiddenLayerCount - 1][i].getForwardOutputValue() * outputNodes[j].getBackwardOutputValue());
            }
        }
        for (int i = 0; i < outputCount; i++) {
            biasArray[hiddenLayerCount][i] -= (rating * outputNodes[i].getBackwardOutputValue());
        }
    }

    private void reset() {
        inputNodes = new NeuronNode[inputCount];
        for (int i = 0; i < inputCount; i++) {
            inputNodes[i] = new NeuronNode(NeuronNode.NODE_TYPE.INPUT);
        }
        hiddenNodes = new NeuronNode[hiddenLayerCount][];
        biasArray = new double[hiddenLayerCount + 1][];
        for (int i = 0; i < hiddenLayerCount; i++) {
            hiddenNodes[i] = new NeuronNode[hiddenCount[i]];
            biasArray[i] = new double[hiddenCount[i]];
            for (int j = 0; j < hiddenCount[i]; j++) {
                hiddenNodes[i][j] = new NeuronNode(NeuronNode.NODE_TYPE.HIDDEN, sigmoidFunctionType);
            }
        }
        outputNodes = new NeuronNode[outputCount];
        biasArray[hiddenLayerCount] = new double[outputCount];
        for (int i = 0; i < outputCount; i++) {
            outputNodes[i] = new NeuronNode(NeuronNode.NODE_TYPE.OUTPUT, sigmoidFunctionType);
        }

        weightArray = new double[hiddenLayerCount + 1][][];
        weightArray[0] = new double[inputCount][hiddenCount[0]];
        for (int i = 1; i < hiddenLayerCount; i++) {
            weightArray[i] = new double[hiddenCount[i - 1]][hiddenCount[i]];
        }
        weightArray[hiddenLayerCount] = new double[hiddenCount[hiddenLayerCount - 1]][outputCount];

    }

    public void paramsReset() {
        for (int i = 0; i < hiddenLayerCount; i++) {
            for (int j = 0; j < hiddenCount[i]; j++) {
                biasArray[i][j] = (Math.random() % 0.02) - 0.01;
            }
        }
        for (int i = 0; i < outputCount; i++) {
            biasArray[hiddenLayerCount][i] = (Math.random() % 0.02) - 0.01;
        }

        for (int i = 0; i < inputCount; i++) {
            for (int j = 0; j < hiddenCount[0]; j++) {
                weightArray[0][i][j] = (Math.random() % 0.02) - 0.01;
            }
        }
        for (int i = 1; i < hiddenLayerCount; i++) {
            for (int j = 0; j < hiddenCount[i - 1]; j++) {
                for (int k = 0; k < hiddenCount[i]; k++) {
                    weightArray[i][j][k] = (Math.random() % 0.02) - 0.01;
                }
            }
        }
        for (int i = 0; i < hiddenCount[hiddenLayerCount - 1]; i++) {
            for (int j = 0; j < outputCount; j++) {
                weightArray[hiddenLayerCount][i][j] = (Math.random() % 0.02) - 0.01;
            }
        }
    }


    public double[] predict(double[] input) {
        forward(input);
        double[] result = new double[outputCount];
        for (int i = 0; i < outputCount; i++) {
            result[i] = outputNodes[i].getForwardOutputValue();
        }
        return result;
    }

    public double predictSin(double input) {
        double result = 0;
        forward(new double[]{input});
        result = outputNodes[0].getForwardOutputValue();
        return result;
    }

    public int predictClassification(double[] input, double[] expect) {
        forward(input);
        int maxOutputIndex = 0;
        double maxOutputValue = 0;
        int maxExpectIndex = 0;
        double maxEcpectVlaue = 0;
        for (int i = 0; i < outputCount; i++) {
            if (outputNodes[i].getForwardOutputValue() > maxOutputValue) {
                maxOutputIndex = i;
                maxOutputValue = outputNodes[i].getForwardOutputValue();
            }
            if (expect[i] > maxEcpectVlaue) {
                maxExpectIndex = i;
                maxEcpectVlaue = expect[i];
            }
        }

        if (maxExpectIndex != maxOutputIndex) {
            return 0;
        } else {
            return 1;
        }
    }

}
