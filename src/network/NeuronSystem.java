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


    public NeuronSystem(int inputCount, int[] hiddenCount, int outputCount, double rating) {
        this.inputCount = inputCount;
        this.hiddenLayerCount = hiddenCount.length;
        this.hiddenCount = hiddenCount;
        this.outputCount = outputCount;
        this.rating = rating;
        reset();
    }

    public void train(double[] input, double[] output) {
//        for (int i = 0; i < input.length; i++) {
//            System.out.print("input " + i + " : " + input[i] + "; ");
//        }
//        System.out.println();
//        for (int i = 0; i < output.length; i++) {
//            System.out.print("target " + i + " : " + output[i] + "; ");
//        }
//        System.out.println();
        forward(input);
//        for (int i = 0; i < outputCount; i++) {
//            System.out.print("system " + i + " : " + outputNodes[i].getForwardOutputValue() + "; ");
//        }
//        System.out.println();
        double errorSum = 0;
        for (int i = 0; i < outputCount; i++) {
            double error = outputNodes[i].getForwardOutputValue() - output[i];
            errorSum += error;
            System.out.printf("error index %d %10.7f ;", i, error);
        }
        System.out.println();
        if (outputCount > 1) {
            System.out.println("average error " + (errorSum / outputCount));
        }
        backward(output);
        updateWeights();
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
                temp += (weightArray[hiddenLayerCount][j][i] * hiddenNodes[ hiddenLayerCount - 1][j].getForwardOutputValue());
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

    private void updateWeights() {
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
                biasArray[i][j] = (Math.random() % 0.02) - 0.01;
            }
            for (int j = 0; j < hiddenCount[i]; j++) {
                hiddenNodes[i][j] = new NeuronNode(NeuronNode.NODE_TYPE.HIDDEN);
            }
        }
        outputNodes = new NeuronNode[outputCount];
        biasArray[hiddenLayerCount] = new double[outputCount];
        for (int i = 0; i < outputCount; i++) {
            biasArray[hiddenLayerCount][i] = (Math.random() % 0.02) - 0.01;
        }
        for (int i = 0; i < outputCount; i++) {
            outputNodes[i] = new NeuronNode(NeuronNode.NODE_TYPE.OUTPUT);
        }

        weightArray = new double[hiddenLayerCount + 1][][];
        weightArray[0] = new double[inputCount][hiddenCount[0]];
        for (int i = 0; i < inputCount; i++) {
            for (int j = 0; j < hiddenCount[0]; j++) {
                weightArray[0][i][j] = (Math.random() % 0.02) - 0.01;
            }
        }
        for (int i = 1; i < hiddenLayerCount; i++) {
            weightArray[i] = new double[hiddenCount[i - 1]][hiddenCount[i]];
            for (int j = 0; j < hiddenCount[i - 1]; j++) {
                for (int k = 0; k < hiddenCount[i]; k++) {
                    weightArray[i][j][k] = (Math.random() % 0.02) - 0.01;
                }
            }
        }
        weightArray[hiddenLayerCount] = new double[hiddenCount[hiddenLayerCount - 1]][outputCount];
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
}
