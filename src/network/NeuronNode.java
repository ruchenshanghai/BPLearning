package network;

public class NeuronNode{
    public enum NODE_TYPE {
        INPUT, HIDDEN, OUTPUT
    }
    public enum SIGMOID_FUNCTION_TYPE {
        logFunction, tanFunction
    }
    private NODE_TYPE nodeType;
    // default type
    private SIGMOID_FUNCTION_TYPE sigmoidFunctionType = SIGMOID_FUNCTION_TYPE.tanFunction;

//    public int layerNo;
    private double forwardInputValue;
    private double forwardOutputValue;
    private double backwardInputValue;
    private double backwardOutputValue;

    private double sigmoidProject(double x) {
        switch (sigmoidFunctionType) {
            case logFunction: {
                double result = Math.exp(-x);
                result = 1 / (1 + result);
                return result;
            }
            case tanFunction: {
                double temp1 = Math.exp(x);
                double temp2 = Math.exp(-x);
                double result = (temp1 - temp2) / (temp1 + temp2);
                return result;
            }
        }
        return 0;
    }
    private double sigmoidDerivative(double inputValue) {
        switch (sigmoidFunctionType) {
            case logFunction: {
                double result = 1 - forwardOutputValue;
                result = forwardOutputValue * result * inputValue;
                return result;
            }
            case tanFunction: {
                double temp = 1 - Math.pow(forwardOutputValue, 2);
                double result = temp * inputValue;
                return result;
            }
        }
        return 0;
    }

    public double getForwardInputValue() {
        return forwardInputValue;
    }
    public void setForwardInputValue(double forwardInputValue) {
        this.forwardInputValue = forwardInputValue;
        setForwardOutputValue(forwardInputValue);
    }
    public double getForwardOutputValue() {
        return forwardOutputValue;
    }
    public void setForwardOutputValue(double forwardInputValue) {
        switch (nodeType) {
            case INPUT: {
                this.forwardOutputValue = forwardInputValue;
                break;
            }
            case HIDDEN:
            case OUTPUT: {
                this.forwardOutputValue = sigmoidProject(forwardInputValue);
                break;
            }
        }
    }
    public double getBackwardInputValue() {
        return backwardInputValue;
    }
    public void setBackwardInputValue(double backwardInputValue) {
        this.backwardInputValue = backwardInputValue;
        setBackwardOutputValue(backwardInputValue);
    }
    public double getBackwardOutputValue() {
        return backwardOutputValue;
    }
    public void setBackwardOutputValue(double backwardInputValue) {
        switch (nodeType) {
            case INPUT: {
                this.backwardOutputValue = backwardInputValue;
                break;
            }
            case HIDDEN:
            case OUTPUT: {
                this.backwardOutputValue = sigmoidDerivative(backwardInputValue);
                break;
            }
        }
    }

    public NeuronNode(NODE_TYPE nodeType, SIGMOID_FUNCTION_TYPE functionType) {
        this.nodeType = nodeType;
        this.sigmoidFunctionType = functionType;
    }

    public NeuronNode(NODE_TYPE nodeType) {
        this.nodeType = nodeType;
    }
}
