import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

/**
 * Created by altynbek.kochkonbaev on 17.09.2020.
 */
public class Main {
    public static void main(String[] args) {


        NeuralNetwork neuralNetwork = new NeuralNetwork();

        Layer  inputLayer = new Layer();
        Layer  hiddenLayer = new Layer();
        Layer  hiddenLayer2 = new Layer();
        Layer  outputLayer = new Layer();


        inputLayer.addNeuron(new Neuron());
        inputLayer.addNeuron(new Neuron());

        hiddenLayer.addNeuron(new Neuron());
        hiddenLayer.addNeuron(new Neuron());
        hiddenLayer.addNeuron(new Neuron());
        hiddenLayer.addNeuron(new Neuron());

        hiddenLayer2.addNeuron(new Neuron());
        hiddenLayer2.addNeuron(new Neuron());
        hiddenLayer2.addNeuron(new Neuron());
        hiddenLayer2.addNeuron(new Neuron());


        outputLayer.addNeuron(new Neuron());




        neuralNetwork.addLayer(0,inputLayer);
        neuralNetwork.addLayer(1,hiddenLayer);

        ConnectionFactory.fullConnect(neuralNetwork.getLayerAt(0),neuralNetwork.getLayerAt(1));
        neuralNetwork.addLayer(2, hiddenLayer2);
        ConnectionFactory.fullConnect(neuralNetwork.getLayerAt(1), neuralNetwork.getLayerAt(2));
        neuralNetwork.addLayer(3, outputLayer);
        ConnectionFactory.fullConnect(neuralNetwork.getLayerAt(2), neuralNetwork.getLayerAt(neuralNetwork.getLayersCount() - 1), false);
        neuralNetwork.setInputNeurons(inputLayer.getNeurons());
        neuralNetwork.setOutputNeurons(outputLayer.getNeurons());





        int inputSize = 2;
        int outputSize = 1;
        DataSet dataSet = new DataSet(inputSize,outputSize);




        DataSetRow dataSetRow1 = new DataSetRow(new double[]{0,1},new double[]{1});

        dataSet.addRow(dataSetRow1);

        DataSetRow dataSetRow2 = new DataSetRow(new double[]{1,1},new double[]{0});
        dataSet.addRow(dataSetRow2);
        DataSetRow dataSetRow3 = new DataSetRow(new double[]{0,0},new double[]{0});
        dataSet.addRow(dataSetRow3);
        DataSetRow dataSetRow4 = new DataSetRow(new double[]{1,0},new double[]{1});
        dataSet.addRow(dataSetRow4);


        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(5000);
        neuralNetwork.learn(dataSet,backPropagation);



        neuralNetwork.setInput(0,1);
        neuralNetwork.calculate();
        double [] d = neuralNetwork.getOutput();

        for(int i =0;i<d.length;i++){
            System.out.println(d[i]);
        }
    }
}
