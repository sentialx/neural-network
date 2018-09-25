using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public class NeuralNetwork {
        public List<Layer> Layers = new List<Layer>();
        private Random random = new Random();

        public NeuralNetwork() {

        }

        public Layer AddLayer (int num) {
            Layer layer = new Layer();

            for (int x = 0; x < num; x++) {
                Neuron neuron = layer.AddNeuron();
                if (Layers.Count > 0) {
                    Layer previousLayer = Layers[Layers.Count - 1];

                    for (int y = 0; y < previousLayer.Neurons.Count; y++) {
                        Neuron previousNeuron = previousLayer.Neurons[y];
                        neuron.AddInputSynapse(previousNeuron, GetRandomNumber(-1, 1));
                    }
                }
                neuron.Bias = GetRandomNumber(-1, 1);
            }

            if (Layers.Count > 0) {
                Layer previousLayer = Layers[Layers.Count - 1];

                for (int x = 0; x < previousLayer.Neurons.Count; x++) {
                    Neuron previousNeuron = previousLayer.Neurons[x];

                    for (int y = 0; y < layer.Neurons.Count; y++) {
                        Neuron neuron = layer.Neurons[y];

                        Synapse synapse = new Synapse(neuron, neuron.InputSynapses[x].Weight);
          
                        previousNeuron.AddOutputSynapse(synapse);
                    }
                }
            }

            Layers.Add(layer);

            return layer;
        }

        public static double Sigmoid (double x) {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double SigmoidDerivative (double x) {
            return x * (1.0 - x);
        }

        public double GetRandomNumber(double minimum, double maximum) {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }

        public void ForwardPropagate (double[] input) {
            for (int x = 0; x < Layers[0].Neurons.Count; x++) {
                Layers[0].Neurons[x].Value = input[x];
                Layers[0].Neurons[x].Activation = input[x];
            }

            for (int x = 1; x < Layers.Count; x++) {
                for (int y = 0; y < Layers[x].Neurons.Count; y++) {
                    Layers[x].Neurons[y].CalculateOutput();
                }
            }
        }

        public void BackPropagate (double[] input, double[] expectedOutput) {
            Layer outputLayer = Layers[Layers.Count - 1];

            for (int x = 0; x < outputLayer.Neurons.Count; x++) {
                Neuron neuron = outputLayer.Neurons[x];
                neuron.Error = (expectedOutput[x] - neuron.Activation) * SigmoidDerivative(neuron.Activation);
            }

            for (int x = Layers.Count - 2; x > 0; x--) {
                Layer layer = Layers[x];

                for (int y = 0; y < layer.Neurons.Count; y++) {
                    Neuron neuron = layer.Neurons[y];
                    double error = 0;

                    for (int z = 0; z < neuron.OutputSynapses.Count; z++) {
                        error += neuron.OutputSynapses[z].Weight * neuron.OutputSynapses[z].Neuron.Error;
                    }
                    error *= SigmoidDerivative(neuron.Activation);

                    neuron.Error = error;
                }
            }

            for (int x = Layers.Count - 1; x > 0; x--) {
                Layer layer = Layers[x];

                for (int y = 0; y < layer.Neurons.Count; y++) {
                    Neuron neuron = layer.Neurons[y];

                    for (int z = 0; z < neuron.InputSynapses.Count; z++) {
                        for (int w = 0; w < input.Length; w++) {
                            neuron.InputSynapses[z].Weight += 0.5 * neuron.Error * input[w];
                        }
                    }

                    neuron.Bias += 0.5 * neuron.Error;

                    Layer previousLayer = Layers[x - 1];
                    if (previousLayer != null) {
                        for (int z = 0; z < previousLayer.Neurons.Count; z++) {
                            Neuron previousNeuron = previousLayer.Neurons[z];
                            previousNeuron.OutputSynapses[y].Weight = neuron.InputSynapses[z].Weight;
                        }
                    }
                }
            }
        }

        public void Train (double[][] inputs, double[][] outputs) {
            for (int x = 0; x < 50000; x++) {
                for (int y = 0; y < inputs.Length; y++) {
                    ForwardPropagate(inputs[y]);
                    BackPropagate(inputs[y], outputs[y]);
                }
            }
        }
    }
}
