using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public class Layer {
        public List<Neuron> Neurons = new List<Neuron>();

        public Layer () {

        }

        public Neuron AddNeuron () {
            Neuron neuron = new Neuron();
            Neurons.Add(neuron);

            return neuron;
        }
    }
}
