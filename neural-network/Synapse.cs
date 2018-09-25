using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public class Synapse {
        public double Weight;
        public Neuron Neuron;

        public Synapse (Neuron neuron, double weight) {
            this.Weight = weight;
            this.Neuron = neuron;
        }
    }
}
