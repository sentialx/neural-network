using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public class Neuron {
        public List<Synapse> InputSynapses = new List<Synapse>();
        public List<Synapse> OutputSynapses = new List<Synapse>();

        public double Value;
        public double Activation;
        public double Error;
        public double Bias;

        public Neuron () {
            
        }

        public void SetValue (double value) {
            this.Value = value;
            this.Activation = NeuralNetwork.Sigmoid(value);
        }

        public Synapse AddInputSynapse (Neuron neuron, double weight) {
            Synapse synapse = new Synapse(neuron, weight);

            InputSynapses.Add(synapse);

            return synapse;
        }

        public Synapse AddOutputSynapse (Synapse synapse) {
            OutputSynapses.Add(synapse);

            return synapse;
        }

        public double CalculateOutput () {
            double output = Bias;

            foreach (Synapse synapse in InputSynapses) {
                output += synapse.Neuron.Activation * synapse.Weight;
            }

            this.SetValue(output);

            return output;
        }
    }
}
