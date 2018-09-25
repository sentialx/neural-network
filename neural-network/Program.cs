using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    class Program {
        static void Main(string[] args) {
            NeuralNetwork nn = new NeuralNetwork();

            nn.AddLayer(2);
            nn.AddLayer(8);
            nn.AddLayer(8);
            nn.AddLayer(1);

            double[][] inputs = {
                new double[] { 0, 1 },
                new double[] { 1, 1 }
            };

            double[][] outputs = {
                new double[] { 0 },
                new double[] { 1 },
            };

            nn.Train(inputs, outputs);

            nn.ForwardPropagate(new double[] { 1, 1 });

            Console.ReadKey();
        }
    }
}
