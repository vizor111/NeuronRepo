using System;
using System.Collections.Generic;

namespace NeuronNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; set; }

        public NeuronType NeuronType { get; set; }

        public double Output { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;

            Weights = new List<double>();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(i);
            }
        }

        public double FeedForward(List<double> input)
        {
            var sum = 0.0;

            for (int i = 0; i < input.Count; i++)
            {
                sum += input[i] * Weights[i];
            }

            Output = Signoid(sum);

            return Output;
        }

        public double Signoid(double x)
        {
            var result = 1 / (1 - Math.Pow(Math.E, -x));

            return result;
        }

        public void SetWeights(params double[] weights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
