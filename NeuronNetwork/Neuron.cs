using System;
using System.Collections.Generic;

namespace NeuronNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; set; }

        public List<double> Inputs { get; set; }

        public NeuronType NeuronType { get; set; }

        public double Output { get; private set; }

        public double Delta { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;

            Weights = new List<double>();
            Inputs = new List<double>();

            InitWeightsRandomValue(inputCount);
        }

        private void InitWeightsRandomValue(int inputCount)
        {
            Random rand = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    var num = rand.NextDouble();

                    Weights.Add(num);
                }

                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> input)
        {
            var sum = 0.0;

            for (int i = 0; i < input.Count; i++)
            {
                Inputs[i] = input[i];
            }

            for (int i = 0; i < input.Count; i++)
            {
                sum += input[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Signoid(sum);
            }
            else
            { 
                Output = sum;
            }

            return Output;
        }

        private double Signoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        private double SignoidDx(double x)
        {
            var signoid = Signoid(x);
            var result = signoid / (1 - signoid);
            return result;
        }

        public void SetWeights(params double[] weights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SignoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
