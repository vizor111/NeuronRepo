using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuronNetwork
{
    public class NeuralNetwork
    {
        public Topology Topology { get; set; }

        public List<Layer> Layers { get; set; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public void FeedForward(List<double> inputSignals)
        {
            Neuron neuron;

            SendSignalsToInputNeurions(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                neuron = Layers.Last().Neurons[0];
            }
            else
            {
                neuron = Layers.Last().Neurons.OrderByDescending(n => n.Output).FirstOrDefault(); ;
            }
        }

        private void SendSignalsToInputNeurions(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
            {
                var signal = new List<double> { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var prevLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(prevLayerSignals);
                }
            }
        }

        private void CreateInputLayer()
        {
            var neurons = new List<Neuron>();

            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                neurons.Add(neuron);
            }

            var inputLayer = new Layer(neurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateOutputLayer()
        {
            var neurons = new List<Neuron>();

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Output);
                neurons.Add(neuron);
            }

            var outputLayer = new Layer(neurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int i = 0; i < Topology.HiddeLayers.Count; i++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();

                for (int j = 0; j < Topology.HiddeLayers[i]; j++)
                {
                    var neuron = new Neuron(lastLayer.Count);
                    hiddenNeurons.Add(neuron);
                }

                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }
    }
}
