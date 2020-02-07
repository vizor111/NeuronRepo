using System.Collections.Generic;

namespace NeuronNetwork
{
    public class Topology
    {
        public int InputCount { get; set; }

        public int OutputCount { get; set; }

        public double LearningRate { get; set; }

        public List<int> HiddeLayers { get; set; }

        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;

            HiddeLayers = new List<int>();
            HiddeLayers.AddRange(layers);
        }
    }
}
