using System.Collections.Generic;

namespace NeuronNetwork
{
    public class Topology
    {
        public int InputCount { get; set; }

        public int OutputCount { get; set; }

        public List<int> HiddeLayers { get; set; }

        public Topology(int inputCount, int outputCount, params int[] layers)
        {
            InputCount = InputCount;
            OutputCount = outputCount;

            HiddeLayers = new List<int>();

            HiddeLayers.AddRange(layers);
        }
    }
}
