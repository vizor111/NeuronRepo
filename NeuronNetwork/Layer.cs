using System.Collections.Generic;

namespace NeuronNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; set; }

        public NeuronType Type { get; set; }

        public int NeuronsCount
        {
            get { return Neurons?.Count ?? 0; }
        }

        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            Neurons = neurons;
            Type = type;
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();

            foreach (var neur in Neurons)
            {
                result.Add(neur.Output);
            }

            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
