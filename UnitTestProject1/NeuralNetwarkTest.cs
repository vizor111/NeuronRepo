using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NeuronNetwork.Tests
{
    [TestClass]
    public class NeuralNetwarkTest
    {
        [TestMethod]
        public void FeedForwardTest()
        {
            var dataset = new List<(double, double[])>
            {
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //                T  A  S  F
               (0, new double[] { 0, 0, 0, 0 }),
               (0, new double[] { 0, 0, 0, 1 }),
               (1, new double[] { 0, 0, 1, 0 }),
               (0, new double[] { 0, 0, 1, 1 }),
               (0, new double[] { 0, 1, 0, 0 }),
               (0, new double[] { 0, 1, 0, 1 }),
               (1, new double[] { 0, 1, 1, 0 }),
               (0, new double[] { 0, 1, 1, 1 }),
               (1, new double[] { 1, 0, 0, 0 }),
               (1, new double[] { 1, 0, 0, 1 }),
               (1, new double[] { 1, 0, 1, 0 }),
               (1, new double[] { 1, 0, 1, 1 }),
               (1, new double[] { 1, 1, 0, 0 }),
               (0, new double[] { 1, 1, 0, 1 }),
               (1, new double[] { 1, 1, 1, 0 }),
               (1, new double[] { 1, 1, 1, 1 })
            };


            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);

            var difference = neuralNetwork.Learn(dataset, 100000);

            var results = new List<double>();
            foreach (var data in dataset)
            {
                var res = neuralNetwork.FeedForward(data.Item2).Output;
                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 3);
                var actual = Math.Round(results[i], 3);
                //Assert.AreEqual(expected, actual);
            }
        }
    }
}
