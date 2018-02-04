# Intel-Movidius-Challenge
*the following is from the competition guidlines*
## Background

Market research estimates there will be as many as [20 billion connected devices in the market by 2020](https://www.gartner.com/newsroom/id/3598917). These devices are expected to generate billions of petabytes of data traffic between [cloud](https://en.wikipedia.org/wiki/Cloud_computing) and [edge](https://en.wikipedia.org/wiki/Edge_computing) devices. In 2017 alone, 8.4 billion connected devices are expected in the market which is sparking a strong need to pre-process data at the edge. This has led many IoT device manufacturers, especially those working on vision based devices like smart cameras, drones, robots, AR/VR, etc., to bring intelligence to the edge.

Through the recent addition of the Movidius VPU technology to its existing AI edge solutions portfolio, Intel is well positioned to provide solutions that help developers and data scientists pioneer the low-power intelligent edge devices segment. The Intel Movidius Neural Compute Stick (NCS) and Neural Compute SDK (NCSDK) is a developer kit that aims at lowering the barrier to entry for developers and data scientists to develop and prototype intelligent edge devices.

In this challenge you will be pushing your network training skills to its limits by fine-tuning [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs) that are targeted for embedded applications. Contestants are expected to leverage the [Neural Compute SDK's](https://movidius.github.io/ncsdk/) (NCSDK) mvNCProfile tool to analyze the bandwidth, execution time and complexity of their network at each layer, and tune it to get the best accuracy and execution time.

## Your Objective
Your task is to design and fine-tune a convolutional neural network (CNN) which can classify images among a predefined set of image labels. The neural network should be deployed on the Intel Movidius Neural Compute Stick for inference purpose. NCSDK provides the tools to convert your pre-trained CNN to a binary file which can be deployed to NCS.

# Submission:
1. Retrain
2. Validate model graph accuracy
```
mvNCCheck meta/network.meta -w meta/network -in=input -on=FineTuning/FineTuning -s12 -is 299 299  -i topcoder_example/data/provisional/provisional_00001.jpg -S 0.05 -M -1 -cs 0,1,2
```
2. build the movidius graph file from meta and weights:
```
mvNCCompile meta/network.meta -w meta/network -in=input -on=FineTuning/FineTuning/BiasAdd -s12 -is 299 299  -o compiled.graph
```
3. Run provided inference script to get inferences.csv
```
python inferences.py "~/data"
```
4. zip files and post to onedrive for download link
- network.meta
- compiled.graph
- inferences.csv
- supported/