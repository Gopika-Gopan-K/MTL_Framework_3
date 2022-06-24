# MTL_Framework_3

The multi-tasking classification and segmentation network used in this work is shown in figure.

![alt text](https://github.com/Gopika-Gopan-K/MTL_Framework_3/blob/1790a77b568c7f6dcb4aea739c8d385b0c11446f/pics/block_diagram.png)

A hierarchical framework consisting of two U-nets, each having spatio-channel attention and ASPP used in this work. The first U-net outputs the lung segmentation. The segmented lung output is the concatenated channel-wise with the input CT slice and passed as the input to second U-net for lesion segmentation. The classification part of the network utilizes the rich feature representation of the second U-net and pass it through a 1x1 convolution with ReLu activation to reduce the channel from 1024 to 128. The features are then flattened input to a Dense with ReLu activation. Additionally, the lung segmentation output from first U-net and lesion segmentation output from second U-net are flattened and passed individually through a Dense layer with ReLu activation. The features are then flattened and concatenated with the flattened features from bottleneck layer. The concatenated features are then input to a Dense layer for 3-class classification. 

The classification part of the framework distinguishes the given CT slice as belonging to : (a) Closed Lung, (b) Open Lung - Normal and (c) Open Lung - Covid. The segmentation part outputs the lungs masks from first U-net and lesion masks from second U-net.

Ensure the CT volumes are in folder named “Data”, lung mask in folder “Lung Mask” and lesion mask in folder “Lesion Mask”. Both the lung mask and lesion mask should have same name as the corresponding CT volume and all three are in .nii format. The CT scans are extracted and lung window of -1000 to 400 is applied before the slices are normalized for further analysis.

