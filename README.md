# ML & JAva
This repo contains demos for my talk `ML & Java`. It focuses on using/comparign 3 ML frameworks and JSR-381. 

The structure of `src/main` directory is as follows:
```shell
|-- java                  
|   `-- org        
|       `-- example
|           |-- dl4j      # Deeplearning4j training + testing examples
|           `-- visrec    # JSR-381 training + testing examples
`-- resources
    |-- dataset
    |   |-- test          # Test dataset
    |   |   |-- hotdog
    |   |   `-- nothotdog
    |   `-- train         # Training dataset
    |       |-- hotdog
    |       `-- nothotdog
    `-- visrec            # JSR-381 config files
```

#### DJL serving
Start the container 
```shell
docker run -itd -p 8080:8080 deepjavalibrary/djl-serving
```
head to `http://localhost:8080` and load `resnet18` Pytorch model from `https://resources.djl.ai/demo/pytorch/traced_resnet18.zip`.