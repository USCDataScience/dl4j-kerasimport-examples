# DL4J Examples for Importing Keras Models

This project contains an example for importing deeplearning models trained on Keras to Deeplearning4j to embed them in JVM applications.

The (first) example used here is InceptionV3 network trained on ImageNet dataset, used for image recognition.
 
 
## Build instructions
+ Using maven : `mvn clean compile`


## Example Usage
+ Import this maven project into your IDE and run the `main` method inside `KerasInceptionV3Net`.
+ or from CLI: `mvn compile exec:java -Dexec.mainClass="edu.usc.irds.dl.dl4j.examples.KerasInceptionV3Net"`
 
#### Sample Code:
```java
String modelFile = "data/inception-model.json";
String weightsFile = "data/inception-model-weights.h5";
String classMapFile = "data/imagenet_class_index.json";
File imagesDir = new File("data/images");
KerasInceptionV3Net net = new KerasInceptionV3Net(modelFile, weightsFile);
try (InputStream is = new FileInputStream(classMapFile)){
    net.loadClassIndex(is);
}
for (String imageFile: imagesDir.list((dir, name) -> name.endsWith(".jpg"))) {
    INDArray imgData = net.preProcessImage(new File(imagesDir, imageFile).getPath());
    long st = System.currentTimeMillis();
    List<Label> results = net.classify(imgData);
    long timeTaken = System.currentTimeMillis() -  st;
    System.out.println(imageFile + ":: " + timeTaken + "ms" + " ::: " + results);
}
```

#### Sample output:
```
2017-02-11 19:10:56 INFO  KerasInceptionV3Net:55 - Time taken to import model 4514 ms
ak47.jpg:: 1225ms ::: [Label(413, 0.8965, assault_rifle)]
cat1.jpg:: 807ms ::: [Label(278, 0.2900, kit_fox), Label(280, 0.0811, grey_fox), Label(279, 0.0584, Arctic_fox)]
cat2.jpg:: 909ms ::: [Label(285, 0.3412, Egyptian_cat), Label(281, 0.1530, tabby), Label(834, 0.0524, suit)]
german_shepherd_dog.jpg:: 839ms ::: [Label(235, 0.5327, German_shepherd), Label(225, 0.3057, malinois)]
handgun.jpg:: 852ms ::: [Label(763, 0.5338, revolver), Label(413, 0.3358, assault_rifle), Label(764, 0.0724, rifle)]
```



## License
   + Apache Licence 2.0

## Developers
+ Thamme Gowda [@thammegowda](http://twitter.com/thammegowda)
+ Chris Mattmann [@chrismattmann](http://twitter.com/chrismattmann)


## Credits / Acknowledgements

The credits goes to these awesome people:

+ Developers of **Deeplearning4j**
  + **Dave kale(turambar)** and **Susan Eraly(eraly)** for their work on Keras Import model features and its documentation at https://deeplearning4j.org/model-import-keras 
  + **Samuel Audet @saudet ** for answering my build questions on their gitter channel at https://gitter.im/deeplearning4j/deeplearning4j
+ **François Chollet** for keras and also making InceptionV3 models available under MIT license
   + Keras model and weights are extracted from his code at: https://github.com/fchollet/deep-learning-models
 
