# Privacy-Preserving Sequential Recommendation with Collaborative Confusion


## Introduction
Sequential recommendation has attracted a lot of attention from both academia and industry, however the privacy risks associated with gathering and transferring users' personal interaction data are often underestimated or ignored. Existing privacy-preserving studies are mainly applied to traditional collaborative filtering or matrix factorization rather than sequential recommendation. Moreover, these studies are mostly based on differential privacy or federated learning, which often lead to significant performance degradation, or have high requirements for communication. 

In this work, we address privacy-preserving from a different perspective. Unlike existing research, we capture collaborative signals of neighbor interaction sequences and directly inject indistinguishable items into the target sequence before the recommendation process begins, thereby increasing the perplexity of the target sequence. Even if the target interaction sequence is obtained by attackers, it is difficult to discern which ones are the actual user interaction records. To achieve this goal, we introduce a novel sequential recommender system called CLOUD, which incorporates a collaborative confusion mechanism to modify the raw interaction sequences before conducting recommendation. Specifically, CLOUD first calculates the similarity between the target interaction sequence and other neighbor sequences to find similar sequences. Then, CLOUD considers the shared representation of the target sequence and similar sequences to determine the operation to be performed: keep, delete, or insert. A copy mechanism is designed to make items from similar sequences have a higher probability to be inserted into the target sequence. Finally, the modified sequence is used to train the recommender and predict the next item. 

We conduct extensive experiments on three benchmark datasets. 
The experimental results show that CLOUD achieves a maximum modification rate of 66.57% on interaction sequences, 
and obtains over 99% recommendation accuracy compared to the state-of-the-art sequential recommendation methods. 
This proves that CLOUD can effectively protect user privacy at minimal recommendation performance cost, which provides a new solution for privacy-preserving for sequential recommendation.
## Requirements

To install the required packages, please run:

```python
pip install -r requirements.txt
```

## Datasets

We use [Beauty](http://jmcauley.ucsd.edu/data/amazon/links.html), [Sports_and_Outdoors ](http://jmcauley.ucsd.edu/data/amazon/links.html)and [Yelp](https://www.yelp.com/dataset) datasets for experiments. We have uploaded the processed datasets here. However, if you download raw datasets from official websites, please refer to *./dataprocessing/readme.md* for the details about dataset processing.

## Experiments

For training, validating and testing model on *Beauty* dataset, please run:

```python
python main.py -m=train
python main.py -m=valid
python main.py -m=test
```

For other datasets, please revise the path of dataset and item_num in *main.py*.

If you want to set the probabilities of keep, delete and insert for generating randomly modified sequences when training, please revise the plist in *main.py*.

If you want to get the performances on the changed sequence group and the unchanged sequence group under a certain epoch, for example, evaluate the 100th epoch, please run:

```python
python evaluate_on_mod.py -e 100
```

Tips:
If there is no such large resource, you need to set a smaller batch size in *main.py*. 
