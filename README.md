<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
![deep](https://github.com/Mishu791/Melanoma_Skin-Cancer_Detection_Model/blob/master/images/deep.png)

* In this project we created a model to identify melanoma in images of skin lesions. And after that we also deployed it in Google Cloud for serving this as a web application for early detection of skin cancer of users. In particular, we used images within the same patient and determine which are likely to represent a melanoma. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.

* Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people.

* The dataset has been obtained from Kaggle. (https://www.kaggle.com/c/siim-isic-melanoma-classification/data) 

### Built With

* [Pytorch](https://pytorch.org/)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/)


### Prerequisites

The non-trival dependencies to be installed 
* wtfml
  ```sh
  !pip install wtfml=0.0.2
  ```
* pretrainedmodels
  ```sh
  !pip install pretrainedmodels
  ```



<!-- USAGE EXAMPLES -->
## Usage

Although the model has beed tested on the test dataset provided in the Kaggle competition. The usage case can earn more credibility if more data is used or can be tested with more patient skin lesion images. Interestingly i tried with my skin image! it gave a prediction of 0.615 which refers to a false positve. I will be working on that for further improvement.

For more about early detection of melanoma, please refer to this (https://www.sciencedaily.com/releases/2017/10/171017091900.htm)



<!-- References -->
## References
* [Melanoma Classification](https://www.kaggle.com/ibtesama/melanoma-classification-with-attention)
* [Albumentation](https://albumentations.ai/docs/examples/pytorch_classification/)
