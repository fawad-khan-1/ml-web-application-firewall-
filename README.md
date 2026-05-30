# machine-learning-web-application-firewall

Title:
A Universal Machine-Learning Framework for Detecting and Preventing File Inclusion Attacks in PHP Web Applications

## Description
This project investigates the use of machine learning technology in the detection and prevention of file inclusion attacks against web applications built with PHP

## Project Overview
Traditional rule-based firewalls such as ModSecurity rely on rigid rule structures to detect signature patterns of file inclusion attacks. But if they encounter patterns that are falling outside of their rulesets, their detection abilities can be negatively impacted. In other words, they are very good at detecting well known attack patterns, but can suffer if the patterns are either too complex, as in the case of higly evasive obfuscated payload patterns, or completely unseen such as the advanced multi-stage attack patterns. This project investigates how a firewall driven by machine learning would fare in a similar situation. The idea here is to train a machine learning model on well known attack patterns of both local and remote file inclusion, and then see how well it can generalize those patterns to make predictions in the case of highly complex and obfuscated patterns, as well as previously unseen patterns such as those found in advanced multi-stage attack payloads.

Even though, file inclusion attacks are not as widespread as other ones such as SQL Injections(SQLi) and Cross Site Scripting(XSS), they can be used to carry out those forms of attacks too. Since the consequences of file inclusion attacks can be extremely serious, often leading to severe system compromises, it was deemed necessary to do further studies on them, and find out how machine learning can be used to aid in their detection and mitigation. This study aims to better the understanding of ways to identify and prevent these attacks.

The machine learning model built for this study is trained on a dataset which contains patterns of both local and remote file inclusion, as well as some of the more well known obfuscated patterns. After the model's initiaal training and testing phase is over, it is further tested in a simulation of an actual live attack. It is attacked with both patterns of higly complex and obfuscated payloads, as well as several advanced multi-stage attack payloads. Its performance is measured with metrics such as accuracy, precision, recall and F1 score. The performance of this machine learning model is compared with the performance of ModSecurity when ModSecurity is also attacked with the same attack vectors. This comparison is done to determine if machine learning performs better or worse than rule-based firewalls when dealing with the same types of attack payloads.

## Research Objective
The primary objective of this study is to evaluate the use of machine learning in detecting and preventing file inclusion attacks against PHP web applications. It studies both local file inclusion, and remote file inclusion.

## Technologies Used
- Python
- Machine Learning
- Scikit-learn
- TensorFlow / Keras
- Pandas
- Numpy
- matplotlib
- seaborn
- FastAPI
- VirtualBox
- Ubuntu Linux VM
- Kali Linux VM

 ## Machine Learning Pipeline
  1. An enhanced file inclusion dataset is loaded into the model
  2. Duplicates and missing values are removed from the dataset
  3. Dataset is vectorized, to convert string values to a numerical format
  4. Dataset is split into training and testing portions using a ratio of 80/20. 80% for training, and 20% for testing
  5. Six well-known, classification machine learning algorithms are trained and tested on the dataset
  6. The test results of each algorithm in the ML model are output, and evaluated
  7. A cross-validation is performed at the end of model training and testing to determine the consistency of each algorithm
  8. Each algorithm in the model is then tested in a simulated live attack scenario which is set up in VirtualBox
  9. Each algorithm is tested with three sets of attack payloads. These three sets are made up of conventional well-known signature patterns, highly complex obfuscated patterns, and advanced multi-stage attack patterns
  10. The performance of each algorithm is recorded in tabular form, and compared with the performance of ModSecurity which is also tested with the same payload patterns
  11. The final determination about the best performing algorithms for handling each form of attack payloads is made at the end
 
 ## Attack Detection Focus
  The focus of this project is on:
  - Local File Inclusion (LFI)
  - Remote File Inclusion (RFI)
  - Highly complex obfuscated patterns
  - Advanced multi-stage attack patterns

 ## Repository Scope
  This repository is focused mainly on the machine learning model built here with six, classification algorithms. This model was built to be a part of a much broader research project. The model code presented here       shows dataset preprocessing,feature extraction, model training and attack classification logic.

 ## Results
  The results of this study show that for attack patterns which are well known, rule-based systems like ModSecurity show the best performance. But for attack patterns that are highly obfuscated, or relatively unknown such as the advanced multi-stage attack patterns, machine learning shows better performance due to its ability to learn patterns, and then use those learned patterns to generalize to previously unseen ones.

 
