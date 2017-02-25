Project: Project_003 - Anomaly detection in network traffic
==============

Participants
============

* Bhartia, Shweta, FG-16-IG-3002, sbhartia, sbhartia@umail.iu.edu
* Sripada, Pramod, FG-16-IG-3018, pramodvspk, ksripada@umail.iu.edu
* Nallani, Ramkaushik, FG-16-IG-3015, rnallani, rnallani@umail.iu.edu

Abstract
========

Biggest threat in twenty first century is Cyber war. Hackers are trying to intrude into network and hack the systems present in that system thereby stealing the sensitive information. This urges the necessity of detecting anomalous or malicious activities in network so that proper actions can be taken. Anomaly detection in network traffic is challenging task. The anomaly detection classifiers or the anomaly detection models built should be sensitive in not misclassifying normal usage as threats. Too many misclassifications of normal usage results in the ignorance of all the notifications which puts entire system at stake. This paper speaks about data mining techniques devised to detect anomalies in network traffic. Supervised learning methods such as classification can be used in detecting known threats, i.e. the threats which have been experienced in historic data. Anomalies fall under "unknown-unknown" category. Hence supervised techniques cannot be used to find the unknown unknown. Anomaly detection in network traffic is challenging task. Data mining techniques make it possible to search large amount of data for characteristic rules and patterns. In this paper, we build an anomaly classifier using classification methods such as Decision Trees and Random Forest Classifiers, discuss its challenges and then present an anomaly detection algorithm which will be built on K means clustering. Anomaly classifier has the ability to classify earlier patterns but the application of K means technique on training dataset results in the formation of clusters. Corresponding cluster centroids are used as patterns for detection of anomalies in new monitoring data. Proper care has to be taken while building the models which should not be overfitting as small changes in the testing data can render the model useless. We precisely described about the data mining methodology devised for anomaly detection in network traffic in this paper.

Refernces
=========

https://gitlab.com/cloudmesh_fall2016/project-003/blob/master/references.bib

Deliverables
============

Use subdirectories to cleanly manage your deliverables where appropriate

* The installation instructions are present in the README.rst file inside the code/python folder
* The data has been uploaded into the repository and been referenced accordingly in the programs
* Python program to run the kddcup analysis
* Spark programs to run the kddcup analysis on different clouds
* Report in original format
* Report in PDF
* Images directory with all images included in the report