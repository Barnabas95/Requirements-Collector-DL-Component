# Requirements-Collector: Automating Requirements Specification from Elicitation Sessions and User Feedback

Requirements-Collector: (Automating Requirements Specification from Elicitation Sessions and User Feedback) Tool - Version 1.0

#### Authors: [Sebastiano Panichella](https://spanichella.github.io/index.html) and [Marcela Ruiz](https://www.marcelaruiz.eu/)

### Tool description: 

*In  the  context  of  digital  transformation,  speeding up  the  time-to-market  of  high-quality  software  products  is  a big  challenge.  Software  quality  correlates  with  the  success  of requirements engineering (RE) sessions and the ability to collect feedback  from  end-users  in  an  efficient,  dynamic  way.  Thus, software  analysts  are  tasked  to  collect  all  relevant  material  of RE  sessions  and  user  feedback,  usually  specified  on  written notes,   flip   charts,   pictures,   and   (mobile   apps)   user   reviews. Afterwards comprehensible requirements need to be specified for software implementation and testing. These activities are mostly performed manually, which causes process delays; and software quality attributes like reliability, usability, comprehensibility, etc., are diminished causing software devaluation. This paper presents the design of the Requirements-Collector tool, for automating the tasks  of  requirements  specification  and  user  feedback  analysis. The proposed tool involves computational mechanisms to enable the  automatic  classification  of  requirements  discussed  in  RE meetings   (stored   in   form   of   audio   recordings)   and   textual feedback  in  form  of  user  reviews.  The  requirements  collector tool  has  the  potential  to  renovate  the  role  of  software  analysts, which can experience substantial reduction of manual tasks, more efficient communication, dedication to more analytical tasks, and assurance  of  software  quality  from  conception  phases.  Results of  this  work  have  shown  that the  requirements  collector tool is  able  to automatically  exploits  machine  and  deep-learning  strategies  to classify RE specifications and user review feedback with reliable accuracy*.

**Requirement-Collector Context:**
![](https://github.com/spanichella/Requirement-Collector-tool/blob/master/Pipeline_AutomatedRE-RE20-P%26D.png)

- [Video Demonstration of the requirements collector including DL and ML components](https://youtu.be/MfPRBBIXvQY)

#### License
This program is a free software you can redistribute it under the terms of the GNU Public License
as published by the Free Software Fundation either version 2 of the License, or (at your option)
any later version.

#### Get started with the *"Deep-Learning-classifier"* component of Requirement-Collector
Please clone this repository and copy the content of the folder to a local folder of your choice.
Inside this folder you find the *"src"* folder, containing all required executable files for enabling the automated classification of requirements  specification  from Requirement meetings transcripts. Example of data are provided to experiment with the automated classification of Audio Transcripts of Requirement meetings. You can run and explore the code by importing in in your favourite IDE. The following figure exemplify the results from running the DL Component.

![](https://github.com/lmruizcar/Requirements-Collector-DL-Component/blob/master/RequirementsCollectorDLOutputExample.png)

### Demonstration video: [![Requirement-Collector-DL-Component Demonstration](https://github.com/lmruizcar/Requirements-Collector-DL-Component/blob/master/RequirementsCollector-DLComponent.PNG)](https://youtu.be/OlI7oXzP4OI)
- [Youtube link](https://youtu.be/OlI7oXzP4OI)
#### Requirements
- Java 13
- Cores: [8]
- Memory: [4.9GB]

## References

[1] M. Ruiz and B. Hasselmann, "Can We Design Software as We Talk: A research idea, International working conference on Exploring Modeling Methods for Systems Analysis and Development, 2020
  
[2] S. Panichella, A. Di Sorbo, E. Guzman, C. A. Visaggio, G. Canfora, andH. C. Gall, “Ardoc: App reviews development oriented classifier,” International Symposium Foundations of Software Engineering, 2016

