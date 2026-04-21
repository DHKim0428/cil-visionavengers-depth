# Monocular depth estimation
(Project Option 3)


Monocular depth estimation is the process of predicting the distance of objects from the camera using only a single RGB image. Unlike stereo vision, which relies on multiple viewpoints to triangulate depth, monocular depth estimation leverages visual cues such as texture gradients, object sizes, occlusions, and perspective to infer the scene’s three-dimensional structure.


## Why Tackle Monocular Depth Estimation?

* Widespread Applications:
  * Autonomous Driving: Vehicles can better understand their surroundings, identify obstacles, and plan safer navigation strategies using accurate depth maps.
  * Robotics: Robots can navigate and interact within complex environments without requiring bulky sensor arrays, leading to more compact and cost-effective solutions.
  * Augmented Reality: AR systems benefit from depth information to seamlessly blend virtual objects with real-world scenes, enhancing user experiences with more realistic interactions.
  * Scene Understanding: Depth estimation supports advanced image segmentation, object detection, and scene reconstruction, contributing to a wide range of computer vision applications.
* Cost and Hardware Efficiency:
  * Single-camera systems are cheaper and simpler to deploy compared to multi-camera setups or depth sensors, making this technology accessible for consumer devices and mass-market applications.

By solving the problem of monocular depth estimation, researchers and practitioners can unlock more intuitive and efficient ways to interpret visual data, leading to innovations across industries and contributing to the advancement of intelligent systems.


## Competition

To join the Kaggle competition, please use the private links shared on the Moodle page.

## Training Data
In this project, you are tasked with predicting depth maps from everyday scenes captured in RGB images. We provide a robust training dataset consisting of 22,605 samples of size 560x560. Each training sample includes an input RGB image and its corresponding target depth map. The depth maps are stored as NumPy arrays containing pixel-wise depth measurements in meters, with values ranging from a minimum of 1 millimeter up to a maximum of 80 meters. But depth scales between images are not meaningful and only represent ordering within individual images


Your model’s performance will be evaluated using a separate test set of 591 samples of size 560x560. Note that the ground truth depth maps for these test samples are not provided, ensuring that your model is truly tested on its ability to generalize.


After generating your predictions, save each depth map as a NumPy array (.npy) file with the appropriate filename. Then, run the provided submission generation code - pointing it to the folder containing your prediction files - to create a CSV file for submission to the Kaggle competition.


The data can be accessed on Kaggle


## Evaluation Metrics
Your approach is evaluated according to scale-invariant Root Mean Squared Error (RMSE). See a detailed description on Kaggle.





## Report
For the project submission, you must write a comprehensive 4-page report and submit the accompanying code. You can base your implementation and ideas on techniques not covered in class, and we encourage thorough comparisons by testing your novel approach against standard methods. Scientific rigor is valued, so please ensure your experiments are well-documented, and your findings are statistically sound.

**For instructions on how to write a scientific paper, see this PDF. The write-up must be a maximum of 4 pages long (excluding references, appendix, and declaration of originality). Please include the declaration of originality in your report.**

You should compare your solution to at least two baseline algorithms. For the baselines, you can use the implementations you developed as part of the programming exercises or come up with your own relevant baselines.

### Grading of the report
To ensure fairness we will use the same grading criteria for all reports. Your reports will be graded according to the following three criteria:

1. Quality of paper (30%)

6.0: Comparable to a (workshop) submission to an international conference (features discussed in the last lecture).

5.5: Background, method, and experiment are clear. May have minor issues in one or two sections. Language is good. Scores and baselines are well documented.

5.0: Explanation of work is clear, and the reader is able to identify the novelty of the work. Minor issues in one or two sections. Minor problems with language. Has all the recommended sections in the howto-paper.

4.5: Able to identify contributions. Major problems in the presentation of results and or ideas and or reproducibility/baselines.

4.0: Hard to identify contribution, but still there. One or two good sections should get students a pass.

3.5: Unable to see novelty. No comparison with any baselines.

2. Creativity of solution (20%)

6.0: Elegant proposal, either making a useful assumption, studying a particular class of data, or using a novel mathematical fact.

5.5: A non-obvious combination of ideas presented in the course or published in a paper (Depending on the difficulty of that idea).

5.0: A novel idea or combination not explicitly presented in the course.

4.5: An idea mentioned in a published paper with small extensions/changes, but not so trivial to implement.

4.0: A trivial idea taken from a published paper.

Quality of implementation (20%)

6.0: Idea is executed well. The experiments make sense in order to answer the proposed research questions. There are no obvious experiments not done that could greatly increase clarity. The submitted code and other supplementary material are understandable, commented, complete, clean and there is a README file that explains it and describes how to reproduce your results.

Subtractions from this grade will be made if
	* the submitted code is unclear, does not run or experiments cannot be reproduced or there is no description of it
    * experiments done are useless to gain understanding or of unclear nature or obviously useful experiments have been left undone
    * comparison to baselines are not done

 

## Computational infrastructure
Each group will have access to one GPU on the ETH student cluster. You will be granted access to the cluster after the registration deadline. You can find documentation for the student cluster at: https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster. The student cluster uses slurm to schedule your jobs. You can find an introduction to Slurm here: https://slurm.schedmd.com/quickstart.html. You can run jupyter notebooks on the student cluster through https://student-jupyter.inf.ethz.ch.

The student computing cluster is a shared resource that sometimes experiences high demand, which can result in jobs taking a considerable amount of time to be scheduled. To avoid last-minute complications, you should begin submitting your jobs well before the deadline rather than waiting until the very end to utilize your allocated compute resources.

### Conda environments
On the cluster, one central installation of conda exists for CIL. Add the following to your ~/.bashrc file and restart your shell to get access to conda.

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cluster/courses/cil/envs/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cluster/courses/cil/envs/etc/profile.d/conda.sh" ]; then
        . "/cluster/courses/cil/envs/etc/profile.d/conda.sh"
    else
        export PATH="/cluster/courses/cil/envs/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
 
To activate the conda environment for your project, run

```bash
module load cuda/12.6.0
# For project 1,
conda activate /cluster/courses/cil/envs/envs/collaborative-filtering
# For project 2,
conda activate /cluster/courses/cil/envs/envs/text-classification
# For project 3,
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation
```

The training data for each project has been deposited at kaggle. For your convenience, the data will be also available here shortly after the project start:

```
# For project 1,
/cluster/courses/cil/collaborative-filtering
# For project 2,
/cluster/courses/cil/text-classification
# For project 3,
/cluster/courses/cil/monocular-depth-estimation
```
