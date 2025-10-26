# Diagnosing Parkinson’s Disease using voice sample data analysis
# Context
Parkinson’s Disease (PD) is a progressive movement disorder of the nervous system, according to the U.S. National Institute of Neurological Disorders and Stroke. Common symptoms include tremors, rigidity, bradykinesia, and postural instability. Notable individuals affected include Muhammad Ali, George H.W. Bush, Michael J. Fox, Ozzy Osbourne, and Pope John Paul II.

There is no cure or specific diagnostic test for PD; existing methods like blood tests and brain scans are invasive and stressful. Increasing research focuses on non-invasive diagnosis through speech analysis, as people with PD often experience dysphonia—reduced loudness, breathiness, roughness, and vocal tremors—detectable via voice frequency patterns.
# Objective
By examining these voice samples, I aim to explore the potential of speech data as a diagnostic tool. In this project, I focus on predicting the motor and total UPDRS scores, which reflect the severity and progression of Parkinson’s disease in patients.
# Dataset
I used a dataset provided by my lecturer in a CSV file containing 5,875 rows and 22 columns. The original study gathered voice recordings from 42 individuals with early-stage Parkinson’s disease, with each person contributing between 101 and 168 samples. These recordings were analysed into various acoustic features using the software *Praat*.

I also used another dataset in a .txt file, supplied in a previous assignment, to validate the models. This dataset includes voice recordings from 20 individuals with Parkinson’s disease and 20 without, each providing 26 voice samples. These samples were likewise processed for acoustic characteristics using *Praat*.

Learn more about *Praat* software: [https://www.fon.hum.uva.nl/praat/](https://www.fon.hum.uva.nl/praat/)
# Methodology
In this assignment, I explore voice sample data from early-stage Parkinson’s disease (PD) patients to uncover patterns and make predictions. My methodology consists of three main stages:

1. **Exploratory Data Analysis (EDA):** I conduct an in-depth examination of the data’s structure and patterns using both descriptive and inferential statistics.
2. **Data Visualisation:** I create graphical representations to highlight complex relationships and strengthen data storytelling.
3. **Predictive Modelling:** I build and refine models to predict UPDRS scores through:

   * Basic Linear Regression
   * Log-Transformation and Collinearity Analysis
   * Gaussian Transformation
   * Model rebuilding after each transformation

My objective is to determine the most effective predictive model to support the development of PD diagnostic techniques using voice sample analysis.
<img width="1023" height="538" alt="image" src="https://github.com/user-attachments/assets/65e8607d-d0a2-4416-8135-6b47e7cf4aa6" />

# Key findings
1. **Baseline Linear Regression**

Developed predictive models for Motor and Total UPDRS scores using a 60:40 train-test split.
Significant features:
    * Motor UPDRS: Jitter(rap) (negative effect) and Shimmer(apq3) (positive effect).
    * Total UPDRS: Jitter(rap) (negative effect) and Jitter(ddp) (positive effect).
Explained variance: ~21.7% (Motor) and ~25% (Total) — both outperforming the baseline model.
