1.Project overview 

  This is a simple project that works on linear regression with Scikit-Learn to      predict the results of students . It evaluates how  study hours and practice exams affect the overall performance  . It also has inclusion of data visualization and model evaluation 
 
2.Project goal 
  
  The goals have been broken down into ;
     ~Determine the impact of study methods on exam performance . 
     ~Build a model to predict scores based on study hours and study methods 
     ~Evaluate the accuracy of the model 
     ~
     
3.Datset 

   A small dataset is created within the script

4.Methodology
   
   ~ Data is prepared ;
                      In this case we create a new dataset  as mentioned above .
                      An additional dataset (practice_exams) is created .
   ~Data is analised ;
                      By scattering plots of Hours vs Scores 
                      Through regression line  Visualization (To show model 
                         prediction)
                      Through comparing the actual scores against predicted
                                scores .
   ~Modelling ;
                      Train a simple linear regression  model (hours - scores )
                      Train several linear regression model (hours + predicted
                              exams-scores )
   ~Evaluation ;
                      I used Mean absolute error and mean squarred error 
                      Compare the actual vs predicted results .
        
5.Tecch start 
   Language : Python 
   Libraries : panda , Numpy , Matplotlib , Scikit-learn
6.Results
7.How to run the project . 
~Clone or download this repository .
~Run the script 
~The script will :
                  Print dataset , predictions and evaluation metrics .
                  Generate and display plots .
                  Prompt for user input to predict scores interactively 