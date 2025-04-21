# 369 Project: Game of Thrones - Which Characters will Live/Die
### Collaborators:
- Rob Murray
- Maya Labuhn
- Klaudia Jefferson


### Report:
• Introduction: What data mining problem are you trying to solve? What impact will it bring if the problem is solved?
Can be used for storytelling purposes, use it to soften the blow when a character dies. 

• Formulation: Which data mining task can it be formulated into? What’s the input and the expected output?

Our tasks focuses on the classification of a dataset.
It takes the input of a data set with attributes: 
name: Character's name
Title: Social status or nobility title
House: House to which the character belongs
Culture: Social group associated with the character
book1/2/3/4/5: Indicator of the character's appearance in each book
Is noble: Nobility status based on title
Age: Character's age (year reference: 305 AC)
male: Gender of the character (1 = Male, 0 = Female)
dateOfBirth: Character's birth year
Spouse: Name of the character's spouse
Father: Name of the character's father
Mother: Name of the character's mother
Heir: Name of the character's heir
Is married: Whether the character is married (1 = Yes, 0 = No)
Is spouse alive: Whether the character's spouse is alive (1 = Yes, 0 = No)
Is mother alive: Whether the character's mother is alive (1 = Yes, 0 = No)
Is heir alive: Whether the character's heir is alive (1 = Yes, 0 = No)
Is father alive: Whether the character's father is alive (1 = Yes, 0 = No)
Number dead relations: Number of known deceased relations of the character
Popularity score: Number of internal incoming and outgoing links to the character's wiki page

Then determines the output classification of the dataset for attribute: 
isAlive: Indicates whether the character is alive in the book (1 = Alive, 0 = Deceased)

• Datasets: Where did you get the dataset? Provide some data statistics. How did you preprocess the data?
Out datasets are based on the series game of thrones, we obtained our data from Kaggle for determining if a character is alive. We preprocessed it by removing
converting the csv to a numpy array (matrix) format for easier applications. 

• Algorithm: Which data mining algorithm did you apply?
For predicting the isAlive classification we applied the Naive Bayes algorithm on our dataset.  

• Experiments: Evaluate the output using an appropriate evaluation metric. Show the results you get and discuss whether they are meaningful.

