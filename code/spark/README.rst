=====================================================
Deployment of project on Databricks community edition
=====================================================
1.	Go to https://databricks.com/try-databricks, click on the start today button under the Community edition
2.	The page redirects to https://accounts.cloud.databricks.com/registration.html#signup/community, complete the registration by providing the details.
3.	After logging into the account, create a new Notebook, click on the Home button then click on the Users, then right click on your email and select the import button which shows the Import Notebooks modal, then click on the Import from File radio button and upload the .ipynb file located at https://gitlab.com/cloudmesh_fall2016/project-003/blob/master/code/spark/kddCupSpark.ipynb
4.	The project is available as iPython Notebook, execute the cells using Shift+Enter
5.      It prompts on creation of a cluster, select a sample cluster name and select an Apache Spark version of Spark 2.0 (Auto-updating Scala 2.10)
6.      Once the cluster is created, executed the cells in the iPython notebook, that provides the output.

=======================================
Deployment of project on Big Data Labs
=======================================
1.	Go to http://bigdata-labs.com/, click on the SIGNIN/SIGNUP button, which redirects to the login page, and click on the Sign up URL on the right bottom the page.
2.	Please provide your Email and Password and Sign up for an account.
3.	Once logged in, open the console by clicking on the _Open Console button.
4.	Once the console is opened, provide the username provide the password from the previous screen.
5.	Execute the below commands to download the data from the kddcup website, unzip it and put it into the users HDFS directory
6.	wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
7.	gunzip kddcup.data.gz
8.	hadoop fs –put kddcup.data /user/<<yourusername>>/ 
9.	Create a new file using vi kddCup.py, and insert the <<file>> contents into kddCup.py.
10.	Modify the line number 71, change the path of data accordingly
11.	Submit the Apache Spark job on the cluster using /usr/hdp/current/spark2-client/bin/spark-submit kddCup.py, and the job will be executed and output will be shown.

=======================================
Deployment of project on Cloudxlab.com
=======================================
1.	Go to https://cloudxlab.com/, click on the Login button on the top right corner.
2.	Click on the Create an Account URL on the next page, sign up by selecting an appropriate plans available.
3.	Once logged in, click on MyLab button on the home page.
4.	Click on the lab credentials tab on the next page which provides you with your credentials and URL to the Web Console.
5.	Open the Web Console, enter the username and password accordingly. 
6.	Execute the below commands to download the data from the kddcup website, unzip it and put it into the users HDFS directory
7.	wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
8.	gunzip kddcup.data.gz
9.	hadoop fs –put kddcup.data /user/<<yourusername>>/ 
10.	Create a new file using vi kddCup.py, and insert the <<file>> contents into kddCup.py.
11.	Modify the line number 71, change the path of data accordingly
12.	Submit the Apache Spark job on the cluster using /usr/spark2.0.1/bin/spark-submit, and the job will be executed and output will be shown.