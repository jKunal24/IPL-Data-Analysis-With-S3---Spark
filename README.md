# IPL-Data-Analysis-With-S3-and-Spark
Analysis of IPL Data from year 2008-2024 

**Storage : *AWS S3* 
**Databricks : *Environment* to use Spark for data processing and handle infrastructure side of things ( local setup of scalas/java, networking tasks, etc. )
**Language : *Python* ( as interface to use Apache Spark )
**Visualization Libraries : *Plotly, Matplotlib & Seaborn*

*Data Cleaning :*
 1.Standardizing/Merging team names of same Franchise with different strings in deliveries file.
 2.Standardizing/Merging venue/stadium names of same stadiums with different strings in Matches file.

*Metrics & Visualizations based on :*
----------------------------------------------------------------------------------------------------------------------------
1.Average inning wise score for all venues/stadiums : Heatmap

2.Highest chased target at all venues/stadiums : Heatmap

3.Distribution of innings_total across all IPL Season : Boxplot

4.Over-wise average run-rate for all teams : Line graph (subplot)

5.Batsman dominated Face-offs ( contest between a batsman & bowler against each-other ) : Plotly Tables

6.Bowler dominated Face-offs ( contest between a batsman & bowler against each-other ) : Plotly Tables

7.Average No. of wickets taken in different parts of the game ( overs : 1-4, 5-8, 9-12, 13-16, 17-20) by all teams : Heatmap

8.Average No. of wickets lost in different parts of the game ( overs : 1-4, 5-8, 9-12, 13-16, 17-20) by all teams : Heatmap

9.Orange Cap Holders in all IPL Seasons : Horizontal Bar plot

10.Purple Cap Holders in all IPL Seasons : Horizontal Bar plot

11.Over-wise ( from 1-20 )avg run-rate for all teams : Heatmap

12.Over-wise ( from 1-20 ) avg runs conceeded by all teams  Heatmap

13.Most Runs in a Single edition of IPL ( Top 10) - Horizontal Bar plot

14.Most Wickets in a Single edition of IPL ( Top 10) - Horizontal Bar plot

15.Most no. of boundaries in IPL (Top 10) - Bar plot

16.Most no. of Sixes in IPL (Top 10)- Bar plot

17.Most no. of Fours in IPL (Top 10)- Bar plot 

18.Most Economical bowlers in powerplay (Top 10 ) **min 50 overs bowled in powerplay - Line plot

19.Highest run scorers in IPL ( Top 10 - with all batting avg ) - Hat graph (matpllotlib)

20.Highest run scorers in IPL ( Top 10 - with all time avg an no. of matches) - Bar plot

21.Highest wicket takers in IPL (Top 10) - Bar plot

22.Most no. of catches in IPL (Top 10 fielders) - Bar plot

23.Most no. of stumpings (Top 10 Wicketkeepers) - Bar plot
