import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
df = pd.read_csv(r'C:\Users\Admin\Desktop\python project ca2\NCHS_-_Leading_Causes_of_Death__United_States.csv')
print(df.head(5))
print(df.info())
print(df.describe())
print("checkin missing value", df.isnull().sum())
print("counting missing value", df.isnull().sum().sum())
print("mean of death rate",df['Age-adjusted Death Rate'].mean().round(1))
print("median of death rate", df["Age-adjusted Death Rate"].median().round(1))
print("standard deviation of death rate",df["Age-adjusted Death Rate"].std().round(2))
print("\nEDA is complete! These insights should help us better understand the dataset.")

## barplot 2017
year = 2017
df_year = df[df['Year'] == year]
top_causes_year = df_year.groupby('Cause Name')['Deaths'].sum().sort_values(ascending=False).nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_causes_year.values, y=top_causes_year.index, palette='magma')
plt.title(f'Top 10 Leading Causes of Death in the U.S. ({year})')
plt.xlabel('Total Deaths (number of individuals)')
plt.ylabel('Cause of Death')
plt.tight_layout()
plt.show()

###barplot 2013
year = 2013
df_year = df[df['Year'] == year]
top_causes_year = df_year.groupby('Cause Name')['Deaths'].sum().sort_values(ascending=True).nsmallest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_causes_year.values, y=top_causes_year.index, palette='magma')
plt.title(f'Top 10 smallest Leading Causes of Death in the U.S. ({year})')
plt.xlabel('Total Deaths (number of individuals)')
plt.ylabel('Cause of Death')
plt.tight_layout()
plt.show()

### top 10 largest  death cause
plt.figure(figsize=(12,6))
top_causes = df.groupby('Cause Name')['Deaths'].sum().nlargest(10)
top_causes.plot(kind='bar', color='skyblue')
plt.title('Top 10 Leading Causes of Death in whole data')
plt.xticks(rotation=45)
plt.show()
########## lineplot
# Analyzing death trends over the years
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Year', y='Deaths', hue='Cause Name', estimator='sum')
plt.title('How Have Death Trends Changed Over Time?')
plt.ylabel('Total Deaths(In Thousand)')
plt.show()
### histmap
# Examining the distribution of age-adjusted death rates
plt.figure(figsize=(8,5))
sns.histplot(df['Age-adjusted Death Rate'], bins=30, kde=True, color='purple')
plt.title('Distribution of Age-Adjusted Death Rate')
plt.xlabel('Age-Adjusted Death Rate')
plt.ylabel("count(in Thousand)")
plt.show()
################# lineplot
# Trend for Heart Disease
cause = "Heart disease"
heart_trend = df[(df['State'] == 'United States') & (df['Cause Name'] == cause)]
heart_trend['Deaths'] = heart_trend['Deaths'] / 1000000
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Deaths', data=heart_trend, marker='o',color="blue")
plt.title(f'Yearly Deaths Due to {cause} in the United States', fontsize=14)
plt.ylabel('Number of Deaths (in million)  ')
plt.grid(False)
plt.show()

#####
# Pie chart for 2017
latest_year = df['Year'].max()
pie_data = (
    df[(df['Year'] == latest_year) & (df['State'] == 'United States')]
    .groupby('Cause Name')['Deaths']
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 8))
pie_data.head(7).plot.pie(autopct='%1.1f%%', startangle=140)
plt.title(f'Death Cause Distribution in U.S. ({latest_year})')
plt.ylabel('')
plt.tight_layout()
plt.show()

########
# Correlation plot (scatter )
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df[df['State'] == 'California'], x='Deaths', y='Age-adjusted Death Rate')
plt.title('Correlation Between Total Deaths and Age-adjusted Death Rate (California)')
plt.tight_layout()
plt.show()

####### box plot
plt.figure(figsize=(10,8))
sns.boxplot(data=df, x="Age-adjusted Death Rate", y="Cause Name")
plt.title('Boxplot of Age-adjusted Death Rate by Cause of Death')
plt.xlabel('Age-adjusted Death Rate')
plt.ylabel('Cause of Death')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

## p-test
# we are comparing cause and states 
cause = "Heart disease"
state1 = "California"
state2 = "Texas"
group1 = df[(df["Cause Name"] == cause) & (df["State"] == state1)]["Age-adjusted Death Rate"]
group2 = df[(df["Cause Name"] == cause) & (df["State"] == state2)]["Age-adjusted Death Rate"]

# Perform Welch's t-test (assumes unequal variances)
t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
# Display the results
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")


### Z-TEST
print("we are using Z-Test on it")
year = 2017
df_year = df[df['Year'] == year]
deaths_by_cause = df_year.groupby('Cause Name')['Deaths'].sum()

cause_of_interest = "Heart disease"
x = deaths_by_cause[cause_of_interest]

other_causes = deaths_by_cause.drop(cause_of_interest)
mean_others = other_causes.mean()
std_others = other_causes.std()
z_score = (x - mean_others) / std_others
p_value = 1 - stats.norm.cdf(z_score)
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant — deaths from Heart disease are significantly higher than average.")
else:
    print("Result: Not statistically significant.")

########## heatmap
selected_states = sorted(df['State'].unique())[:8]
selected_causes = sorted(df['Cause Name'].unique())[:8]

filtered_df = df[df['State'].isin(selected_states) & df['Cause Name'].isin(selected_causes)]
pivot_df = filtered_df.pivot_table(
    index="Cause Name",
    columns="State",
    values="Age-adjusted Death Rate",
    aggfunc="mean"
)
pivot_df = pivot_df.loc[selected_causes, selected_states]
correlation_matrix = pivot_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, linecolor='gray')

plt.title("8x8 Correlation Heatmap Between Selected States (Based on 8 Causes)")
plt.tight_layout()
plt.show()







            
