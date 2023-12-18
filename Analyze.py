import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read my data and drop unnecessary columns
attendance_csv = pd.read_csv('School_Attendance_by_Student_Group_and_District__2021-2022.csv')
attendance_csv = attendance_csv.drop(attendance_csv[['Reporting period', 'Date update']], axis=1)

# Create an array of the district names and group names to help figure out how many unieque districts we are working
# with and to help with indexing later
district_names = attendance_csv['District name'].unique()
group_names = attendance_csv['Student group'].unique()


# Exploratory question 1: What trends do we see in attendance rates over time in Connecticut since the pandemic?

# Create a function that takes in as inputs a list of student groups and a district, defaulting to all student groups
# and the state data and returns a data frame showing the districts change in attendance rates by subgroup


def attendance_change(list_of_student_groups=group_names, district='Connecticut'):
    district_csv = attendance_csv[attendance_csv['District name'] == district]
    change_dictionary = {}
    for subgroup in list_of_student_groups:
        if subgroup in district_csv['Student group'].unique():
            subgroup_data = district_csv[district_csv['Student group'] == subgroup]
            subgroup_attendance_change_19_20_to_20_21 = float(subgroup_data['2019-2020 attendance rate'].iloc[0]) - \
                                                        float(subgroup_data['2020-2021 attendance rate'].iloc[0])
            subgroup_attendance_change_20_21_to_21_22 = float(subgroup_data['2020-2021 attendance rate'].iloc[0]) - \
                                                        float(subgroup_data[
                                                                  '2021-2022 attendance rate - year to date'].iloc[0])
            subroup_cumulative_attendance_change = float(subgroup_data['2019-2020 attendance rate'].iloc[0]) - \
                                                   float(subgroup_data['2021-2022 attendance rate - year to date'].iloc[
                                                             0])
            subgroup_changes = {'decrease in attendance rates from 19-20 to 20-21':
                                    round(subgroup_attendance_change_19_20_to_20_21, 4),
                                'decrease in attendance rates from 20-21 to 21-22':
                                    round(subgroup_attendance_change_20_21_to_21_22, 4),
                                'cumulative decrease in attendance rates':
                                    round(subroup_cumulative_attendance_change, 4)}
        else:
            subgroup_changes = {'decrease in attendance rates from 19-20 to 20-21': np.nan,
                                'decrease in attendance rates from 20-21 to 21-22': np.nan,
                                'cumulative decrease in attendance rates': np.nan}
        subgroup_dict = {f'{subgroup} attendance decrease': subgroup_changes}
        change_dictionary.update(subgroup_dict)
    change_df = pd.DataFrame.from_dict(change_dictionary)
    return change_df


# Use my function to create a data frame that shows trends in Connecticut to answer the first of my exploratory
# TL;DR Attendance rates are down across all subgroups
Connecticut_df = attendance_change()
Connecticut_df.to_csv('Connectict_attendance_changes.csv')
# print(Connecticut_df.head())

# Exploratory Question 2: are these trends universal across all school districts
# To help answer this, I apply my function to each school district to create a dataframe that has changes in attendance
# for all districts for all 3 years.
# First create a list of dataframes
attendance_change_dfs = [attendance_change(district=name) for name in district_names]
# Then use that list of dataframes with the list of district names to create a new data frame
changes_across_districts_df = pd.concat(attendance_change_dfs, keys=district_names)
changes_across_districts_df.index.names = ['name', 'year']
changes_across_districts_df.to_csv('Changes_across_districts.csv')
# print(changes_across_districts_df.head(6))
# Now that I have all my districts and years in one dataframe, I want to create a cross-section to determine if any
# districts saw an in crease in cumulative attendance or an increase in attendance for one or more subgroups

cumulative_change_df = changes_across_districts_df.xs('cumulative decrease in attendance rates', level='year')
cumulative_change_df.to_csv('Cumulative_Changes.csv')
# Use the cross-section to create a subset of the dataframe consisting of schools that saw an increase in attendance
districts_with_overall_increased_attendance = cumulative_change_df[cumulative_change_df['All Students attendance ' \
                                                                                        'decrease'] < 0]
districts_with_overall_increased_attendance.to_csv('overall_attendance_increases.csv')
# Use the cross-section to create a subset of the dataframe consisting in schools that had one or more student groups
# see an increase in attendance
districts_with_group_increased_attendance = cumulative_change_df[(cumulative_change_df < 0).any(axis=1)]
districts_with_group_increased_attendance.to_csv('group_attendance_increases.csv')

# Exploratory Question 3: How are districts that had higher than average attendance in the year leading up to the
# pandemic faring currently

# First I create 2 pivot tables based on my starting CSV in order to work with summary statistics of the attendance
# rates from the 2019-2020 school year and the 2021-2022 school year. This will allow me to identify which schools were
# in the top or bottom x percentile to find high attendance and low attendance schools
attendance_rate_by_district_19_20 = attendance_csv.pivot(index='District name', columns='Student group',
                                                         values='2019-2020 attendance rate')
attendance_rate_by_district_21_22 = attendance_csv.pivot(index='District name', columns='Student group',
                                                         values='2021-2022 attendance rate - year to date')

# Next I identify the top quartile of school attendance rates from the 19-20 school year (I will also make a list of
# these schools because while I am interested in how the top quartile between years looks, I am also interested in what
# happens to the schools from the top quartile of schools)
top_quartile_19_20 = attendance_rate_by_district_19_20['All Students'].quantile(0.75)
top_quartile_df_19_20 = attendance_rate_by_district_19_20[attendance_rate_by_district_19_20['All Students'] >=
                                                          top_quartile_19_20].sort_values(by='All Students')
top_quartile_districts_19_20 = list(top_quartile_df_19_20.index.values)
# top_quartile_df_19_20.to_csv('top_quartile_19_20.csv')
top_quartile_21_22 = attendance_rate_by_district_21_22['All Students'].quantile(0.75)
top_quartile_df_21_22 = attendance_rate_by_district_21_22[attendance_rate_by_district_21_22['All Students'] >=
                                                          top_quartile_21_22].sort_values(by='All Students')
# top_quartile_df_21_22.to_csv('top_quartile_21_22.csv')
top_quartile_districts_21_22 = list(top_quartile_df_21_22.index.values)

top_19_20_quartile_current_attendance = attendance_rate_by_district_21_22.loc[top_quartile_districts_19_20]

# Create a list of school that were in the top quartile in 2019-2020 and 2021-2022. we can find the length to determine
# how many schools from 19-20's tpo quartile were also in 21-22's top quartile
top_quartile_both_years = []
for district in top_quartile_districts_19_20:
    if district in top_quartile_districts_21_22:
        top_quartile_both_years.append(district)

print(len(top_quartile_both_years))


# To get an idea of what is happening to the attendance rates I create a function that calculates the median and mean
# of a subgroup  using 'All Students' as the default subgroup. I then print the values to see what is happening to
# the median and mean of attendance rates among the top quartile of districts

def calc_med_mean(df, subgroup='All Students'):
    return df[subgroup].median(), df[subgroup].mean()


# print(calc_med_mean(top_quartile_df_19_20, 'All Students'))
# print(calc_med_mean(top_quartile_df_21_22, 'All Students'))
# print(calc_med_mean(top_19_20_quartile_current_attendance))

# Next I do the same steps to the top decile to see if they appear to behave differently than the top quartile
top_decile_19_20 = attendance_rate_by_district_19_20['All Students'].quantile(0.9)
top_decile_df_19_20 = attendance_rate_by_district_19_20[attendance_rate_by_district_19_20['All Students'] >=
                                                        top_decile_19_20].sort_values(by='All Students')
top_decile_districts_19_20 = list(top_decile_df_19_20.index.values)

top_decile_21_22 = attendance_rate_by_district_21_22['All Students'].quantile(0.90)
top_decile_df_21_22 = attendance_rate_by_district_21_22[attendance_rate_by_district_21_22['All Students'] >=
                                                        top_decile_21_22].sort_values(by='All Students')
top_decile_districts_21_22 = list(top_decile_df_21_22.index.values)
top_decile_districts_21_22 = list(top_decile_df_21_22.index.values)
top_decile_both_years = []
for district in top_decile_districts_19_20:
    if district in top_decile_districts_21_22:
        top_decile_both_years.append(district)
# print(len(top_decile_both_years))
top_19_20_decile_current_attendance = attendance_rate_by_district_21_22.loc[top_decile_districts_19_20]

# print(calc_med_mean(top_decile_df_19_20))
# print(calc_med_mean(top_decile_df_21_22))
# print(calc_med_mean(top_19_20_decile_current_attendance))

# Exploratory Question 4: How are districts that had lower than average attendance in the year leading up to the
# pandemic performing currently?
# To answer this question, I go through the same steps I went through to answer question 3, but this time I use the
# bottom quartile and decile instead of the top

# Bottom quartile 2019-2020
bot_quartile_19_20 = attendance_rate_by_district_19_20['All Students'].quantile(0.25)
bot_quartile_df_19_20 = attendance_rate_by_district_19_20[attendance_rate_by_district_19_20['All Students'] <=
                                                          bot_quartile_19_20].sort_values(by='All Students')
bot_quartile_districts_19_20 = list(bot_quartile_df_19_20.index.values)
bot_19_20_quartile_current_attendance = attendance_rate_by_district_21_22.loc[bot_quartile_districts_19_20]
# Bottom quartile 2021-2022
bot_quartile_21_22 = attendance_rate_by_district_21_22['All Students'].quantile(0.25)
bot_quartile_df_21_22 = attendance_rate_by_district_21_22[attendance_rate_by_district_21_22['All Students'] <=
                                                          bot_quartile_21_22].sort_values(by='All Students')
bot_quartile_districts_21_22 = list(bot_quartile_df_21_22.index.values)

bot_quartile_both_years = []
for district in bot_quartile_districts_19_20:
    if district in bot_quartile_districts_21_22:
        bot_quartile_both_years.append(district)
# print(len(bot_quartile_both_years))
#
# print(calc_med_mean(bot_quartile_df_19_20, 'All Students'))
# print(calc_med_mean(bot_quartile_df_21_22, 'All Students'))
# print(calc_med_mean(bot_19_20_quartile_current_attendance))

# Bottom decile 2019-2020
bot_decile_19_20 = attendance_rate_by_district_19_20['All Students'].quantile(0.10)
bot_decile_df_19_20 = attendance_rate_by_district_19_20[attendance_rate_by_district_19_20['All Students'] <=
                                                        bot_decile_19_20].sort_values(by='District name')
bot_decile_districts_19_20 = list(bot_decile_df_19_20.index.values)
bot_decile_current_attendance = attendance_rate_by_district_21_22.loc[bot_decile_districts_19_20]
# Bottom decile 21022
bot_decile_21_22 = attendance_rate_by_district_21_22['All Students'].quantile(0.10)
bot_decile_df_21_22 = attendance_rate_by_district_21_22[attendance_rate_by_district_21_22['All Students'] <=
                                                        bot_decile_21_22].sort_values(by='District name')
bot_decile_districts_21_22 = list(bot_decile_df_21_22.index.values)

# print(calc_med_mean(bot_decile_df_19_20, 'All Students'))
# print(calc_med_mean(bot_decile_df_21_22, 'All Students'))
# print(calc_med_mean(bot_19_20_quartile_current_attendance))

# Exploratory question 5: Does the 2019-2020 size of the district appear to have an impact on the change in attendance
# rates?

# First, create pivot tables for 2019-2020 based on size of the school district
population_data_by_district_19_20 = attendance_csv.pivot(index='District name', columns='Student group',
                                                         values='2019-2020 student count')

# Find the bottom quartile value and top quartile value in order to determine small districts, large districts, and
# average districts
bot_quartile_population_size = population_data_by_district_19_20['All Students'].quantile(.25)
top_quartile_population_size = population_data_by_district_19_20['All Students'].quantile(.75)
# Create a list of small districts, a list of average sized districts, and a list of large districts
small_district_df = population_data_by_district_19_20[population_data_by_district_19_20['All Students'] <
                                                      bot_quartile_population_size]

small_district_names = list(small_district_df.index.values)

large_district_df = population_data_by_district_19_20[population_data_by_district_19_20['All Students'] >
                                                      top_quartile_population_size]
large_district_names = list(large_district_df.index.values)

med_district_df = population_data_by_district_19_20[(population_data_by_district_19_20['All Students'] >=
                                                     bot_quartile_population_size) &
                                                    (population_data_by_district_19_20['All Students'] <=
                                                     top_quartile_population_size)]
med_district_names = list(med_district_df.index.values)

# Use the techniques from exploratory question 2 where I created a dataframe with hierarchical indexing and do the same
# thing but instead create 3 such dataframes based on the district size
small_district_changes_df = [attendance_change(district=name) for name in small_district_names]
large_district_changes_df = [attendance_change(district=name) for name in large_district_names]
med_district_changes_dfs = [attendance_change(district=name) for name in med_district_names]

changes_across_small_districts_df = pd.concat(small_district_changes_df, keys=small_district_names)
changes_across_small_districts_df.index.names = ['name', 'year']
cumulative_small_district_changes = changes_across_small_districts_df.xs('cumulative decrease in attendance rates',
                                                                         level='year')

changes_across_large_districts_df = pd.concat(large_district_changes_df, keys=large_district_names)
changes_across_large_districts_df.index.names = ['name', 'year']
cumulative_large_district_changes = changes_across_large_districts_df.xs('cumulative decrease in attendance rates',
                                                                         level='year')

changes_across_med_districts_df = pd.concat(med_district_changes_dfs, keys=med_district_names)
changes_across_med_districts_df.index.names = ['name', 'year']
cumulative_med_district_changes = changes_across_med_districts_df.xs('cumulative decrease in attendance rates',
                                                                     level='year')

# calculate and print the median and mean changes in attendance rates for each group of school districts
# print(calc_med_mean(cumulative_small_district_changes, subgroup='All Students attendance decrease'))
# print(calc_med_mean(cumulative_med_district_changes, subgroup='All Students attendance decrease'))
# print(calc_med_mean(cumulative_large_district_changes, subgroup='All Students attendance decrease'))
# print(calc_med_mean(cumulative_change_df, subgroup='All Students attendance decrease'))
# print(len(small_district_names))
# print(len(large_district_names))
# print(len(med_district_names))

# Now let's look at just the largest districts
large_districts_sorted = large_district_df.sort_values(by='All Students', ascending=False)
largest_district_names = list(large_districts_sorted.index.values)[1:11]
# 1:11 instead of 0:10 bc Connecticut is the first entry and I don't want to include the data of the entire state
largest_districts_dfs = [attendance_change(district=name) for name in largest_district_names]
changes_across_largest_districts_df = pd.concat(largest_districts_dfs, keys=largest_district_names)
changes_across_largest_districts_df.index.names = ['name', 'year']
cumulative_change_largest_districts = changes_across_largest_districts_df.xs('cumulative decrease in attendance rates',
                                                                             level='year')
# print(calc_med_mean(cumulative_change_largest_districts, subgroup='All Students attendance decrease'))


# Graphics 1: First graph I want to generate is part of answering my first question. I want to create a boxplot of
# attendance rates for each group of kids for each year, next to a boxplot for cumulative change in attendance

fig1 = plt.figure(figsize=(18, 10))

attendance_ax1 = fig1.add_subplot(1, 3, 1)
attendance_ax1.set_xticklabels(group_names, rotation=90)
attendance_ax1.set(ylim=(0.7, 1))

attendance_ax2 = fig1.add_subplot(1, 3, 2)
attendance_ax2.set_xticklabels(group_names, rotation=90)
attendance_ax2.set(ylim=(0.7, 1))

attendance_ax3 = fig1.add_subplot(1, 3, 3)
attendance_ax3.set_xticklabels(labels=group_names, rotation=90)
attendance_ax3.set_ylabel('Cumulative decrease in attendance rates')
attendance_ax3.set_xlabel("Student group")

attendance_boxplot_19_20 = sns.boxplot(x=attendance_csv['Student group'],
                                       y=attendance_csv['2019-2020 attendance rate'], ax=attendance_ax1,
                                       hue=attendance_csv['Student group'])
attendance_boxplot_19_20.set(title="Attendance Rates in the 2019-200 School Year")

attendance_boxplot_21_22 = sns.boxplot(x=attendance_csv['Student group'],
                                       y=attendance_csv['2021-2022 attendance rate - year to date'], ax=attendance_ax2,
                                       hue=attendance_csv['Student group'])
attendance_boxplot_21_22.set(title="Attendance Rates in the 2021-2022 School Year")

cumulative_change_boxplot = sns.boxplot(data=cumulative_change_df, ax=attendance_ax3)
cumulative_change_boxplot.set(title='Cumulative Decrease in Attendance Rates')

plt.subplots_adjust(bottom=0.35)
fig1.savefig('Attendance_Rates.pdf')

# Graphics 2: For the second graph I show the relation between attendance in the 2019-2020 school year, and
# 2021-2022 school year. We can see a strong correlation between attendance rates for the two years with schools
# that had a higher attendance rate in 2019-2020 also having higher attendance rates in 2021-2022

fig2 = plt.figure(figsize=(8, 8))
fig2_ax = fig2.add_subplot(1, 1, 1)
attendance_scatterplot = sns.scatterplot(
    x=attendance_csv[attendance_csv['Student group'] == 'All Students']['2019-2020 attendance rate'],
    y=attendance_csv[attendance_csv['Student group'] == 'All Students']['2021-2022 attendance rate - year to date'],
    ax=fig2_ax)
attendance_scatterplot.set(
    title='Relationship between attendance rates in the 2019-2020 school year \n and the 2021-2022 school year')

fig2.savefig('Relationship_Between_first_year_last_year_scatterplot.pdf')
# Graph 3: For my final graph I show the relationship between school size and changes in attendance rates. To do this
# I want two graphs: a scatterplot of school size and change in attendance rates, and a boxplot that shows changes in
# attendance rates categorized by schools of different sizes (small, med, large). To do this I start by joining each
# of the district size dataframe with the all students attendance decrease in its corresponding cumulative change
# dataframe. I will then add a column categorizing the school as small, medium, or large and then concatenate the
# dataframes to be able to plot district size and changes in attendance rates

small_district_join = small_district_df.join(cumulative_small_district_changes['All Students attendance decrease'],
                                             how='inner')
small_district_join['Size'] = 'Small'

med_district_join = med_district_df.join(cumulative_med_district_changes['All Students attendance decrease'],
                                         how='inner')
med_district_join['Size'] = 'Medium'

large_district_join = large_district_df.join(cumulative_large_district_changes['All Students attendance decrease'],
                                             how='inner').drop('Connecticut')
large_district_join['Size'] = 'Large'

all_attendance_decrease_with_size = pd.concat([small_district_join, med_district_join, large_district_join])
# Make a copy that also has the largest category
all_attendance_decrease_with_largest_size = all_attendance_decrease_with_size.copy()
for name in largest_district_names:
    all_attendance_decrease_with_largest_size.loc[name, 'Size'] = 'Largest'

fig3 = plt.figure(figsize=(10, 10))
fig3_ax1 = fig3.add_subplot(2, 2, 1)
fig3_ax2 = fig3.add_subplot(2, 2, 3)
fig3_ax3 = fig3.add_subplot(2, 2, 4)


#print(all_attendance_decrease_with_size['All Students'], all_attendance_decrease_with_size['All Students'])
regression_graph = sns.regplot(data=all_attendance_decrease_with_size, x='All Students',
                               y='All Students attendance decrease', ax=fig3_ax1)
regression_graph.set(xlabel='Student Population Size')
pal=['#E7873C', '#7D52D7', '#B73737', '#086809']
three_category_boxplot = sns.boxplot(data=all_attendance_decrease_with_size, x='Size',
                                     y='All Students attendance decrease', ax=fig3_ax2, palette=pal)

my_order = ['Small', 'Medium', 'Large', 'Largest']

print(my_order)
four_category_boxplot = sns.boxplot(data=all_attendance_decrease_with_largest_size, x='Size',
                                    y='All Students attendance decrease', ax=fig3_ax3,palette=pal, order=my_order)

fig3.savefig('Relation_between_district_size_and_attendance_decrease.pdf')
plt.show()
