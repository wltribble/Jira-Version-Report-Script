import os
from jira import JIRA
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta
from google.colab import userdata # imports my environment variables (I built this in Google's browser-based Jupyter notebook tool)

# Load necessary secrets
JIRA_API_TOKEN = userdata.get('JIRA_API_TOKEN') # Replace with your Jira API Token
JIRA_EMAIL = userdata.get('MY_EMAIL') # Replace with your Jira account email
JIRA_SERVER = userdata.get('JIRA_SERVER')  # Replace with your Jira URL
PROJECT_KEY = userdata.get('PROJECT_KEY')  # Replace with your Jira project key

# Initialize Jira client
jira = JIRA(
    server=JIRA_SERVER,
    basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN)
)

def list_unreleased_versions():
    """Fetch and display unreleased versions."""
    versions = jira.project_versions(PROJECT_KEY)
    unreleased_versions = [v for v in versions if not v.released]
    for idx, version in enumerate(unreleased_versions):
        print(f"{idx}: {version.name}")
    return unreleased_versions

def get_version_issues(version_name):
    """Retrieve issues associated with the specified version."""
    jql = f'project = {PROJECT_KEY} AND fixVersion = "{version_name}" ORDER BY created ASC'
    issues = jira.search_issues(jql, maxResults=False, expand='changelog')
    return issues

def calculate_project_status_to_date(version_name):
    """Iterate over each issue in a version and accumulate the changes to their statuses to show project status so far"""
    issues = get_version_issues(version_name)
    if not issues:
        print(f"No issues found for version '{version_name}'.")
        return

    # Get the start date of the selected version
    for v in jira.project_versions(PROJECT_KEY):
        if str(v.name) == version_name:
            version = v
    version_start_date = str(pd.to_datetime(version.startDate) - timedelta(days=1))
    version_release_date = str(pd.to_datetime(version.releaseDate))

    # Find the earliest change date among issues in the version
    earliest_date = min(
        datetime.strptime(history.created, '%Y-%m-%dT%H:%M:%S.%f%z').date()
        for issue in issues
        for history in issue.changelog.histories
    )
    latest_date = datetime.now().date()

    # Prepare data storage for daily tracking
    data = []
    cumulative_story_points_completed = 0
    cumulative_estimated_story_points = 0
    cumulative_unestimated_stories = 0
    total_story_count = sum(1 for issue in issues if issue.fields.issuetype.name == 'Story')

    # Dictionary to track last known values for each issue
    last_known_values = {issue.key: {'story_points': float(0.0), 'completed': False} for issue in issues}

    # Loop through each day and calculate cumulative metrics for that day
    current_date = earliest_date
    while current_date <= latest_date:
        for issue in issues:
            # Initialize variables for the current state of this issue
            story_points = float(0.0)
            completed = False

            # Sort the histories to ensure chronological processing
            sorted_histories = sorted(
                issue.changelog.histories,
                key=lambda h: datetime.strptime(h.created, '%Y-%m-%dT%H:%M:%S.%f%z')
            )

            # Find the most recent status and story points as of the current date
            for history in sorted_histories:
                history_date = datetime.strptime(history.created, '%Y-%m-%dT%H:%M:%S.%f%z').date()
                if history_date > current_date:
                    break  # Stop if we exceed the current date

                for item in history.items:
                    if item.field == 'status':
                        completed = str(item.toString) == "Done"
                    elif item.field == 'Story point estimate':  # This is a custom field in my project, make sure to update for yours
                        story_points = float(item.toString) if item.toString else float(0.0)

            # Retrieve last known values for comparison
            last_values = last_known_values[issue.key]
            last_story_points = last_values['story_points']
            last_completed = last_values['completed']

            # Only update cumulative metrics if there has been a change
            if issue.fields.issuetype.name == 'Story':
                # Check for changes in story points and completed status
                if story_points != last_story_points:
                    if last_story_points == float(0.0):
                        delta_story_points = story_points
                    else:
                        delta_story_points = story_points - last_story_points
                    cumulative_estimated_story_points += delta_story_points
                    last_values['story_points'] = story_points  # Update last known value

                if completed != last_completed:
                    if completed:  # Add story points to completed if it is now "Done"
                        cumulative_story_points_completed += story_points if story_points else 0
                    elif last_completed:  # If it's no longer "Done," subtract from completed
                        cumulative_story_points_completed -= last_story_points if last_story_points else 0
                    last_values['completed'] = completed  # Update last known completed status

                # Track unestimated stories if story_points is None
                cumulative_unestimated_stories = sum(1 for key, values in last_known_values.items() if values['story_points'] is None)

        # Calculate the percentage of unestimated stories for this day
        unestimated_percentage = (cumulative_unestimated_stories / total_story_count) * 100 if total_story_count > 0 else 0

        # Store data for this day
        data.append((current_date, cumulative_story_points_completed, cumulative_estimated_story_points, unestimated_percentage))

        # Move to the next day
        current_date += timedelta(days=1)

    # Convert data to DataFrame for cumulative plotting
    historical_data = pd.DataFrame(data, columns=['Date', 'Story Points Completed', 'Estimated Story Points', 'Unestimated Stories (%)'])

    # Convert 'Date' column to datetime and filter data from the version start date onward
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data = historical_data[historical_data['Date'] >= pd.to_datetime(version_start_date)]
    historical_data['Days'] = (historical_data['Date'] - historical_data['Date'].min()).dt.days

    return historical_data, version_release_date

def projection_of_progress(historical_data, version_release_date): 
    """Take project status so far and use it to forecast progress till completion"""
    # Set up the regression model
    X = sm.add_constant(historical_data['Days'])  # Add an intercept for the linear model
    y = historical_data['Story Points Completed']
    model = sm.OLS(y, X).fit()  # Fit the linear regression model

    # Get start and end points for projection
    current_date = datetime.now().date()

    # Generate future dates and extend DataFrame to include projection period
    future_dates = pd.date_range(pd.to_datetime(current_date), pd.to_datetime(version_release_date), freq='D')
    future_days = (future_dates - historical_data['Date'].min()).days
    projected_data = pd.DataFrame({'Date': future_dates, 'Days': future_days})

    # Start the projection from the last value of cumulative_story_points_completed
    start_value = historical_data['Story Points Completed'].iloc[-1]
    X_future = sm.add_constant(projected_data['Days'])
    projected_points = model.predict(X_future)

    # Adjust the predicted points to start from the current cumulative value
    projected_data['Projected Story Points'] = projected_points + (start_value - projected_points.iloc[0])
    
    # Extend the current cumulative_estimated_story_points to the end date
    projected_data['Estimated Story Points'] = historical_data['Estimated Story Points'].iloc[-1]

    # Calculate 95% confidence intervals for the projection (before applying the shift)
    predictions = model.get_prediction(X_future)
    conf_int = predictions.conf_int(alpha=0.05)

    # Apply the shift to the confidence intervals to match the projected trendline
    projected_data['Lower Bound'] = conf_int[:, 0] + (start_value - projected_points.iloc[0])
    projected_data['Upper Bound'] = conf_int[:, 1] + (start_value - projected_points.iloc[0])

    # Final guarantee that dates are in datetime
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')
    projected_data['Date'] = pd.to_datetime(projected_data['Date'], errors='coerce')

    return historical_data, projected_data

def plot_version_report(historical_data, projected_data, version_release_date, version_name):
    """Take historical project status + projected future progress and plot them"""
    # Combine historical and projected data for plotting
    combined_df = pd.concat([historical_data, projected_data], ignore_index=True)

    # Plotting with dual Y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot the historical data
    ax1.plot(historical_data['Date'], historical_data['Estimated Story Points'], color='darkgray', label='Cumulative Estimated Story Points', zorder=1)
    ax1.fill_between(historical_data['Date'], historical_data['Estimated Story Points'], color='darkgray', alpha=0.2, zorder=1.5)
    ax1.plot(historical_data['Date'], historical_data['Story Points Completed'], color='blue', label='Cumulative Story Points Completed', zorder=3)
    ax1.fill_between(historical_data['Date'], historical_data['Story Points Completed'], color='blue', alpha=0.2, zorder=2.5)

    # Plot the projected trend line and estimated points
    ax1.plot(projected_data['Date'], projected_data['Projected Story Points'], color='navy', linewidth=3, linestyle='-', zorder=4)
    ax1.fill_between(projected_data['Date'], projected_data['Lower Bound'], projected_data['Upper Bound'], color='navy', alpha=0.2)
    ax1.plot(projected_data['Date'], projected_data['Estimated Story Points'], color='darkgray', zorder=1)
    ax1.fill_between(projected_data['Date'], projected_data['Estimated Story Points'], color='darkgray', alpha=0.2, zorder=1.5)

    # Draw and label a vertical dotted line for the version release date
    ax1.axvline(x=pd.to_datetime(version_release_date), color='gray', linestyle='--', linewidth=2)
    label_padding = timedelta(days=0.25)
    ax1.text(
        pd.to_datetime(version_release_date) - label_padding,  # x-coordinate: the release date
        ax1.get_ylim()[1] * 0.97,  # y-coordinate: slightly below the top of the y-axis
        'Release Date',  # Label text
        color='gray',  # Text color to match the line
        ha='right',  # Horizontal alignment to the right of the x-coordinate
        fontsize=10,  # Font size
        fontweight='bold'  # Make the label bold
    )

    # Draw and label a vertical dotted line for today
    ax1.axvline(x=datetime.now().date(), color='black', linestyle=":", linewidth=2)
    ax1.text(
        (datetime.now().date() + label_padding),
        ax1.get_ylim()[1] * 0.97,  # y-coordinate: slightly below the top of the y-axis
        ' Today',  # Label text
        color='Black',  # Text color to match the line
        ha='left',  # Horizontal alignment to the left of the x-coordinate
        fontsize=10,  # Font size
        fontweight='bold'  # Make the label bold
    )

    # Label Y-axis for Story Points
    ax1.set_ylabel("Story Points")
    ax1.set_ylim(0, combined_df['Estimated Story Points'].max() * 1.1)  # Add 10% padding to max

    # Plot on secondary Y-axis (Percentage of Stories Unestimated)
    ax2 = ax1.twinx()
    ax2.plot(historical_data['Date'], historical_data['Unestimated Stories (%)'], color='red', label='Percentage of Stories Unestimated', linestyle='--', zorder=2)
    ax2.set_ylabel("Percentage of Stories Unestimated")
    ax2.set_ylim(0, 100)  # Set scale from 0 to 100

    # Format x-axis to show dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 7 days; adjust as needed
    fig.autofmt_xdate()  # Rotate date labels for readability

    # Draw trendline of work completed, if work has been done
    start_date = historical_data['Date'].min()  # Earliest date in the Date column
    end_value = historical_data['Story Points Completed'].iloc[-1]  # Last value of "Story Points Completed"
    if end_value != 0:
        ax1.plot([start_date, historical_data['Date'].max()], [0, end_value], color='navy', linewidth=3, linestyle='-', zorder=4)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Customize and display the plot
    plt.title(f"Version Report for Version: {version_name}")
    ax1.set_xlabel("Date")
    ax1.grid(True)
    plt.show()

if __name__ == "__main__":
    unreleased_versions = list_unreleased_versions()
    if not unreleased_versions:
        print("No unreleased versions found.")
    else:
        try:
            selection = int(input("Select a version by number: "))
            selected_version = unreleased_versions[selection]
            print(f"Generating report for version: {selected_version.name}")
            historical_data, version_release_date = calculate_project_status_to_date(selected_version.name)
            historical_data, projected_data = projection_of_progress(historical_data, version_release_date)
            plot_version_report(historical_data, projected_data, version_release_date, selected_version.name)
        except (IndexError, ValueError):
            print("Invalid selection.")
