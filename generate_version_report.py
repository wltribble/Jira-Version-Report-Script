import os
from jira import JIRA
import matplotlib.pyplot as plt
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

def generate_version_report(version_name):
    """Take chosen Version, iterate through its issues, iterate through their change histories, monitor changes in their estimates and statuses, and plot the graph"""
    issues = get_version_issues(version_name)
    if not issues:
        print(f"No issues found for version '{version_name}'.")
        return

    # Find the earliest change date
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
    last_known_values = {issue.key: {'story_points': None, 'completed': False} for issue in issues}

    # Loop through each day and calculate cumulative metrics for that day
    current_date = earliest_date
    while current_date <= latest_date:
        for issue in issues:
            # Initialize variables for the current state of this issue
            story_points = None
            completed = False

            # Find the most recent status and story points as of the current date
            for history in issue.changelog.histories:
                history_date = datetime.strptime(history.created, '%Y-%m-%dT%H:%M:%S.%f%z').date()
                if history_date > current_date:
                    break  # Stop if we exceed the current date

                for item in history.items:
                    if item.field == 'status':
                        completed = item.toString == "Done"
                    elif item.field == 'Story point estimate':  # This is a custom field in my project, make sure to update for yours
                        story_points = getattr(issue.fields, 'customfield_10016', None) # This is a custom field in my project, make sure to update for yours

            # Retrieve last known values for comparison
            last_values = last_known_values[issue.key]
            last_story_points = last_values['story_points']
            last_completed = last_values['completed']

            # Only update cumulative metrics if there has been a change
            if issue.fields.issuetype.name == 'Story':
                # Check for changes in story points and completed status
                if story_points != last_story_points:
                    if last_story_points is None:
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
    df = pd.DataFrame(data, columns=['Date', 'Story Points Completed', 'Estimated Story Points', 'Unestimated Stories (%)'])

    # Plotting with dual Y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot on primary Y-axis (Story Points)
    ax1.plot(df.index, df['Estimated Story Points'], color='darkgray', label='Cumulative Estimated Story Points', zorder=1)
    ax1.fill_between(df.index, df['Estimated Story Points'], color='darkgray', alpha=0.2, zorder=2)
    ax1.plot(df.index, df['Story Points Completed'], color='blue', label='Cumulative Story Points Completed', zorder=3)
    ax1.fill_between(df.index, df['Story Points Completed'], color='blue', alpha=0.2, zorder=2.5)
    ax1.set_ylabel("Story Points")
    ax1.set_ylim(0, df['Estimated Story Points'].max() * 1.1)  # Add 10% padding to max

    # Plot on secondary Y-axis (Percentage of Stories Unestimated)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Unestimated Stories (%)'], color='red', label='Percentage of Stories Unestimated', linestyle='--', zorder=2)
    ax2.set_ylabel("Percentage of Stories Unestimated")
    ax2.set_ylim(0, 100)  # Set scale from 0 to 100

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Customize and display the plot
    plt.title(f"Enhanced Version Report for {version_name}")
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
            generate_version_report(selected_version.name)
        except (IndexError, ValueError):
            print("Invalid selection.")
