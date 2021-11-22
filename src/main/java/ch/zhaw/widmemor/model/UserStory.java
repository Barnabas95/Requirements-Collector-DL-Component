package ch.zhaw.widmemor.model;

/**
 * This class is a representation of a user story as defined by the Storyscreen application for ease of transmitting it
 * back to the Storyscreen application
 **/
public class UserStory {

    private Integer projectId;
    private String goal;
    private String reason;

    //Wil be filled with placeholder as this is to be filled once the user story is created in the Storyscreen
    private int estimate;
    private int priority;

    //This is the "result" from the evaluation/algorithm
    private String shortDescription;
    //Storyscreen uses a dropdown list. Probably best to send found UserRole as String and then convert it while
    // processing it in the Storyscreen
    private int userRole;
    //Filled with placeholder
    private int themeId;


    public UserStory(Integer projectId, String shortDescription) {
        this.projectId = projectId;
        this.goal = "Goal";
        this.reason = "Reason";
        this.estimate = 1;
        this.priority = 1;
        this.shortDescription = shortDescription;
        this.userRole = 1;
    }

    public Integer getProjectId() {
        return projectId;
    }

    public void setProjectId(Integer projectId) {
        this.projectId = projectId;
    }

    public String getShortDescription() {
        return shortDescription;
    }

    public void setShortDescription(String shortDescription) {
        this.shortDescription = shortDescription;
    }

    public String getGoal() {
        return goal;
    }

    public void setGoal(String goal) {
        this.goal = goal;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public int getEstimate() {
        return estimate;
    }

    public void setEstimate(int estimate) {
        this.estimate = estimate;
    }

    public int getPriority() {
        return priority;
    }

    public void setPriority(int priority) {
        this.priority = priority;
    }

    public int getUserRole() {
        return userRole;
    }

    public void setUserRole(int userRole) {
        this.userRole = userRole;
    }

    public int getThemeId() {
        return themeId;
    }

    public void setThemeId(int themeId) {
        this.themeId = themeId;
    }
}
