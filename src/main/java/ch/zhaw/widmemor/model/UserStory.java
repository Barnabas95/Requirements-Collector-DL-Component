package ch.zhaw.widmemor.model;

/**
 * This class is a representation of a user story as defined by the Storyscreen application for ease of transmitting it
 * back to the Storyscreen application
**/
public class UserStory {

    private Integer projectId;
    private String description;

    public UserStory(Integer projectId, String description) {
        this.projectId = projectId;
        this.description = description;
    }

    public Integer getProjectId() {
        return projectId;
    }

    public void setProjectId(Integer projectId) {
        this.projectId = projectId;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }
}
